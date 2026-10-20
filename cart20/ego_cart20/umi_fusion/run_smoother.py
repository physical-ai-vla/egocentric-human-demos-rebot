#!/usr/bin/env python3
"""[2026-10-13] Batch-smoother driver for one HandUMI episode -> <episode>/derived/umi_fusion/smoothed_{left,right}.npz + smooth_qa.json
Inputs: MASt3R restart_v1 (runs_restart/<tag>/ss1.csv), Orbbec canonical.npz (p_W_*, valid_*, R_world_body_* = AHRS + solved yaw,
already in the table world), wrist IMU (gyro + accel), Kalibr camera<-IMU rotation.  Wrist frames lag the IMU by 1 frame
(canonical eval, gyro vs MASt3R rate r = 0.94-0.98) -> wrist time = capture_ns - WRIST_LAG_S.
Two passes: pass 1 robust; pass 2 drops Orbbec frames with |res| > gate (LEFT_RIGHT_MISMATCH / outlier) and re-solves.
usage: run_smoother.py <episode dir> [--sides left right]
"""
import argparse, dataclasses, json, pathlib, sys, time

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ego_cart20.sources.hfold_export import X_TOOL, x_cam_tcp, read_mcap            # noqa: E402
from ego_cart20.umi_fusion.run_episode import R_cb_kalibr, load_restart, tag_of     # noqa: E402
from ego_cart20.umi_fusion.smoother import SmootherCfg, smooth_side                 # noqa: E402
from ego_cart20.umi_fusion.qa import coverage, longest_run_s                        # noqa: E402
from ego_cart20.umi_fusion.relpose import inter_gripper_relative_pose, resample_poses  # noqa: E402
from ego_cart20.sources.trim import trim_end_ns, trim_info                          # noqa: E402

WRIST_LAG_S = 1 / 30.0
RC_PRIOR = {"tips_v001": {"left": (0.007, 0.131, 0.048), "right": (-0.003, 0.131, 0.035)}}   # [2026-10-15] median over 6 eps, slow frames
TAGGATE = {}        # [2026-10-15] side -> bool mask over canonical frames (orbbec_tag_gate.py), True = drop the Orbbec anchor
RCB = {}            # [2026-10-15] side -> R_cb override (--rcb data_v003); empty = Kalibr from the export settings (17.5 / 23.7 deg off, see
                    # ego_collector/configs/calibration/camera_imu_Rcb_data_v003.json)


def imu_full(ep, side):
    from mcap.reader import make_reader
    t, w, a = [], [], []
    with open(ep / "sensors.mcap", "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages(topics=[f"/{side}/imu"]):
            d = json.loads(msg.data); t.append(d["host_receive_ns"]); w.append([d["gx"], d["gy"], d["gz"]]); a.append([d["ax"], d["ay"], d["az"]])
    o = np.argsort(t); return np.array(t, np.int64)[o], np.array(w, float)[o], np.array(a, float)[o]


def interp_rot(tq, ts, Rs, ok):
    ts, Rs = ts[ok], Rs[ok]; tq = np.clip(tq, ts[0], ts[-1]); return Slerp(ts, Rot.from_matrix(Rs))(tq).as_matrix()


def interp_pos(tq, ts, P, ok, max_gap=0.1):
    tv, Pv = ts[ok], P[ok]; j = np.clip(np.searchsorted(tv, tq) - 1, 0, len(tv) - 2)
    w = np.clip((tq - tv[j]) / np.maximum(tv[j + 1] - tv[j], 1e-9), 0, 1); z = (1 - w)[:, None] * Pv[j] + w[:, None] * Pv[j + 1]
    near = ((tv[j + 1] - tv[j]) <= max_gap) & (tq >= tv[0]) & (tq <= tv[-1]); return z, near


def camera_delay(t_cam, seg, Rm, gt, gw, R_cb, max_lag=5):
    """[2026-10-13] per-episode camera-IMU delay: correlate |w| from MASt3R (consecutive same-segment frames) with |w| from the
    gyro, over lags of -max_lag..max_lag frames. -> (delay_s, r, table). Positive delay = camera frames are late."""
    k = np.flatnonzero((seg[:-1] >= 0) & (seg[:-1] == seg[1:])); dt = np.diff(t_cam)[k]
    wm = Rot.from_matrix(np.einsum("nji,njk->nik", Rm[k], Rm[k + 1])).magnitude() / dt
    gmag = np.linalg.norm(gw, axis=1); fps = 1 / np.median(np.diff(t_cam)); tab = {}
    for L in range(-max_lag, max_lag + 1):
        tm = 0.5 * (t_cam[k] + t_cam[k + 1]) - L / fps; wg = np.interp(tm, gt, gmag)
        tab[L] = float(np.corrcoef(wm, wg)[0, 1]) if len(k) > 30 else np.nan
    L = max(tab, key=lambda x: -np.inf if not np.isfinite(tab[x]) else tab[x]); return L / fps, tab[L], tab


def run_side(ep, side, fm, cz, cfg, t0):
    tag = tag_of(ep, side); seg, pm, Rm = load_restart(tag); vf = fm[f"{side}_wrist"]; cap = dict(zip(vf[:, 0].tolist(), vf[:, 1].tolist()))
    idx = np.array([i for i in range(len(seg)) if i in cap]); t_raw = (np.array([cap[i] for i in idx]) - t0) / 1e9
    seg, pm, Rm = seg[idx], pm[idx], Rm[idx]
    gt, gw, ga = imu_full(ep, side); gt = (gt - t0) / 1e9; gw = gw - gw[gt < gt[0] + 1.0].mean(0)
    R_cb = RCB.get(side, R_cb_kalibr(tag)); R_bc = R_cb.T
    delay, r_corr, tab = camera_delay(t_raw, seg, Rm, gt, gw, R_cb)
    use = delay if (np.isfinite(r_corr) and r_corr >= 0.8) else WRIST_LAG_S          # fall back to the measured 1-frame lag
    t = t_raw - use
    tb = (cz["t_ns"] - t0) / 1e9; okb = cz[f"valid_{side}"].astype(bool)
    if side in TAGGATE: okb &= ~TAGGATE[side]                                        # [2026-10-15] anchor not on the unit (AprilTag gate)
    Rwb = cz[f"R_world_body_{side}"]; okR = np.isfinite(Rwb).all(axis=(1, 2)) & (np.abs(np.linalg.det(Rwb) - 1) < 1e-3)
    R0 = interp_rot(t, tb, Rwb, okR) @ R_bc
    z, near = interp_pos(t, tb, cz[f"p_W_{side}"], okb)
    rc0 = np.array(cfg.r_c0); p0 = np.where(near[:, None], z - np.einsum("nij,j->ni", R0, rc0), np.nan)
    good = np.isfinite(p0[:, 0]); p0 = np.array([np.interp(t, t[good], p0[good, k]) for k in range(3)]).T
    t_run = time.time()
    r1 = smooth_side(t, R0, p0, seg, Rm, pm, gt, gw, R_cb, gt, ga, z, near, cfg)
    keep = ~(np.nan_to_num(r1["orb_res_mm"], nan=0) > cfg.gate_mm)
    drop = {(a, b) for a, b, e in r1["pair_res"] if e > cfg.m3_gate_sigma}
    r2 = smooth_side(t, r1["R"], r1["p"], seg, Rm, pm, gt, gw, R_cb, gt, ga, z, near, cfg, orb_keep=keep, pair_drop=drop, scale_init=r1["scale_log"])
    r2["t"] = t; r2["seg"] = seg; r2["near"] = near; r2["rejected"] = near & ~keep; r2["sec"] = time.time() - t_run; r2["tag"] = tag
    r2["camera_delay"] = dict(estimated_ms=round(delay * 1e3, 1), corr=round(r_corr, 3), used_ms=round(use * 1e3, 1),
                              corr_by_lag_frames={int(k): round(v, 3) for k, v in tab.items()})
    return r2


def cut_side(r, t_max):
    """[2026-10-14] drop frames after the episode end trim (SPACE reach). The solve itself used the whole take; only what
    is reported and saved stops at t_max (s, same origin as r["t"])."""
    k = r["t"] <= t_max; r = dict(r)
    for key in ("t", "seg", "near", "rejected", "R", "p", "sw", "orb_res_mm", "orb_used"): r[key] = r[key][k]
    return r


def qa_side(r, cfg):
    t, seg, near, rej = r["t"], r["seg"], r["near"], r["rejected"]; used = near & ~rej; res = r["orb_res_mm"]
    m3 = seg >= 0
    dist_orb = np.full(len(t), np.inf)                                     # time to the nearest accepted Orbbec frame
    if used.any(): tu = t[used]; j = np.clip(np.searchsorted(tu, t), 1, len(tu) - 1); dist_orb = np.minimum(np.abs(tu[j - 1] - t), np.abs(tu[j] - t))
    full = m3 | (dist_orb <= 0.5)                                          # pose constrained by MASt3R or by a nearby Orbbec anchor
    full &= dist_orb <= 3.0                                                # [2026-10-13 user] anchor-free > 3 s -> INVALID (no long extrapolation)
    T = np.tile(np.eye(4), (len(t), 1, 1)); T[:, :3, :3] = r["R"]; T[:, :3, 3] = r["p"]
    step = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) * 1e3
    fast = np.flatnonzero(step / np.maximum(np.diff(t), 1e-3) / 1e3 > 2.5)  # implausible speed -> INVALID +-3 frames
    for k in fast: full[max(k - 3, 0):k + 5] = False
    dR = np.degrees(Rot.from_matrix(np.einsum("nji,njk->nik", T[:-1, :3, :3], T[1:, :3, :3])).magnitude())
    bnd = np.flatnonzero((np.diff(seg) != 0) & (seg[1:] >= 0) & (seg[:-1] >= 0))
    segs = []
    for s in np.unique(seg[seg >= 0]):
        m = seg == s; o = m & used
        segs.append(dict(segment=int(s), dur_s=round(float(m.sum() / 30), 1), orbbec_frames=int(o.sum()), rejected=int((m & rej).sum()),
                         res_mm_p50=round(float(np.nanmedian(res[o])), 1) if o.any() else None,
                         scale=round(float(np.exp(np.median([r["scale_log"][r["sw"][k]] for k in np.flatnonzero(m)]))), 4),
                         status="ACCEPTED" if o.sum() >= 10 else "WEAK_NO_ORBBEC"))
    return dict(tag=r["tag"], seconds=round(r["sec"], 1), info=r["info"], camera_delay=r["camera_delay"], speed_invalidated_frames=int(len(fast)), r_c=np.round(r["r_c"], 4).tolist(), bias=np.round(r["bias"], 5).tolist(),
                coverage=coverage(t, m3, used, full),
                e_p_mm=dict(p50=round(float(np.nanpercentile(res[used], 50)), 1), p90=round(float(np.nanpercentile(res[used], 90)), 1),
                            p95=round(float(np.nanpercentile(res[used], 95)), 1)) if used.any() else None,
                orbbec_rejected=dict(frames=int(rej.sum()), frac_of_orbbec=round(float(rej.sum() / max(near.sum(), 1)), 4), flag="LEFT_RIGHT_MISMATCH/outlier"),
                step_mm=dict(p99=round(float(np.percentile(step, 99)), 1), max=round(float(step.max()), 1),
                             boundary_max=round(float(step[bnd].max()), 1) if len(bnd) else None),
                rot_step_deg=dict(p99=round(float(np.percentile(dR, 99)), 2), max=round(float(dR.max()), 2),
                                  boundary_max=round(float(dR[bnd].max()), 2) if len(bnd) else None),
                segments=segs), T, full


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--sides", nargs="+", default=["left", "right"])
    ap.add_argument("--rcb", choices=["kalibr", "data_v003"], default="kalibr"); ap.add_argument("--out-name", default="umi_fusion")
    ap.add_argument("--rc-prior", choices=["default", "tips_v001"], default="default",
                    help="tips_v001: camera->body-point lever r_c measured from the Orbbec jaw-tape tips (orbbec_tag_check.py, 10-15); "
                         "the free r_c converged to ~[0,-0.1,+0.15] m = camera 20-35 cm off the unit")
    ap.add_argument("--sigma-rc", type=float, default=None)
    ap.add_argument("--tag-gate", action="store_true", help="drop canonical frames flagged by ego_collector/scripts/umi_fusion/orbbec_tag_gate.py"); a = ap.parse_args()
    if a.rcb != "kalibr":
        cal = pathlib.Path.home() / "ego_collector/configs/calibration"
        RCB.update({s: np.load(cal / f"camera_imu_{s}_Rcb_{a.rcb}.npy") for s in ("left", "right")})
    ep = pathlib.Path(a.episode).resolve(); out = ep / "derived" / a.out_name; out.mkdir(parents=True, exist_ok=True)
    fm, _ = read_mcap(ep); cz = np.load(ep / "derived/body_track/canonical.npz", allow_pickle=True); cfg = SmootherCfg()
    if a.tag_gate:
        g = np.load(ep / "derived/body_track/tag_gate.npz"); assert len(g["t_ns"]) == len(cz["t_ns"]); TAGGATE.update(left=g["bad_left"], right=g["bad_right"])
    t0 = int(min(fm["left_wrist"][0, 1], fm["right_wrist"][0, 1], cz["t_ns"][0])); Xct, _ = x_cam_tcp(); qa = dict(episode=str(ep), sides={}, rcb=a.rcb, rc_prior=a.rc_prior, sigma_rc=a.sigma_rc, tag_gate=a.tag_gate, tag_gate_frac={k: round(float(v.mean()), 4) for k, v in TAGGATE.items()}); keepT = {}
    t_end = trim_end_ns(ep); qa["trim"] = trim_info(ep)
    for side in a.sides:
        cs = cfg
        if a.rc_prior != "default": cs = dataclasses.replace(cs, r_c0=tuple(RC_PRIOR[a.rc_prior][side]))
        if a.sigma_rc is not None: cs = dataclasses.replace(cs, sigma_rc_m=a.sigma_rc)
        r = run_side(ep, side, fm, cz, cs, t0)
        if t_end is not None: r = cut_side(r, (t_end - t0) / 1e9)
        q, T, full = qa_side(r, cfg); qa["sides"][side] = q; T_tcp = T @ Xct @ X_TOOL; keepT[side] = (r["t"], T_tcp, full)
        np.savez_compressed(out / f"smoothed_{side}.npz", t=r["t"], T_W_c=T, T_W_tcp=T_tcp, full_6dof=full, segment=r["seg"], orbbec_used=r["orb_used"],
                            orbbec_rejected=r["rejected"], orb_res_mm=r["orb_res_mm"], r_c=r["r_c"], scale_log=r["scale_log"])
        print(json.dumps({side: {k: q[k] for k in ("seconds", "coverage", "e_p_mm", "orbbec_rejected", "step_mm", "rot_step_deg", "r_c", "camera_delay", "speed_invalidated_frames")}}, default=float), flush=True)
    if len(keepT) == 2:
        (tL, TL0, fL), (tR, TR0, fR) = keepT["left"], keepT["right"]; tc = np.arange(max(tL[0], tR[0]), min(tL[-1], tR[-1]), 1 / 15)
        TL, vL = resample_poses(tL, TL0, fL, tc); TR, vR = resample_poses(tR, TR0, fR, tc); Tlr = inter_gripper_relative_pose(TL, TR); v = vL & vR
        js = np.linalg.norm(np.diff(Tlr[:, :3, 3], axis=0), axis=1) * 1e3; js[~(v[1:] & v[:-1])] = np.nan; d = np.linalg.norm(Tlr[:, :3, 3], axis=1)
        qa["inter_gripper"] = dict(valid_frac=round(float(v.mean()), 4), distance_m_p05_p50_p95=np.round(np.percentile(d[v], [5, 50, 95]), 3).tolist(),
                                   jump_mm_15hz_p95=round(float(np.nanpercentile(js, 95)), 1), jump_mm_15hz_max=round(float(np.nanmax(js)), 1))
        np.savez_compressed(out / "smoothed_inter_gripper_15hz.npz", t=tc, T_LR=Tlr, valid=v)
        print(json.dumps({"inter_gripper": qa["inter_gripper"]}))
    (out / "smooth_qa.json").write_text(json.dumps(qa, indent=1, default=float))


if __name__ == "__main__":
    main()
