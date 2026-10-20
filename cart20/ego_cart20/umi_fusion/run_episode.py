#!/usr/bin/env python3
"""[2026-10-13] Real-data UMI pose fusion for one HandUMI episode -> <episode>/derived/umi_fusion/{fused.npz, qa.json}.

Per side (wrist-frame timeline = wrist video capture_ns of frame_idx):
  MASt3R restart_v1 segments (~/c8/robotlike/runs_restart/<tag>/ss1.csv, own Sim3 frame per segment)
  + Orbbec canonical rigid point (<episode>/derived/body_track/canonical.npz p_W_*, table world; valid & not swap-flagged)
  + wrist gyro (sensors.mcap /<side>/imu, bias from the first second = REC hold), Kalibr camera<-IMU rotation (export settings)
  -> fusion.fuse_side: per-segment Sim3 + shared r_c (robust), IMU rotation bridge, Orbbec position in non-MASt3R frames.
TCP: T_W_tcp = T_W_c @ X_cam_tcp (camera_tcp_v2) @ X_TOOL (reBot dataset-TCP axes, as hfold_export).
QA: coverage (mast3r / orbbec / both / either / full_6dof + longest gaps), alignment residual per segment, inter-gripper
continuity on a 15 Hz common timeline, episode gate.
usage: run_episode.py <episode dir> [--r-c-prior-json f]
"""
import argparse, csv, json, pathlib, re, sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ego_cart20.sources.hfold_export import X_TOOL, x_cam_tcp, read_mcap                     # noqa: E402
from ego_cart20.umi_fusion.fusion import SideInputs, fuse_side, _interp_body                # noqa: E402
from ego_cart20.umi_fusion.qa import coverage, episode_gate, overlap_error                  # noqa: E402
from ego_cart20.umi_fusion.relpose import inter_gripper_relative_pose, resample_poses       # noqa: E402

H = pathlib.Path.home(); RR = H / "c8/robotlike/runs_restart"; EXP = H / "c8/robotlike/export"
SWAP_FLAG_BITS = 0                                         # canonical flags are advisory; validity already excludes INVALID frames


def tag_of(ep, side):
    ts = ep.parent.name.split("_", 1)[1]; return f"{ts}_{ep.name.split('_')[1]}_{side}"


def load_restart(tag):
    rows = list(csv.DictReader(open(RR / tag / "ss1.csv"))); n = max(int(r["frame_idx"]) for r in rows) + 1
    seg = np.full(n, -1); p = np.zeros((n, 3)); R = np.tile(np.eye(3), (n, 1, 1))
    for r in rows:
        if r["is_lost"] != "false": continue
        q = [float(r[k]) for k in ("q_x", "q_y", "q_z", "q_w")]
        if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 0.5: continue
        i = int(r["frame_idx"]); seg[i] = int(r["segment"]); p[i] = [float(r["x"]), float(r["y"]), float(r["z"])]; R[i] = Rot.from_quat(q).as_matrix()
    return seg, p, R


def imu(ep, side):
    from mcap.reader import make_reader
    t, w = [], []
    with open(ep / "sensors.mcap", "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages(topics=[f"/{side}/imu"]):
            d = json.loads(msg.data); t.append(d["host_receive_ns"]); w.append([d["gx"], d["gy"], d["gz"]])
    t = np.array(t, np.int64); w = np.array(w, float); o = np.argsort(t); return t[o], w[o]


def R_cb_kalibr(tag):
    m = re.search(r"IMU\.T_b_c1:.*?data:\s*\[([^\]]+)\]", (EXP / tag / "orbslam_setting.yaml").read_text(), re.S)
    return np.array([float(v) for v in m.group(1).split(",")]).reshape(4, 4)[:3, :3].T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--r-c-prior-json"); a = ap.parse_args()
    ep = pathlib.Path(a.episode).resolve(); out = ep / "derived/umi_fusion"; out.mkdir(parents=True, exist_ok=True)
    fm, _ = read_mcap(ep); cz = np.load(ep / "derived/body_track/canonical.npz", allow_pickle=True)
    prior = json.load(open(a.r_c_prior_json)) if a.r_c_prior_json else {}
    Xct, _ = x_cam_tcp(); res = {}; qa = dict(episode=str(ep), sides={})
    t0 = None
    for side in ("left", "right"):
        tag = tag_of(ep, side); seg, p, R = load_restart(tag); vf = fm[f"{side}_wrist"]
        cap = dict(zip(vf[:, 0].tolist(), vf[:, 1].tolist())); n = len(seg)
        keep = np.array([i in cap for i in range(n)]); idx = np.flatnonzero(keep)
        t_ns = np.array([cap[i] for i in idx], np.int64); t0 = t0 or int(min(t_ns[0], cz["t_ns"][0]))
        gt, gw = imu(ep, side); bias = gw[gt < gt[0] + 1e9].mean(0)
        ok = cz[f"valid_{side}"].astype(bool)
        inp = SideInputs(t=(t_ns - t0) / 1e9, seg=seg[idx], p_seg=p[idx], R_seg=R[idx], body_t=(cz["t_ns"] - t0) / 1e9,
                         body_p=cz[f"p_W_{side}"], body_ok=ok, gyro_t=(gt - t0) / 1e9, gyro_w=gw - bias, R_cb=R_cb_kalibr(tag),
                         r_c_prior=np.array(prior[side]) if side in prior else None)
        f = fuse_side(inp)
        T_tcp = f.T_W_c @ Xct @ X_TOOL
        pb, okb = _interp_body(inp, inp.t)
        m3 = f.rotation_source == "MAST3R"; full = f.pose_source != "INVALID"
        p_pred_body = f.T_W_c[:, :3, 3] + np.einsum("nij,j->ni", f.T_W_c[:, :3, :3], f.r_c if f.r_c is not None else np.zeros(3))
        err = overlap_error(inp.t, p_pred_body, pb, m3 & okb, R_W=f.T_W_c[:, :3, :3])
        step = np.linalg.norm(np.diff(T_tcp[:, :3, 3], axis=0), axis=1) * 1e3; step[~(full[1:] & full[:-1])] = np.nan
        dR = np.degrees(Rot.from_matrix(np.einsum("nji,njk->nik", T_tcp[:-1, :3, :3], T_tcp[1:, :3, :3])).magnitude()); dR[~(full[1:] & full[:-1])] = np.nan
        cov = coverage(inp.t, m3, okb, full)
        qa["sides"][side] = dict(tag=tag, coverage=cov, alignment=f.fit, overlap_body_error=err,
                                 tcp_step_mm=dict(p99=round(float(np.nanpercentile(step, 99)), 1), max=round(float(np.nanmax(step)), 1)),
                                 tcp_rot_step_deg=dict(p99=round(float(np.nanpercentile(dR, 99)), 2), max=round(float(np.nanmax(dR)), 2)),
                                 sources={k: int((f.pose_source == k).sum()) for k in ("MAST3R_ORBBEC", "ORBBEC_IMU", "INVALID")},
                                 gate=episode_gate(cov, err))
        res[side] = dict(t=inp.t, T_tcp=T_tcp, full=full, f=f)
        np.savez_compressed(out / f"fused_{side}.npz", t_ns=t_ns, frame_idx=idx, T_W_c=f.T_W_c, T_W_tcp=T_tcp, pose_source=f.pose_source.astype(str),
                            position_source=f.position_source.astype(str), rotation_source=f.rotation_source.astype(str),
                            confidence_position=f.confidence_position, confidence_rotation=f.confidence_rotation, imu_gap_s=f.imu_gap_s,
                            r_c=f.r_c if f.r_c is not None else np.full(3, np.nan))
    # inter-gripper on a 15 Hz common timeline
    L, Rr = res["left"], res["right"]; tc = np.arange(max(L["t"][0], Rr["t"][0]), min(L["t"][-1], Rr["t"][-1]), 1 / 15)
    TL, vL = resample_poses(L["t"], L["T_tcp"], L["full"], tc); TR, vR = resample_poses(Rr["t"], Rr["T_tcp"], Rr["full"], tc)
    Tlr = inter_gripper_relative_pose(TL, TR); v = vL & vR; d = np.linalg.norm(Tlr[:, :3, 3], axis=1)
    js = np.linalg.norm(np.diff(Tlr[:, :3, 3], axis=0), axis=1) * 1e3; js[~(v[1:] & v[:-1])] = np.nan
    qa["inter_gripper"] = dict(valid_frac=round(float(v.mean()), 4), distance_m=dict(p05=round(float(np.percentile(d[v], 5)), 3), p50=round(float(np.percentile(d[v], 50)), 3),
                               p95=round(float(np.percentile(d[v], 95)), 3)), jump_mm_15hz=dict(p95=round(float(np.nanpercentile(js, 95)), 1), max=round(float(np.nanmax(js)), 1)))
    np.savez_compressed(out / "inter_gripper_15hz.npz", t=tc, T_LR=Tlr, valid=v, T_W_left=TL, T_W_right=TR)
    (out / "qa.json").write_text(json.dumps(qa, indent=1, default=float)); print(json.dumps(qa, indent=1, default=float))


if __name__ == "__main__":
    main()
