"""DPVO wrist-tracker evaluation against the Orbbec-only UMI body track (2026-10-15, Agent C software-T265 direction).

Inputs per side: <run>.npz from dpvo_eval_run.py (online + final camera poses, per-frame latency, GPU memory) and the episode's
derived/body_track/canonical.npz (Orbbec-only body position + orientation per wrist frame, independent of the wrist camera).
DPVO is monocular (no scale) and the camera is not the body centroid, so each trajectory is aligned with a Sim(3) + lever arm
(p_ref = s R c + t + R R_cam r), fitted on the whole episode (ATE) and on the FIRST 3 s only (drift: what a live tracker sees).

Reported (the list the user asked for): fps / latency / GPU memory (from the run), tracking losses (online jumps > 3 m/s or
> 30 deg/frame, or a frozen pose; 1-2 frame stalls and >= 5 frame collapses counted separately), rotation drift split into yaw / tilt, ATE, drift at 10/30 s after a 3 s alignment, error right after still periods (restart),
rotation error during fast rotation (> 120 deg/s), online vs final.
Initial alignment is Sim(3) (+ lever arm) on the first 3 s; that scale is then FIXED for the 10/30 s drift. Also scale(t)/scale_init
on sliding 3 s windows, and rotation drift reported separately against two references (Orbbec body orientation and the IMU AHRS),
because the Orbbec orientation is not treated as ground truth.
    python scripts/dpvo_eval_metrics.py <runs_dir> [--session <UMIalign session dir>]"""
from __future__ import annotations
import argparse
import json
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation as R

SESSION = pathlib.Path.home() / "ego_collector/datasets/human_handumi_raw/UMIalign/UMIalign_20261014_112920"


def umeyama(src, dst, scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0); A, B = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(B.T @ A / len(src)); d = np.sign(np.linalg.det(U @ Vt)); Dg = np.diag([1, 1, d])
    Rm = U @ Dg @ Vt; s = (S * np.diag(Dg)).sum() / A.var(0).sum() if scale else 1.0
    return s, Rm, mu_d - s * Rm @ mu_s


def align_lever(c, Rc, p, iters=15):
    """p_i ~ s R c_i + t + R Rc_i r   ->   (s, R, t, r). Alternates Umeyama (r fixed) and linear LS for (t, r)."""
    r = np.zeros(3); s, Rm, t = umeyama(c, p)
    for _ in range(iters):
        s, Rm, t = umeyama(c + np.einsum("nij,j->ni", Rc, r) / max(s, 1e-9), p)
        A = np.concatenate([np.tile(np.eye(3), (len(c), 1)), np.concatenate([Rm @ X for X in Rc])], 1)
        b = (p - s * c @ Rm.T).reshape(-1); x, *_ = np.linalg.lstsq(A, b, rcond=None); t, r = x[:3], x[3:]
    return s, Rm, t, r


def chordal_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, 0)); d = np.sign(np.linalg.det(U @ Vt)); return U @ np.diag([1, 1, d]) @ Vt


def fit_rot(Rc, Rref, Rw0=None, k=5, iters=5):
    """Rref_i ~ Rw Rc_i Rcb  (Rw: DPVO world -> reference world, Rcb: camera -> body/IMU mount).
    Closed form first: relative rotations conjugate, Rref_i Rref_j^T = Rw (Rc_i Rc_j^T) Rw^T, so their rotation vectors satisfy
    v_ref = Rw v_dpvo -> Kabsch on those pairs (k-frame steps, > 2 deg). Then Rcb = chordal mean, then a few alternations."""
    a = R.from_matrix(np.einsum("nij,nkj->nik", Rc[k:], Rc[:-k])).as_rotvec(); b = R.from_matrix(np.einsum("nij,nkj->nik", Rref[k:], Rref[:-k])).as_rotvec()
    m = (np.linalg.norm(a, axis=1) > np.radians(2)) & (np.linalg.norm(b, axis=1) > np.radians(2))
    if m.sum() >= 5:
        U, _, Vt = np.linalg.svd(b[m].T @ a[m]); d = np.sign(np.linalg.det(U @ Vt)); Rw = U @ np.diag([1, 1, d]) @ Vt
    else: Rw = np.eye(3) if Rw0 is None else Rw0
    for _ in range(iters):
        Rcb = chordal_mean(np.einsum("nji,kj,nkl->nil", Rc, Rw, Rref))            # Rc^T Rw^T Rref
        Rw = chordal_mean(np.einsum("nij,kj,nlk->nil", Rref, Rcb, Rc))             # Rref Rcb^T Rc^T
    return Rw, Rcb


def rot_err_deg(Rw, Rc, Rcb, Rref):
    E = np.einsum("ij,njk,kl,nml->nim", Rw, Rc, Rcb, Rref)                         # (Rw Rc Rcb) Rref^T
    return np.degrees(np.linalg.norm(R.from_matrix(E).as_rotvec(), axis=1))


def yaw_tilt_err_deg(Rw, Rc, Rcb, Rref):
    """World-frame error E = (Rw Rc Rcb) Rref^T split into tilt (roll/pitch: where E sends 'up') and yaw (about 'up').
    Both reference worlds are gravity-aligned (z up), so tilt is checkable by gravity and yaw is the part an IMU drifts in."""
    E = np.einsum("ij,njk,kl,nml->nim", Rw, Rc, Rcb, Rref)
    tilt = np.degrees(np.arccos(np.clip(E[:, 2, 2], -1, 1)))
    yaw = np.degrees(np.abs(np.arctan2(E[:, 1, 0] - E[:, 0, 1], E[:, 0, 0] + E[:, 1, 1])))
    return yaw, tilt


def apply(s, Rm, t, r, c, Rc): return s * c @ Rm.T + t + np.einsum("ij,njk,k->ni", Rm, Rc, r)


def ang_speed(Rs, dt):
    return np.r_[0, np.degrees(np.linalg.norm(R.from_matrix(np.einsum("nji,njk->nik", Rs[:-1], Rs[1:])).as_rotvec(), axis=1)) / dt]


def best_lag(a, b, max_lag=10):
    best = (0, -2)
    for L in range(-max_lag, max_lag + 1):
        x, y = (a[L:], b[:len(b) - L]) if L >= 0 else (a[:L], b[-L:])
        n = min(len(x), len(y)); x, y = x[:n], y[:n]; m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 50:
            cc = np.corrcoef(x[m], y[m])[0, 1]
            if cc > best[1]: best = (L, cc)
    return best


def evaluate(run: pathlib.Path, session: pathlib.Path) -> dict:
    name = run.stem; ep = int(name.split("_")[2]); side = name.split("_")[3]
    z = np.load(run); ref = np.load(session / f"episode_{ep:06d}/derived/body_track/canonical.npz")
    t = (ref["t_ns"] - ref["t_ns"][0]) / 1e9; P = ref[f"p_W_{side}"]; Rb = ref[f"R_world_body_{side}"]; ok = ref[f"valid_{side}"] & np.all(np.isfinite(P), 1)
    dt = float(np.median(np.diff(t))); out = dict(run=name, frames=int(len(z["lat_ms"])), gpu_mem_mb=float(z["mem_mb"]),
                                               fps=round(1000 / float(np.mean(z["lat_ms"])), 1), lat_ms_p50=round(float(np.median(z["lat_ms"])), 1),
                                               lat_ms_p95=round(float(np.percentile(z["lat_ms"], 95)), 1))
    for kind in ("online", "final"):
        if kind == "online": idx = z["online_idx"].astype(int); Q = z["online"]
        else: idx = np.asarray(z["final_t"]).astype(int); Q = z["final"]
        if len(Q) < 50: out[kind] = dict(error="too few poses"); continue
        c = Q[:, :3]; Rc = R.from_quat(Q[:, 3:7]).as_matrix()
        # time offset by rotation-speed correlation (frame index k of the wrist video vs canonical index)
        n = len(t); w_ref = ang_speed(Rb, dt); w_dp = np.full(n, np.nan); kk = idx[idx < n]
        w_full = ang_speed(Rc, dt); w_dp[kk] = w_full[: len(kk)]
        lag, cc = best_lag(w_ref, w_dp); ri = idx + lag; m = (ri >= 0) & (ri < n); m[m] &= ok[ri[m]]
        c, Rc, ri = c[m], Rc[m], ri[m]; p = P[ri]
        # losses: implausible online jumps or a frozen pose (DPVO never says "lost")
        s0, R0, t0_, r0 = align_lever(c, Rc, p); cw = apply(s0, R0, t0_, r0, c, Rc)
        step = np.linalg.norm(np.diff(cw, axis=0), axis=1) / (np.diff(ri) * dt); rstep = ang_speed(Rc, dt)[1:]
        frozen = (np.linalg.norm(np.diff(c, axis=0), axis=1) < 1e-9) & (rstep < 1e-6)
        bad = (step > 3.0) | (rstep > 30 / dt) | frozen
        runs_bad, cur = [], 0
        for b_ in bad:
            if b_: cur += 1
            elif cur: runs_bad.append(cur); cur = 0
        if cur: runs_bad.append(cur)
        stalls = int(sum(1 for L in runs_bad if L <= 2)); collapses = int(sum(1 for L in runs_bad if L >= 5))   # 3-4 frames: counted in losses only
        losses = len(runs_bad)
        e = np.linalg.norm(cw - p, axis=1) * 1000
        Rw0, Rcb0 = fit_rot(Rc, Rb[ri], R0); rot_err = rot_err_deg(Rw0, Rc, Rcb0, Rb[ri])     # global fit (camera mount solved)
        # drift: align on the first 3 s only, then error at +10 s / +30 s (what a live tracker accumulates)
        tt = t[ri]; first = tt <= tt[0] + 3.0
        drift = {}
        if first.sum() > 30:
            sA, RA, tA, rA = align_lever(c[first], Rc[first], p[first]); cA = apply(sA, RA, tA, rA, c, Rc); eA = np.linalg.norm(cA - p, axis=1) * 1000
            for H in (10, 30):
                sel = np.abs(tt - (tt[0] + H)) < 0.5
                drift[f"{H}s_mm"] = round(float(np.median(eA[sel])), 1) if sel.any() else None
            drift["scale_ratio_first3s_vs_all"] = round(float(sA / s0), 3)
            # scale(t)/scale_init: Sim(3)+lever refit on sliding 3 s windows (1 s step) vs the first-3 s scale
            sc = []
            for w0 in np.arange(tt[0], tt[-1] - 3.0, 1.0):
                sel = (tt >= w0) & (tt < w0 + 3.0)
                if sel.sum() > 30 and np.ptp(p[sel], 0).max() > 0.03:                # enough motion for a scale
                    sc.append((w0 - tt[0], align_lever(c[sel], Rc[sel], p[sel])[0] / sA))
            if sc:
                S = np.array(sc); at = lambda H: (round(float(S[np.argmin(np.abs(S[:, 0] - H)), 1]), 3) if np.abs(S[:, 0] - H).min() < 1.5 else None)
                drift["scale_t_over_init"] = {"10s": at(10), "20s": at(20), "30s": at(30), "min": round(float(S[:, 1].min()), 3), "max": round(float(S[:, 1].max()), 3), "n_windows": len(S)}
            # rotation drift, fitted on the first 3 s only, vs TWO references (Orbbec body orientation, IMU AHRS)
            for refname, Rref in (("orbbec", Rb[ri]), ("imu", ref[f"R_gravity_imu_{side}"][ri])):
                Rw, Rcb = fit_rot(Rc[first], Rref[first], RA); er = rot_err_deg(Rw, Rc, Rcb, Rref); ey, et = yaw_tilt_err_deg(Rw, Rc, Rcb, Rref)
                for H in (10, 30):
                    sel = np.abs(tt - (tt[0] + H)) < 0.5
                    drift[f"rot_{refname}_{H}s_deg"] = round(float(np.median(er[sel])), 1) if sel.any() else None
                    drift[f"yaw_{refname}_{H}s_deg"] = round(float(np.median(ey[sel])), 1) if sel.any() else None
                    drift[f"tilt_{refname}_{H}s_deg"] = round(float(np.median(et[sel])), 1) if sel.any() else None
        # restart after stillness: reference speed < 2 cm/s for >= 2 s, then error in the 2 s after motion resumes
        v = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1) / dt]; still = v < 0.02; starts, run_len = [], 0
        for i in range(len(still)):
            run_len = run_len + 1 if still[i] else 0
            if i + 1 < len(still) and run_len * dt >= 2.0 and not still[i + 1]: starts.append((i + 1, run_len * dt))
        rs = []
        for i0, L in starts:
            selb = (ri >= i0 - int(1 / dt)) & (ri < i0); sela = (ri >= i0) & (ri < i0 + int(2 / dt))
            if selb.any() and sela.any(): rs.append(dict(still_s=round(L, 1), err_before_mm=round(float(np.median(e[selb])), 1), err_after_max_mm=round(float(e[sela].max()), 1)))
        Rw1, Rcb1 = fit_rot(Rc, ref[f"R_gravity_imu_{side}"][ri], R0); rot_err_imu = rot_err_deg(Rw1, Rc, Rcb1, ref[f"R_gravity_imu_{side}"][ri])
        fast = w_ref[ri] > 120
        out[kind] = dict(n=int(len(c)), lag_frames=int(lag), lag_corr=round(float(cc), 2), coverage=round(len(c) / max(1, ok.sum()), 3),
                         losses=losses, stalls_1_2f=stalls, collapses_5f=collapses, longest_bad_frames=int(max(runs_bad, default=0)), ate_mm_p50=round(float(np.median(e)), 1), ate_mm_p90=round(float(np.percentile(e, 90)), 1),
                         rot_err_deg_p50=round(float(np.median(np.abs(rot_err))), 1), rot_err_deg_p90=round(float(np.percentile(np.abs(rot_err), 90)), 1),
                         rot_err_imu_deg_p50=round(float(np.median(rot_err_imu)), 1), rot_err_imu_deg_p90=round(float(np.percentile(rot_err_imu, 90)), 1),
                         fast_rot_frames=int(fast.sum()), fast_rot_err_deg_p90=round(float(np.percentile(np.abs(rot_err[fast]), 90)), 1) if fast.sum() > 5 else None,
                         lever_r_m=np.round(r0 / max(s0, 1e-9), 3).tolist(), drift=drift, restarts=rs[:6],
                         longest_still_s=round(max([L for _, L in starts], default=0.0), 1))
    out["verdict"] = verdict(out)
    return out


# 10-15 user decision rules: a FIXED scale error (5-10 %) is fine (it feels like a gain); scale that WANDERS over time is a
# fail; no jump on restart; small loss / rotation drift in fast rotation; low latency; hard episodes must not collapse.
GATES = dict(scale_band=0.10, restart_jump_mm=30.0, fast_rot_p90_deg=10.0, rot_drift_30s_deg=10.0, lat_p95_ms=50.0, min_coverage=0.9, max_losses=0)


def verdict(o: dict) -> dict:
    v, on = {}, o.get("online", {})
    if "error" in on: return dict(overall="FAIL", reason=on["error"])
    st = (on.get("drift") or {}).get("scale_t_over_init")
    v["scale_stable"] = None if not st else bool(1 - GATES["scale_band"] <= st["min"] and st["max"] <= 1 + GATES["scale_band"])
    rs = on.get("restarts") or []
    v["restart_no_jump"] = None if not rs else bool(all(r["err_after_max_mm"] - r["err_before_mm"] <= GATES["restart_jump_mm"] for r in rs))
    v["fast_rotation_ok"] = None if on.get("fast_rot_err_deg_p90") is None else bool(on["fast_rot_err_deg_p90"] <= GATES["fast_rot_p90_deg"])
    rd = [x for x in ((on.get("drift") or {}).get("rot_orbbec_30s_deg"), (on.get("drift") or {}).get("rot_imu_30s_deg")) if x is not None]
    v["rot_drift_ok"] = None if not rd else bool(min(rd) <= GATES["rot_drift_30s_deg"])        # either reference agreeing is enough
    v["latency_ok"] = bool(o["lat_ms_p95"] <= GATES["lat_p95_ms"])
    v["no_collapse"] = bool(on.get("coverage", 0) >= GATES["min_coverage"] and on.get("collapses_5f", 99) <= GATES["max_losses"])   # short stalls are logged, not failed
    known = [x for x in v.values() if x is not None]
    v["overall"] = "PASS" if known and all(known) else "FAIL"; v["untested"] = [k for k, x in v.items() if x is None]
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs"); ap.add_argument("--session", default=str(SESSION)); a = ap.parse_args()
    res = [evaluate(f, pathlib.Path(a.session)) for f in sorted(pathlib.Path(a.runs).glob("*.npz"))]
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n판정 (온라인 자세 기준):")
    for r in res: print(f"  {r['run']:32s} {r['verdict']['overall']:4s} " + " ".join(f"{k}={'통과' if x else '실패'}" for k, x in r["verdict"].items() if isinstance(x, bool)) + (f" | 미검증 {r['verdict']['untested']}" if r['verdict'].get('untested') else ""))
    (pathlib.Path(a.runs) / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
