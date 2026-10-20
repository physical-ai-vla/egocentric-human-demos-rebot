"""Pure-visual MASt3R-SLAM (restart_v1, RGB only) on a trackcam board recording vs the AprilGrid ground truth (2026-10-15).
No IMU anywhere. Per segment the estimate is aligned ONCE and then FROZEN:
  rotation  = mean of R_gt(t) R_est(t)^T over the init window (orientation-only, works even while the camera is still),
  scale + t = least squares on the init window (the first INIT_S seconds of the segment in which the GT camera moved >= 5 cm).
Metrics (user GO criteria in brackets):
  fps            frames / MASt3R wall seconds (single-thread runner, 4090)                       [>= 20-25]
  step10_err_cm  |s R d_est - d_gt| over frame pairs <= 3 s apart whose GT displacement is 8-12 cm  [< 1-2 cm]
  drift30_cm     position error 30 s after the end of an init window (re-fitted at every 5 s start) [< 2 cm]
  fake_rot_cm    translation error over 0.5 s windows in which the GT camera turns > 20 deg          [< 1 cm]
  scale_cv       spread of per-5 s-window scale (Sim3 of displacements) / frozen scale              [small]
  restarts       segments - 1 over the take                                                         [0-1 per 60-90 s]
ORB-SLAM3 (--orb3 <log.csv> from scripts/orb3_driver/orb3_trackcam.cc): segment = map_id, lost = state not OK/OK_KLT, only frames after the
IMU initialisation are scored, and --metric freezes the scale at 1.0 (Mono-Inertial claims metric scale; the fitted scale is reported too).
Also reported: time to first OK pose and to IMU initialisation, processing_ms p50/p95.
    python scripts/trackcam_m3_vs_board.py datasets/trackcam_imu/<ts> <run dir with ss1.csv>
    python scripts/trackcam_m3_vs_board.py datasets/trackcam_imu/<ts> <out dir> --orb3 <log.csv> [--metric]"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

INIT_S = 5.0


def load_est(run: Path):
    rows = list(csv.DictReader(open(run / "ss1.csv")))
    fi = np.array([int(r["frame_idx"]) for r in rows]); seg = np.array([int(r["segment"]) for r in rows])
    lost = np.array([r["is_lost"] == "true" or r["state"] not in ("TRACKING", "INIT") for r in rows])
    P = np.array([[float(r[k]) for k in "xyz"] for r in rows]); Q = np.array([[float(r[k]) for k in ("q_x", "q_y", "q_z", "q_w")] for r in rows])
    return fi, seg, lost, P, Q


FIXED_SCALE = None


def load_orb3(path: Path, t_cam):
    rows = list(csv.DictReader(open(path))); f_of = {int(t): i for i, t in enumerate(t_cam)}
    rows = [r for r in rows if int(r["t_ns"]) in f_of]
    fi = np.array([f_of[int(r["t_ns"])] for r in rows]); st = np.array([int(r["tracking_state"]) for r in rows])
    imu = np.array([int(r["imu_initialized"]) for r in rows]); seg = np.array([int(r["map_id"]) for r in rows])
    ok = np.isin(st, (2, 5)) & (imu == 1)
    P = np.array([[float(r[k]) if r[k] else np.nan for k in "xyz"] for r in rows]); Q = np.array([[float(r[k]) if r[k] else 0.0 for k in ("qx", "qy", "qz", "qw")] for r in rows])
    Q[~ok] = [0, 0, 0, 1]; P[~ok] = 0
    tt = (t_cam[fi] - t_cam[0]) / 1e9; pm = np.array([float(r["processing_ms"]) for r in rows])
    first_ok = tt[np.isin(st, (2, 5))][0] if np.isin(st, (2, 5)).any() else None; first_imu = tt[imu == 1][0] if (imu == 1).any() else None
    extra = dict(first_ok_s=None if first_ok is None else round(float(first_ok), 2), imu_init_s=None if first_imu is None else round(float(first_imu), 2),
                 maps=int(len(np.unique(seg))), relocalizations=int(max(int(r["relocalization_count"]) for r in rows)),
                 lost_frames_after_init=int(((~np.isin(st, (2, 5))) & (tt > (first_imu or 1e9))).sum()),
                 max_lost_ms=round(float(max(float(r["lost_duration_ms"]) for r in rows)), 1),
                 processing_ms=dict(p50=round(float(np.median(pm)), 1), p95=round(float(np.percentile(pm, 95)), 1), max=round(float(pm.max()), 1)),
                 output_hz=round(float(np.isin(st, (2, 5)).sum() / max(1e-3, tt[-1] - tt[0])), 1))
    return fi, seg, ~ok, P, Q, extra


def fit_scale_t(Pe, Pg, Rm):
    A = Pe @ Rm.T; mA, mG = A.mean(0), Pg.mean(0); a, g = A - mA, Pg - mG
    s = float((a * g).sum() / max((a * a).sum(), 1e-12)) if FIXED_SCALE is None else FIXED_SCALE; return s, mG - s * mA


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec"); ap.add_argument("run"); ap.add_argument("--orb3"); ap.add_argument("--metric", action="store_true"); a = ap.parse_args()
    global FIXED_SCALE
    if a.metric: FIXED_SCALE = 1.0
    E, run = Path(a.rec), Path(a.run); t_cam = np.load(E / "tracking_cam_t_ns.npy"); gt = np.load(E / "board_traj.npz")
    g_idx = {int(t): i for i, t in enumerate(gt["t_ns"])}; f2g = np.array([g_idx.get(int(t), -1) for t in t_cam])
    n_frames = len(t_cam); dur = (t_cam[-1] - t_cam[0]) / 1e9
    if a.orb3:
        fi, seg, lost, P, Q, extra = load_orb3(Path(a.orb3), t_cam); run.mkdir(parents=True, exist_ok=True)
        out = dict(take=E.name, backend="orb_slam3_mono_inertial", scale_mode="metric(1.0)" if a.metric else "fitted", duration_s=round(dur, 1), frames=n_frames, **extra)
    else:
        fi, seg, lost, P, Q = load_est(run); info = json.loads((run / "ss1.json").read_text())
        segs = json.loads((run / "ss1.segments.json").read_text())["segments"] if (run / "ss1.segments.json").exists() else [dict(segment=0)]
        secs = sum(json.loads((run / f"ss1.seg{s['segment']:02d}.json").read_text()).get("seconds", np.nan) for s in segs) if (run / "ss1.seg00.json").exists() else info.get("seconds", np.nan)
        out = dict(take=E.name, duration_s=round(dur, 1), frames=n_frames, est_frames=int(len(fi)), lost_frames=int(lost.sum()), restarts=len(segs) - 1,
                   fps=round(len(fi) / secs, 1) if secs and np.isfinite(secs) else None, ms_per_frame=round(1000 * secs / len(fi), 1) if secs and np.isfinite(secs) else None)
    step, drift, fake, scl, per_seg = [], [], [], [], []
    for s in np.unique(seg):
        m = (seg == s) & ~lost & (fi < n_frames); idx = fi[m]; gi = f2g[idx]; ok = gi >= 0
        idx, gi, Pe, Qe = idx[ok], gi[ok], P[m][ok], Q[m][ok]
        if len(idx) < 60: continue
        t = (t_cam[idx] - t_cam[0]) / 1e9; Pg = gt["T_B_C"][gi, :3, 3]; Rg = R.from_matrix(gt["T_B_C"][gi, :3, :3]); Re = R.from_quat(Qe)
        moved = np.linalg.norm(Pg - Pg[0], axis=1) >= 0.05
        def align_at(k0):
            w = (t >= t[k0]) & (t < t[k0] + INIT_S)
            if w.sum() < 30 or np.ptp(Pg[w], axis=0).max() < 0.03: return None
            Rm = (Rg[w] * Re[w].inv()).mean().as_matrix(); sc, tt = fit_scale_t(Pe[w], Pg[w], Rm); return Rm, sc, tt, w
        k0 = int(np.argmax(moved)) if moved.any() else 0; A0 = align_at(max(0, k0 - 30))
        if A0 is None: per_seg.append(dict(segment=int(s), note="no init window")); continue
        Rm, sc, tt, w0 = A0; Pa = Pe @ Rm.T * sc + tt
        # 10 cm steps (frozen scale)
        for i in range(0, len(t), 3):
            j = np.flatnonzero((t > t[i]) & (t <= t[i] + 3.0)); dg = np.linalg.norm(Pg[j] - Pg[i], axis=1); j = j[(dg > 0.08) & (dg < 0.12)]
            if len(j): k = j[0]; step.append(np.linalg.norm((Pa[k] - Pa[i]) - (Pg[k] - Pg[i])))
        # 30 s drift: re-align on every 5 s start, error 30 s after the window
        for st in np.arange(t[0], t[-1] - 30 - INIT_S, 5.0):
            k = int(np.searchsorted(t, st)); A = align_at(k)
            if A is None: continue
            Rk, sk, tk, wk = A; te = t[k] + INIT_S + 30.0; e = np.abs(t - te) < 0.5
            if e.sum() < 5: continue
            drift.append(float(np.median(np.linalg.norm(Pe[e] @ Rk.T * sk + tk - Pg[e], axis=1))))
        # fake translation during rotation
        for i in range(0, len(t), 3):
            k = int(np.searchsorted(t, t[i] + 0.5))
            if k < len(t) and abs(t[k] - t[i] - 0.5) < 0.1 and np.degrees((Rg[i].inv() * Rg[k]).magnitude()) > 20:
                fake.append(np.linalg.norm((Pa[k] - Pa[i]) - (Pg[k] - Pg[i])))
        # scale per 5 s window
        for st in np.arange(t[0], t[-1] - 5, 2.5):
            ww = (t >= st) & (t < st + 5)
            if ww.sum() < 30 or np.ptp(Pg[ww], axis=0).max() < 0.05: continue
            s5, _ = fit_scale_t(Pe[ww], Pg[ww], Rm); scl.append(s5 / sc)
        ate = np.linalg.norm(Pa - Pg, axis=1)
        per_seg.append(dict(segment=int(s), frames=int(len(t)), t0=round(float(t[0]), 1), t1=round(float(t[-1]), 1), frozen_scale=round(sc, 4),
                            ate_frozen_cm_p50=round(float(np.median(ate)) * 100, 2), ate_frozen_cm_p90=round(float(np.percentile(ate, 90)) * 100, 2)))
    q = lambda v: dict(n=len(v), p50=round(float(np.median(v)) * 100, 2), p90=round(float(np.percentile(v, 90)) * 100, 2)) if len(v) else None
    out.update(step10_err_cm=q(step), drift30_cm=q(drift), fake_rot_cm=q(fake),
               scale_window_ratio=dict(n=len(scl), p10=round(float(np.percentile(scl, 10)), 3), p50=round(float(np.median(scl)), 3), p90=round(float(np.percentile(scl, 90)), 3)) if scl else None,
               segments=per_seg)
    (run / (("orb3_eval_metric.json" if a.metric else "orb3_eval_fitted.json") if a.orb3 else "board_eval.json")).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
