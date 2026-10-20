#!/usr/bin/env python3
"""[2026-10-15] Per-episode wrist-camera <-> IMU rotation check (was the Teensy loose on the HandUMI body during collection?).

The smoother uses ONE Kalibr rotation per side (export orbslam_setting IMU.T_b_c1 -> R_cb) for every episode. If the IMU board
was loose, the true camera<-IMU rotation would differ from it and wander between (or within) episodes. Check, per episode x side:
  omega_c  = rotvec(R_wc[i]^T R_wc[i+1]) / dt   from MASt3R restart_v1 (consecutive frames of one segment, camera frame)
  omega_b  = mean gyro over the same interval, shifted by a scanned delay (IMU frame)
  R_est    = argmin sum w |omega_c - R omega_b|^2 (Kabsch, |omega_b| > 30 deg/s, 2 rounds of 3-MAD outlier removal)
Reported: angle(R_est, R_cb_kalibr), angle between first-half and second-half R_est (within-episode wander), per-axis vector
correlation of omega_c vs R_est omega_b (the board check got 0.99/0.99/0.97 with a rigid IMU), best delay, sample count.
Read-only on the dataset. usage: imu_cam_extrinsic_check.py <session dir> ... [--out csv] [-j 6]
"""
import argparse, csv, json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial.transform import Rotation as Rot

H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_cart20"))
from ego_cart20.sources.hfold_export import read_mcap                          # noqa: E402
from ego_cart20.umi_fusion.run_episode import R_cb_kalibr, load_restart, tag_of  # noqa: E402
from ego_cart20.umi_fusion.run_smoother import imu_full                          # noqa: E402

MIN_RATE = np.radians(30.0); DELAYS_MS = np.arange(-40, 101, 5)


def kabsch(a, b, w):
    """R minimising sum w |a - R b|^2 (rows are vectors)."""
    M = (b * w[:, None]).T @ a; U, _, Vt = np.linalg.svd(M); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def fit(wc, wb):
    keep = np.linalg.norm(wb, axis=1) > MIN_RATE
    if keep.sum() < 60: return None, keep
    R = None
    for _ in range(3):
        w = np.ones(keep.sum()); R = kabsch(wc[keep], wb[keep], w)
        r = np.linalg.norm(wc - wb @ R.T, axis=1); mad = 1.4826 * np.median(r[keep]) + 1e-6
        keep = keep & (r < 3 * mad + np.radians(10))
    return R, keep


def vec_corr(wc, wb, R, k):
    p = wb[k] @ R.T; return [float(np.corrcoef(wc[k, i], p[:, i])[0, 1]) for i in range(3)]


def one(job):
    ep, side = map(str, job); ep = pathlib.Path(ep); tag = tag_of(ep, side); row = dict(episode=f"{ep.parent.name}/{ep.name}", side=side)
    try:
        seg, _, Rm = load_restart(tag); fm, _ = read_mcap(ep); cap = dict(zip(fm[f"{side}_wrist"][:, 0].tolist(), fm[f"{side}_wrist"][:, 1].tolist()))
        idx = np.array([i for i in range(len(seg)) if i in cap]); t = np.array([cap[i] for i in idx], np.int64) / 1e9; seg, Rm = seg[idx], Rm[idx]
        k = np.flatnonzero((seg[:-1] >= 0) & (seg[:-1] == seg[1:]) & (np.diff(idx) == 1)); dt = np.diff(t)[k]
        wc = Rot.from_matrix(np.einsum("nji,njk->nik", Rm[k], Rm[k + 1])).as_rotvec() / dt[:, None]
        gt, gw, _ = imu_full(ep, side); gt = gt / 1e9; gw = gw - gw[gt < gt[0] + 1.0].mean(0)
        cg = np.concatenate([[np.zeros(3)], np.cumsum(0.5 * (gw[1:] + gw[:-1]) * np.diff(gt)[:, None], 0)])   # integral of gyro
        def wb_at(d):                                                    # mean gyro over [t_k, t_k+1] shifted by delay d (s)
            a, b = t[k] - d, t[k + 1] - d
            return np.stack([(np.interp(b, gt, cg[:, i]) - np.interp(a, gt, cg[:, i])) for i in range(3)], 1) / (b - a)[:, None]
        best = None
        for dms in DELAYS_MS:
            wb = wb_at(dms / 1e3); R, kk = fit(wc, wb)
            if R is None: continue
            q = float(np.mean(vec_corr(wc, wb, R, kk)))
            if best is None or q > best[0]: best = (q, dms, R, wb, kk)
        if best is None: row["status"] = "too_few_samples"; return row
        q, dms, R, wb, kk = best; Rk = R_cb_kalibr(tag)
        half = np.arange(len(k)) < len(k) // 2; Rh = [fit(wc[m], wb[m])[0] for m in (half, ~half)]
        vc = vec_corr(wc, wb, R, kk); res = np.degrees(np.linalg.norm(wc[kk] - wb[kk] @ R.T, axis=1))
        row.update(status="ok", n=int(kk.sum()), delay_ms=int(dms), vec_corr_x=round(vc[0], 3), vec_corr_y=round(vc[1], 3), vec_corr_z=round(vc[2], 3),
                   ang_vs_kalibr_deg=round(float(np.degrees(Rot.from_matrix(R @ Rk.T).magnitude())), 2),
                   halves_deg=round(float(np.degrees(Rot.from_matrix(Rh[0] @ Rh[1].T).magnitude())), 2) if all(x is not None for x in Rh) else None,
                   resid_p50_dps=round(float(np.median(res)), 1), rate_rms_dps=round(float(np.degrees(np.sqrt(np.mean(np.sum(wb[kk] ** 2, 1))))), 1),
                   R_est_rotvec_deg=json.dumps(np.round(np.degrees(Rot.from_matrix(R).as_rotvec()), 2).tolist()),
                   R_kalibr_rotvec_deg=json.dumps(np.round(np.degrees(Rot.from_matrix(Rk).as_rotvec()), 2).tolist()))
    except Exception as ex:
        row.update(status=f"error {type(ex).__name__}: {ex}"[:200])
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("sessions", nargs="+"); ap.add_argument("--out", default="imu_cam_check.csv"); ap.add_argument("-j", type=int, default=6)
    a = ap.parse_args(); jobs = []
    for s in map(pathlib.Path, a.sessions):
        for ep in sorted(s.glob("episode_*")):
            if not (ep / ".complete").exists(): continue
            if json.loads((ep / "episode_meta.json").read_text()).get("exclude_from_training"): continue
            jobs += [(str(ep), side) for side in ("left", "right")]
    with ProcessPoolExecutor(a.j) as ex: rows = list(ex.map(one, jobs))
    cols = sorted({k for r in rows for k in r}, key=lambda c: ["episode", "side", "status", "n", "delay_ms", "vec_corr_x", "vec_corr_y", "vec_corr_z",
                                                             "ang_vs_kalibr_deg", "halves_deg", "resid_p50_dps", "rate_rms_dps"].index(c) if c in
                  ["episode", "side", "status", "n", "delay_ms", "vec_corr_x", "vec_corr_y", "vec_corr_z", "ang_vs_kalibr_deg", "halves_deg", "resid_p50_dps", "rate_rms_dps"] else 99)
    with open(a.out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} sides -> {a.out}; ok {sum(r['status'] == 'ok' for r in rows)}")
