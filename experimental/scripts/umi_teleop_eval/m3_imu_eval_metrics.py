"""MASt3R-SLAM translation + IMU orientation, evaluated as the pose teleop would get (2026-10-15, Agent C wrist-tracker decision).

Fused pose per wrist frame: position = MASt3R-SLAM (restart_v1 ss1.csv; a new segment = new world + new scale after a loss),
orientation = the IMU AHRS (canonical.npz R_gravity_imu_<side>). Scored against the Orbbec-only body track (canonical.npz).
Per segment, the scale is fitted ONCE on its first 3 s (Sim(3) + camera lever arm, using the reference orientation for the lever)
and then FROZEN: drift at +10/+30 s and scale(t)/scale_init from 2 s displacement ratios. Segment restarts count as re-anchors
(clutch-like) and as tracking breaks; lost frames inside a segment count as stalls (1-2 frames) or collapses (>= 5 frames).
Orientation quality = IMU vs Orbbec agreement (also during fast rotation, > 120 deg/s).
Throughput comes from the run json (seconds / frames, single-thread MASt3R on the 4090).
    python scripts/m3_imu_eval_metrics.py 20261014_112920_000001_right ... [--runs ~/c8/robotlike/runs_restart]"""
from __future__ import annotations
import argparse
import csv
import json
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation as R

H = pathlib.Path.home()
SESSION = H / "ego_collector/datasets/human_handumi_raw/UMIalign/UMIalign_20261014_112920"
RUNS = H / "c8/robotlike/runs_restart"


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); A, B = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(B.T @ A / len(src)); d = np.sign(np.linalg.det(U @ Vt)); Dg = np.diag([1, 1, d])
    Rm = U @ Dg @ Vt; return (S * np.diag(Dg)).sum() / A.var(0).sum(), Rm, mu_d - (S * np.diag(Dg)).sum() / A.var(0).sum() * Rm @ mu_s


def align(c, p, Rb, iters=10):
    """camera position c (MASt3R world) vs body position p:  p_i + Rb_i r = s R c_i + t   (r = body->camera lever, body frame)."""
    r = np.zeros(3)
    for _ in range(iters):
        s, Rm, t = umeyama(c, p + np.einsum("nij,j->ni", Rb, r))
        A = Rb.reshape(-1, 3); b = (s * c @ Rm.T + t - p).reshape(-1); r, *_ = np.linalg.lstsq(A, b, rcond=None)
    return s, Rm, t, r


def ang_speed(Rs, dt):
    return np.r_[0, np.degrees(np.linalg.norm(R.from_matrix(np.einsum("nji,njk->nik", Rs[:-1], Rs[1:])).as_rotvec(), axis=1)) / dt]


def rel_rot_err(Ra, Rb):
    """Orientation agreement after removing the constant mount/world offsets: error of each frame's rotation since the start."""
    Ea = np.einsum("ji,njk->nik", Ra[0], Ra); Eb = np.einsum("ji,njk->nik", Rb[0], Rb)
    # body-frame deltas; the IMU->body mount rotation M is constant: Eb = M^T Ea M -> compare the rotation ANGLES (mount-invariant)
    return np.abs(np.degrees(np.linalg.norm(R.from_matrix(Ea).as_rotvec(), axis=1)) - np.degrees(np.linalg.norm(R.from_matrix(Eb).as_rotvec(), axis=1)))


def evaluate(name: str, runs: pathlib.Path, session: pathlib.Path) -> dict:
    ep = int(name.split("_")[2]); side = name.split("_")[3]; rd = runs / name
    rows = list(csv.DictReader(open(rd / "ss1.csv")))
    J = [json.load(open(f)) for f in sorted(rd.glob("ss1.seg*.json"))]; J = [j for j in J if j.get("seconds") and j.get("n_frames")]
    secs = sum(j["seconds"] for j in J); nfr = sum(j["n_frames"] for j in J)
    ref = np.load(session / f"episode_{ep:06d}/derived/body_track/canonical.npz")
    t = (ref["t_ns"] - ref["t_ns"][0]) / 1e9; dt = float(np.median(np.diff(t))); n = len(t)
    P, Rb, Ri, ok = ref[f"p_W_{side}"], ref[f"R_world_body_{side}"], ref[f"R_gravity_imu_{side}"], ref[f"valid_{side}"]
    k = np.array([int(r["frame_idx"]) for r in rows]); seg = np.array([int(r["segment"]) for r in rows])
    lost = np.array([r["is_lost"] == "true" or r["state"] not in ("TRACKING", "INIT") for r in rows])
    c = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    out = dict(run=name, mast3r_fps=round(nfr / secs, 1) if secs else None, frames=int(len(rows)), segments=int(seg.max() + 1),
               tracked_frac=round(float((~lost).mean()), 3))
    # breaks: lost runs inside segments + segment restarts
    runs_bad, cur = [], 0
    for b in lost:
        if b: cur += 1
        elif cur: runs_bad.append(cur); cur = 0
    if cur: runs_bad.append(cur)
    out.update(stalls_1_2f=sum(1 for L in runs_bad if L <= 2), collapses_5f=sum(1 for L in runs_bad if L >= 5), longest_lost_frames=max(runs_bad, default=0),
               segment_restarts=int(seg.max()))
    # orientation (IMU) vs Orbbec: angle-of-rotation agreement, overall and in fast rotation
    m_ok = ok & np.all(np.isfinite(P), 1) & np.all(np.isfinite(Ri.reshape(n, -1)), 1) & np.all(np.isfinite(Rb.reshape(n, -1)), 1)
    wr = ang_speed(Rb, dt); e_rot = rel_rot_err(Ri[m_ok], Rb[m_ok]); fast = wr[m_ok] > 120
    wi = ang_speed(Ri, dt); corr = float(np.corrcoef(wi[m_ok], wr[m_ok])[0, 1])
    out.update(imu_vs_orbbec_rotspeed_corr=round(corr, 2), fast_rot_frames=int(fast.sum()),
               imu_fast_rot_angle_err_p90=round(float(np.percentile(e_rot[fast], 90)), 1) if fast.sum() > 5 else None)
    # translation per segment, scale frozen after its first 3 s
    segs = []
    for s_id in range(seg.max() + 1):
        sel = (seg == s_id) & ~lost & (k < n) & np.all(np.isfinite(c), 1); kk = k[sel]; kk_ok = m_ok[kk]; kk = kk[kk_ok]
        if len(kk) < 120: continue
        cc = c[sel][kk_ok]; pp = P[kk]; RR = Rb[kk]; tt = t[kk]; first = tt <= tt[0] + 3.0
        if np.ptp(pp[first], 0).max() < 0.03: first = tt <= tt[0] + 6.0                     # too little motion to fix a scale
        s0, R0, t0, r0 = align(cc[first], pp[first], RR[first]); cw = s0 * cc @ R0.T + t0 - np.einsum("nij,j->ni", RR, r0)
        e = np.linalg.norm(cw - pp, axis=1) * 1000
        sg = dict(segment=s_id, dur_s=round(float(tt[-1] - tt[0]), 1), err_mm_p50=round(float(np.median(e)), 1), err_mm_p90=round(float(np.percentile(e, 90)), 1))
        for Hs in (10, 30):
            m = np.abs(tt - (tt[0] + Hs)) < 0.5
            sg[f"drift_{Hs}s_mm"] = round(float(np.median(e[m])), 1) if m.any() else None
        kd = int(2 / dt); rat = []
        for j in range(0, len(cw) - kd, kd):
            dr = np.linalg.norm(pp[j + kd] - pp[j])
            if dr > 0.05: rat.append(np.linalg.norm(cw[j + kd] - cw[j]) / dr)
        if rat: sg["scale_ratio_2s"] = dict(p10=round(float(np.percentile(rat, 10)), 2), p50=round(float(np.median(rat)), 2), p90=round(float(np.percentile(rat, 90)), 2), n=len(rat))
        segs.append(sg)
    out["segments_eval"] = segs
    # verdict (same user rules as DPVO): scale band 0.9-1.1 (2 s ratios p10..p90 of the long segments), no collapse, fps for teleop
    long = [s for s in segs if s["dur_s"] >= 10 and "scale_ratio_2s" in s]
    v = dict(scale_stable=bool(long) and all(0.9 <= s["scale_ratio_2s"]["p10"] and s["scale_ratio_2s"]["p90"] <= 1.1 for s in long),
             no_collapse=out["collapses_5f"] == 0 and out["segment_restarts"] == 0,
             realtime=bool(out["mast3r_fps"] and out["mast3r_fps"] >= 25), imu_rotation_ok=out["imu_fast_rot_angle_err_p90"] is None or out["imu_fast_rot_angle_err_p90"] <= 10)
    v["overall"] = "PASS" if all(v.values()) else "FAIL"; out["verdict"] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+"); ap.add_argument("--runs", default=str(RUNS)); ap.add_argument("--session", default=str(SESSION)); a = ap.parse_args()
    res = [evaluate(nm, pathlib.Path(a.runs), pathlib.Path(a.session)) for nm in a.names]
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
