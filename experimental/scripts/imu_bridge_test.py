#!/usr/bin/env python3
"""[2026-09-18] Can the wrist IMU carry the trajectory between metric cube anchors?

Double-integrated accelerometer is never an absolute position source: a small bias grows quadratically. The only question
worth asking is narrower and testable directly from the recordings: starting from a known metric anchor, how far has the
IMU-predicted position drifted by the time the NEXT anchor arrives? That error, as a function of gap length, decides whether
a PnP + IMU factor graph is worth building at all.

The test needs no fusion code. Gravity is removed using the orientation we already trust (MASt3R, whose rotation agreed to
0.45 deg across reconstructions), the accelerometer bias is taken from a still window, and the doubly-integrated position is
compared against the straight-line distance the hand actually moved over the same interval.
    .venv/bin/python scripts/imu_bridge_test.py [--episodes 5] [--side left]"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

ROOT = Path(__file__).resolve().parents[1]; M = Path.home() / "ego_data/manifests"
G = 9.80665


def calib(side):
    f = ROOT / "configs/calibration" / ("camera_imu_left_v001.yaml" if side == "left" else "camera_imu_right_v002.yaml")
    c = yaml.safe_load(open(f)); T = np.array(c["T_camera_imu"], float)
    return T[:3, :3], float(c.get("time_offset_ms", 0.0)) * 1e-3, f.name


def imu(ep, side):
    from mcap.reader import make_reader
    rows = []
    with open(f"{ep}/sensors.mcap", "rb") as fh:
        for _, ch, msg in make_reader(fh).iter_messages(topics=[f"/{side}/imu"]):
            d = json.loads(msg.data)
            rows.append((d["device_timestamp_us"], d["gx"], d["gy"], d["gz"], d["ax"], d["ay"], d["az"]))
    if len(rows) < 200: return None
    A = np.asarray(rows, float); o = np.argsort(A[:, 0]); A = A[o]
    k = np.r_[True, np.diff(A[:, 0]) > 0]; A = A[k]
    return A[:, 0] * 1e-6, A[:, 1:4], A[:, 4:7]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=5); ap.add_argument("--side", default="left")
    ap.add_argument("--out", default=str(M / "kin_audit/imu_bridge.md")); a = ap.parse_args()
    R_ci, toff, calname = calib(a.side)
    dirs = [l.strip() for l in open(M / "E30_dirs.txt")][: a.episodes]
    GAPS = (0.2, 0.5, 1.0, 2.0, 4.0)
    res = {g: [] for g in GAPS}; moved = {g: [] for g in GAPS}
    for ep in dirs:
        r = imu(ep, a.side)
        if r is None: continue
        t, gyro, acc = r
        df = pd.read_parquet(f"{ep}/derived/pose_mast3r_filtered/{a.side}_camera_pose.parquet")
        tv = df["t_ns"].to_numpy(np.int64) * 1e-9; Q = df[["qx", "qy", "qz", "qw"]].to_numpy(); P = df[["x", "y", "z"]].to_numpy()
        nq = np.linalg.norm(Q, axis=1); ok = df["valid"].to_numpy(bool) & (nq > 1e-6)
        if ok.sum() < 100: continue
        tv, Qv, Pv = tv[ok], Q[ok] / nq[ok][:, None], P[ok]
        # IMU clock -> the pose clock: the device clock is regular, the host clock is not
        ti = t + (np.median(tv) - np.median(t)) + toff
        m = (ti >= tv[0]) & (ti <= tv[-1])
        ti, acc_m, gyro_m = ti[m], acc[m], gyro[m]
        if len(ti) < 500: continue
        Rw = Slerp(tv, Rot.from_quat(Qv))(ti)                       # orientation we trust, used only to remove gravity
        a_w = Rw.apply(acc_m @ R_ci.T)                              # accel -> world, via the frozen camera-IMU extrinsic
        still = np.linalg.norm(acc_m, axis=1)
        q10 = np.percentile(np.abs(still - G), 10)
        bias = a_w[np.abs(still - G) <= q10].mean(0) if (np.abs(still - G) <= q10).any() else np.array([0, 0, G])
        lin = a_w - bias                                            # gravity + static bias removed
        dt = np.diff(ti, prepend=ti[0]); dt[0] = dt[1]
        for gap in GAPS:
            n = int(gap / np.median(dt))
            if n < 10 or len(ti) < 3 * n: continue
            for s in range(0, len(ti) - n, max(n, int(len(ti) / 12))):
                seg = lin[s:s + n]; d = dt[s:s + n]
                v = np.cumsum(seg * d[:, None], 0)                  # start from rest at the anchor
                p = np.cumsum(v * d[:, None], 0)
                pred = np.linalg.norm(p[-1])
                t0, t1 = ti[s], ti[s + n]
                j0 = np.searchsorted(tv, t0); j1 = min(np.searchsorted(tv, t1), len(Pv) - 1)
                if j1 <= j0: continue
                true = np.linalg.norm(Pv[j1] - Pv[j0])
                res[gap].append(abs(pred - true)); moved[gap].append(true)
    L = [f"# Can the wrist IMU bridge the gaps between metric anchors? ({a.side}, {len(dirs)} episodes) -- 2026-09-18", "",
         f"Doubly-integrated accelerometer starting from rest at an anchor, gravity removed with the MASt3R orientation,",
         f"static bias from the quietest samples. Extrinsic `{calname}`. The comparison is against the distance actually travelled.", "",
         "| bridge length | IMU position error p50 | p95 | distance actually travelled p50 |", "|---|---|---|---|"]
    for g in GAPS:
        if len(res[g]) < 5: continue
        L.append(f"| {g:.1f} s | {np.percentile(res[g],50)*1000:.0f} mm | {np.percentile(res[g],95)*1000:.0f} mm | {np.percentile(moved[g],50)*1000:.0f} mm |")
    L += ["", f"(n per row: {[len(res[g]) for g in GAPS]})"]
    Path(a.out).expanduser().write_text("\n".join(L) + "\n"); print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
