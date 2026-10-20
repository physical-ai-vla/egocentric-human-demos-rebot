#!/usr/bin/env python3
"""[2026-10-12] How long can the wrist IMU gyro carry the camera ROTATION across a MASt3R gap?

Inside MASt3R-valid stretches (ground truth here = MASt3R camera rotation), start gyro integration at R(t0) and compare to
MASt3R at t0 + dt for dt in DTS:  e_R(dt) = angle(R_imu(t0->t1)^T @ R_m3(t0->t1)).
Rotation only (IMU never makes position). Gyro bias = mean over the first still second of the take (the REC hold).
Camera<-IMU rotation: (a) Kalibr T_b_c1 from the export's orbslam_setting.yaml, (b) re-fitted from the data (Wahba on
angular-velocity vectors; FIT on the first half of the valid frames, reported on the second half) -- (b) exists because the
right unit's Kalibr extrinsic is known to be ~18 deg off (camera_imu_right_rotfix_20261006).
Frame <-> CORI: ss1 frame_idx i <-> CORI sample i + lag (lag scanned in frames, picked by the 1-frame fit residual).

usage: imu_bridge_validate.py <export dir> <ss1.csv> [--out report.json]
"""
import argparse, csv, json, pathlib, re, sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

DTS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0)


def load_ss1(p):
    rows = list(csv.DictReader(open(p))); n = max(int(r["frame_idx"]) for r in rows) + 1
    R = np.tile(np.eye(3), (n, 1, 1)); ok = np.zeros(n, bool)
    for r in rows:
        if r["is_lost"] != "false": continue
        q = [float(r[k]) for k in ("q_x", "q_y", "q_z", "q_w")]
        if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 0.5: continue
        i = int(r["frame_idx"]); R[i] = Rot.from_quat(q).as_matrix(); ok[i] = True
    return R, ok


def T_b_c(setting):
    m = re.search(r"IMU\.T_b_c1:.*?data:\s*\[([^\]]+)\]", pathlib.Path(setting).read_text(), re.S)
    return np.array([float(v) for v in m.group(1).split(",")]).reshape(4, 4)


def gyro_integrate(t_g, w, t_query, bias):
    """R_b(t0 -> t) cumulative from t_g[0]; returns body orientation at each t_query (piecewise-constant exp map)."""
    w = w - bias; dt = np.diff(t_g, append=t_g[-1]); Rs = [np.eye(3)]
    for k in range(len(t_g) - 1):
        Rs.append(Rs[-1] @ Rot.from_rotvec(w[k] * dt[k]).as_matrix())
    Rs = np.array(Rs); j = np.clip(np.searchsorted(t_g, t_query) - 1, 0, len(t_g) - 1)
    frac = np.clip((t_query - t_g[j]) / np.maximum(dt[j], 1e-9), 0, 1)
    return np.einsum("nij,njk->nik", Rs[j], Rot.from_rotvec((w[j] * (dt[j] * frac)[:, None])).as_matrix())


def wahba(a, b):
    """R minimising sum |R a_i - b_i|^2 (rows)."""
    U, _, Vt = np.linalg.svd(b.T @ a); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); return U @ D @ Vt


def ang(Ra):
    return np.degrees(Rot.from_matrix(Ra).magnitude())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("export"); ap.add_argument("ss1"); ap.add_argument("--out"); a = ap.parse_args()
    exp = pathlib.Path(a.export); d = json.load(open(exp / "imu_data.json"))["1"]["streams"]
    t_c = np.array([s["cts"] for s in d["CORI"]["samples"]]) / 1e3
    t_g = np.array([s["cts"] for s in d["GYRO"]["samples"]]) / 1e3; w = np.array([s["value"] for s in d["GYRO"]["samples"]], float)
    R_m, ok = load_ss1(a.ss1)
    bias = w[t_g < t_g[0] + 1.0].mean(0)
    R_bc_k = T_b_c(exp / "orbslam_setting.yaml")[:3, :3]
    fps = 1.0 / np.median(np.diff(t_c))
    best = None
    for lag in range(-4, 5):                                       # frame i <-> CORI i + lag
        i = np.flatnonzero(ok); i = i[(i + lag >= 0) & (i + lag < len(t_c))]
        tq = t_c[i + lag]; R_b = gyro_integrate(t_g, w, tq, bias)
        pair = np.flatnonzero(np.diff(i) == 1)                     # consecutive valid frames
        if len(pair) < 50: continue
        wb = np.array([Rot.from_matrix(R_b[k].T @ R_b[k + 1]).as_rotvec() for k in pair])           # body increment
        wc = np.array([Rot.from_matrix(R_m[i[k]].T @ R_m[i[k + 1]]).as_rotvec() for k in pair])      # camera increment
        mv = np.linalg.norm(wc, axis=1) > np.radians(0.3)
        R_cb = wahba(wb[mv], wc[mv]); res = np.degrees(np.linalg.norm(wb[mv] @ R_cb.T - wc[mv], axis=1))
        if best is None or np.median(res) < best[1]: best = (lag, float(np.median(res)), i, tq, R_b)
    lag, res1, i, tq, R_b = best
    half = len(i) // 2
    # (b) re-fit on the FIRST half only
    def incr(idx):
        p = idx[np.flatnonzero(np.diff(i[idx]) == 1)]
        wb = np.array([Rot.from_matrix(R_b[k].T @ R_b[k + 1]).as_rotvec() for k in p])
        wc = np.array([Rot.from_matrix(R_m[i[k]].T @ R_m[i[k + 1]]).as_rotvec() for k in p]); mv = np.linalg.norm(wc, axis=1) > np.radians(0.3)
        return wb[mv], wc[mv]
    wb1, wc1 = incr(np.arange(half)); R_cb_fit = wahba(wb1, wc1)
    R_cb_k = R_bc_k.T
    # (c) Kalibr rotation + gyro bias re-fitted on the FIRST half (what the bridge can do: fit b on the MASt3R stretch before a gap)
    p1 = np.flatnonzero(np.diff(i[:half]) == 1); tq1 = tq[:half]
    dtk = np.diff(tq1)[p1]
    wc_rate = np.array([Rot.from_matrix(R_m[i[k]].T @ R_m[i[k + 1]]).as_rotvec() for k in p1]) / dtk[:, None]     # camera rate
    wb_raw = np.array([w[(t_g >= tq1[k]) & (t_g < tq1[k + 1])].mean(0) if ((t_g >= tq1[k]) & (t_g < tq1[k + 1])).any() else np.full(3, np.nan) for k in p1])
    g = np.isfinite(wb_raw).all(1); b_fit = (wb_raw[g] - wc_rate[g] @ R_cb_k).mean(0)       # w_b = R_cb^T w_c + b
    R_b2 = gyro_integrate(t_g, w, tq, b_fit)
    out = dict(export=str(exp), ss1=a.ss1, fps=round(fps, 2), lag_frames=lag, gyro_bias_rad_s=bias.tolist(), valid_frames=int(ok.sum()),
               kalibr_vs_fit_deg=round(float(ang(R_cb_k.T @ R_cb_fit)), 2), bias_hold=bias.tolist(), eval="second half of the valid frames", dt={})
    ev = np.arange(half, len(i))
    for dt in DTS:
        k = int(round(dt * fps)); rec = {}
        for name, R_cb, RB in (("kalibr", R_cb_k, R_b), ("fit", R_cb_fit, R_b), ("kalibr_biasfit", R_cb_k, R_b2)):
            e = []
            for s in ev:
                if s + k >= len(i) or i[s + k] - i[s] != k: continue          # stay inside one MASt3R-valid stretch
                Rc_imu = R_cb @ (RB[s].T @ RB[s + k]) @ R_cb.T; Rc_m3 = R_m[i[s]].T @ R_m[i[s + k]]
                e.append(ang(Rc_imu.T @ Rc_m3))
            e = np.array(e)
            rec[name] = dict(n=int(len(e)), p50=round(float(np.percentile(e, 50)), 2), p90=round(float(np.percentile(e, 90)), 2),
                             p99=round(float(np.percentile(e, 99)), 2), max=round(float(e.max()), 2)) if len(e) else None
        out["dt"][str(dt)] = rec
    print(json.dumps(out, indent=1))
    if a.out: pathlib.Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
