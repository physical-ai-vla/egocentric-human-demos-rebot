"""IMU <-> UMI tag rig: rotation R_I_rig and time offset from a tag_rig_record take that logged the XIAO IMUs (2026-10-17).
Rig poses from the global Orbbec tags (configs/calibration/umi_tag_rig_v001.json), body angular velocity w_rig = log(R_k^T R_{k+2}) / dt
(2-frame central step, the tag rotation noise is ~1 deg per frame), gyro on the host clock (fit_device_to_host). Steps:
  1. offset: cross-correlate |w_rig| and |w_imu| (both resampled on a 5 ms grid), search +-300 ms
  2. rotation: Kabsch on the vectors (only |w| > 30 deg/s), w_imu = R_I_rig w_rig; then the offset re-scanned with the vectors
Report: offset_ms (t_cam_host = t_imu_host + offset), rotation, per-axis correlation, residual RMS (deg/s).
    python scripts/tag_rig_imu.py datasets/tag_rig/<take> [<take> ...] [--write]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SIDES = {"UMI 오른손": "right", "UMI 왼손": "left"}


def rig_track(E: Path, rig: dict, K, size: float):
    s = size / 2; obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), prm)
    t = np.load(E / "t_ns.npy"); out = {n: [] for n in rig}
    Trt = {n: {int(k): np.array(v) for k, v in rig[n]["T_rig_tag"].items()} for n in rig}
    for k in range(len(t)):
        cs, ids, _ = det.detectMarkers(cv2.imread(str(E / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE))
        for n in rig:
            est = []
            for c, i in zip(cs if ids is not None else [], ids.ravel() if ids is not None else []):
                if int(i) not in Trt[n]: continue
                px = c.reshape(4, 2).astype(np.float64)
                if np.mean(np.linalg.norm(px - np.roll(px, 1, 0), axis=1)) < 20: continue
                ok, rv, tv = cv2.solvePnP(obj, px, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok: est.append(cv2.Rodrigues(rv)[0] @ Trt[n][int(i)][:3, :3].T)          # R_C_rig
            if est: out[n].append((int(t[k]), R.from_matrix(np.array(est)).mean().as_matrix()))
    return out


def rig_omega(track):
    t = np.array([a for a, _ in track]) / 1e9; Rm = np.array([b for _, b in track]); W, T = [], []
    for k in range(len(t) - 2):
        dt = t[k + 2] - t[k]
        if not 0.08 < dt < 0.25: continue
        W.append(R.from_matrix(Rm[k].T @ Rm[k + 2]).as_rotvec() / dt); T.append((t[k] + t[k + 2]) / 2)
    return np.array(T), np.array(W)


def main() -> None:
    from handumi_collector.pose.timing import fit_device_to_host
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="+"); ap.add_argument("--rig", default=str(ROOT / "configs/calibration/umi_tag_rig_v001.json"))
    ap.add_argument("--tag-size", type=float, default=0.048); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    rig = json.loads(Path(a.rig).read_text()); res_all = {n: [] for n in rig}
    for take in a.takes:
        E = Path(take); meta = json.loads((E / "meta.json").read_text()); I = meta["intrinsics"]; K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]])
        tracks = rig_track(E, rig, K, a.tag_size)
        for n, side in SIDES.items():
            if n not in rig or not (E / f"imu_{side}.npz").exists() or len(tracks.get(n, [])) < 50: continue
            if len(np.load(E / f"imu_{side}.npz")["device_us"]) < 100: print(f"[{n}] {E.name}: IMU 샘플 없음 -- 건너뜀"); continue
            z = np.load(E / f"imu_{side}.npz"); ti = fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]) / 1e9; g = z["gyro"].astype(float)
            # 10-17 v2: hand-eye on RELATIVE rotations over ~0.4 s (instantaneous tag rates were too noisy: 55-120 deg/s residual)
            tr = np.array([x[0] for x in tracks[n]]) / 1e9; Rr = np.array([x[1] for x in tracks[n]])
            bias = np.median(g[: max(50, int(0.05 * len(g)))], 0) if np.linalg.norm(np.median(g[:200], 0)) < 0.05 else np.zeros(3)
            Rq = [R.identity()]                                   # gyro integrated on the host clock (body frame, right-multiplied)
            for j in range(len(ti) - 1): Rq.append(Rq[-1] * R.from_rotvec((g[j] - bias) * (ti[j + 1] - ti[j])))
            Rq = R.concatenate(Rq); qx = Rq.as_quat()
            def imu_rot(tt):
                j = np.clip(np.searchsorted(ti, tt), 1, len(ti) - 1); return Rq[j]
            pairs = [(k, m) for k in range(len(tr)) for m in [int(np.searchsorted(tr, tr[k] + 0.4))] if m < len(tr) and 0.3 < tr[m] - tr[k] < 0.6]
            A = np.array([R.from_matrix(Rr[k].T @ Rr[m]).as_rotvec() for k, m in pairs])           # rig body relative rotation
            big = np.linalg.norm(A, axis=1) > np.radians(8); pairs = [p_ for p_, b_ in zip(pairs, big) if b_]; A = A[big]
            def fit(off):
                Bv = np.array([(imu_rot(tr[k] - off).inv() * imu_rot(tr[m] - off)).as_rotvec() for k, m in pairs])
                H = A.T @ Bv; U, S, Vt = np.linalg.svd(H); X = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T   # Bv = X A
                r = np.linalg.norm(A @ X.T - Bv, axis=1); return float(np.degrees(np.median(r))), X, r
            if len(pairs) < 20: print(f"[{n}] {E.name}: 회전 쌍 부족 ({len(pairs)})"); continue
            offs = np.arange(-0.15, 0.15, 0.005); sc = [fit(o)[0] for o in offs]; off = float(offs[int(np.argmin(sc))]); med, Rir, r = fit(off)
            rms = float(np.degrees(np.sqrt(np.mean(r ** 2))))
            print(f"[{n}] {E.name}: 시간 오프셋 {off * 1000:+.0f} ms | 회전 R_I_rig rotvec {np.round(np.degrees(R.from_matrix(Rir).as_rotvec()), 1)}° | 0.4초 상대회전 잔차 중앙 {med:.2f}° RMS {rms:.2f}° (쌍 {len(pairs)})")
            res_all[n].append(dict(take=E.name, offset_ms=off * 1000, R_I_rig=Rir.tolist(), resid_med_deg=med, resid_rms_deg=rms, n=len(pairs)))
    for n, L in res_all.items():
        if len(L) >= 2:
            d = np.degrees((R.from_matrix(L[0]["R_I_rig"]) * R.from_matrix(L[1]["R_I_rig"]).inv()).magnitude())
            print(f"[{n}] 두 녹화 사이 회전 차이 {d:.1f}°, 오프셋 차이 {abs(L[0]['offset_ms'] - L[1]['offset_ms']):.0f} ms")
    if a.write:
        cfg = json.loads(Path(a.rig).read_text())
        for n, L in res_all.items():
            if not L: continue
            best = max(L, key=lambda x: x["n"]); cfg[n]["imu"] = dict(R_I_rig=best["R_I_rig"], offset_ms=best["offset_ms"], from_take=best["take"], all=L)
        Path(a.rig).write_text(json.dumps(cfg, indent=1, ensure_ascii=False)); print("기록", a.rig)


if __name__ == "__main__":
    main()
