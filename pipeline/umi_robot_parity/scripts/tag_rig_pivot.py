"""UMI jaw tip (TCP) in the tag-rig frame by PIVOT calibration (2026-10-17, no stereo cams: user lead decision). Take = tag_rig_record
--proto pivot: the jaw tip is held on one desk point while the UMI is tilted in every direction, right UMI then left UMI. Rig poses
come from the tags (rig = T_rig_tag of a --rig take solved by tag_rig_solve.py); the tip p_rig and the fixed point p_C (Orbbec frame)
solve   R_i p_rig + t_i = p_C   for all frames i (linear least squares, 2 rounds of 3-sigma outlier rejection).
Quality: residual RMS / p90 (mm) and the tilt spread (deg; < 20 deg spread = the tip is poorly observable along the axis).
    python scripts/tag_rig_pivot.py datasets/tag_rig/<pivot take> --rig datasets/tag_rig/<rig take>/tag_rig.json [--write]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
PHASES = {"UMI 오른손": "오른손", "UMI 왼손": "왼손"}


def main() -> None:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--rig", required=True); ap.add_argument("--tag-size", type=float, default=0.048); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    E = Path(a.take); meta = json.loads((E / "meta.json").read_text()); rigs = json.loads(Path(a.rig).read_text())["rigs"]
    I = meta["intrinsics"]; K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]]); t = (np.load(E / "t_ns.npy") - meta["t_start_monotonic_ns"]) / 1e9
    s = a.tag_size / 2; obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), prm)
    proto = meta["protocol"]; out = {}
    for name, word in PHASES.items():
        if name not in rigs: print(f"[{name}] 리그 정보 없음 -- 먼저 --proto rig 녹화 + tag_rig_solve"); continue
        Trt = {int(k): np.array(v) for k, v in rigs[name]["T_rig_tag"].items()}
        ph = next((p for i, p in enumerate(proto) if p["text"].startswith("집게 끝을 떼지") and word in proto[i - 1]["text"]), None)
        if ph is None: print(f"[{name}] 피벗 구간 없음"); continue
        Rs, ts = [], []
        for k in np.where((t >= ph["t0"] + 1) & (t < ph["t0"] + ph["dur"] - 0.5))[0]:
            img = cv2.imread(str(E / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE); cs, ids, _ = det.detectMarkers(img); est = []
            for c, i in zip(cs if ids is not None else [], ids.ravel() if ids is not None else []):
                i = int(i)
                if i not in Trt: continue
                px = c.reshape(4, 2).astype(np.float64)
                if np.mean(np.linalg.norm(px - np.roll(px, 1, 0), axis=1)) < 20: continue
                ok, rv, tv = cv2.solvePnP(obj, px, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok: continue
                T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rv)[0]; T[:3, 3] = tv.ravel(); est.append(T @ np.linalg.inv(Trt[i]))   # T_C_rig
            if est:
                q = R.from_matrix(np.array([e[:3, :3] for e in est])).mean(); Rs.append(q.as_matrix()); ts.append(np.median([e[:3, 3] for e in est], 0))
        if len(Rs) < 20: print(f"[{name}] 피벗 프레임 부족 ({len(Rs)})"); continue
        Rs, ts = np.array(Rs), np.array(ts); keep = np.ones(len(Rs), bool)
        for thr in (0.05, 0.025, 0.015, 0.010):        # 10-17: progressively tighter gate (3-sigma kept all 256 frames, RMS 10 mm)
            A = np.concatenate([np.c_[Rs[keep][j], -np.eye(3)] for j in range(keep.sum())]); b = -ts[keep].reshape(-1)
            x = np.linalg.lstsq(A, b, rcond=None)[0]; p_rig, p_C = x[:3], x[3:]
            res = np.linalg.norm(np.einsum("nij,j->ni", Rs, p_rig) + ts - p_C, axis=1)
            if (res < thr).sum() < 30: break
            keep = res < thr
        Rk = R.from_matrix(Rs[keep]); spread = float(2 * np.degrees(np.percentile((Rk * Rk.mean().inv()).magnitude(), 90)))   # ~ tilt range around the mean pose
        rms, p90 = float(np.sqrt(np.mean(res[keep] ** 2)) * 1000), float(np.percentile(res[keep], 90) * 1000)
        print(f"[{name}] 피벗 프레임 {keep.sum()}/{len(Rs)} | 집게 끝 (리그 좌표, cm) {np.round(p_rig * 100, 2)} |p| {np.linalg.norm(p_rig) * 100:.1f} cm | 잔차 RMS {rms:.1f} mm, p90 {p90:.1f} mm | 기울기 범위 {spread:.0f}°")
        out[name] = dict(p_rig_m=p_rig.tolist(), p_cam_m=p_C.tolist(), frames=int(keep.sum()), residual_rms_mm=rms, residual_p90_mm=p90, tilt_spread_deg=spread, rig_ref=rigs[name]["ref"])
    (E / "tag_rig_pivot.json").write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(f"저장 {E / 'tag_rig_pivot.json'}")
    if a.write and out:
        path = ROOT / "configs/calibration/umi_tag_rig_v001.json"; prev = json.loads(path.read_text()) if path.exists() else {}
        for name, v in out.items(): prev[name] = dict(T_rig_tag=rigs[name]["T_rig_tag"], ref=rigs[name]["ref"], tcp_in_rig_m=v["p_rig_m"], pivot=v, rig_take=str(a.rig), pivot_take=str(E))
        path.write_text(json.dumps(prev, indent=1, ensure_ascii=False)); print(f"기록 {path}")


if __name__ == "__main__":
    main()
