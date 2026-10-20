"""Joint pivot over several tag_rig_record --proto pivot takes (2026-10-17): ONE jaw-tip position per UMI in its rig frame, one fixed
desk point per take (the dot may move between takes), progressively tightened inlier gate. Reuses tag_rig_pivot's per-frame rig poses.
Also prints the per-take solution and the leave-one-take-out spread (repeatability).
    python scripts/tag_rig_pivot_joint.py datasets/tag_rig/<take1> <take2> ... [--write]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from tag_rig_imu import rig_track          # noqa: E402  (R only; translations re-derived below)
import cv2                                  # noqa: E402

WORD = {"UMI 오른손": "오른손", "UMI 왼손": "왼손"}


def poses(E: Path, name: str, rig: dict, size: float):
    meta = json.loads((E / "meta.json").read_text()); I = meta["intrinsics"]; K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]])
    t = (np.load(E / "t_ns.npy") - meta["t_start_monotonic_ns"]) / 1e9; proto = meta["protocol"]
    ph = next((p for i, p in enumerate(proto) if p["text"].startswith("집게 끝을 떼지") and WORD[name] in proto[i - 1]["text"]), None)
    if ph is None: return np.zeros((0, 3, 3)), np.zeros((0, 3))
    s = size / 2; obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), prm)
    Trt = {int(k): np.array(v) for k, v in rig[name]["T_rig_tag"].items()}; Rs, ts = [], []
    from tag_rig_pose import rig_pose        # 10-19: one PnP over every visible corner (rotation jitter 1.78 -> 0.39 deg)
    for k in np.where((t >= ph["t0"] + 1) & (t < ph["t0"] + ph["dur"] - 0.5))[0]:
        cs, ids, _ = det.detectMarkers(cv2.imread(str(E / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE)); r = rig_pose(cs, ids, Trt, K, size, min_side=20)
        if r is not None and r["rmse_px"] < 2.0: Rs.append(r["T"][:3, :3]); ts.append(r["T"][:3, 3])
    return np.array(Rs), np.array(ts)


def solve(blocks):
    """blocks: list of (Rs, ts). Unknowns: p_rig (3) + one p_C per block (3 each)."""
    keeps = [np.ones(len(b[0]), bool) for b in blocks]; nb = len(blocks)
    for thr in (0.05, 0.025, 0.015, 0.010):
        rows, rhs = [], []
        for bi, ((Rs, ts), kp) in enumerate(zip(blocks, keeps)):
            for j in np.where(kp)[0]:
                Arow = np.zeros((3, 3 + 3 * nb)); Arow[:, :3] = Rs[j]; Arow[:, 3 + 3 * bi: 6 + 3 * bi] = -np.eye(3); rows.append(Arow); rhs.append(-ts[j])
        x = np.linalg.lstsq(np.concatenate(rows), np.concatenate(rhs), rcond=None)[0]; p = x[:3]
        res = [np.linalg.norm(np.einsum("nij,j->ni", Rs, p) + ts - x[3 + 3 * bi: 6 + 3 * bi], axis=1) for bi, (Rs, ts) in enumerate(blocks)]
        new = [r < thr for r in res]
        if sum(k.sum() for k in new) < 40: break
        keeps = new
    allr = np.concatenate([r[k] for r, k in zip(res, keeps)])
    return p, [x[3 + 3 * bi: 6 + 3 * bi] for bi in range(nb)], float(np.sqrt(np.mean(allr ** 2)) * 1000), [int(k.sum()) for k in keeps]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="+"); ap.add_argument("--rig", default=str(ROOT / "configs/calibration/umi_tag_rig_v001.json"))
    ap.add_argument("--tag-size", type=float, default=0.048); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    rig = json.loads(Path(a.rig).read_text()); out = {}
    for name in WORD:
        blocks, names = [], []
        for tk in a.takes:
            Rs, ts = poses(Path(tk), name, rig, a.tag_size)
            if len(Rs) >= 30: blocks.append((Rs, ts)); names.append(Path(tk).name)
            else: print(f"[{name}] {Path(tk).name}: 프레임 부족 ({len(Rs)}) -- 제외")
        if not blocks: continue
        for (Rs, ts), nm in zip(blocks, names):
            p1, c1, rms1, n1 = solve([(Rs, ts)]); print(f"[{name}] {nm} 단독: 집게 끝 {np.round(p1 * 100, 2)} cm | RMS {rms1:.1f} mm | 사용 {n1[0]}/{len(Rs)}")
        p, cs, rms, ns = solve(blocks)
        loo = [np.linalg.norm(solve([b for j, b in enumerate(blocks) if j != i])[0] - p) * 1000 for i in range(len(blocks))] if len(blocks) > 2 else []
        print(f"[{name}] 합동 ({len(blocks)}개 녹화): 집게 끝 {np.round(p * 100, 2)} cm, |p| {np.linalg.norm(p) * 100:.1f} cm | RMS {rms:.1f} mm | 사용 {ns} | 하나 빼고 다시 풀 때 변화 {np.round(loo, 1)} mm")
        out[name] = dict(tcp_in_rig_m=p.tolist(), residual_rms_mm=rms, takes=names, frames=ns, leave_one_out_mm=[float(x) for x in loo])
    if a.write and out:
        cfg = json.loads(Path(a.rig).read_text())
        for n, v in out.items(): cfg[n]["tcp_in_rig_m"] = v["tcp_in_rig_m"]; cfg[n]["pivot_joint"] = v
        Path(a.rig).write_text(json.dumps(cfg, indent=1, ensure_ascii=False)); print("기록", a.rig)


if __name__ == "__main__":
    main()
