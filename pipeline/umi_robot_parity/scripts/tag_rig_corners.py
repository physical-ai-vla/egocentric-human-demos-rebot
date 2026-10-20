"""Baseline accuracy of the raw UMI TCP measurement (2026-10-19, GPT review: do this BEFORE the IMU smoother so the gain can be attributed).
Take = tag_rig_record --proto corners: each outer corner of the fixed desk AprilGrid touched 4x (straight / tilted forward / left / right)
per UMI, 4 s hold. TCP per hold = median over the LAST 1.5 s of T_C_rig (multi-corner PnP, scripts/tag_rig_pose.py) * tcp_in_rig.
Ground truth = the AprilGrid outer square (6x6 tags, spacing 0.3): side = 7.5 * board tag size (--board-tag, the PRINTED black-square
side; the printer shrank the UMI tags 50 -> 48 mm). If the Orbbec also sees the board (markerBorderBits=1), the board pose gives the
corners in the camera frame -> absolute errors and per-axis bias in the board frame.
Reports: per-corner repeatability (std) and tilt spread, the 4 edges + 2 diagonals vs truth (scale ratio), out-of-plane error, absolute
error / board-frame bias when the board is visible, left vs right UMI agreement on the same corners.
    python scripts/tag_rig_corners.py datasets/tag_rig/<corners take> --board-tag 0.024"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from tag_rig_pose import rig_pose                       # noqa: E402
NAMES = {"오른손": "UMI 오른손", "왼손": "UMI 왼손"}
CORNERS = ("왼쪽 아래", "오른쪽 아래", "오른쪽 위", "왼쪽 위")


def board_pose(gray, K, tag, det1, cols=3, rows=4, id0=300, sp=0.3):
    """AprilGrid (cols x rows, ids id0.., 1-bit border) -> T_C_board or None. Board frame: origin = outer corner of tag id0, x along cols, y up rows."""
    cs, ids, _ = det1.detectMarkers(gray)
    if ids is None: return None
    P3, P2 = [], []
    for c, i in zip(cs, ids.ravel()):
        i = int(i) - id0
        if not 0 <= i < cols * rows: continue
        r, col = divmod(i, cols); x0, y0 = col * tag * (1 + sp), r * tag * (1 + sp)
        P3.append(np.array([[x0, y0 + tag, 0], [x0 + tag, y0 + tag, 0], [x0 + tag, y0, 0], [x0, y0, 0]], float)); P2.append(c.reshape(4, 2))
    if len(P3) < 3: return None
    P3, P2 = np.concatenate(P3), np.concatenate(P2).astype(np.float64); ok, rv, tv = cv2.solvePnP(P3, P2, K, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
    if not ok: return None
    rv, tv = cv2.solvePnPRefineLM(P3, P2, K, np.zeros(5), rv, tv); T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rv)[0]; T[:3, 3] = tv.ravel(); return T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--rig", default=str(ROOT / "configs/calibration/umi_tag_rig_v001.json"))
    ap.add_argument("--tag-size", type=float, default=0.048); ap.add_argument("--board-tag", type=float, default=0.05); ap.add_argument("--board", default="3x4@300", help="colsxrows@id0 (10-19 big board aprilgrid_3x4_50mm_id300; old 6x6 = 6x6@0 with --board-tag 0.025)"); a = ap.parse_args()   # 10-19 board reprinted aprilgrid_6x6_25mm_v2 at true 100%
    E = Path(a.take); meta = json.loads((E / "meta.json").read_text()); cfg = json.loads(Path(a.rig).read_text())
    I = meta["intrinsics"]; K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]]); t = (np.load(E / "t_ns.npy") - meta["t_start_monotonic_ns"]) / 1e9
    prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11); det = cv2.aruco.ArucoDetector(D, prm)
    p1 = cv2.aruco.DetectorParameters(); p1.markerBorderBits = 1; p1.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG; det1 = cv2.aruco.ArucoDetector(D, p1)
    rig = {h: {int(k): np.array(v) for k, v in cfg[n]["T_rig_tag"].items()} for h, n in NAMES.items()}; tcp = {h: np.array(cfg[n]["tcp_in_rig_m"]) for h, n in NAMES.items()}
    bc, rest = a.board.split("x"); br, bid0 = rest.split("@"); bc, br, bid0 = int(bc), int(br), int(bid0)
    W = (bc + 0.3 * (bc - 1)) * a.board_tag; Hh = (br + 0.3 * (br - 1)) * a.board_tag
    truth = {"왼쪽 아래": np.array([0, 0, 0.0]), "오른쪽 아래": np.array([W, 0, 0.0]), "오른쪽 위": np.array([W, Hh, 0.0]), "왼쪽 위": np.array([0, Hh, 0.0])}
    holds, boards = [], []
    for p in meta["protocol"]:
        if "멈추세요" not in p["text"] or "," not in p["text"]: continue
        hand, corner, tilt = [x.strip() for x in p["text"].split(",")[:3]]; pts, rms = [], []
        for k in np.where((t >= p["t0"] + p["dur"] - (3.0 if p["dur"] >= 5.5 else 1.5)) & (t < p["t0"] + p["dur"] - 0.1))[0]:
            g = cv2.imread(str(E / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE); cs, ids, _ = det.detectMarkers(g)
            r = rig_pose(cs, ids, rig[hand], K, a.tag_size)
            if r is not None and r["rmse_px"] < 2.0: pts.append((r["T"] @ np.r_[tcp[hand], 1.0])[:3]); rms.append(r["rmse_px"])
            if len(boards) < 60 and k % 3 == 0:
                B = board_pose(g, K, a.board_tag, det1, bc, br, bid0)
                if B is not None: boards.append(B)
        holds.append(dict(hand=hand, corner=corner, tilt=tilt, p=np.median(pts, 0) if len(pts) >= 3 else None, n=len(pts), jitter_mm=float(np.std(np.linalg.norm(np.array(pts) - np.median(pts, 0), axis=1)) * 1000) if len(pts) >= 3 else None))
    ok = [h for h in holds if h["p"] is not None]
    print(f"[모서리] 정지 {len(holds)}회 중 측정 성공 {len(ok)} | 보드 정답 {W * 1000:.1f} x {Hh * 1000:.1f} mm (태그 {a.board_tag * 1000:.1f} mm, {a.board}) | 보드가 Orbbec에 보인 프레임 {len(boards)}")
    rep = {"holds": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in h.items()} for h in holds]}
    Tb = None
    if len(boards) >= 5:
        from scipy.spatial.transform import Rotation as R
        Tb = np.eye(4); Tb[:3, :3] = R.from_matrix(np.array([b[:3, :3] for b in boards])).mean().as_matrix(); Tb[:3, 3] = np.median([b[:3, 3] for b in boards], 0)
    for hand in ("오른손", "왼손"):
        H = [h for h in ok if h["hand"] == hand]
        if not H: continue
        print(f"== {hand} UMI ({len(H)}회)")
        means = {}
        for c in CORNERS:
            P = np.array([h["p"] for h in H if h["corner"] == c])
            if len(P) == 0: continue
            means[c] = P.mean(0); spread = max(np.linalg.norm(x - y) for x, y in itertools.combinations(P, 2)) * 1000 if len(P) > 1 else 0.0
            print(f"   {c}: {len(P)}회, 반복 표준편차 {np.linalg.norm(P - P.mean(0), axis=1).std() * 1000 + np.linalg.norm(P - P.mean(0), axis=1).mean() * 0:.1f} mm, 평균에서 최대 {np.linalg.norm(P - P.mean(0), axis=1).max() * 1000:.1f} mm, 기울기 바꿀 때 최대 차이 {spread:.1f} mm")
            for h in H:
                if h["corner"] == c: print(f"      {h['tilt']:8s} 평균에서 {np.linalg.norm(h['p'] - P.mean(0)) * 1000:5.1f} mm (프레임 {h['n']}, 흔들림 {h['jitter_mm']:.1f} mm)")
        if len(means) == 4 and Tb is not None:          # 10-20: name -> true corner by the board pose (the board was placed rotated 90 deg)
            bm = {c: (np.linalg.inv(Tb) @ np.r_[means[c], 1])[:3] for c in CORNERS}
            best = min(itertools.permutations(CORNERS), key=lambda perm: sum(np.linalg.norm(bm[c][:2] - truth[q][:2]) for c, q in zip(CORNERS, perm)))
            tmap = {c: truth[q] for c, q in zip(CORNERS, best)}
            if list(best) != list(CORNERS): print("   (보드 자세로 모서리 대응을 다시 맞춤: " + ", ".join(f"{c}→보드 {q}" for c, q in zip(CORNERS, best)) + ")")
        else: tmap = dict(truth)
        if len(means) == 4:
            pairs = [("왼쪽 아래", "오른쪽 아래"), ("오른쪽 아래", "오른쪽 위"), ("오른쪽 위", "왼쪽 위"), ("왼쪽 위", "왼쪽 아래"), ("왼쪽 아래", "오른쪽 위"), ("오른쪽 아래", "왼쪽 위")]
            for x, y in pairs:
                m_ = np.linalg.norm(means[x] - means[y]) * 1000; tr = np.linalg.norm(tmap[x] - tmap[y]) * 1000
                print(f"   {'대각선' if (x, y) in pairs[4:] else '변'} {x}–{y}: 측정 {m_:6.1f} mm / 정답 {tr:6.1f} mm → 비율 {m_ / tr:.3f} ({(m_ / tr - 1) * 100:+.1f}%)")
            Pm = np.array([means[c] for c in CORNERS]); cen = Pm.mean(0); _, _, vt = np.linalg.svd(Pm - cen); n = vt[2]
            allP = np.array([h["p"] for h in H]); print(f"   평면에서 벗어남: 모서리 평균 최대 {np.abs((Pm - cen) @ n).max() * 1000:.1f} mm, 개별 측정 최대 {np.abs((allP - cen) @ n).max() * 1000:.1f} mm")
            if Tb is not None:
                err = np.array([(np.linalg.inv(Tb) @ np.r_[means[c], 1])[:3] - tmap[c] for c in CORNERS]) * 1000
                print(f"   절대 오차(보드 좌표 mm): " + ", ".join(f"{c} {np.round(e, 1)}" for c, e in zip(CORNERS, err)) + f" | 평균 편향 {np.round(err.mean(0), 1)} | 크기 {np.linalg.norm(err, axis=1).round(1)}")
    for c in CORNERS:
        R_ = [h["p"] for h in ok if h["hand"] == "오른손" and h["corner"] == c]; L_ = [h["p"] for h in ok if h["hand"] == "왼손" and h["corner"] == c]
        if R_ and L_: print(f"   {c}: 오른손 평균 vs 왼손 평균 차이 {np.linalg.norm(np.mean(R_, 0) - np.mean(L_, 0)) * 1000:.1f} mm")
    outp = E / "corners_report.json"
    try: outp.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    except PermissionError: outp = ROOT / "datasets/tag_rig" / f"corners_report_{E.name}.json"; outp.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("저장", outp)
    if False: (E / "corners_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print("저장", E / "corners_report.json")


if __name__ == "__main__":
    main()
