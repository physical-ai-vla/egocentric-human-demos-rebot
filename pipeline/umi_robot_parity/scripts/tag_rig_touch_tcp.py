"""TCP (jaw tip in the tag-rig frame) from CORNER TOUCHES on a board of known geometry (2026-10-20, GPT review: tag-only, before the IMU
smoother). Each hold sample i: p_B_rig(i) + R_B_rig(i) t = corner_B(i). One t per UMI by robust least squares (Huber, f_scale 5 mm).
Board frame from the Orbbec board pose (median over the take); hold names -> true corners by the board pose (best permutation).
Quality per hold (used as an automatic gate): mean visible UMI tags, multi-corner PnP RMSE, mean viewing angle, in-hold jitter of the
TCP point, single-tag fraction. Evaluation: leave-one-corner-out (4 folds: solve on 3 corners, held-out corner error), fold-to-fold t
spread, and the full-data re-evaluation: same-corner spread across poses (main KPI), absolute bias, edges / diagonals.
    python scripts/tag_rig_touch_tcp.py datasets/tag_rig/<corners take> [--write]"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from tag_rig_pose import rig_pose                      # noqa: E402
from tag_rig_corners import board_pose, NAMES, CORNERS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--rig", default=str(ROOT / "configs/calibration/umi_tag_rig_v001.json"))
    ap.add_argument("--tag-size", type=float, default=0.048); ap.add_argument("--board-tag", type=float, default=0.05); ap.add_argument("--board", default="3x4@300")
    ap.add_argument("--gate-tags", type=float, default=1.5, help="min mean visible UMI tags per hold"); ap.add_argument("--gate-jitter-mm", type=float, default=15.0)
    ap.add_argument("--write", action="store_true"); ap.add_argument("--fused", action="store_true", help="10-20: rig poses from scripts/tag_rig_fuse.py (IMU orientation + tag centres)")
    ap.add_argument("--depth", action="store_true", help="tag centre range from the aligned depth"); ap.add_argument("--rot", default="imu", choices=["imu", "tag"]); a = ap.parse_args()
    E = Path(a.take); meta = json.loads((E / "meta.json").read_text()); cfg = json.loads(Path(a.rig).read_text())
    I = meta["intrinsics"]; K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]]); t = (np.load(E / "t_ns.npy") - meta["t_start_monotonic_ns"]) / 1e9
    D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11); prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    det = cv2.aruco.ArucoDetector(D, prm); p1 = cv2.aruco.DetectorParameters(); p1.markerBorderBits = 1; p1.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG; det1 = cv2.aruco.ArucoDetector(D, p1)
    bc, rest = a.board.split("x"); br, bid0 = rest.split("@"); bc, br, bid0 = int(bc), int(br), int(bid0)
    W = (bc + 0.3 * (bc - 1)) * a.board_tag; Hh = (br + 0.3 * (br - 1)) * a.board_tag
    truth = {"왼쪽 아래": np.array([0, 0, 0.0]), "오른쪽 아래": np.array([W, 0, 0.0]), "오른쪽 위": np.array([W, Hh, 0.0]), "왼쪽 위": np.array([0, Hh, 0.0])}
    rig = {h: {int(k): np.array(v) for k, v in cfg[n]["T_rig_tag"].items()} for h, n in NAMES.items()}; tcp0 = {h: np.array(cfg[n]["tcp_in_rig_m"]) for h, n in NAMES.items()}
    fused = {}
    if a.fused:
        from tag_rig_fuse import fuse_take
        for h_ in ("오른손", "왼손"):
            fr = fuse_take(E, h_, cfg, a.tag_size, use_depth=a.depth, rot_source=a.rot)
            if fr is not None: fused[h_] = {int(round((tn / 1e9) * 1e3)): (T, n) for tn, T, n in zip(fr["t_ns"], fr["T_C_rig"], fr["ntags"])}
        print("[터치 TCP] IMU 융합 자세 사용: " + ", ".join(f"{h_} {len(v)}프레임" for h_, v in fused.items()))
    holds, boards = [], []
    for p in meta["protocol"]:
        if "멈추세요" not in p["text"] or "," not in p["text"]: continue
        hand, corner = [x.strip() for x in p["text"].split(",")[:2]]; S = []
        for k in np.where((t >= p["t0"] + p["dur"] - 3.0) & (t < p["t0"] + p["dur"] - 0.1))[0]:
            g = cv2.imread(str(E / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE); cs, ids, _ = det.detectMarkers(g)
            if a.fused:
                key = int(round((t[k] + meta["t_start_monotonic_ns"] / 1e9) * 1e3)); fv = fused.get(hand, {}).get(key)
                if fv is not None: S.append(dict(T=fv[0], ntag=int(fv[1]), rmse=0.0, view=0.0))
                if len(boards) < 80 and k % 3 == 0:
                    B = board_pose(g, K, a.board_tag, det1, bc, br, bid0)
                    if B is not None: boards.append(B)
                continue
            r = rig_pose(cs, ids, rig[hand], K, a.tag_size)
            if r is not None and r["rmse_px"] < 2.0:
                Tt = {int(i): None for i in r["tags"]}; ray = r["T"][:3, 3] / np.linalg.norm(r["T"][:3, 3])
                views = [float(np.degrees(np.arccos(abs((r["T"] @ rig[hand][i])[:3, 2] @ ray)))) for i in r["tags"]]
                S.append(dict(T=r["T"], ntag=len(r["tags"]), rmse=r["rmse_px"], view=float(np.mean(views))))
            if len(boards) < 80 and k % 3 == 0:
                B = board_pose(g, K, a.board_tag, det1, bc, br, bid0)
                if B is not None: boards.append(B)
        holds.append(dict(hand=hand, corner=corner, text=p["text"], S=S))
    Tb = np.eye(4); Tb[:3, :3] = R.from_matrix(np.array([b[:3, :3] for b in boards])).mean().as_matrix(); Tb[:3, 3] = np.median([b[:3, 3] for b in boards], 0); Tbi = np.linalg.inv(Tb)
    print(f"[터치 TCP] {E.name}: 정지 {len(holds)}회, 보드 자세 {len(boards)}프레임 | 보드 {W * 1000:.0f} x {Hh * 1000:.0f} mm")
    rep = {}
    for hand in ("오른손", "왼손"):
        H = [h for h in holds if h["hand"] == hand and len(h["S"]) >= 5]
        if not H: continue
        for h in H:                                          # board-frame rig poses + quality
            h["RB"] = np.array([(Tbi @ s["T"])[:3, :3] for s in h["S"]]); h["pB"] = np.array([(Tbi @ s["T"])[:3, 3] for s in h["S"]])
            tipB = h["pB"] + np.einsum("nij,j->ni", h["RB"], tcp0[hand])
            # 10-20: keep only the frames where the UMI had SETTLED (the operator often reaches the cross late -> the first part of the
            # window was still moving: one hold drifted 37 mm then sat within 1-2 mm). Reference = median of the last 40 % of the window.
            ref = np.median(tipB[int(0.6 * len(tipB)):], 0); keep = np.linalg.norm(tipB - ref, axis=1) < 0.006
            if keep.sum() >= 5:
                h["RB"], h["pB"], h["S"], tipB = h["RB"][keep], h["pB"][keep], [s_ for s_, k_ in zip(h["S"], keep) if k_], tipB[keep]
            h["jit"] = float(np.linalg.norm(tipB - np.median(tipB, 0), axis=1).std() * 1000); h["settled"] = int(keep.sum())
            h["ntag"] = float(np.mean([s["ntag"] for s in h["S"]])); h["rmse"] = float(np.median([s["rmse"] for s in h["S"]])); h["view"] = float(np.median([s["view"] for s in h["S"]]))
            h["single"] = float(np.mean([s["ntag"] == 1 for s in h["S"]]))
        # name -> true corner by the board pose, using the current t
        means = {c: np.mean([np.median(h["pB"] + np.einsum("nij,j->ni", h["RB"], tcp0[hand]), 0) for h in H if h["corner"] == c], 0) for c in CORNERS if any(h["corner"] == c for h in H)}
        best = min(itertools.permutations(CORNERS), key=lambda perm: sum(np.linalg.norm(means[c][:2] - truth[q][:2]) for c, q in zip(CORNERS, perm) if c in means))
        cmap = dict(zip(CORNERS, best))
        print(f"== {hand} UMI: 정지 {len(H)}회 | 모서리 대응 " + ", ".join(f"{c}→{q}" for c, q in cmap.items()))
        for h in H:
            h["gate"] = h["ntag"] >= a.gate_tags and h["jit"] <= a.gate_jitter_mm
            print(f"   {h['text'][:26]:26s} 태그 {h['ntag']:.1f} (단일 {h['single']:.0%}) rmse {h['rmse']:.2f}px 각도 {h['view']:.0f}° 흔들림 {h['jit']:5.1f}mm → {'사용' if h['gate'] else '제외'}")
        G = [h for h in H if h["gate"]]
        if len({h["corner"] for h in G}) < 2: print("   사용할 정지가 부족 -- 건너뜀"); continue

        def solve(sub):
            Rb = np.concatenate([h["RB"] for h in sub]); pb = np.concatenate([h["pB"] for h in sub]); tgt = np.concatenate([np.tile(truth[cmap[h["corner"]]], (len(h["S"]), 1)) for h in sub])
            w = np.concatenate([np.full(len(h["S"]), 1 / np.sqrt(len(h["S"]))) for h in sub])          # each hold weighs the same
            f = lambda x: ((pb + np.einsum("nij,j->ni", Rb, x) - tgt) * w[:, None]).ravel()
            return least_squares(f, tcp0[hand], loss="huber", f_scale=0.005 / np.sqrt(30)).x

        def evaluate(x, sub):
            pts = {h["text"]: np.median(h["pB"] + np.einsum("nij,j->ni", h["RB"], x), 0) for h in sub}
            per = {c: np.array([pts[h["text"]] for h in sub if h["corner"] == c]) for c in CORNERS}; per = {c: v for c, v in per.items() if len(v)}
            spread = {c: (max(np.linalg.norm(u - v) for u, v in itertools.combinations(P, 2)) * 1000 if len(P) > 1 else 0.0) for c, P in per.items()}
            err = {c: (P.mean(0) - truth[cmap[c]]) * 1000 for c, P in per.items()}
            return spread, err, {c: P.mean(0) for c, P in per.items()}
        x_all = solve(G); sp0, er0, _ = evaluate(tcp0[hand], G); sp1, er1, m1 = evaluate(x_all, G)
        print(f"   TCP 기존 {np.round(tcp0[hand] * 100, 2)} cm → 터치 보정 {np.round(x_all * 100, 2)} cm (변화 {np.linalg.norm(x_all - tcp0[hand]) * 1000:.1f} mm)")
        print(f"   [전체 재평가] 같은 모서리 자세별 최대 차이(mm): 기존 " + ", ".join(f"{sp0[c]:.0f}" for c in sp0) + " → 보정 " + ", ".join(f"{sp1[c]:.0f}" for c in sp1))
        print(f"   [전체 재평가] 절대 오차 크기(mm): 기존 " + ", ".join(f"{np.linalg.norm(er0[c]):.0f}" for c in er0) + " → 보정 " + ", ".join(f"{np.linalg.norm(er1[c]):.0f}" for c in er1)
              + f" | 편향 기존 {np.round(np.mean(list(er0.values()), 0), 1)} → 보정 {np.round(np.mean(list(er1.values()), 0), 1)}")
        if len(m1) == 4:
            pr = [("왼쪽 아래", "오른쪽 아래"), ("오른쪽 아래", "오른쪽 위"), ("오른쪽 위", "왼쪽 위"), ("왼쪽 위", "왼쪽 아래"), ("왼쪽 아래", "오른쪽 위"), ("오른쪽 아래", "왼쪽 위")]
            print("   [전체 재평가] 변/대각선 오차: " + ", ".join(f"{(np.linalg.norm(m1[x] - m1[y]) / np.linalg.norm(truth[cmap[x]] - truth[cmap[y]]) - 1) * 100:+.1f}%" for x, y in pr))
        folds = []
        for c in CORNERS:                                    # leave-one-corner-out
            tr = [h for h in G if h["corner"] != c]; te = [h for h in G if h["corner"] == c]
            if len(set(h["corner"] for h in tr)) < 2 or not te: continue
            x = solve(tr); spT, erT, _ = evaluate(x, te); folds.append(x)
            print(f"   [빼고 검증] {c} 제외 → TCP {np.round(x * 100, 2)} cm | 그 모서리 절대 오차 {np.linalg.norm(erT[c]):.1f} mm {np.round(erT[c], 1)} | 자세별 최대 차이 {spT[c]:.1f} mm")
        if folds: print(f"   fold 간 TCP 흩어짐: 최대 {max(np.linalg.norm(f - x_all) for f in folds) * 1000:.1f} mm (전체 해 기준)")
        rep[hand] = dict(tcp_touch_m=x_all.tolist(), tcp_prev_m=tcp0[hand].tolist(), used_holds=len(G), total_holds=len(H),
                         loco_tcp_m=[f.tolist() for f in folds], spread_prev_mm=sp0, spread_new_mm=sp1)
    out = ROOT / "datasets/tag_rig" / f"touch_tcp_{E.name}.json"; out.write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print("저장", out)
    if a.write:
        cfg = json.loads(Path(a.rig).read_text())
        for hand, v in rep.items():
            n = NAMES[hand]; cfg[n]["tcp_pivot_m"] = cfg[n].get("tcp_pivot_m", cfg[n]["tcp_in_rig_m"]); cfg[n]["tcp_in_rig_m"] = v["tcp_touch_m"]; cfg[n]["touch_tcp"] = dict(take=E.name, **{k: v[k] for k in ("used_holds", "total_holds", "loco_tcp_m")})
        Path(a.rig).write_text(json.dumps(cfg, indent=1, ensure_ascii=False)); print("기록", a.rig)


if __name__ == "__main__":
    main()
