"""Solve the UMI AprilTag rig from a tag_rig_record take (2026-10-16): per-tag PnP + RGB-D quality, tag-to-tag rigid transforms, and the
still-window agreement that gates the global anchor (user plan: >= 2 tags agree within 2-3 cm / 5-10 deg -> correction allowed).
Per detection: IPPE_SQUARE PnP (tag size = black square, default 0.05 m; Orbbec colour treated as pinhole with the SDK fx fy cx cy),
reprojection error, pixel side, viewing angle (tag normal vs camera ray), depth check = median aligned depth at the 4 corners vs the PnP
corner z (mm), plane residual of the depth points inside the tag.
Rig frame = tag 200 (or the most seen tag). T_rig_tag_i = robust mean over good co-visible frames (chordal rotation mean, median
translation), chained through a common tag when a pair is never co-visible.
    python scripts/tag_rig_solve.py datasets/tag_rig/<ts> [--tag-size 0.05]  -> <take>/tag_rig.json + printed report"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

GROUPS = {"UMI 오른손": (200, 201, 202), "UMI 왼손": (210, 211, 212), "로봇 오른팔": (220, 221, 222), "로봇 왼팔": (230, 231, 232)}
GOOD = dict(reproj_px=1.5, side_px=25, view_deg=65)


def T_from(rvec, tvec):
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rvec)[0]; T[:3, 3] = np.ravel(tvec); return T


def mean_T(Ts):
    q = R.from_matrix(np.array([T[:3, :3] for T in Ts])).mean(); T = np.eye(4); T[:3, :3] = q.as_matrix(); T[:3, 3] = np.median([T_[:3, 3] for T_ in Ts], 0); return T


def dT(A, B):
    E = np.linalg.inv(A) @ B; return float(np.linalg.norm(E[:3, 3])), float(np.degrees(R.from_matrix(E[:3, :3]).magnitude()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--tag-size", type=float, default=0.048); a = ap.parse_args()
    E = Path(a.take); meta = json.loads((E / "meta.json").read_text()); I = meta["intrinsics"]; ds = float(meta["depth_scale"] or 1.0)
    K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]]); dist = np.zeros(5)
    depth = np.load(E / "depth.npz")["depth"]; t = np.load(E / "t_ns.npy"); t0 = meta["t_start_monotonic_ns"]; ts = (t - t0) / 1e9
    s = a.tag_size / 2; obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), prm)
    F = []                                                                     # per frame: {id: dict(T, quality...)}
    for k in range(len(t)):
        img = cv2.imread(str(E / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE); dep = depth[k].astype(np.float64) * ds
        if dep.max() > 100: dep = dep / 1000.0                                # depth_scale in mm units -> metres
        c, ids, _ = det.detectMarkers(img); fr = {}
        for cc, i in zip(c if ids is not None else [], ids.ravel() if ids is not None else []):
            i = int(i)
            if i not in meta["tag_ids"]: continue
            px = cc.reshape(4, 2).astype(np.float64); ok, rv, tv = cv2.solvePnP(obj, px, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok: continue
            T = T_from(rv, tv); rp = np.linalg.norm(cv2.projectPoints(obj, rv, tv, K, dist)[0].reshape(4, 2) - px, axis=1).mean()
            side = float(np.mean(np.linalg.norm(px - np.roll(px, 1, 0), axis=1))); n = T[:3, 2]; ray = T[:3, 3] / np.linalg.norm(T[:3, 3])
            view = float(np.degrees(np.arccos(min(1.0, abs(n @ ray)))))
            zc = (T[:3, :3] @ obj.T + T[:3, 3:4])[2]; dz = []
            for (u, v), zp in zip(px, zc):
                u0, v0 = int(round(u)), int(round(v)); win = dep[max(0, v0 - 2):v0 + 3, max(0, u0 - 2):u0 + 3]; win = win[win > 0.05]
                if win.size: dz.append((np.median(win) - zp) * 1000)
            mask = np.zeros(img.shape, np.uint8); cv2.fillConvexPoly(mask, px.astype(np.int32), 1); vv, uu = np.nonzero(mask); zz = dep[vv, uu]; good = zz > 0.05
            plane_res = None
            if good.sum() > 30:
                X = np.c_[(uu[good] - K[0, 2]) * zz[good] / K[0, 0], (vv[good] - K[1, 2]) * zz[good] / K[1, 1], zz[good]]
                cen = X.mean(0); _, sv, vt = np.linalg.svd(X - cen, full_matrices=False); plane_res = float(np.sqrt(np.mean(((X - cen) @ vt[2]) ** 2)) * 1000)
            fr[i] = dict(T=T, reproj=float(rp), side=side, view=view, depth_dz_mm=float(np.median(dz)) if dz else None, n_depth=len(dz), plane_res_mm=plane_res)
        F.append(fr)
    ok = lambda d: d["reproj"] < GOOD["reproj_px"] and d["side"] > GOOD["side_px"] and d["view"] < GOOD["view_deg"]
    allq = [d for f in F for d in f.values()]
    if allq:
        print(f"[태그 리그] 프레임 {len(F)} | 품질 p50: 재투영 {np.median([d['reproj'] for d in allq]):.2f}px, 한 변 {np.median([d['side'] for d in allq]):.0f}px, 보는 각도 {np.median([d['view'] for d in allq]):.0f}°, "
              f"depth−PnP z {np.nanmedian([d['depth_dz_mm'] if d['depth_dz_mm'] is not None else np.nan for d in allq]):+.1f} mm, 평면 잔차 {np.nanmedian([d['plane_res_mm'] if d['plane_res_mm'] is not None else np.nan for d in allq]):.1f} mm")
    rep = dict(take=E.name, tag_size_m=a.tag_size, rigs={})
    st = next((p_ for p_ in meta["protocol"] if p_["text"].startswith("그대로")), None)
    w = [k for k in range(len(F)) if st and st["t0"] + 1 <= ts[k] < st["t0"] + st["dur"] - 1]
    for name, G in GROUPS.items():                                   # 10-17: one rig per UMI / robot gripper
        cnt = {i: sum(i in f for f in F) for i in G}
        if sum(cnt.values()) == 0: continue
        ref = G[0] if cnt[G[0]] else max(cnt, key=cnt.get); pair = {}
        for f in F:
            g = [i for i in f if i in G and ok(f[i])]
            for i in g:
                for j in g:
                    if i != j: pair.setdefault((i, j), []).append(np.linalg.inv(f[i]["T"]) @ f[j]["T"])      # T_tagi_tagj
        # 10-17: connect tags through the BEST-supported pairs (maximum spanning tree on co-visible frame counts). The first version
        # took any pair with >= 5 frames in dict order: right UMI tag 201 was attached via 200-201 (9 frames) instead of 202-201 (57)
        # and the pivot frames (mostly 201+202) scattered 15 cm.
        T_rig = {ref: np.eye(4)}
        while True:
            cand = [(len(Lp), i, j) for (i, j), Lp in pair.items() if i in T_rig and j not in T_rig and len(Lp) >= 5]
            if not cand: break
            _, i, j = max(cand); T_rig[j] = T_rig[i] @ mean_T(pair[(i, j)])
        print(f"[{name}] 검출 {cnt} | 기준 {ref} | 함께 본 좋은 쌍 " + str({f'{i}-{j}': len(Lp) for (i, j), Lp in pair.items() if i < j}) + f" | 리그에 들어간 태그 {sorted(T_rig)}")
        for i, T in T_rig.items():
            if i != ref and pair.get((ref, i)):
                res = np.median([dT(T, M) for M in pair[(ref, i)]], 0)
                print(f"   T_{ref}_{i}: 이동 {np.round(T[:3, 3] * 100, 1)} cm, 회전 {np.degrees(R.from_matrix(T[:3, :3]).magnitude()):.1f}°, 쌍 잔차 p50 {res[0] * 100:.2f} cm / {res[1]:.1f}°")
        rr = dict(ref=ref, counts=cnt, T_rig_tag={str(i): T.tolist() for i, T in T_rig.items()})
        dis, poses = [], []
        for k in w:
            est = {i: F[k][i]["T"] @ np.linalg.inv(T_rig[i]) for i in F[k] if i in T_rig and ok(F[k][i])}
            ids = sorted(est)
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)): dis.append(dT(est[ids[x]], est[ids[y]]))
            if est: poses.append(mean_T(list(est.values())))
        if dis:
            d = np.array(dis); rr.update(still_disagree_cm_p50=float(np.median(d[:, 0]) * 100), still_disagree_cm_p90=float(np.percentile(d[:, 0], 90) * 100),
                                         still_disagree_deg_p50=float(np.median(d[:, 1])), still_disagree_deg_p90=float(np.percentile(d[:, 1], 90)))
            print(f"   정지 구간 태그 간 불일치: 위치 p50 {rr['still_disagree_cm_p50']:.2f} / p90 {rr['still_disagree_cm_p90']:.2f} cm, 회전 p50 {rr['still_disagree_deg_p50']:.1f} / p90 {rr['still_disagree_deg_p90']:.1f}°")
        if poses:
            P = np.array([T[:3, 3] for T in poses]); rr.update(still_pos_std_mm=float(np.linalg.norm(P - P.mean(0), axis=1).std() * 1000), still_frames=len(poses))
            print(f"   정지 구간 리그 위치 흔들림(std) {rr['still_pos_std_mm']:.1f} mm ({len(poses)} 프레임, 2개 이상 {sum(1 for k in w if sum(ok(F[k][i]) for i in F[k] if i in G) >= 2)}/{len(w)})")
        rep["rigs"][name] = rr
    (E / "tag_rig.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print(f"   저장 {E / 'tag_rig.json'}")


if __name__ == "__main__":
    main()
