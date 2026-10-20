#!/usr/bin/env python3
"""[2026-10-15] Is the UMI pose wrong, or only its projection? AprilTag + jaw-tape check in the overhead Orbbec colour.

Per head frame (every --step-th), per side whose top tag (L10 / R20, tag36h11) is detected:
  tag_uv           detected tag centre (px), tag_W = depth at the centre (5x5 median) back-projected and mapped to W (R_WC, t_WC)
  tips_uv / tips_W orange jaw tape (HSV) within TIP_R px of the tag, largest blob centroid; depth -> W   (= where the TCP should be)
  body_uv          canonical Orbbec body point p_W projected (the smoother's position anchor)
  cam_uv[_old]     smoothed wrist-camera origin T_W_c (rcb3 / old)       tcp_uv[_old]  smoothed TCP origin T_W_tcp (rcb3 / old)
  errors           px: tag<->body; 3D (mm): |tag_W - p_W|, |tcp_W - tips_W| (rcb3 / old), |cam_W - tag_W|
  dt_ms            head frame time - nearest canonical sample time;  speed_mps = canonical body speed (slow = < 0.15 m/s)
Constant offsets (tag vs body point, camera vs tag) are expected; what matters is their size and whether they stay constant.
-> <out>/frames.csv, <out>/montage_<n>.jpg (slow frames, both sides), printed summary.
usage: orbbec_tag_check.py <episode> ... --out DIR [--step 3]
"""
import argparse, csv, json, pathlib, sys

import cv2, numpy as np, pupil_apriltags as pa

H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_cart20"))
from ego_cart20.sources.hfold_export import read_mcap        # noqa: E402

TAG = {10: "left", 20: "right"}; TIP_R = 110; SLOW = 0.15; OLD, NEW = "umi_fusion", "umi_fusion_rcb3"


def backproject(K, R_WC, t_WC, uv, depth_img):
    u, v = int(round(uv[0])), int(round(uv[1])); h, w = depth_img.shape
    if not (2 <= u < w - 2 and 2 <= v < h - 2): return None
    d = depth_img[v - 2:v + 3, u - 2:u + 3].astype(float); d = d[d > 0]
    if len(d) < 5: return None
    z = np.median(d) / 1000.0; pc = np.array([(uv[0] - K[0, 2]) * z / K[0, 0], (uv[1] - K[1, 2]) * z / K[1, 1], z])
    return R_WC @ pc + t_WC


def proj(K, R_WC, t_WC, P):
    pc = (np.atleast_2d(P) - t_WC) @ R_WC
    return np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], 1)


def tips(img, c):
    x0, y0 = int(max(c[0] - TIP_R, 0)), int(max(c[1] - TIP_R, 0)); roi = img[y0:int(c[1] + TIP_R), x0:int(c[0] + TIP_R)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV); m = cv2.inRange(hsv, (3, 130, 110), (22, 255, 255))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    k = 1 + int(np.argmax(st[1:, 4]))
    return None if st[k, 4] < 12 else cen[k] + [x0, y0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episodes", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--old", default="umi_fusion"); ap.add_argument("--new", default="umi_fusion_rcb3")
    a = ap.parse_args(); global OLD, NEW; OLD, NEW = a.old, a.new; out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    det = pa.Detector(families="tag36h11", quad_decimate=1.0, refine_edges=True, decode_sharpening=0.5); rows, shots = [], []
    for e in map(pathlib.Path, a.episodes):
        sm = json.loads((e.parent / "session_meta.json").read_text())["devices"]["head_depth"]["intrinsics"]
        K = np.array([[sm["fx"], 0, sm["cx"]], [0, sm["fy"], sm["cy"]], [0, 0, 1.0]])
        fm, _ = read_mcap(e); cz = np.load(e / "derived/body_track/canonical.npz", allow_pickle=True); R_WC, t_WC = cz["R_WC"], cz["t_WC"]
        t0 = int(min(fm["left_wrist"][0, 1], fm["right_wrist"][0, 1], cz["t_ns"][0]))
        Z = {v: {s: np.load(e / f"derived/{v}/smoothed_{s}.npz") for s in ("left", "right")} for v in (OLD, NEW)}
        hv = {int(f): int(c) for f, c in fm["head_depth"]}; cap = cv2.VideoCapture(str(e / "head_depth.mp4")); fi = -1
        while True:
            ok, img = cap.read(); fi += 1
            if not ok: break
            if fi % a.step or fi not in hv: continue
            g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC); dets = [d for d in det.detect(g) if d.tag_id in TAG and not d.hamming]
            if not dets: continue
            dp = e / "head_depth_depth" / f"{fi:06d}.png"; depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED) if dp.exists() else None
            tc = hv[fi]; ci = int(np.argmin(np.abs(cz["t_ns"] - tc))); vis = img.copy(); slow_all = True
            for d in dets:
                side = TAG[d.tag_id]; c = d.center / 2; r = dict(episode=f"{e.parent.name}/{e.name}", frame=fi, side=side, t_s=round((tc - t0) / 1e9, 3), dt_ms=round((cz["t_ns"][ci] - tc) / 1e6, 1),
                                                                  tag_u=round(c[0], 1), tag_v=round(c[1], 1))
                pW = cz[f"p_W_{side}"][ci]; okb = bool(cz[f"valid_{side}"][ci]) and np.isfinite(pW).all()
                sp = np.linalg.norm(cz[f"p_W_{side}"][min(ci + 3, len(cz["t_ns"]) - 1)] - cz[f"p_W_{side}"][max(ci - 3, 0)]) / max((cz["t_ns"][min(ci + 3, len(cz["t_ns"]) - 1)] - cz["t_ns"][max(ci - 3, 0)]) / 1e9, 1e-3)
                r["speed_mps"] = round(float(sp), 3) if np.isfinite(sp) else None; slow_all &= bool(np.isfinite(sp) and sp < SLOW)
                tagW = backproject(K, R_WC, t_WC, c, depth) if depth is not None else None; tp = tips(img, c); tipW = backproject(K, R_WC, t_WC, tp, depth) if (tp is not None and depth is not None) else None
                if tp is not None: r.update(tips_u=round(tp[0], 1), tips_v=round(tp[1], 1))
                cv2.polylines(vis, [(d.corners / 2).astype(np.int32)], True, (255, 255, 0), 1)
                if okb:
                    bu = proj(K, R_WC, t_WC, pW)[0]; r.update(body_u=round(bu[0], 1), body_v=round(bu[1], 1), err_tag_body_px=round(float(np.linalg.norm(bu - c)), 1))
                    cv2.circle(vis, tuple(map(int, bu)), 4, (0, 255, 255), -1)
                    if tagW is not None: r["d_tag_body_mm"] = round(float(np.linalg.norm(tagW - pW) * 1e3), 1)
                for v, suf, th in ((OLD, "_old", 1), (NEW, "", 2)):
                    z = Z[v][side]; i = int(np.argmin(np.abs(z["t"] - (tc - t0) / 1e9)))
                    if not z["full_6dof"][i]: continue
                    camW, tcpW = z["T_W_c"][i][:3, 3], z["T_W_tcp"][i][:3, 3]; cu, tu = proj(K, R_WC, t_WC, camW)[0], proj(K, R_WC, t_WC, tcpW)[0]
                    r.update({f"cam_u{suf}": round(cu[0], 1), f"cam_v{suf}": round(cu[1], 1), f"tcp_u{suf}": round(tu[0], 1), f"tcp_v{suf}": round(tu[1], 1)})
                    if tagW is not None: r[f"d_cam_tag_mm{suf}"] = round(float(np.linalg.norm(camW - tagW) * 1e3), 1)
                    if tipW is not None: r[f"d_tcp_tips_mm{suf}"] = round(float(np.linalg.norm(tcpW - tipW) * 1e3), 1)
                    if tp is not None: r[f"err_tcp_tips_px{suf}"] = round(float(np.linalg.norm(tu - tp)), 1)
                    cv2.drawMarker(vis, tuple(map(int, cu)), (255, 0, 255), cv2.MARKER_SQUARE, 10, th); cv2.drawMarker(vis, tuple(map(int, tu)), (0, 0, 255), cv2.MARKER_CROSS, 14, th)
                if tp is not None: cv2.drawMarker(vis, tuple(map(int, tp)), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
                rows.append(r)
            if slow_all and len(dets) == 2: shots.append((f"{e.parent.name[-6:]} ep{e.name[-2:]} f{fi}", vis))
        print("done", e.parent.name, e.name, flush=True)
    cols = sorted({k for r in rows for k in r}, key=lambda c: (c not in ("episode", "frame", "side", "t_s", "dt_ms", "speed_mps"), c))
    with open(out / "frames.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    sel = shots[:: max(1, len(shots) // 24)][:24]
    for j in range(0, len(sel), 12):
        tiles = []
        for name, im in sel[j:j + 12]:
            im = im.copy(); cv2.putText(im, name, (6, 20), 0, 0.55, (255, 255, 255), 2); tiles.append(cv2.resize(im, (424, 240)))
        while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f"montage_{j // 12}.jpg"), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))
    print(f"{len(rows)} tag observations, {len(shots)} slow two-tag frames -> {out}")


if __name__ == "__main__":
    main()
