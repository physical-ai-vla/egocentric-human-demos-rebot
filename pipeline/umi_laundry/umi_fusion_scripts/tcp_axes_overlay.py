#!/usr/bin/env python3
"""[2026-10-15] Absolute TCP orientation sanity check (the smoother residuals cannot see a wrong camera<->IMU rotation).

Projects the smoothed TCP frame of BOTH pose sources into the overhead Orbbec colour image of the same instant:
  thin lines = derived/umi_fusion (Kalibr R_cb), thick lines = derived/umi_fusion_rcb3 (data_v003 R_cb);
  red = TCP x (contract: camera optical axis = the direction the jaws point), green = y, blue = z; 8 cm long.
  yellow dot = Orbbec canonical body point p_W (projection check: it must sit on the HandUMI unit).
Frames per episode: rest (1.5 s), approach (0.7 s before the first grasp), grasp (first open->closed), max pitch, max roll
(of the rcb3 TCP x axis). One sheet per episode -> <out>/<session>_<ep>.jpg.
usage: tcp_axes_overlay.py <episode> ... --out DIR
"""
import argparse, json, pathlib, sys

import cv2, numpy as np

H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_cart20"))
from ego_cart20.sources.hfold_export import read_mcap        # noqa: E402
from ego_cart20.sources.jaw_visual import binary_from_px     # noqa: E402

AX = 0.12; CROP = 150; COLORS = [(0, 0, 255), (0, 200, 0), (255, 80, 0)]


def proj(K, R_WC, t_WC, P):
    pc = (P - t_WC) @ R_WC                                    # world -> Orbbec camera (R_WC: camera -> world)
    return np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], 1), pc[:, 2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episodes", nargs="+"); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for e in map(pathlib.Path, a.episodes):
        sm = json.loads((e.parent / "session_meta.json").read_text())["devices"]["head_depth"]["intrinsics"]
        K = np.array([[sm["fx"], 0, sm["cx"]], [0, sm["fy"], sm["cy"]], [0, 0, 1.0]])
        fm, _ = read_mcap(e); cz = np.load(e / "derived/body_track/canonical.npz", allow_pickle=True)
        t0 = int(min(fm["left_wrist"][0, 1], fm["right_wrist"][0, 1], cz["t_ns"][0])); R_WC, t_WC = cz["R_WC"], cz["t_WC"]
        hv = fm["head_depth"]; cap = cv2.VideoCapture(str(e / "head_depth.mp4")); tiles = []
        Z = {v: {s: np.load(e / f"derived/{v}/smoothed_{s}.npz") for s in ("left", "right")} for v in ("umi_fusion", "umi_fusion_rcb3")}
        for side in ("left", "right"):
            zb = Z["umi_fusion_rcb3"][side]; t = zb["t"]; ok = zb["full_6dof"]; x = zb["T_W_tcp"][:, :3, 0]
            j = np.load(e / f"derived/jaw/jaw_{side}.npz", allow_pickle=True); b, vv = binary_from_px(j["t_ns"], j["d_px"]); tj = (j["t_ns"] - t0) / 1e9
            cl = tj[1:][(b[:-1] == 1) & (b[1:] == 0) & vv[1:]]; cl = cl[cl > 3.5]
            pitch = np.degrees(np.arcsin(np.clip(-x[:, 2], -1, 1))); roll = np.degrees(np.arctan2(zb["T_W_tcp"][:, 2, 1], zb["T_W_tcp"][:, 2, 2]))
            picks = {"rest": 1.5, "approach": (cl[0] - 0.7) if len(cl) else None, "grasp": cl[0] if len(cl) else None,
                     "max pitch": float(t[ok][np.argmax(np.abs(pitch[ok]))]), "max roll": float(t[ok][np.argmax(np.abs(roll[ok]))])}
            for name, tq in picks.items():
                if tq is None: continue
                k = int(np.argmin(np.abs(hv[:, 1] - (t0 + tq * 1e9)))); cap.set(cv2.CAP_PROP_POS_FRAMES, int(hv[k, 0])); okf, im = cap.read()
                if not okf: continue
                i = int(np.argmin(np.abs(t - tq)))
                for v, th in (("umi_fusion", 1), ("umi_fusion_rcb3", 3)):
                    T = Z[v][side]["T_W_tcp"][min(i, len(Z[v][side]["t"]) - 1)]; P = np.vstack([T[:3, 3], T[:3, 3] + AX * T[:3, :3].T])
                    uv, d = proj(K, R_WC, t_WC, P)
                    if (d > 0).all():
                        for c in range(3): cv2.line(im, tuple(map(int, uv[0])), tuple(map(int, uv[c + 1])), COLORS[c], th, cv2.LINE_AA)
                pb = cz[f"p_W_{side}"][int(np.argmin(np.abs(cz["t_ns"] - (t0 + tq * 1e9))))]
                if np.isfinite(pb).all():
                    uv, d = proj(K, R_WC, t_WC, pb[None]); cv2.circle(im, tuple(map(int, uv[0])), 5, (0, 255, 255), -1)
                c0, _ = proj(K, R_WC, t_WC, zb["T_W_tcp"][i][:3, 3][None]); cx, cy = map(int, c0[0])
                im = cv2.copyMakeBorder(im, CROP, CROP, CROP, CROP, cv2.BORDER_CONSTANT); crop = im[cy:cy + 2 * CROP, cx:cx + 2 * CROP]
                crop = cv2.resize(crop, (2 * CROP * 2, 2 * CROP * 2), interpolation=cv2.INTER_CUBIC)
                cv2.putText(crop, f"{side} {name} t={tq:.1f}s", (6, 22), 0, 0.65, (255, 255, 255), 2)
                cv2.putText(crop, f"rcb3 pitch {pitch[i]:+.0f} roll {roll[i]:+.0f}", (6, 46), 0, 0.6, (255, 255, 255), 2)
                tiles.append(crop)
        while len(tiles) % 5: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f"{e.parent.name[-15:]}_{e.name[-2:]}.jpg"), np.vstack([np.hstack(tiles[r:r + 5]) for r in range(0, len(tiles), 5)]))
        print("wrote", e.parent.name, e.name)


if __name__ == "__main__":
    main()
