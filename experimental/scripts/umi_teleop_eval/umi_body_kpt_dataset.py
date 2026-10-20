"""Tagless auto-labels for the UMI body-keypoint detector on the Orbbec colour stream (2026-10-15, Agent C).

Teacher pose per frame = the live teleop pose of that recording (ticks.npz human_T_right: kpt+depth position, IMU orientation),
only where the hand was TRACKING and the wrist rotated < max_w deg/s (IMU<->camera time error stays sub-pixel-ish). The rigid body
keypoints (UMI/H frame) come from umi_body_kpts.json (multi-view triangulation of user clicks). Each kept frame gets a YOLO-pose
label: box = keypoints' bbox + margin, keypoints = projections (visibility 2 when inside the image, 0 outside).
    python scripts/umi_body_kpt_dataset.py <episode_dir> [--out ~/finger_distill/ds_umi_body] [--every 2] [--val 0.2]"""
from __future__ import annotations
import argparse, json, pathlib, random
import cv2
import numpy as np

KPTS = ["jaw_tip", "jaw_root", "cam_lens", "body_top_back"]      # handle_bottom dropped: 19.6 px triangulation residual


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--out", default=str(pathlib.Path.home() / "finger_distill/ds_umi_body"))
    ap.add_argument("--every", type=int, default=2); ap.add_argument("--val", type=float, default=0.2); ap.add_argument("--max-w", type=float, default=60.0)
    ap.add_argument("--margin", type=float, default=0.25); a = ap.parse_args()
    E = pathlib.Path(a.episode); B = json.load(open(E / "umi_body_kpts.json")); fx, fy, cx, cy = B["K"]
    P = np.array([B["points"][k]["P_H_tri_m"] for k in KPTS])
    import sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from visual_teleop import config as vcfg
    c = vcfg.load(str(pathlib.Path(__file__).resolve().parents[1] / "visual_teleop/config/teleop_kpt.yaml")); T_W_C, _ = vcfg.T_W_C(c); T_C_W = np.linalg.inv(T_W_C)
    tk = np.load(E / "ticks.npz", allow_pickle=True); TT, tt, st = tk["human_T_right"], tk["t_ns"], tk["hand_status_right"]
    tc = np.load(E / "frames/rgbd_color_t_ns.npy"); cap = cv2.VideoCapture(str(E / "frames/rgbd_color.mp4"))
    out = pathlib.Path(a.out).expanduser()
    for s in ("train", "val"): (out / "images" / s).mkdir(parents=True, exist_ok=True); (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    random.seed(0); n_keep = n_skip = 0; W, H = 1280, 720
    blocks = {}                                                          # split by 10 s blocks (no near-duplicate leakage)
    for k, t in enumerate(tc):
        ok, img = cap.read()
        if not ok: break
        if k % a.every: continue
        i = int(np.searchsorted(tt, t)); i = min(max(i, 1), len(tt) - 1)
        if abs(tt[i] - t) > 50e6 or st[i] != "TRACKING" or st[i - 1] != "TRACKING" or not np.all(np.isfinite(TT[i])): n_skip += 1; continue
        w = np.degrees(np.arccos(np.clip((np.trace(TT[i - 1][:3, :3].T @ TT[i][:3, :3]) - 1) / 2, -1, 1))) / max(1e-3, (tt[i] - tt[i - 1]) / 1e9)
        if w > a.max_w: n_skip += 1; continue
        T = TT[i]; Xc = (T_C_W[:3, :3] @ (P @ T[:3, :3].T + T[:3, 3]).T).T + T_C_W[:3, 3]
        if np.any(Xc[:, 2] < 0.2): n_skip += 1; continue
        u = fx * Xc[:, 0] / Xc[:, 2] + cx; v = fy * Xc[:, 1] / Xc[:, 2] + cy
        vis = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if vis.sum() < 2 or not vis[0]: n_skip += 1; continue
        x0, x1, y0, y1 = u[vis].min(), u[vis].max(), v[vis].min(), v[vis].max(); mw, mh = (x1 - x0) * a.margin + 12, (y1 - y0) * a.margin + 12
        x0, x1, y0, y1 = max(0, x0 - mw), min(W - 1, x1 + mw), max(0, y0 - mh), min(H - 1, y1 + mh)
        b = int((t - tc[0]) / 10e9); split = blocks.setdefault(b, "val" if random.random() < a.val else "train")
        name = f"{E.name}_{k:06d}"; cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        kp = " ".join(f"{uu / W:.6f} {vv / H:.6f} {2 if ok_ else 0}" for uu, vv, ok_ in zip(u, v, vis))
        (out / "labels" / split / f"{name}.txt").write_text(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f} {kp}\n")
        n_keep += 1
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames: {{0: umi}}\nkpt_shape: [{len(KPTS)}, 3]\nflip_idx: {list(range(len(KPTS)))}\n")
    (out / "keypoints.json").write_text(json.dumps(dict(names=KPTS, P_H_m=P.tolist(), source=str(E / "umi_body_kpts.json")), indent=1))
    nt = len(list((out / "images/train").glob("*.jpg"))); nv = len(list((out / "images/val").glob("*.jpg")))
    print(f"labels: kept {n_keep}, skipped {n_skip} -> train {nt} / val {nv}  ({out})")


if __name__ == "__main__":
    main()
