#!/usr/bin/env python3
"""[2026-10-15] Markerless UMI jaw-tip keypoint dataset from a visual-teleop recording (user: no tag, no tape at runtime).
Labels come from the tag-guided tape tracker the hub ran during recording (umi_pseudo_labels.npz, source == 1); tag/tape are only
the TEACHER. One class 'umi', 1 keypoint = jaw tip; box = square around the tip, side from depth (farther -> smaller).
Last 20 % of the recording = val (temporal split). Frames where the teacher had no tape = dropped (not negatives).
usage: umi_kpt_dataset.py <rec_dir> [...] --out ~/finger_distill/ds_umi --stride 2"""
import argparse, os, pathlib
import cv2, numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("recs", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args(); out = pathlib.Path(a.out).expanduser(); n = dict(train=0, val=0)
    for split in ("train", "val"): (out / "images" / split).mkdir(parents=True, exist_ok=True); (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for rec in a.recs:
        rec = pathlib.Path(rec).expanduser(); L = np.load(rec / "umi_pseudo_labels.npz"); t_vid = np.load(rec / "frames" / "rgbd_color_t_ns.npy")
        lab = {int(t): (uv, s) for t, uv, s in zip(L["t_ns"], L["tape_uv"], L["source"])}
        cap = cv2.VideoCapture(str(rec / "frames" / "rgbd_color.mp4")); cut = int(0.8 * len(t_vid))
        for i, t in enumerate(t_vid):
            ok, img = cap.read()
            if not ok: break
            if i % a.stride: continue
            uv, s = lab.get(int(t), (None, 0))
            if s != 1 or uv is None or not np.all(np.isfinite(uv)): continue
            h, w = img.shape[:2]; dpng = rec / "frames" / "rgbd_depth_mm" / f"{int(t)}.png"; z = np.nan
            if dpng.exists():
                d = cv2.imread(str(dpng), cv2.IMREAD_UNCHANGED); u, v = int(uv[0]), int(uv[1]); win = d[max(0, v - 4):v + 5, max(0, u - 4):u + 5].astype(float)
                win = win[win > 0]; z = float(np.median(win)) if len(win) else np.nan
            side = float(np.clip(110000.0 / z, 70, 260)) if np.isfinite(z) else 140.0          # ~0.15 m UMI head at fx ~ 730
            bw, bh = side / w, side / h; cx, cy = np.clip(uv[0] / w, bw / 2, 1 - bw / 2), np.clip(uv[1] / h, bh / 2, 1 - bh / 2)
            split = "val" if i >= cut else "train"; name = f"{rec.name}_{i:06d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "labels" / split / f"{name}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {uv[0] / w:.6f} {uv[1] / h:.6f} 2\n")
            n[split] += 1
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nkpt_shape: [1, 3]\nflip_idx: [0]\nnames:\n  0: umi\n")
    print("ds_umi:", n)


if __name__ == "__main__":
    main()
