#!/usr/bin/env python3
"""[2026-10-14] Thumb/index-ONLY wrist-cam detector (visual teleop spec §8): re-label the WiLoR-distilled 4-kpt dataset
(~/finger_distill/ds: [wrist, thumb tip, index MCP, index tip]) to 2 keypoints [thumb tip, index tip] with a box around the
two TIPS only (padded), so detection no longer needs the wrist / palm in view (partial hand at the image edge).
Hard negatives (empty label files) come from --neg-video clips: wrist cam pointed at faces / room / cloth / the OTHER hand.

Pinch metric for this model (fingertip_detector backend student2): tip distance in % of image width. The camera is rigid on
the wrist, so pixel distance tracks the pinch; thresholds are calibrated per mount (gripper.close/open_threshold).

  ~/finger_distill/.venv/bin/python scripts/finger_distill_2kpt.py build --neg-video ~/finger_distill/neg/neg_room_01.mp4
  ~/finger_distill/.venv/bin/python scripts/finger_distill_2kpt.py train --epochs 60
Writes ~/finger_distill/ds2kpt and ~/finger_distill/student2kpt.pt."""
import argparse, os, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path.home() / "finger_distill"
SRC, DST = ROOT / "ds", ROOT / "ds2kpt"
PAD, MIN_HALF = 0.6, 0.035          # half box = max(PAD * tip distance, MIN_HALF) in normalised width/height units


def relabel(line: str, aspect: float) -> str | None:
    v = [float(x) for x in line.split()]
    if len(v) < 5 + 12: return None
    k = np.array(v[5:17]).reshape(4, 3); th, ix = k[1], k[3]
    if th[2] <= 0 or ix[2] <= 0: return None                      # need both tips labelled
    c = (th[:2] + ix[:2]) / 2; d = np.hypot((th[0] - ix[0]) * aspect, th[1] - ix[1])
    hw = max(PAD * d, MIN_HALF); w, h = 2 * hw / aspect, 2 * hw
    w, h = min(w, 1.0), min(h, 1.0); cx, cy = np.clip(c[0], w / 2, 1 - w / 2), np.clip(c[1], h / 2, 1 - h / 2)
    return " ".join(f"{x:.6f}" for x in [0, cx, cy, w, h, *th[:2], 2, *ix[:2], 2]).replace("0.000000 ", "0 ", 1)


def build(neg_videos: list[str], neg_every: int) -> None:
    import cv2
    if DST.exists(): shutil.rmtree(DST)
    n_pos = n_neg = n_skip = 0
    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True); (DST / "labels" / split).mkdir(parents=True)
        for img in sorted((SRC / "images" / split).iterdir()):
            lab = SRC / "labels" / split / (img.stem + ".txt")
            if not lab.exists() or not lab.read_text().strip(): n_skip += 1; continue   # WiLoR-miss frames are NOT negatives
            im = cv2.imread(str(img)); aspect = im.shape[1] / im.shape[0]
            out = [r for r in (relabel(l, aspect) for l in lab.read_text().splitlines() if l.strip()) if r]
            if not out: n_skip += 1; continue
            os.symlink(img, DST / "images" / split / img.name); (DST / "labels" / split / (img.stem + ".txt")).write_text("\n".join(out) + "\n"); n_pos += 1
    for vi, v in enumerate(neg_videos):
        cap = cv2.VideoCapture(str(Path(v).expanduser())); i = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            i += 1
            if i % neg_every: continue
            split = "val" if (i // neg_every) % 5 == 0 else "train"; name = f"neg{vi:02d}_{i:06d}"
            cv2.imwrite(str(DST / "images" / split / f"{name}.jpg"), f); (DST / "labels" / split / f"{name}.txt").write_text(""); n_neg += 1
    (DST / "data.yaml").write_text(f"path: {DST}\ntrain: images/train\nval: images/val\nkpt_shape: [2, 3]\nflip_idx: [0, 1]\nnames:\n  0: thumb_index\n")
    print(f"ds2kpt: {n_pos} positives, {n_neg} hard negatives, {n_skip} source frames skipped (no WiLoR hand)")


def train(epochs: int) -> None:
    from ultralytics import YOLO
    m = YOLO(str(ROOT / "yolov8n-pose.pt"))
    r = m.train(data=str(DST / "data.yaml"), epochs=epochs, imgsz=640, device="mps", project=str(ROOT / "runs"), name="student2kpt", exist_ok=True,
                fliplr=0.0, mosaic=0.5, verbose=False)
    best = Path(r.save_dir) / "weights" / "best.pt"; shutil.copy(best, ROOT / "student2kpt.pt"); print("student2kpt ->", ROOT / "student2kpt.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "train"]); ap.add_argument("--neg-video", action="append", default=[])
    ap.add_argument("--neg-every", type=int, default=3); ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    build(a.neg_video, a.neg_every) if a.cmd == "build" else train(a.epochs)
