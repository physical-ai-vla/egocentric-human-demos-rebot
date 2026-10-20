#!/usr/bin/env python3
"""[2026-10-13] Record raw wrist-camera video for the WiLoR -> small fingertip model distillation (finger_grip_teleop).
usage: finger_distill_record.py --cam Arducam --secs 180 --out ~/finger_distill/rec_<time>.mp4"""
import argparse, pathlib, sys, time
import cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from finger_grip_teleop import open_camera
ap = argparse.ArgumentParser(); ap.add_argument("--cam", default="Arducam"); ap.add_argument("--secs", type=float, default=180); ap.add_argument("--out", required=True); a = ap.parse_args()
cap, name, idx = open_camera(a.cam); ok, f = cap.read(); h, w = f.shape[:2]
vw = cv2.VideoWriter(str(pathlib.Path(a.out).expanduser()), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)); t0 = time.time(); n = 0
TIPS = ["open/close slowly", "open/close fast", "rotate the wrist while pinching", "half-open holds", "other hand in view too", "different backgrounds / table / cloth"]
while time.time() - t0 < a.secs:
    ok, f = cap.read()
    if not ok: continue
    vw.write(f); n += 1; el = time.time() - t0
    v = cv2.resize(f, (960, 540)); cv2.putText(v, f"REC {el:5.1f}/{a.secs:.0f}s  {TIPS[int(el // 20) % len(TIPS)]}", (20, 50), 0, 1.0, (0, 0, 255), 3)
    cv2.imshow("finger distill rec (q stop)", v)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
vw.release(); print(f"saved {n} frames -> {a.out}")
