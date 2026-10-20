#!/usr/bin/env python3
"""[2026-10-13] Train the realtime fingertip student (YOLOv8n-pose, 4 kpts) on the WiLoR pseudo-labels from finger_distill_label.py.
Mac MPS, local (small: ~2.7k images). Horizontal flip off (handedness / wearer-side geometry matters).
usage: finger_distill_train.py ~/finger_distill/ds [--epochs 40] -> ~/finger_distill/student.pt"""
import argparse, pathlib, shutil
from ultralytics import YOLO
ap = argparse.ArgumentParser(); ap.add_argument("ds"); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--imgsz", type=int, default=640); a = ap.parse_args()
ds = pathlib.Path(a.ds).expanduser(); home = pathlib.Path.home() / "finger_distill"
m = YOLO(str(home / "yolov8n-pose.pt"))
r = m.train(data=str(ds / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=16, device="mps", fliplr=0.0, mosaic=0.5, degrees=10, scale=0.3,
            hsv_v=0.5, project=str(home / "runs"), name="student", exist_ok=True, workers=4, patience=15, plots=False, verbose=False)
best = home / "runs/student/weights/best.pt"; shutil.copy(best, home / "student.pt"); print("student ->", home / "student.pt")
