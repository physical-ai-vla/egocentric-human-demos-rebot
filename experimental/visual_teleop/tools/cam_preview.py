"""Live preview of one camera to a JPEG for the dashboard's /view/<name> page (2026-10-15, user: aim the tripod C922).
    .venv/bin/python -m visual_teleop.tools.cam_preview --index 3 --name c922
Writes /tmp/vt_<name>.jpg ~10 times a second until Ctrl-C. Read-only camera use."""
from __future__ import annotations
import argparse
import time
import cv2


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--index", type=int, required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    last = 0.0
    while True:
        ok, f = cap.read()
        if not ok: time.sleep(0.05); continue
        if time.time() - last > 0.1:
            last = time.time(); h, w = f.shape[:2]
            cv2.line(f, (w // 2, 0), (w // 2, h), (0, 255, 255), 1); cv2.line(f, (0, h // 2), (w, h // 2), (0, 255, 255), 1)   # centre cross
            cv2.putText(f, f"{a.name} (OpenCV {a.index}) {w}x{h}", (10, 30), 0, 0.9, (0, 255, 255), 2)
            cv2.imwrite(f"/tmp/vt_{a.name}.jpg.tmp.jpg", f); __import__("os").replace(f"/tmp/vt_{a.name}.jpg.tmp.jpg", f"/tmp/vt_{a.name}.jpg")


if __name__ == "__main__":
    main()
