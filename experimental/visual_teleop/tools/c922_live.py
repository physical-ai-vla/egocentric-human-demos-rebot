"""Live C922 view in the browser (2026-10-15, user: "c922 화면을 틀어줘"): MJPEG at http://127.0.0.1:8740 with the markerless
jaw-tip detection drawn on top (green circle + confidence) and the running detection rate, to aim the fixed camera.
    .venv/bin/python -m visual_teleop.tools.c922_live [--index 3] [--no-kpt]"""
from __future__ import annotations
import argparse, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

LATEST = {"jpg": None}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--index", type=int, default=3); ap.add_argument("--port", type=int, default=8740)
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720); ap.add_argument("--no-kpt", action="store_true")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.index, cv2.CAP_AVFOUNDATION); cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    kpt = None
    if not a.no_kpt:
        from ..vision.umi_kpt import UmiKptTracker
        kpt = UmiKptTracker(os.path.expanduser("~/finger_distill/umi_kpt.onnx"), np.eye(4), conf=0.3)

    def grab():
        hist = []
        while True:
            ok, f = cap.read()
            if not ok: time.sleep(0.05); continue
            if kpt is not None:
                out = kpt.infer(f); hist = (hist + [bool(out)])[-90:]
                if out:
                    c, u, v = max(out); cv2.circle(f, (int(u), int(v)), 20, (0, 255, 0), 3); cv2.putText(f, f"jaw tip {c:.2f}", (int(u) + 24, int(v)), 0, 0.9, (0, 255, 0), 2)
                cv2.putText(f, f"detect 3s: {100 * np.mean(hist):.0f}%", (10, 30), 0, 0.9, (0, 255, 255), 2)
            LATEST["jpg"] = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
    threading.Thread(target=grab, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
                try:
                    while True:
                        j = LATEST["jpg"]
                        if j: self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError): return
            body = "<html><head><meta charset=utf-8><title>C922 live</title></head><body style='margin:0;background:#111'><img src='/stream' style='width:100vw'></body></html>".encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)
        def log_message(self, *x): pass
    print(f"[c922_live] http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
