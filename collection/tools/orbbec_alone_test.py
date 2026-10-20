"""[2026-10-12] Orbbec alone for N s (no wrist cams, no UI): prints delivered fps every 5 s.
sudo .venv/bin/python tools/orbbec_alone_test.py [seconds]"""
import sys, time, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from ego_collector.camera.capture import CaptureConfig
from ego_collector.camera.orbbec_capture import OrbbecCapture
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 180
cap = OrbbecCapture(CaptureConfig(index=0, width=848, height=480, fps=30)); cap.open()
t0 = time.time(); tw = t0; n = 0; worst = 99.0; rates = []
try:
    while time.time() - t0 < dur:
        if cap.poll() is not None: n += 1
        else: time.sleep(0.002)
        if time.time() - tw >= 5:
            r = n / (time.time() - tw); rates.append(r); worst = min(worst, r)
            print(f"{time.time() - t0:5.0f} s  {r:5.1f} fps", flush=True); n = 0; tw = time.time()
finally:
    cap.close()
print(f"RESULT mean {sum(rates)/max(len(rates),1):.1f} fps, worst 5 s window {worst:.1f} fps")
