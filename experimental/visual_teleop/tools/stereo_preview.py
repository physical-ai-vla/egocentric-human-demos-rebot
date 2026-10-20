"""Live side-by-side preview of the two tracking fisheyes (aiming, connector / stall tests) (2026-10-15).
Same capture threads and preview as stereo_imu_record (one thread per camera, so a stalled camera shows a red STALL banner instead of
freezing the whole page), nothing recorded. Prints a line whenever a camera stalls or recovers. Stop it before recording.
    .venv/bin/python -m visual_teleop.tools.stereo_preview [--exposure 10] [--port 8745]"""
from __future__ import annotations
import argparse, tempfile, time
from pathlib import Path
from visual_teleop.tools.stereo_check import index_for_location
from visual_teleop.tools.stereo_imu_record import Cam, PREVIEW, lock, start_preview


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left-location", default="0x03110000"); ap.add_argument("--right-location", default="0x03140000")
    ap.add_argument("--exposure", type=int, default=10); ap.add_argument("--port", type=int, default=8745)
    ap.add_argument("--left-index", type=int, default=-1); ap.add_argument("--right-index", type=int, default=-1); a = ap.parse_args()   # 10-16: location probe failed after a replug
    il = a.left_index if a.left_index >= 0 else index_for_location(a.left_location); ir = a.right_index if a.right_index >= 0 else index_for_location(a.right_location); print(f"[preview] 왼쪽 {a.left_location}=index {il}, 오른쪽 {a.right_location}=index {ir}", flush=True)
    tmp = Path(tempfile.mkdtemp()); L, R = Cam("cam_left", il, tmp, 2), Cam("cam_right", ir, tmp, 2)
    print("[preview] 노출", lock(a.left_location, a.exposure), lock(a.right_location, a.exposure), flush=True)
    L.start(); R.start(); time.sleep(1.0); start_preview(L, R, a.port); PREVIEW["status"] = "preview only (not recording)"
    last, stalled = {"왼쪽": 0, "오른쪽": 0}, {"왼쪽": False, "오른쪽": False}; t0 = time.monotonic()
    while True:
        time.sleep(0.2)
        for n, c in (("왼쪽", L), ("오른쪽", R)):
            age = (time.monotonic_ns() - c.tl) / 1e9 if c.tl else 0.0
            if age > 1.0 and not stalled[n]: stalled[n] = True; print(f"[preview] {time.monotonic() - t0:6.1f}s {n} 카메라 멈춤", flush=True)
            if age < 0.2 and stalled[n]: stalled[n] = False; print(f"[preview] {time.monotonic() - t0:6.1f}s {n} 카메라 복구", flush=True)


if __name__ == "__main__":
    main()
