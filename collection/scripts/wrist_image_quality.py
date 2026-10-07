#!/usr/bin/env python3
"""[2026-09-19] Is a wrist camera over-exposed, or out of focus? Two different faults, two different fixes.

"Too bright" and "blurry" are both reported by eye, and by eye they are easy to confuse: an over-exposed frame
loses edge contrast and reads as soft. They need opposite responses -- one is a UVC control, the other is the
lens -- so this separates them with numbers:

    exposure    mean luma, p99, and the fraction of pixels clipped at the top (>=250) or crushed (<=5)
    focus       variance of the Laplacian on a centre crop, plus the same normalised by local contrast so a
                dark frame is not mistaken for a soft one

Both are measured on the SAME centre crop, because a fisheye's corners are soft by construction and would
swamp a real focus change.

Run it on recorded episodes to get a reference from a take that was accepted, then on a new take or on the
live cameras, and compare like with like.

    .venv/bin/python scripts/wrist_image_quality.py --session datasets/human_handumi_raw/HRGBD_qa10/<stamp>
    .venv/bin/python scripts/wrist_image_quality.py --live            # needs the cameras free
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2, numpy as np

CROP = 0.5          # centre fraction measured; keeps fisheye corner softness out of the focus number
NFRAMES = 24


def measure(bgr: np.ndarray) -> dict:
    h, w = bgr.shape[:2]
    ch, cw = int(h * CROP), int(w * CROP)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    g = cv2.cvtColor(bgr[y0:y0 + ch, x0:x0 + cw], cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    sd = float(g.std())
    return dict(mean=float(g.mean()), p99=float(np.percentile(g, 99)),
                clip=float((g >= 250).mean()), crush=float((g <= 5).mean()),
                lapvar=float(lap.var()),
                # normalised: how much high-frequency energy per unit of scene contrast. A dark but sharp
                # frame and a bright but sharp frame land in the same place; only real defocus moves it.
                sharp=float(lap.var() / max(sd * sd, 1e-6)))


def from_video(path: Path, n: int = NFRAMES) -> list[dict]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append(measure(fr))
    cap.release()
    return out


def from_live(index: int, n: int = NFRAMES) -> tuple[list[dict], str]:
    """Returns the rows AND the frame size actually delivered.

    The size is not decoration. Laplacian variance scales with pixel count, so a live grab that the driver
    silently served at a lower resolution than the recording would look catastrophically different from the
    reference for a reason that has nothing to do with the lens. Any comparison that does not state the size
    on both sides is not a comparison.
    """
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    out, size = [], "no frame"
    for i in range(n + 10):
        ok, fr = cap.read()
        if not ok:
            continue
        size = f"{fr.shape[1]}x{fr.shape[0]}"
        if i >= 10:                             # discard the auto-settle frames the driver emits on open
            out.append(measure(fr))
    cap.release()
    return out, size


def summarise(label: str, rows: list[dict], size: str = "") -> None:
    if not rows:
        print(f"  {label:>28s}   -- no frames --"); return
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    print(f"  {label:>28s} {a['mean'].mean():7.1f} {a['p99'].mean():7.1f} "
          f"{100*a['clip'].mean():7.2f}% {100*a['crush'].mean():7.2f}% "
          f"{a['lapvar'].mean():9.1f} {a['sharp'].mean():8.3f}  {size}")


def header() -> None:
    print(f"  {'source':>28s} {'mean':>7s} {'p99':>7s} {'clipped':>8s} {'crushed':>8s} {'lapvar':>9s} {'sharp':>8s}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, action="append", default=[])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--indices", default="0,1,2", help="live capture indices to sweep")
    ap.add_argument("--frames", type=int, default=NFRAMES)
    a = ap.parse_args()

    print(f"\ncentre {int(CROP*100)}% crop, {a.frames} frames per source")
    print("  'sharp' = Laplacian variance / scene variance -- brightness-independent; a drop here is focus.")
    print("  'clipped' = pixels pinned at the top of the range -- that is over-exposure, not focus.\n")
    for s in a.session:
        eps = sorted(p for p in s.iterdir() if p.is_dir() and p.name.startswith("episode_"))
        print(f"{s.name}   {len(eps)} episodes"); header()
        for cam in ("left_wrist", "right_wrist"):
            rows = []
            for ep in eps:
                v = ep / f"{cam}.mp4"
                if v.exists():
                    rows += from_video(v, max(a.frames // max(len(eps), 1), 3))
            summarise(cam, rows)
        print()
    if a.live:
        print("LIVE"); header()
        for i in (int(x) for x in a.indices.split(",")):
            rows, size = from_live(i, a.frames)
            summarise(f"index {i}", rows, size)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
