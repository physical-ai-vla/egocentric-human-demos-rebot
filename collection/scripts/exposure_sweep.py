#!/usr/bin/env python3
"""[2026-09-18] Find the exposure to freeze for a wrist camera under the current lighting.

The recording profile pins exposure and white balance so the policy sees one consistent image distribution; the pinned value
(20) was set under different lighting and is now blowing out. This sweeps exposure, measures what actually matters, and
proposes a value. Auto exposure is used only to read what the camera itself thinks is correct, never for recording.

Brightness alone is not the criterion: a longer shutter fixes the histogram and then smears the hand during fast motion, so
the sweep also reports a sharpness proxy (variance of the Laplacian) and prefers the SHORTEST exposure that still keeps the
shadows out of the floor.
    .venv/bin/python scripts/exposure_sweep.py [--camera left|right|both] [--values 5,10,20,40,80,160]"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, cv2, yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"left": "FisheyeCamLeft", "right": "Arducam 1080P Low Light"}


def find_index(match_name):
    """the UVC index whose frame size matches a 1080p wrist camera; macOS gives no name through OpenCV"""
    out = []
    for i in range(6):
        c = cv2.VideoCapture(i)
        if c.isOpened():
            ok, f = c.read()
            if ok and f is not None and f.shape[0] >= 720: out.append((i, f.shape))
        c.release()
    return out


def measure(cap, n=8):
    """discard the first frames so the sensor settles, then measure the settled ones"""
    for _ in range(6): cap.read()
    stats = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok: continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        stats.append((float(g.mean()), float((g >= 250).mean()), float((g <= 5).mean()),
                      float(cv2.Laplacian(g, cv2.CV_64F).var())))
    if not stats: return None
    S = np.array(stats)
    return dict(mean=S[:, 0].mean(), clipped=S[:, 1].mean() * 100, black=S[:, 2].mean() * 100, sharp=S[:, 3].mean())


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--values", default="5,10,20,40,80,160,320")
    ap.add_argument("--list", action="store_true"); a = ap.parse_args()
    if a.list or a.index is None:
        print("1080p-capable UVC indices (open each and look at the preview to identify it):")
        for i, shape in find_index(None): print(f"  index {i}: {shape[1]}x{shape[0]}")
        print("\nre-run with --index N"); return 0
    cap = cv2.VideoCapture(a.index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if not cap.isOpened(): print(f"index {a.index} did not open"); return 1
    # what the camera itself chooses, for reference only
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    time.sleep(1.2); auto = measure(cap)
    auto_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    print(f"auto exposure (reference only): mean {auto['mean']:.0f}, clipped {auto['clipped']:.2f} %, "
          f"black {auto['black']:.2f} %, sharpness {auto['sharp']:.0f}, camera reports exposure {auto_exp:g}\n")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    rows = []
    print(f"{'exposure':>9s} {'mean':>7s} {'clipped %':>10s} {'black %':>8s} {'sharpness':>10s}")
    for v in [float(x) for x in a.values.split(",")]:
        cap.set(cv2.CAP_PROP_EXPOSURE, v)
        time.sleep(0.5); m = measure(cap)
        if m is None: continue
        rows.append((v, m))
        print(f"{v:9g} {m['mean']:7.0f} {m['clipped']:10.2f} {m['black']:8.2f} {m['sharp']:10.0f}")
    cap.release()
    ok = [(v, m) for v, m in rows if m["clipped"] < 1.0 and m["black"] < 2.0 and 70 <= m["mean"] <= 150]
    print()
    if ok:
        v, m = min(ok, key=lambda r: r[0])          # shortest shutter that still behaves: least motion blur
        print(f"proposed exposure {v:g}: mean {m['mean']:.0f}, clipped {m['clipped']:.2f} %, black {m['black']:.2f} %")
        print("chosen as the SHORTEST exposure meeting the criteria, so the hand does not smear during fast motion")
    else:
        print("no value met clipped < 1 %, black < 2 % and mean 70-150; widen --values or change the lighting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
