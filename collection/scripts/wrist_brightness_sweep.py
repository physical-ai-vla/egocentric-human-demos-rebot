#!/usr/bin/env python3
"""[2026-09-19] Pick the `brightness` pedestal for a wrist camera by sweeping it against a fixed scene.

Retuning from a single live glance is how the wrong value gets frozen: the wrist cameras are hand-worn, so an
unworn camera is aimed at whatever it happens to be lying against, and its exposure statistics say more about
that than about the setting. So this REQUIRES the operator to hold the cameras on the workspace, as worn, and
then sweeps the one control that is still free.

Exposure and gain are already at their floor (exposure-time-abs 1, gain 0), so `brightness` -- a luma pedestal
applied after the sensor -- is the only lever left. It cannot recover a highlight the sensor already clipped,
which is why the target is stated as BOTH ends: match the reference mean without crushing the shadows.

Reference, from the accepted take HRGBD_qa10_20260918_172229 (centre 50% crop):
    left_wrist    mean 154.1   clipped 0.00%   crushed 0.81%
    right_wrist   mean 153.7   clipped 0.00%   crushed 0.95%

    .venv/bin/python scripts/wrist_brightness_sweep.py --cam right --index 1
    .venv/bin/python scripts/wrist_brightness_sweep.py --cam left  --index 2 --values -64,-56,-48,-40,-32
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.wrist_image_quality import from_live                              # noqa: E402

UVC = Path.home() / "robot-cockpit" / "bin" / "uvc-util"
REFERENCE = {"left": dict(mean=154.1, crushed=0.0081), "right": dict(mean=153.7, crushed=0.0095)}
UVC_INDEX = {"left": 0, "right": 1}          # uvc-util listing position, NOT the OpenCV index


def set_brightness(uvc_index: int, value: int) -> None:
    subprocess.run([str(UVC), "-I", str(uvc_index), f"-sbrightness={value}"], check=True,
                   capture_output=True, text=True)


def read_brightness(uvc_index: int) -> str:
    return subprocess.run([str(UVC), "-I", str(uvc_index), "-obrightness"],
                          capture_output=True, text=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", choices=["left", "right"], required=True)
    ap.add_argument("--index", type=int, required=True, help="OpenCV capture index (profile: left 2, right 1)")
    ap.add_argument("--uvc-index", type=int, default=None, help="uvc-util listing index (default: left 0, right 1)")
    ap.add_argument("--values", default="-64,-56,-48,-40,-32,-24,-16")
    ap.add_argument("--frames", type=int, default=12)
    a = ap.parse_args()
    ui = a.uvc_index if a.uvc_index is not None else UVC_INDEX[a.cam]
    vals = [int(x) for x in a.values.split(",")]
    was = read_brightness(ui)
    ref = REFERENCE[a.cam]

    print(f"\n{a.cam}_wrist   OpenCV index {a.index}, uvc-util index {ui}, current brightness {was}")
    print("HOLD THE CAMERA ON THE WORKSPACE, AS WORN, AND KEEP IT THERE FOR THE WHOLE SWEEP.")
    print(f"target: mean ~{ref['mean']:.0f}, clipped 0.00%, crushed <= {100*ref['crushed']:.2f}%  (the accepted take)\n")
    print(f"  {'brightness':>10s} {'mean':>7s} {'p99':>7s} {'clipped':>8s} {'crushed':>8s} {'size':>10s}")
    rows = []
    try:
        for v in vals:
            set_brightness(ui, v)
            r, size = from_live(a.index, a.frames)
            if not r:
                print(f"  {v:>10d}   -- no frames --"); continue
            m = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
            rows.append((v, m))
            print(f"  {v:>10d} {m['mean']:7.1f} {m['p99']:7.1f} {100*m['clip']:7.2f}% {100*m['crush']:7.2f}% {size:>10s}")
    finally:
        if was.lstrip("-").isdigit():
            set_brightness(ui, int(was))
            print(f"\nrestored brightness to {was} -- nothing is frozen until the profile is edited")

    ok = [(v, m) for v, m in rows if m["clip"] < 0.0005 and m["crush"] <= ref["crushed"] * 1.5]
    if ok:
        v, m = min(ok, key=lambda vm: abs(vm[1]["mean"] - ref["mean"]))
        print(f"\nbest match: brightness {v}  (mean {m['mean']:.1f} vs reference {ref['mean']:.1f}, "
              f"clipped {100*m['clip']:.2f}%, crushed {100*m['crush']:.2f}%)")
        print(f"to freeze it: set {a.cam}_wrist uvc_controls.brightness = {v} in configs/handumi/hardware_handumi_rgbd.yaml")
        print("and bump calibration_version -- the recorded takes are only comparable within one version.")
    else:
        print("\nNo swept value met both ends. The sensor itself is over- or under-exposed at this scene "
              "brightness and the pedestal cannot fix it; change the lighting, not the control.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
