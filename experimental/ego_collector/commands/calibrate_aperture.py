"""Metric grasp calibration from the finger AprilTags of a *processed* episode.

Record a short episode holding the pinch fully OPEN for a few seconds and fully CLOSED
(fingertips touching) for a few seconds, run process-tags, then:

    ego-collector calibrate-aperture <ep> --open 2:6 --closed 8:12          # seconds windows
    ego-collector calibrate-aperture <ep> --auto                            # percentiles over the whole episode

Writes configs/aperture_calibration.yaml (open_m / closed_m per side), used by
generate-actions --grasp-source tags. Later, retargeting maps grasp 0..1 onto the
robot gripper's [closed, open] range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ego_collector.hands.grasp import ApertureCalibration
from ego_collector.io.parquet import read_parquet
from ego_collector.recording.episode import EpisodePaths

DEFAULT_OUT = Path("configs/aperture_calibration.yaml")
SIDES = ("left", "right")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector calibrate-aperture", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode", type=Path)
    p.add_argument("--open", dest="open_w", default=None, help="a:b seconds of fully OPEN pinch")
    p.add_argument("--closed", dest="closed_w", default=None, help="a:b seconds of fully CLOSED pinch (fingertips touching)")
    p.add_argument("--auto", action="store_true", help="No windows: open = 95th pct, closed = 5th pct of the whole episode")
    p.add_argument("--percentile", type=float, default=10.0)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    paths = EpisodePaths(args.episode)
    if not paths.finger_pose.exists():
        raise SystemExit(f"{paths.finger_pose} missing: run process-tags first (with thumb/index tags in configs/wrist_extrinsics.yaml)")
    fp = read_parquet(paths.finger_pose)
    t = (fp["timestamp_ns"].to_numpy() - fp["timestamp_ns"].iloc[0]) / 1e9

    def window(spec: str) -> np.ndarray:
        a, b = spec.split(":")
        return (t >= float(a or 0)) & (t <= float(b or 1e9))

    open_s: dict[str, np.ndarray] = {}
    closed_s: dict[str, np.ndarray] = {}
    for side in SIDES:
        a = fp[f"{side}_aperture_m"].to_numpy(dtype=np.float64)
        if args.auto:
            finite = a[np.isfinite(a)]
            if len(finite) >= 10:
                open_s[side] = np.full(10, np.percentile(finite, 95))
                closed_s[side] = np.full(10, np.percentile(finite, 5))
        else:
            if not (args.open_w and args.closed_w):
                raise SystemExit("pass --open a:b and --closed c:d (seconds), or --auto")
            open_s[side] = a[window(args.open_w)]
            closed_s[side] = a[window(args.closed_w)]
    cal = ApertureCalibration.from_samples(open_s, closed_s, percentile=args.percentile)
    cal.save(args.output)
    for side in SIDES:
        o, c = cal.bounds(side)
        n_o = int(np.isfinite(open_s.get(side, np.array([]))).sum())
        n_c = int(np.isfinite(closed_s.get(side, np.array([]))).sum())
        tag = "" if n_o >= 5 and n_c >= 5 else "  (defaults kept: not enough valid samples)"
        print(f"{side:5s} open {o*1000:5.1f} mm  closed {c*1000:5.1f} mm  (samples open {n_o} / closed {n_c}){tag}")
    print(f"-> {args.output}")
    if all(cal.bounds(s) == ApertureCalibration().bounds(s) for s in SIDES):
        print("note: nothing calibrated; check finger tag visibility in the episode (replay/tracking-test)")


if __name__ == "__main__":
    main()
