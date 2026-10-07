from __future__ import annotations

import argparse
from pathlib import Path

from ego_collector.actions.generate import generate_actions
from ego_collector.filtering.pose_filter import OneEuroConfig
from ego_collector.hands.grasp import ApertureCalibration, GraspCalibration
from ego_collector.recording.episode import EpisodePaths


def load_grasp_calibration(path: Path) -> GraspCalibration:
    if path.exists():
        return GraspCalibration.load(path)
    print(f"warning: {path} missing; using default grasp calibration (run calibrate-grasp)")
    return GraspCalibration()


def load_aperture_calibration(path: Path) -> ApertureCalibration:
    if path.exists():
        return ApertureCalibration.load(path)
    print(f"warning: {path} missing; using default aperture calibration (run calibrate-aperture)")
    return ApertureCalibration()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector generate-actions", description="Pseudo actions (absolute + delta, 14-D) and bimanual features")
    p.add_argument("episodes", type=Path, nargs="+")
    p.add_argument("--grasp-calibration", type=Path, default=Path("configs/grasp_calibration.yaml"))
    p.add_argument("--aperture-calibration", type=Path, default=Path("configs/aperture_calibration.yaml"))
    p.add_argument("--grasp-source", choices=("auto", "tags", "mediapipe"), default="auto", help="tags = metric aperture from the finger AprilTags; mediapipe = normalized hand landmarks; auto = tags when available")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--no-filter", action="store_true", help="Skip the *_filtered pose columns")
    p.add_argument("--frame", choices=("world", "head", "both"), default="both", help="world = table frame (needs world tags); head = egocentric/head-relative ablation (no world tags needed)")
    args = p.parse_args(argv)
    cal = load_grasp_calibration(args.grasp_calibration)
    acal = load_aperture_calibration(args.aperture_calibration) if args.grasp_source != "mediapipe" else None
    for ep in args.episodes:
        frames = ("world", "head") if args.frame == "both" else (args.frame,)
        s = generate_actions(EpisodePaths(ep), grasp_cal=cal, aperture_cal=acal, grasp_source=args.grasp_source, horizon=args.horizon, filter_config=None if args.no_filter else OneEuroConfig(), frames=frames)
        valid = "  ".join(f"{f} {s[f'{f}_action_valid_fraction']*100:.1f}%" for f in frames)
        print(f"{ep.name}: action valid [{valid}]  grasp({s['grasp_source']}) L {s['left_grasp_valid_fraction']*100:.0f}% R {s['right_grasp_valid_fraction']*100:.0f}%")


if __name__ == "__main__":
    main()
