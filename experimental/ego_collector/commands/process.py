"""Full offline processing: process-tags -> process-hands -> generate-actions -> qa."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ego_collector.actions.generate import generate_actions
from ego_collector.commands.generate_actions import load_aperture_calibration, load_grasp_calibration
from ego_collector.commands.process_tags import add_common, load_configs
from ego_collector.filtering.pose_filter import OneEuroConfig
from ego_collector.hands.mediapipe_tracker import DEFAULT_MODEL, process_hands
from ego_collector.qa.episode import QAThresholds, run_qa
from ego_collector.recording.episode import EpisodePaths
from ego_collector.tracking.process import TagProcessingConfig, process_tags

log = logging.getLogger("ego_collector.process")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector process", description=__doc__)
    add_common(p)
    p.add_argument("--backend", choices=("auto", "pupil", "opencv"), default="auto")
    p.add_argument("--grasp-calibration", type=Path, default=Path("configs/grasp_calibration.yaml"))
    p.add_argument("--aperture-calibration", type=Path, default=Path("configs/aperture_calibration.yaml"))
    p.add_argument("--grasp-source", choices=("auto", "tags", "mediapipe"), default="auto")
    p.add_argument("--hand-model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--thresholds", type=Path, default=Path("configs/qa.yaml"))
    p.add_argument("--skip-hands", action="store_true", help="Tags + actions + QA only (grasp = NaN)")
    p.add_argument("--mirror-handedness", action="store_true")
    p.add_argument("--delegate", choices=("auto", "cpu", "gpu"), default="auto")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--frame", choices=("world", "head", "both"), default="both")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)
    intr, wm, wr = load_configs(args)
    cal = load_grasp_calibration(args.grasp_calibration)
    acal = load_aperture_calibration(args.aperture_calibration) if args.grasp_source != "mediapipe" else None
    th = QAThresholds.from_yaml(args.thresholds)
    if args.skip_hands:
        th.require_both_hands = False
    if args.no_world:
        th.require_world = False
    for ep in args.episodes:
        paths = EpisodePaths(ep)
        process_tags(paths, intr=intr, world_map=wm, wrists=wr, config=TagProcessingConfig(backend=args.backend), progress=not args.no_progress)
        if not args.skip_hands:
            process_hands(paths, model_path=args.hand_model, mirror_handedness=args.mirror_handedness, progress=not args.no_progress, delegate=args.delegate)
        generate_actions(paths, grasp_cal=cal, aperture_cal=acal, grasp_source=args.grasp_source, horizon=args.horizon, filter_config=OneEuroConfig(), frames=("world", "head") if args.frame == "both" else (args.frame,))
        r = run_qa(paths, th)
        failed = [k for k, c in r.checks.items() if not c["pass"] and c["hard"]]
        print(f"{ep.name}: {r.tier.upper()}  failed={failed or '-'}")


if __name__ == "__main__":
    main()
