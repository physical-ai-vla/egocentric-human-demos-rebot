"""``python -m ego_collector <command>`` — recording + offline processing stages.

record | calibrate-camera | process-tags | process-hands | calibrate-grasp |
generate-actions | qa | process | visualize | replay | label | tracking-test
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

COMMANDS: dict[str, tuple[str, str]] = {
    # name: (module, help)
    "record": ("ego_collector.commands.record", "Record raw episodes from the head camera (R/S/D/Q)"),
    "calibrate-camera": ("ego_collector.commands.calibrate_camera", "ChArUco intrinsics -> configs/camera/<name>.yaml"),
    "process-tags": ("ego_collector.commands.process_tags", "AprilTag detection, world camera pose, wrist poses"),
    "process-hands": ("ego_collector.commands.process_hands", "MediaPipe hand landmarks -> hands/hand_pose.parquet"),
    "process-hands3d": ("ego_collector.commands.process_hands3d", "WiLoR 3D hand pose (offline) -> wrist SE3 + aperture parquets, overlay, plots"),
    "process-hawor": ("ego_collector.commands.process_hawor", "HaWoR world track -> tracking parquets -> pseudo-actions (Action v1)"),
    "calibrate-grasp": ("ego_collector.commands.calibrate_grasp", "Open/closed pinch aperture calibration (MediaPipe, normalized)"),
    "calibrate-aperture": ("ego_collector.commands.calibrate_aperture", "Open/closed pinch calibration from the finger AprilTags (metric)"),
    "inventory": ("ego_collector.commands.inventory", "Per-layout episode inventory + permutation-coverage check (batch QC)"),
    "plan": ("ego_collector.commands.plan", "Cube-stacking collection plan (layouts x orders) + progress"),
    "report": ("ego_collector.commands.report", "Dataset-level QC report + pilot gate over all episodes"),
    "curate": ("ego_collector.commands.curate", "Index-style curation: quality, embedding dedup, caption, tier A/B/D"),
    "generate-actions": ("ego_collector.commands.generate_actions", "Pseudo actions + bimanual features"),
    "qa": ("ego_collector.commands.qa", "Tracking/hand QA -> qa/report.json + dataset tier"),
    "process": ("ego_collector.commands.process", "tags -> hands -> actions -> qa"),
    "visualize": ("ego_collector.commands.visualize", "3D trajectories + grasp plots (rerun or matplotlib)"),
    "replay": ("ego_collector.commands.replay", "Frame-by-frame overlay replay"),
    "label": ("ego_collector.commands.label", "Set outcome / recovery / notes on an episode"),
    "tracking-test": ("ego_collector.commands.tracking_test", "Static jitter / repeatability / availability report"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ego_collector", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = "-v" in argv or "--verbose" in argv
    argv = [a for a in argv if a not in ("-v", "--verbose")]
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s - %(message)s", datefmt="%H:%M:%S")
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return
    name = argv[0]
    if name not in COMMANDS:
        raise SystemExit(f"unknown command {name!r}; one of {', '.join(COMMANDS)}")
    module = importlib.import_module(COMMANDS[name][0])
    module.main(argv[1:])


if __name__ == "__main__":
    main()
