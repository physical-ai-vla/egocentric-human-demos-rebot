from __future__ import annotations

import argparse
from pathlib import Path

from ego_collector.camera.intrinsics import CameraIntrinsics
from ego_collector.recording.episode import EpisodePaths
from ego_collector.tracking.process import TagProcessingConfig, process_tags
from ego_collector.tracking.world_pose import WorldTagMap
from ego_collector.tracking.wrist_pose import WristExtrinsics


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("episodes", type=Path, nargs="+", help="Episode directories (datasets/raw/episode_XXXXXX)")
    p.add_argument("--camera-config", type=Path, default=Path("configs/camera/c922.yaml"))
    p.add_argument("--world-map", type=Path, default=Path("configs/world_tag_map.yaml"))
    p.add_argument("--wrist-extrinsics", type=Path, default=Path("configs/wrist_extrinsics.yaml"))
    p.add_argument("--no-world", action="store_true", help="No world tags on the table: head/camera-frame tracking only")


def load_configs(args):
    intr = CameraIntrinsics.load(args.camera_config)
    wm = None if getattr(args, "no_world", False) else WorldTagMap.from_yaml(args.world_map)
    wr = WristExtrinsics.from_yaml(args.wrist_extrinsics)
    return intr, wm, wr


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ego_collector process-tags", description="AprilTag detection -> world camera pose -> wrist poses (offline).")
    add_common(p)
    p.add_argument("--backend", choices=("auto", "pupil", "opencv"), default="auto")
    p.add_argument("--max-world-reproj-px", type=float, default=3.0)
    p.add_argument("--max-wrist-reproj-px", type=float, default=3.0)
    p.add_argument("--min-world-tags", type=int, default=1)
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    intr, wm, wr = load_configs(args)
    cfg = TagProcessingConfig(max_world_reproj_px=args.max_world_reproj_px, max_wrist_reproj_px=args.max_wrist_reproj_px, min_world_tags=args.min_world_tags, backend=args.backend)
    for ep in args.episodes:
        summary = process_tags(EpisodePaths(ep), intr=intr, world_map=wm, wrists=wr, config=cfg, progress=not args.no_progress)
        print(f"{ep.name}: world {summary['world_pose_valid_fraction']*100:.1f}%{'' if summary['world_tags'] else ' (no world tags)'}  L {summary['left_wrist_valid_fraction']*100:.1f}%  R {summary['right_wrist_valid_fraction']*100:.1f}%  reproj {summary['world_reprojection_error_median']:.2f}px  ({summary['detector_backend']}/{summary['tag_family']})")


if __name__ == "__main__":
    main()
