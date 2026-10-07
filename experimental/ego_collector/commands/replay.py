from __future__ import annotations

import argparse
from pathlib import Path

from ego_collector.camera.intrinsics import CameraIntrinsics
from ego_collector.recording.episode import EpisodePaths
from ego_collector.visualization.replay import EpisodeReplay


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector replay", description="Frame-by-frame replay with tag / wrist / hand / grasp overlays.")
    p.add_argument("episode", type=Path)
    p.add_argument("--camera-config", type=Path, default=Path("configs/camera/c922.yaml"))
    p.add_argument("--scale", type=float, default=0.6)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--export", type=Path, default=None, help="Write the overlay video instead of opening a window")
    args = p.parse_args(argv)
    intr = CameraIntrinsics.load(args.camera_config) if args.camera_config.exists() else None
    rep = EpisodeReplay(EpisodePaths(args.episode), intr)
    if args.export:
        print(rep.render_video(args.export, scale=args.scale))
    else:
        rep.run(scale=args.scale, start=args.start)


if __name__ == "__main__":
    main()
