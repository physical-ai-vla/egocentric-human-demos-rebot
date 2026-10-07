from __future__ import annotations

import argparse
from pathlib import Path

from ego_collector.recording.episode import EpisodePaths
from ego_collector.visualization.trajectory import log_rerun, plot_matplotlib


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector visualize", description="3D wrist/head trajectories + grasp signals.")
    p.add_argument("episode", type=Path)
    p.add_argument("--backend", choices=("auto", "rerun", "matplotlib"), default="auto")
    p.add_argument("--save", type=Path, default=None, help="Save the matplotlib figure (PNG) instead of showing it")
    p.add_argument("--no-video", action="store_true", help="Rerun: skip logging RGB frames")
    args = p.parse_args(argv)
    paths = EpisodePaths(args.episode)
    backend = args.backend
    if backend == "auto":
        try:
            import rerun  # noqa: F401

            backend = "rerun" if args.save is None else "matplotlib"
        except ImportError:
            backend = "matplotlib"
    if backend == "rerun":
        log_rerun(paths, with_video=not args.no_video)
    else:
        out = plot_matplotlib(paths, out=args.save, show=args.save is None)
        if out:
            print(out)


if __name__ == "__main__":
    main()
