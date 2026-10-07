from __future__ import annotations

import argparse
from pathlib import Path

from ego_collector.hands.mediapipe_tracker import DEFAULT_MODEL, process_hands
from ego_collector.recording.episode import EpisodePaths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector process-hands", description="MediaPipe hand landmarks (offline) -> hands/hand_pose.parquet")
    p.add_argument("episodes", type=Path, nargs="+")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--mirror-handedness", action="store_true")
    p.add_argument("--delegate", choices=("auto", "cpu", "gpu"), default="auto")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)
    for ep in args.episodes:
        s = process_hands(EpisodePaths(ep), model_path=args.model, mirror_handedness=args.mirror_handedness, progress=not args.no_progress, delegate=args.delegate)
        print(f"{ep.name}: L {s['left_hand_visible_fraction']*100:.1f}%  R {s['right_hand_visible_fraction']*100:.1f}%")


if __name__ == "__main__":
    main()
