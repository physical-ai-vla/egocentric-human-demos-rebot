from __future__ import annotations

import argparse
import json
from pathlib import Path

from ego_collector.recording.episode import EpisodePaths, list_episodes
from ego_collector.recording.metadata import OUTCOMES


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector label", description="Set outcome / recovery / notes (metadata.json). Nothing is deleted.")
    p.add_argument("episode", type=Path, nargs="?", help="Episode dir; omit with --list to summarize a raw root")
    p.add_argument("--outcome", choices=OUTCOMES, default=None)
    p.add_argument("--recovery", action="store_true", default=None)
    p.add_argument("--no-recovery", dest="recovery", action="store_false")
    p.add_argument("--operator", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--list", type=Path, default=None, help="Raw root to summarize (tier / outcome per episode)")
    args = p.parse_args(argv)
    if args.list:
        for ep in list_episodes(args.list):
            m = ep.read_metadata()
            print(f"{ep.root.name}  {m.get('task','?'):26s} {m.get('dataset_tier','?'):16s} {m.get('outcome','?'):8s} rec={m.get('recovery', False)}  {m.get('frame_count', 0)}f")
        return
    if args.episode is None:
        raise SystemExit("episode required (or --list ROOT)")
    paths = EpisodePaths(args.episode)
    fields = {}
    if args.outcome is not None:
        fields["outcome"] = args.outcome
    if args.recovery is not None:
        fields["recovery"] = bool(args.recovery)
    if args.operator is not None:
        fields["operator_id"] = args.operator
    if args.notes is not None:
        fields["notes"] = args.notes
    meta = paths.update_metadata(**fields) if fields else paths.read_metadata()
    print(json.dumps({k: meta.get(k) for k in ("episode_id", "task", "outcome", "recovery", "dataset_tier", "operator_id", "notes")}, indent=2))


if __name__ == "__main__":
    main()
