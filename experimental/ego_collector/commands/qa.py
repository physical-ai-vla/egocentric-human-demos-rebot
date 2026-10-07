from __future__ import annotations

import argparse
from pathlib import Path

from ego_collector.qa.episode import QAThresholds, run_qa
from ego_collector.recording.episode import EpisodePaths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector qa", description="Episode QA -> qa/report.json, dataset_tier in metadata")
    p.add_argument("episodes", type=Path, nargs="+")
    p.add_argument("--thresholds", type=Path, default=Path("configs/qa.yaml"))
    args = p.parse_args(argv)
    th = QAThresholds.from_yaml(args.thresholds)
    for ep in args.episodes:
        r = run_qa(EpisodePaths(ep), th)
        failed = [k for k, c in r.checks.items() if not c["pass"] and c["hard"]]
        print(f"{ep.name}: {r.tier.upper():16s} failed={failed or '-'} warnings={len(r.warnings)}")
        for w in r.warnings:
            print(f"   ! {w}")


if __name__ == "__main__":
    main()
