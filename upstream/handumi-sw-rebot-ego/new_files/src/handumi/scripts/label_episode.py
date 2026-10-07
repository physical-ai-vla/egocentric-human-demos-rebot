#!/usr/bin/env python3
"""Attach episode-level metadata (outcome, recovery, operator, cube plan) to a dataset.

Stored as a sidecar ``meta/handumi_episodes.json`` in the dataset root — the raw
rows are never rewritten. Failure and recovery episodes are kept and labelled,
not deleted.

::

    handumi dataset label outputs/cubes_A --episode 12 --outcome success --operator op01
    handumi dataset label outputs/cubes_A --episode 13 --outcome failure --recovery --note "dropped blue"
    handumi dataset label outputs/cubes_A --episode 12 --plan outputs/plans/A.json   # merges order/layout
    handumi dataset label outputs/cubes_A --show
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SIDECAR = Path("meta") / "handumi_episodes.json"
OUTCOMES = ("success", "failure", "partial", "unknown")


def load_labels(root: Path) -> dict[str, dict]:
    path = root / SIDECAR
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_labels(root: Path, labels: dict[str, dict]) -> None:
    path = root / SIDECAR
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2, sort_keys=True))


def label_episode(
    root: Path,
    episode: int,
    *,
    outcome: str | None = None,
    recovery: bool | None = None,
    operator: str | None = None,
    note: str | None = None,
    plan: dict | None = None,
    calibration_id: str | None = None,
) -> dict:
    labels = load_labels(root)
    entry = labels.get(str(episode), {"episode_id": episode})
    if outcome is not None:
        entry["outcome"] = outcome
    if recovery is not None:
        entry["recovery"] = bool(recovery)
    if operator is not None:
        entry["operator_id"] = operator
    if note is not None:
        entry["note"] = note
    if calibration_id is not None:
        entry["calibration_id"] = calibration_id
    if plan is not None:
        entry.update(
            {
                "task": "3_cube_stack",
                "instruction": plan["instruction"],
                "cube_order": plan["order"],
                "regime": plan["regime"],
                "cube_position_mode": "fixed" if plan["regime"] == "A" else "random",
                "target_position_mode": "random" if plan["regime"] == "C" else "fixed",
                "cube_xy": plan["cube_xy"],
                "pan_xy": plan["pan_xy"],
            }
        )
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    labels[str(episode)] = entry
    save_labels(root, labels)
    return entry


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--outcome", choices=OUTCOMES, default=None)
    p.add_argument("--recovery", action="store_true", default=None)
    p.add_argument("--no-recovery", dest="recovery", action="store_false")
    p.add_argument("--operator", default=None)
    p.add_argument("--note", default=None)
    p.add_argument("--calibration-id", default=None)
    p.add_argument("--plan", type=Path, default=None, help="cube plan JSON; the entry with the same episode index is merged")
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)
    if args.show or args.episode is None:
        labels = load_labels(args.root)
        counts: dict[str, int] = {}
        for e in labels.values():
            counts[e.get("outcome", "unlabelled")] = counts.get(e.get("outcome", "unlabelled"), 0) + 1
        print(json.dumps(labels, indent=2, sort_keys=True) if args.show else "")
        print(f"{len(labels)} labelled episodes: {counts}")
        if args.episode is None:
            return
    plan = None
    if args.plan is not None:
        plans = json.loads(args.plan.read_text())
        plan = next((pl for pl in plans if pl["episode"] == args.episode), None)
        if plan is None:
            raise SystemExit(f"episode {args.episode} not in {args.plan}")
    entry = label_episode(
        args.root,
        args.episode,
        outcome=args.outcome,
        recovery=args.recovery,
        operator=args.operator,
        note=args.note,
        plan=plan,
        calibration_id=args.calibration_id,
    )
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
