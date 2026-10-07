#!/usr/bin/env python3
"""Plan and announce 3-cube stacking episodes (order, layout, instruction).

::

    # 60 episodes of regime A (fixed cubes/pan, random order), plans -> JSON
    handumi task cube plan --regime A --episodes 60 --seed 1 --output outputs/plans/A.json

    # Show the plan for the next episode (prints layout + the --task string)
    handumi task cube show outputs/plans/A.json --episode 12

    # Print just the instruction so it can be piped into `handumi record --task`
    handumi record --task "$(handumi task cube show outputs/plans/A.json --episode 12 --task-only)" ...

The recorder stores one task string per episode; keep the plan JSON next to the
dataset so cube/pan positions can be audited for the A/B/C splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from handumi.tasks.cube_stack import make_plans, read_plans, write_plans


def cmd_plan(args: argparse.Namespace) -> int:
    plans = make_plans(
        regime=args.regime,
        episodes=args.episodes,
        seed=args.seed,
        start_episode=args.start_episode,
        template_index=None if args.random_template else args.template,
    )
    write_plans(plans, args.output)
    counts: dict[str, int] = {}
    for p in plans:
        key = "-".join(p.order)
        counts[key] = counts.get(key, 0) + 1
    print(f"Wrote {len(plans)} regime-{args.regime} plans -> {args.output}")
    print("order balance:", ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    plans = read_plans(args.plans)
    match = [p for p in plans if p.episode == args.episode]
    if not match:
        raise SystemExit(f"episode {args.episode} not in {args.plans}")
    plan = match[0]
    if args.task_only:
        print(plan.instruction)
    else:
        print(plan.layout_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Generate episode plans as JSON.")
    plan.add_argument("--regime", choices=("A", "B", "C"), required=True)
    plan.add_argument("--episodes", type=int, required=True)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--start-episode", type=int, default=0)
    plan.add_argument("--template", type=int, default=0, help="Instruction template index (default 0: 'Stack a, b, c.').")
    plan.add_argument("--random-template", action="store_true", help="Randomize the phrasing per episode.")
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(func=cmd_plan)
    show = sub.add_parser("show", help="Print one episode's layout and instruction.")
    show.add_argument("plans", type=Path)
    show.add_argument("--episode", type=int, required=True)
    show.add_argument("--task-only", action="store_true")
    show.set_defaults(func=cmd_show)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
