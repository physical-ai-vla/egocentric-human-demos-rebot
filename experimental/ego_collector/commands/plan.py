"""Cube-stacking collection plan: layouts x stacking orders, and progress against it.

    ego-collector plan create --layouts 10 --per-cell 1 --extra 40 --episodes-note "phase1"
    ego-collector plan status [--root datasets/raw]

`create` writes configs/collection_plan.yaml: L01..L10 x all 6 orders (base grid) plus
`--extra` single-order episodes on fresh layouts L11.. (new positions). With 3 colors that
is 60 + 40 = 100 episodes, ~16-17 per order. `status` scans recorded episode metadata
(task + layout + order, set via `record --layout L03 --order RBP`) and prints done/target
per cell, per-order totals, and the next cells to record.
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import yaml

from ego_collector.recording.episode import list_episodes

DEFAULT_PLAN = Path("configs/collection_plan.yaml")
DEFAULT_TEMPLATE = "Stack the cubes in this order: {a} at the bottom, {b} in the middle, {c} on top."


def order_codes(colors: list[str]) -> list[str]:
    return ["".join(c[0].upper() for c in p) for p in permutations(colors)]


def instruction_for(order: str, colors: list[str], template: str) -> str:
    by_initial = {c[0].upper(): c for c in colors}
    a, b, c = (by_initial[ch] for ch in order)
    return template.format(a=a, b=b, c=c)


def cmd_create(args) -> None:
    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    if len(colors) != 3 or len({c[0].upper() for c in colors}) != 3:
        raise SystemExit("--colors needs exactly 3 colors with distinct initials, e.g. red,blue,purple")
    orders = order_codes(colors)
    entries = []
    for li in range(1, args.layouts + 1):
        for o in orders:
            entries.append({"layout": f"L{li:02d}", "order": o, "target": args.per_cell})
    for j in range(args.extra):
        entries.append({"layout": f"L{args.layouts + 1 + j:02d}", "order": orders[j % len(orders)], "target": 1})
    plan = {
        "task": args.task,
        "colors": colors,
        "template": args.template,
        "note": args.episodes_note,
        "entries": entries,
    }
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(yaml.safe_dump(plan, sort_keys=False))
    total = sum(e["target"] for e in entries)
    per_order = {o: sum(e["target"] for e in entries if e["order"] == o) for o in orders}
    print(f"{args.plan}: {len(entries)} cells, {total} episodes  per-order {per_order}")
    print("base grid = repeat the SAME layout with different orders (prompt-conditioned trajectories);")
    print(f"extra {args.extra} = fresh layouts for position variation. Record with: record --task {args.task} --layout L01 --order {orders[0]} ...")


def load_plan(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} missing: run `plan create` first")
    return yaml.safe_load(path.read_text()) or {}


def cmd_status(args) -> None:
    plan = load_plan(args.plan)
    colors, template = plan["colors"], plan.get("template", DEFAULT_TEMPLATE)
    orders = order_codes(colors)
    done: dict[tuple[str, str], list[str]] = {}
    off_plan = 0
    for ep in list_episodes(args.root):
        m = ep.read_metadata()
        if m.get("task") != plan["task"]:
            continue
        key = (str(m.get("layout") or ""), str(m.get("order") or ""))
        if not key[0] or not key[1]:
            off_plan += 1
            continue
        done.setdefault(key, []).append(f"{m.get('outcome','unknown')}/{m.get('dataset_tier','unprocessed')}")
    layouts = sorted({e["layout"] for e in plan["entries"]})
    target = {(e["layout"], e["order"]): e["target"] for e in plan["entries"]}
    print(f"plan {args.plan}  task={plan['task']}  colors={colors}")
    print("       " + "  ".join(f"{o:>4s}" for o in orders))
    next_cells = []
    n_done = n_target = 0
    for L in layouts:
        row = []
        for o in orders:
            t = target.get((L, o), 0)
            d = len(done.get((L, o), []))
            n_target += t
            n_done += min(d, t)
            row.append("   ." if t == 0 else f"{d}/{t}".rjust(4))
            if t and d < t:
                next_cells.append((L, o))
        print(f"{L:6s} " + "  ".join(row))
    per_order = {o: sum(len(v) for (l, oo), v in done.items() if oo == o) for o in orders}
    print(f"total {n_done}/{n_target}  per-order {per_order}" + (f"  (+{off_plan} episodes without layout/order)" if off_plan else ""))
    for L, o in next_cells[: args.suggest]:
        print(f"next: record --task {plan['task']} --layout {L} --order {o} --instruction \"{instruction_for(o, colors, template)}\"")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector plan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="Write the collection plan yaml")
    c.add_argument("--task", default="cube_stack")
    c.add_argument("--colors", default="red,blue,purple", help="3 cube colors with distinct initials")
    c.add_argument("--layouts", type=int, default=10, help="Base layouts; each gets all 6 orders")
    c.add_argument("--per-cell", type=int, default=1)
    c.add_argument("--extra", type=int, default=40, help="Additional single-order episodes on fresh layouts (position variation)")
    c.add_argument("--template", default=DEFAULT_TEMPLATE)
    c.add_argument("--episodes-note", default="")
    c.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    c.set_defaults(func=cmd_create)
    s = sub.add_parser("status", help="Progress against the plan + next cells")
    s.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    s.add_argument("--root", type=Path, default=Path("datasets/raw"))
    s.add_argument("--suggest", type=int, default=3)
    s.set_defaults(func=cmd_status)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
