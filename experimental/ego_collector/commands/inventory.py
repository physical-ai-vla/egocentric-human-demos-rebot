"""Per-layout episode inventory + permutation-coverage check.

Made for batch recording sessions: after each layout's batch, run

    ego-collector inventory --last 8
    ego-collector inventory --layout L11

to see exactly which episodes/orders were stored, catch order drift from
restarts immediately (the L07/L08 RBP-oversampling incident), and get the
exact make-up command for any missing permutations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ORDERS = ("RBP", "RPB", "BRP", "BPR", "PRB", "PBR")


def _episodes(root: Path) -> list[tuple[Path, dict]]:
    out = []
    for d in sorted(root.glob("episode_*")):
        mp = d / "metadata.json"
        if mp.exists():
            try:
                out.append((d, json.loads(mp.read_text())))
            except json.JSONDecodeError:
                out.append((d, {}))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector inventory", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("datasets/raw"))
    p.add_argument("--last", type=int, default=None, help="show only the N most recent episodes")
    p.add_argument("--layout", default=None, help="restrict to one layout id (e.g. L11)")
    p.add_argument("--exclude", default="45,46,47,69,75,154", help="episode ids to ignore (discarded takes)")
    args = p.parse_args(argv)

    excluded = {int(x) for x in args.exclude.split(",") if x.strip()}
    eps = [(d, m) for d, m in _episodes(args.root) if m.get("episode_id") not in excluded]
    if args.layout:
        eps = [(d, m) for d, m in eps if m.get("layout") == args.layout]
    if args.last:
        eps = eps[-args.last :]
    if not eps:
        print("no episodes matched")
        return

    print(f"{'episode':16s} {'layout':7s} {'order':6s} {'frames':>6s} {'dur':>6s}  {'tier':16s}")
    for d, m in eps:
        print(
            f"{d.name:16s} {str(m.get('layout','?')):7s} {str(m.get('order','?')):6s} "
            f"{m.get('frame_count', 0):6d} {m.get('duration_s', 0):5.1f}s  {str(m.get('dataset_tier','?')):16s}"
        )

    # per-layout permutation coverage over the selection
    by_layout: dict[str, Counter] = {}
    for _, m in eps:
        by_layout.setdefault(str(m.get("layout", "?")), Counter())[str(m.get("order", "?"))] += 1

    print("\n== permutation coverage ==")
    for layout in sorted(by_layout):
        c = by_layout[layout]
        missing = [o for o in ORDERS if c[o] == 0]
        dups = {o: n for o, n in c.items() if n > 1}
        row = " ".join(f"{o}:{c[o]}" for o in ORDERS)
        status = "OK 6/6" if not missing and not dups else ""
        print(f"  {layout}: {row}  {status}")
        if dups:
            print(f"    duplicates: {dups}")
        if missing:
            task = next((m.get("task") for _, m in eps if m.get("layout") == layout), "cube_stack")
            op = next((m.get("operator_id") for _, m in eps if m.get("layout") == layout), "op01")
            print(
                f"    MISSING {missing} -> make-up:\n"
                f"    uv run ego-collector record --camera 2 --task {task} --operator {op} "
                f"--layout {layout} --batch {len(missing)} --record-s 12 --rest-s 10 "
                f"--orders {','.join(missing)} --colors red,blue,purple"
            )


if __name__ == "__main__":
    main()
