"""Dataset-level QC over all episodes + the pilot gate.

    ego-collector report [--root datasets/raw] [--task cube_stack] [--side right] [--gate pilot] [--json out.json]

Reads each episode's metadata (tracking/actions summaries written by process-tags /
generate-actions / qa) - run `process` first. The pilot gate encodes the go/no-go
thresholds for the first 20-episode batch; automatic checks only - aperture-matches-pinch
and trajectory smoothness must be eyeballed in `replay` / `visualize`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ego_collector.recording.episode import list_episodes


def collect(root: Path, task: str | None) -> list[dict]:
    rows = []
    for ep in list_episodes(root):
        m = ep.read_metadata()
        if task and m.get("task") != task:
            continue
        tr = m.get("tracking", {}) or {}
        rows.append(
            {
                "episode": ep.root.name,
                "task": m.get("task", ""),
                "layout": m.get("layout", ""),
                "order": m.get("order", ""),
                "outcome": m.get("outcome", "unknown"),
                "tier": m.get("dataset_tier", "unprocessed"),
                "duration_s": float(m.get("duration_s", 0.0) or 0.0),
                "world": tr.get("world_pose_valid_fraction"),
                "left_wrist": tr.get("left_wrist_valid_fraction"),
                "right_wrist": tr.get("right_wrist_valid_fraction"),
                "left_aperture": tr.get("left_aperture_valid_fraction"),
                "right_aperture": tr.get("right_aperture_valid_fraction"),
                "grasp_source": (m.get("actions", {}) or {}).get("grasp_source"),
                "failed_checks": m.get("qa_failed_checks", []),
            }
        )
    return rows


def _mean(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return float(np.mean(v)) if v else float("nan")


def _fmt(v, pct=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "   --"
    return f"{v*100:4.0f}%" if pct else f"{v:5.1f}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector report", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("datasets/raw"))
    p.add_argument("--task", default=None)
    p.add_argument("--side", choices=("left", "right", "both"), default="right", help="Hand(s) the gate requires (single-hand collection: the recorded hand)")
    p.add_argument("--gate", choices=("none", "pilot"), default="none")
    p.add_argument("--min-episodes", type=int, default=20)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)
    rows = collect(args.root, args.task)
    if not rows:
        raise SystemExit(f"no episodes under {args.root}" + (f" with task={args.task}" if args.task else ""))
    print(f"{'episode':16s} {'layout':6s} {'order':5s} {'outcome':8s} {'tier':15s} {'dur':>5s} {'world':>5s} {'wristL':>6s} {'wristR':>6s} {'apL':>5s} {'apR':>5s}  failed")
    for r in rows:
        print(
            f"{r['episode']:16s} {r['layout']:6s} {r['order']:5s} {r['outcome']:8s} {r['tier']:15s} {r['duration_s']:5.1f} "
            f"{_fmt(r['world'])} {_fmt(r['left_wrist']):>6s} {_fmt(r['right_wrist']):>6s} {_fmt(r['left_aperture']):>5s} {_fmt(r['right_aperture']):>5s}  {','.join(r['failed_checks']) or '-'}"
        )
    n = len(rows)
    usable = [r for r in rows if r["tier"] == "action_labelled"]
    success = [r for r in rows if r["outcome"] == "success"]
    sides = ("left", "right") if args.side == "both" else (args.side,)
    agg = {
        "episodes": n,
        "usable": len(usable),
        "usable_fraction": len(usable) / n,
        "success_fraction": len(success) / n,
        "world_mean": _mean(rows, "world"),
        **{f"{s}_wrist_mean": _mean(rows, f"{s}_wrist") for s in ("left", "right")},
        **{f"{s}_aperture_mean": _mean(rows, f"{s}_aperture") for s in ("left", "right")},
    }
    per_order: dict[str, int] = {}
    for r in rows:
        if r["order"]:
            per_order[r["order"]] = per_order.get(r["order"], 0) + 1
    print(f"\n{n} episodes  usable {len(usable)} ({agg['usable_fraction']*100:.0f}%)  success {agg['success_fraction']*100:.0f}%  per-order {per_order or '-'}")
    print(f"means: world {_fmt(agg['world_mean'])}  " + "  ".join(f"{s} wrist {_fmt(agg[f'{s}_wrist_mean'])} aperture {_fmt(agg[f'{s}_aperture_mean'])}" for s in sides))
    if args.gate == "pilot":
        gates = {"episodes_ge_min": n >= args.min_episodes, "world_visible_ge_90": agg["world_mean"] >= 0.90}
        for s in sides:
            gates[f"{s}_wrist_dropout_lt_10"] = agg[f"{s}_wrist_mean"] >= 0.90
            gates[f"{s}_aperture_valid_ge_80"] = np.isfinite(agg[f"{s}_aperture_mean"]) and agg[f"{s}_aperture_mean"] >= 0.80
        gates["usable_ge_70"] = agg["usable_fraction"] >= 0.70
        print("\npilot gate: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items()))
        print("verdict:", "PASS -> scale to 60/100 episodes" if all(gates.values()) else "FAIL -> fix tags/FOV/lighting before scaling")
        print("manual checks still required: aperture matches real pinch (replay), trajectory visually smooth (visualize)")
        agg["gates"] = gates
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"episodes": rows, "aggregate": agg}, indent=2, default=float))
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
