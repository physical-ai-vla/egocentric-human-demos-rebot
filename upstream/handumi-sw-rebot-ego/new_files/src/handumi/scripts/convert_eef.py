#!/usr/bin/env python3
"""Convert a raw HandUMI dataset into a robot-independent EEF (TCP) action dataset.

Unlike ``handumi convert`` (joint-space IK for one robot), this keeps the data
in the table frame: ``observation.state`` = left/right TCP pose7 + jaw
openings (16), ``action`` = local SE(3) increments + jaw (14, ``--action delta``)
or the next TCP state (16, ``--action absolute``). Videos are copied unchanged.

Usage
-----
::

    handumi convert-eef outputs/cubes_A --output outputs/cubes_A_eef
    handumi convert-eef outputs/cubes_A --action absolute --horizon 3 \
        --tcp-calibration configs/calibration/controller_tcp/meta_piper_tip_temp.yaml

The controller->TCP transform is resolved in this order: dataset snapshot
(``meta/info.json`` ``handumi.controller_tcp_calibration``), ``--tcp-calibration``,
``--robot`` (``configs/robots/<robot>.yaml``), else identity with a warning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from handumi.calibration.control_tcp import (
    ControllerTcpCalibration,
    calibration_path_for_robot_device,
    controller_tcp_calibration_from_metadata,
    controller_tcp_calibration_metadata,
    load_controller_tcp_calibration,
)
from handumi.dataset.eef_actions import (
    EEF_STATE_NAMES,
    EEF_STATE_SEMANTICS,
    EefEpisode,
    action_feature,
    build_eef_episode,
)
from handumi.dataset.reader import handumi_metadata, recording_device
from handumi.dataset.writer import (
    EpisodeResult,
    load_info,
    update_handumi_metadata,
    write_dataset,
)


def _load_source_tasks(source_root: Path) -> dict[int, str]:
    from handumi.scripts.conversion import _load_source_tasks as impl

    return impl(source_root)


def resolve_tcp_calibration(
    source_info: dict[str, Any],
    *,
    explicit: Path | None,
    robot: str | None,
    device: str,
) -> tuple[ControllerTcpCalibration | None, dict[str, Any]]:
    """Pick the controller->TCP transform and describe where it came from."""
    snapshot = handumi_metadata(source_info).get("controller_tcp_calibration")
    if isinstance(snapshot, dict) and snapshot.get("applied_to_state") is not True:
        try:
            cal = controller_tcp_calibration_from_metadata(snapshot)
            return cal, {"source": "dataset_snapshot", **{k: v for k, v in snapshot.items() if k != "calibration"}}
        except Exception as exc:  # fall through to explicit paths
            print(f"  note: dataset TCP snapshot unusable ({exc}); trying files.", file=sys.stderr)
    path = explicit
    if path is None and robot:
        path, why = calibration_path_for_robot_device(robot, device)
        print(f"  controller->TCP: {why}")
    if path is not None:
        if not Path(path).exists():
            raise SystemExit(f"Controller->TCP calibration not found: {path}")
        cal = load_controller_tcp_calibration(Path(path))
        meta = controller_tcp_calibration_metadata(
            Path(path), applied_to_state=True, source_robot=robot, tracking_device=device
        )
        return cal, {"source": str(path), **meta}
    print(
        "  WARNING: no controller->TCP calibration; TCP == controller anchor. "
        "Pass --tcp-calibration or --robot for real tips.",
        file=sys.stderr,
    )
    return None, {"source": "identity"}


def write_eef_dataset(
    *,
    output_root: Path,
    source_root: Path,
    source_info: dict[str, Any],
    episodes: list[tuple[int, str, EefEpisode]],
    fps: int,
    action_repr: str,
    horizon: int,
    gripper_units: str,
    gripper_max_width_m: float,
    tcp_calibration_metadata: dict[str, Any],
    robot_type: str = "handumi_eef",
) -> None:
    """Write LeRobot v3 with 16-D state and 14/16-D action; patch the action feature."""
    results = [
        EpisodeResult(
            episode_index=i,
            states=ep.states.astype(np.float32),
            actions=ep.actions.astype(np.float32),
            task=task,
            source_episode_index=src_idx,
            source_kind=0,
        )
        for i, (src_idx, task, ep) in enumerate(episodes)
    ]
    write_dataset(
        output_root=output_root,
        source_root=source_root,
        source_info=source_info,
        episodes=results,
        robot_type=robot_type,
        joint_names=list(EEF_STATE_NAMES),
        fps=fps,
    )
    # write_dataset assumes state and action share one feature spec; the delta
    # action is 14-D, so patch the action feature after the fact.
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"] = action_feature(action_repr)  # type: ignore[arg-type]
    info_path.write_text(json.dumps(info, indent=2))
    update_handumi_metadata(
        output_root,
        {
            "state_semantics": EEF_STATE_SEMANTICS,
            "eef_actions": {
                "action_repr": action_repr,
                "horizon": int(horizon),
                "frame": "table" if handumi_metadata(source_info).get("spatial_session_calibration") else "recording_workspace",
                "gripper_units": gripper_units,
                "gripper_max_width_m": float(gripper_max_width_m),
                "rotation_parametrization": "rotvec_rad" if action_repr == "delta" else "quat_xyzw",
                "controller_tcp_calibration": tcp_calibration_metadata,
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="Raw HandUMI LeRobot dataset root.")
    p.add_argument("--output", type=Path, default=None, help="Default: <source>_eef")
    p.add_argument("--episodes", type=str, default=None, help="Comma list / ranges, e.g. 0,2,5-9")
    p.add_argument("--action", choices=("delta", "absolute"), default="delta")
    p.add_argument("--horizon", type=int, default=1, help="Action looks h frames ahead.")
    p.add_argument("--gripper-units", choices=("normalized", "m"), default="normalized")
    p.add_argument("--gripper-max-width-m", type=float, default=None, help="Default: robot yaml or 0.08.")
    p.add_argument("--tcp-calibration", type=Path, default=None)
    p.add_argument("--robot", default=None, help="Resolve TCP calibration + jaw width from configs/robots.")
    p.add_argument("--task", default=None, help="Override the task string for every episode.")
    p.add_argument("--skip-quality-filter", action="store_true")
    p.add_argument("--quality-config", type=Path, default=Path("configs/quality.yaml"))
    return p


def _parse_episodes(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [i for i in out if 0 <= i < total]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source_root = args.source
    if not (source_root / "meta" / "info.json").exists():
        raise SystemExit(f"Not a LeRobot dataset root: {source_root}")
    source_info = load_info(source_root)
    output_root = args.output or source_root.with_name(source_root.name + "_eef")
    fps = int(source_info.get("fps", 30))
    device = recording_device(source_info) or "meta"

    gripper_max = args.gripper_max_width_m
    if gripper_max is None and args.robot:
        from handumi.robots.registry import load_robot_config

        gripper_max = load_robot_config(args.robot).gripper_max_width_m
    gripper_max = float(gripper_max if gripper_max is not None else 0.08)

    calibration, cal_meta = resolve_tcp_calibration(
        source_info, explicit=args.tcp_calibration, robot=args.robot, device=device
    )

    from handumi.dataset import load_raw_episode

    quality_config = None
    if not args.skip_quality_filter:
        from handumi.dataset.quality import EpisodeQualityConfig

        quality_config = EpisodeQualityConfig.from_yaml(args.quality_config)

    task_map = _load_source_tasks(source_root)
    indices = _parse_episodes(args.episodes, int(source_info.get("total_episodes", 0)))
    episodes: list[tuple[int, str, EefEpisode]] = []
    for n, idx in enumerate(indices, start=1):
        print(f"Episode {n}/{len(indices)} (source {idx})")
        try:
            raw = load_raw_episode(root=source_root, episode=idx, download_videos=False)
        except Exception as exc:
            print(f"  SKIP: load failed — {exc}", file=sys.stderr)
            continue
        if quality_config is not None:
            from handumi.dataset.quality import validate_episode

            report = validate_episode(raw.states, fps=raw.fps, signals=raw.signals, episode_index=idx, config=quality_config)
            if not report.accepted:
                codes = ", ".join(f.code for f in report.findings if f.severity == "reject")
                print(f"  SKIP: quality — {codes}", file=sys.stderr)
                continue
        try:
            ep = build_eef_episode(
                raw.states,
                calibration=calibration,
                action_repr=args.action,
                horizon=args.horizon,
                gripper_units=args.gripper_units,
                gripper_max_width_m=gripper_max,
            )
        except ValueError as exc:
            print(f"  SKIP: {exc}", file=sys.stderr)
            continue
        task = args.task or task_map.get(idx, "HandUMI recording")
        episodes.append((idx, task, ep))
        print(f"  {len(ep.states)} rows, task={task!r}")
    if not episodes:
        raise SystemExit("No episodes converted.")
    write_eef_dataset(
        output_root=output_root,
        source_root=source_root,
        source_info=source_info,
        episodes=episodes,
        fps=fps,
        action_repr=args.action,
        horizon=args.horizon,
        gripper_units=args.gripper_units,
        gripper_max_width_m=gripper_max,
        tcp_calibration_metadata=cal_meta,
    )
    total = sum(len(ep.states) for _, _, ep in episodes)
    print(f"\nWrote {len(episodes)} episodes / {total} rows -> {output_root}")


if __name__ == "__main__":
    main()
