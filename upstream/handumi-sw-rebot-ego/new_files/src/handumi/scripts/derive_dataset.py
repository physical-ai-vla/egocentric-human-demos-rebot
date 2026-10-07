#!/usr/bin/env python3
"""Write derived per-episode signals next to a raw HandUMI dataset (raw is untouched).

::

    handumi dataset derive outputs/cubes_A --robot rebot_b601
    # -> outputs/cubes_A/derived/episode_000.parquet ... (+ derived/summary.json)

Columns: local EEF deltas, bimanual relation (T_left_right, hand distance/rate),
gripper events (closing/opening/grasp_start/release/grasped) and, when motor
current was recorded (``observation.feetech.{side}_current_ma``), the contact
proxy (baseline/delta/contact_estimate/possible_slip).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from handumi.dataset.derived import derive_episode, head_relative_tcp
from handumi.dataset.eef_actions import tcp_trajectories
from handumi.dataset.raw import LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX
from handumi.dataset.writer import load_info
from handumi.scripts.convert_eef import _parse_episodes, resolve_tcp_calibration


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path)
    p.add_argument("--episodes", default=None)
    p.add_argument("--robot", default=None)
    p.add_argument("--tcp-calibration", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None, help="Default: <source>/derived")
    args = p.parse_args(argv)

    import pandas as pd

    from handumi.dataset import load_raw_episode
    from handumi.dataset.reader import recording_device

    info = load_info(args.source)
    out_dir = args.output or (args.source / "derived")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = recording_device(info) or "meta"
    calibration, cal_meta = resolve_tcp_calibration(info, explicit=args.tcp_calibration, robot=args.robot, device=device)
    fps = float(info.get("fps", 30))
    summary: dict[str, dict] = {}
    for idx in _parse_episodes(args.episodes, int(info.get("total_episodes", 0))):
        try:
            raw = load_raw_episode(root=args.source, episode=idx, download_videos=False)
        except Exception as exc:
            print(f"episode {idx}: SKIP ({exc})", file=sys.stderr)
            continue
        left, right = tcp_trajectories(raw.states, calibration)
        sig = raw.signals
        lw = sig.get("observation.feetech.left_width_mm", raw.states[:, LEFT_GRIPPER_INDEX] * 1000.0)
        rw = sig.get("observation.feetech.right_width_mm", raw.states[:, RIGHT_GRIPPER_INDEX] * 1000.0)
        cols = derive_episode(
            left_tcp=left,
            right_tcp=right,
            left_width_mm=np.asarray(lw, dtype=np.float64),
            right_width_mm=np.asarray(rw, dtype=np.float64),
            fps=fps,
            left_current_ma=sig.get("observation.feetech.left_current_ma"),
            right_current_ma=sig.get("observation.feetech.right_current_ma"),
        )
        # Head-camera pose in world and head-relative TCPs (AprilTag backend rows carry both).
        head_pose = sig.get("observation.tracking.workspace_from_device_pose")
        if head_pose is not None and np.asarray(head_pose).ndim == 2:
            for i, name in enumerate(("x", "y", "z", "qx", "qy", "qz", "qw")):
                cols[f"head.pose_world.{name}"] = np.asarray(head_pose, dtype=np.float32)[:, i]
            for side in ("left", "right"):
                dev = sig.get(f"observation.tracking.{side}_device_controller_pose")
                if dev is None:
                    continue
                offset = getattr(calibration, side) if calibration is not None else np.array([0, 0, 0, 0, 0, 0, 1.0])
                tcp_head = head_relative_tcp(dev, offset)
                for i, name in enumerate(("x", "y", "z", "qx", "qy", "qz", "qw")):
                    cols[f"{side}.tcp_head.{name}"] = tcp_head[:, i]
        df = pd.DataFrame(cols)
        path = out_dir / f"episode_{idx:03d}.parquet"
        df.to_parquet(path)
        summary[str(idx)] = {
            "rows": int(len(df)),
            "grasps_left": int(df["left.gripper.grasp_start"].sum()),
            "grasps_right": int(df["right.gripper.grasp_start"].sum()),
            "min_hand_distance_m": float(df["bimanual.hand_distance"].min()),
            "has_current": "left.motor.contact_estimate" in df.columns,
        }
        print(f"episode {idx}: {len(df)} rows -> {path.name}  {summary[str(idx)]}")
    (out_dir / "summary.json").write_text(json.dumps({"tcp_calibration": cal_meta, "episodes": summary}, indent=2))
    print(f"summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
