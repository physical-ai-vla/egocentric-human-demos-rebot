"""HaWoR npz -> tracking parquets -> pseudo-actions (Action v1: wrist SE3 + aperture). G3 stage.

    ego-collector process-hawor datasets/raw/episode_* [--trim-s 2.0] [--max-speed 1.5]

Requires <episode>/hands3d/hawor_export.npz (from hawor_export.py on the GPU box).
Writes world-frame tracking parquets (the SLAM world = the old AprilTag world), then runs
the standard action generation: actions/pseudo_actions.parquet has absolute pose7 AND
relative deltas (dx..drz = T_t^-1 T_t+1) per hand + grasp from the metric aperture.
Full MANO joints stay in the npz as derived metadata - not part of the action.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ego_collector.actions.generate import generate_actions
from ego_collector.commands.generate_actions import load_aperture_calibration, load_grasp_calibration
from ego_collector.filtering.pose_filter import OneEuroConfig
from ego_collector.hands3d.adapter import HaworGates, hawor_to_tracking
from ego_collector.recording.episode import EpisodePaths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector process-hawor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", type=Path, nargs="+")
    p.add_argument("--trim-s", type=float, default=2.0, help="Drop the first seconds (SLAM init / exposure warmup)")
    p.add_argument("--max-speed", type=float, default=1.5, help="Wrist teleport gate, m/s")
    p.add_argument("--max-cam-dist", type=float, default=1.2, help="Max wrist distance from the head camera, m")
    p.add_argument("--grasp-calibration", type=Path, default=Path("configs/grasp_calibration.yaml"))
    p.add_argument("--aperture-calibration", type=Path, default=Path("configs/aperture_calibration.yaml"))
    p.add_argument("--horizon", type=int, default=1)
    args = p.parse_args(argv)
    gates = HaworGates(trim_s=args.trim_s, max_speed_m_s=args.max_speed, max_cam_dist_m=args.max_cam_dist)
    acal = load_aperture_calibration(args.aperture_calibration)
    gcal = load_grasp_calibration(args.grasp_calibration)
    for ep in args.episodes:
        paths = EpisodePaths(ep)
        s = hawor_to_tracking(paths, gates=gates)
        a = generate_actions(paths, grasp_cal=gcal, aperture_cal=acal, grasp_source="tags", horizon=args.horizon, filter_config=OneEuroConfig(), frames=("world", "head"))
        print(f"{ep.name}: valid L {s['left_wrist_valid_fraction']*100:.1f}% R {s['right_wrist_valid_fraction']*100:.1f}%"
              f"  (infilled L {s['left_infilled_fraction']*100:.0f}% R {s['right_infilled_fraction']*100:.0f}%, gated-out L {s['left_gated_out_fraction']*100:.1f}% R {s['right_gated_out_fraction']*100:.1f}%)"
              f"  action valid world {a['world_action_valid_fraction']*100:.1f}%")


if __name__ == "__main__":
    main()
