"""HaWoR adapter: gates (trim/teleport/reach) + tracking schema + action generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ego_collector.actions.generate import generate_actions
from ego_collector.hands.grasp import ApertureCalibration, GraspCalibration
from ego_collector.hands3d.adapter import HaworGates, gate_mask, hawor_to_tracking
from ego_collector.io.parquet import write_parquet
from ego_collector.recording.episode import EpisodePaths
from scipy.spatial.transform import Rotation

FPS, T = 30, 150  # 5 s


def make_episode(root):
    paths = EpisodePaths(root / "episode_000001")
    (paths.root / "hands3d").mkdir(parents=True)
    ts = 10**9 + np.arange(T) * 33_333_333
    write_parquet(paths.timestamps, {"frame_index": np.arange(T), "timestamp_ns": ts})
    paths.write_metadata({"episode_id": 1, "task": "cube_stack", "instruction": "t", "frame_count": T, "duration_s": T / FPS, "fps_measured": 30.0, "dropped_frames": 0})
    t = np.arange(T) / FPS
    d = {"img_focal": 1400.0, "slam_scale": 1.0,
         "R_c2w": np.tile(np.eye(3), (T, 1, 1)), "t_c2w": np.zeros((T, 3)),
         "start_idx": 0, "end_idx": T}
    for s, x0 in (("left", -0.15), ("right", 0.15)):
        w = np.stack([x0 + 0.05 * np.sin(t), 0.05 * t / 5, 0.45 + 0 * t], axis=1)
        if s == "right":
            w[80] += np.array([3.0, 0, 0])  # teleport frame
        ap = 0.08 - 0.03 * (1 + np.sin(2 * np.pi * t / 5)) / 2
        d[f"{s}_wrist_pos"] = w
        d[f"{s}_thumb_tip"] = w + [0, 0.1, 0]
        d[f"{s}_index_tip"] = w + np.stack([ap, np.full(T, 0.1), np.zeros(T)], axis=1)
        d[f"{s}_aperture_m"] = ap
        d[f"{s}_root_orient_aa"] = np.stack([np.zeros(T), 0.3 * t, np.zeros(T)], axis=1)  # rotvec about y
        d[f"{s}_trans"] = w
        d[f"{s}_valid"] = np.ones(T, bool)
        d[f"{s}_joints"] = np.tile(w[:, None, :], (1, 21, 1))
    np.savez(paths.root / "hands3d" / "hawor_export.npz", **d)
    return paths


def test_gate_mask_trim_teleport_reach():
    ts = np.arange(90) / 30
    w = np.tile([0.0, 0.0, 0.5], (90, 1))
    w[70] = [5, 5, 5]  # teleport AND out of reach
    ok = gate_mask(w, np.ones(90, bool), np.zeros((90, 3)), ts, HaworGates(trim_s=1.0))
    assert not ok[:30].any() and ok[35]
    assert not ok[69] and not ok[70] and not ok[71] and ok[75]


def test_adapter_to_actions(tmp_path):
    paths = make_episode(tmp_path)
    s = hawor_to_tracking(paths, gates=HaworGates(trim_s=1.0))
    assert s["world_tags"] and s["left_wrist_valid_fraction"] > 0.7
    assert s["right_gated_out_fraction"] > 0  # the injected teleport got gated
    wp = pd.read_parquet(paths.wrist_pose)
    assert bool(wp["world_pose_valid"].all()) and not wp["left_wrist_valid"][:25].any()
    i = 60
    assert abs(wp["left_wrist_x"][i] - (-0.15 + 0.05 * np.sin(2))) < 0.02  # world pose written
    a = generate_actions(paths, grasp_cal=GraspCalibration(), aperture_cal=ApertureCalibration(left_open_m=0.08, left_closed_m=0.05, right_open_m=0.08, right_closed_m=0.05), grasp_source="tags")
    assert a["grasp_source"] == "tags" and a["world_action_valid_fraction"] > 0.5
    pa = pd.read_parquet(paths.pseudo_actions)
    ok = pa["left_valid"].to_numpy(bool) & pa["left_delta_valid"].to_numpy(bool)
    d = np.linalg.norm(pa.loc[ok, ["left_dx", "left_dy", "left_dz"]].to_numpy(), axis=1)
    assert np.nanmax(d) < 0.02  # smooth relative deltas, no teleports survived
    assert np.isfinite(pa.loc[ok, "left_grasp"]).all()
