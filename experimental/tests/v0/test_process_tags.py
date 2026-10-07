"""End-to-end: synthetic head video (moving camera, moving wrists) -> process_tags -> parquets."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pandas as pd

from ego_collector.io.parquet import write_parquet
from ego_collector.recording.episode import EpisodePaths
from ego_collector.tracking.detector import OpenCVTagDetector
from ego_collector.tracking.process import TagProcessingConfig, process_tags
from ego_collector.tracking.transforms import pose7_to_T, rotation_angle_deg, translation_distance
from ego_collector.tracking.wrist_pose import WristExtrinsics
from synth import H, INTR, W, head_pose, render, scene_faces, world_map, wrist_pose

FPS = 30
N = 24


def make_episode(root) -> tuple[EpisodePaths, list[dict]]:
    paths = EpisodePaths(root / "episode_000001")
    paths.root.mkdir(parents=True)
    wm = world_map()
    writer = cv2.VideoWriter(str(paths.video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt = []
    t0 = time.monotonic_ns()
    ts = []
    for i in range(N):
        a = i / (N - 1)
        T_wc = head_pose(x=0.02 * np.sin(2 * np.pi * a), y=-0.30 + 0.04 * a, z=0.62 - 0.02 * a, tilt_deg=47 + 2 * np.sin(4 * a), yaw_deg=3 * np.cos(3 * a))
        left = wrist_pose(-0.20 + 0.15 * a, 0.30, 0.08 + 0.10 * a)
        right = wrist_pose(0.20 - 0.10 * a, 0.35 - 0.05 * a, 0.12)
        wrists = {10: (0.05, left), 20: (0.05, right)} if i not in (10, 11) else {20: (0.05, right)}  # left wrist leaves the view for 2 frames
        img = render(scene_faces(wm, T_wc, wrists), INTR)
        writer.write(img)
        ts.append(t0 + i * 33_333_333)
        gt.append({"T_wc": T_wc, "left": left if 10 in wrists else None, "right": right})
    writer.release()
    write_parquet(paths.timestamps, {"frame_index": np.arange(N), "capture_index": np.arange(N), "timestamp_ns": np.asarray(ts, dtype=np.int64)})
    paths.write_metadata({"episode_id": 1, "task": "synthetic", "instruction": "synthetic"})
    return paths, gt


def test_process_tags_end_to_end(tmp_path):
    paths, gt = make_episode(tmp_path)
    summary = process_tags(paths, intr=INTR, world_map=world_map(), wrists=WristExtrinsics.default("tag36h11"), config=TagProcessingConfig(backend="opencv"), detector=OpenCVTagDetector("tag36h11"), progress=False)
    assert summary["frames"] == N
    assert summary["world_pose_valid_fraction"] > 0.95
    assert abs(summary["left_wrist_valid_fraction"] - (N - 2) / N) < 1e-6
    assert summary["right_wrist_valid_fraction"] > 0.95
    cam = pd.read_parquet(paths.camera_pose)
    wr = pd.read_parquet(paths.wrist_pose)
    tags = pd.read_parquet(paths.apriltags)
    assert len(cam) == N and len(wr) == N and set(tags["role"]) >= {"world", "left", "right"}
    assert list(cam["timestamp_ns"]) == list(wr["timestamp_ns"])
    for i in range(N):
        c = cam.iloc[i]
        assert bool(c["world_pose_valid"]) and c["world_tag_count"] >= 2
        T_est = pose7_to_T([c[f"head_{k}"] for k in ("x", "y", "z", "qx", "qy", "qz", "qw")])
        assert translation_distance(T_est, gt[i]["T_wc"]) * 1000 < 8.0
        assert rotation_angle_deg(T_est, gt[i]["T_wc"]) < 0.8
        w = wr.iloc[i]
        if gt[i]["left"] is None:
            assert not bool(w["left_wrist_valid"]) and np.isnan(w["left_wrist_x"])
        else:
            T_l = pose7_to_T([w[f"left_wrist_{k}"] for k in ("x", "y", "z", "qx", "qy", "qz", "qw")])
            assert translation_distance(T_l, gt[i]["left"]) * 1000 < 12.0
        T_r = pose7_to_T([w[f"right_wrist_{k}"] for k in ("x", "y", "z", "qx", "qy", "qz", "qw")])
        assert translation_distance(T_r, gt[i]["right"]) * 1000 < 12.0
    meta = json.loads(paths.metadata.read_text())
    assert meta["tracking"]["frames"] == N and meta["tag_family"] == "tag36h11"
    # raw detections keep corners so poses can be re-solved later
    row = tags.iloc[0]
    assert len(row["corners"]) == 8 and len(row["center"]) == 2
