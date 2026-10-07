"""Finger tags end-to-end: process_tags (finger_pose) -> tag-aperture actions -> QA with sides=right."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ego_collector.actions.generate import generate_actions
from ego_collector.hands.grasp import ApertureCalibration, GraspCalibration
from ego_collector.io.parquet import write_parquet
from ego_collector.qa.episode import QAThresholds, run_qa
from ego_collector.tracking.detector import OpenCVTagDetector
from ego_collector.tracking.process import TagProcessingConfig, process_tags
from ego_collector.tracking.transforms import T_from_xyz_rpy
from ego_collector.tracking.wrist_pose import WristExtrinsics
from synth import H, INTR, W, head_pose, render, scene_faces, world_map, wrist_pose

import cv2

FPS = 30
N = 20
F = 0.016


def make_episode(root):
    """Right hand only: wrist + thumb/index tags; aperture closes linearly 80 mm -> 20 mm."""
    from ego_collector.recording.episode import EpisodePaths

    paths = EpisodePaths(root / "episode_000001")
    paths.root.mkdir(parents=True)
    wm = world_map()
    writer = cv2.VideoWriter(str(paths.video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    ts, gt_ap = [], []
    t0 = 1_000_000_000
    for i in range(N):
        a = i / (N - 1)
        ap = 0.080 - 0.060 * a
        cx, y, z = 0.10 - 0.04 * a, 0.30, 0.12
        th = T_from_xyz_rpy([cx - ap / 2, y, z], [np.radians(50), 0, 0])
        ix = T_from_xyz_rpy([cx + ap / 2, y, z], [np.radians(50), 0, 0])
        faces = {20: (0.05, wrist_pose(cx + 0.06, y + 0.03, z + 0.02)), 21: (F, th), 22: (F, ix)}
        img = render(scene_faces(wm, head_pose(), faces), INTR)
        writer.write(img)
        ts.append(t0 + i * 33_333_333)
        gt_ap.append(ap)
    writer.release()
    write_parquet(paths.timestamps, {"frame_index": np.arange(N), "capture_index": np.arange(N), "timestamp_ns": np.asarray(ts, dtype=np.int64)})
    paths.write_metadata({"episode_id": 1, "task": "cube_stack", "instruction": "t", "frame_count": N, "dropped_frames": 0, "duration_s": (N - 1) / FPS, "fps_measured": 30.0})
    return paths, np.asarray(gt_ap)


def test_tag_grasp_end_to_end(tmp_path):
    paths, gt_ap = make_episode(tmp_path)
    wr = WristExtrinsics.default("tag36h11", with_fingers=True, finger_size_m=F)
    s = process_tags(paths, intr=INTR, world_map=world_map(), wrists=wr, config=TagProcessingConfig(backend="opencv"), detector=OpenCVTagDetector("tag36h11"), progress=False)
    assert s["fingers"] and s["right_aperture_valid_fraction"] > 0.9, s
    fp = pd.read_parquet(paths.finger_pose)
    ap = fp["right_aperture_m"].to_numpy()
    ok = np.isfinite(ap)
    err_mm = np.abs(ap[ok] - gt_ap[ok]) * 1000
    # spec budget: aperture error 5-10 mm. The p90 tail covers the closest-pinch frames where the
    # two 20 mm tags nearly touch and IPPE depth error peaks (real fingers keep them further apart).
    assert np.median(err_mm) < 3.0 and np.percentile(err_mm, 90) < 10.0, err_mm
    # actions from the tag aperture (auto -> tags), calibrated to the synthetic open/closed range
    cal = ApertureCalibration(right_open_m=0.080, right_closed_m=0.020)
    a = generate_actions(paths, grasp_cal=GraspCalibration(), aperture_cal=cal, grasp_source="auto")
    assert a["grasp_source"] == "tags" and a["right_grasp_valid_fraction"] > 0.9
    pa = pd.read_parquet(paths.pseudo_actions)
    g = pa["right_grasp"].to_numpy()
    fin = np.isfinite(g)
    assert g[fin][0] > 0.9 and g[fin][-1] < 0.1  # opens ~1 -> closes ~0
    assert "right_aperture_m" in pa.columns
    # QA: single-hand dataset (sides=right); MediaPipe hands never processed but tag grasp passes
    r = run_qa(paths, QAThresholds(sides="right", min_wrist_valid_fraction=0.9, min_duration_s=0.5))
    hard_failed = [k for k, c in r.checks.items() if not c["pass"] and c["hard"]]
    assert r.tier == "action_labelled", (r.tier, hard_failed)
    assert "right_aperture_valid_fraction" in r.checks and r.checks["right_aperture_valid_fraction"]["pass"]
    assert "left_wrist_valid_fraction" not in r.checks  # left not required
    # sides=both would fail on the missing left hand
    r2 = run_qa(paths, QAThresholds(sides="both", min_wrist_valid_fraction=0.9, min_duration_s=0.5))
    assert r2.tier == "video_only"
