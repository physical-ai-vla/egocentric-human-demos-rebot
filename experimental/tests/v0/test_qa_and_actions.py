"""Synthetic episode -> process_tags -> generate_actions -> qa tiers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ego_collector.actions.generate import generate_actions
from ego_collector.hands.grasp import GraspCalibration
from ego_collector.qa.episode import QAThresholds, run_qa
from ego_collector.tracking.detector import OpenCVTagDetector
from ego_collector.tracking.process import TagProcessingConfig, process_tags
from ego_collector.tracking.wrist_pose import WristExtrinsics
from synth import INTR, world_map
from test_process_tags import N, make_episode


def test_actions_and_qa_tiers(tmp_path):
    paths, gt = make_episode(tmp_path)
    paths.update_metadata(frame_count=N, dropped_frames=0, duration_s=(N - 1) / 30, fps_measured=30.0, fps_target=30)
    process_tags(paths, intr=INTR, world_map=world_map(), wrists=WristExtrinsics.default("tag36h11"), config=TagProcessingConfig(backend="opencv"), detector=OpenCVTagDetector("tag36h11"), progress=False)
    s = generate_actions(paths, grasp_cal=GraspCalibration())
    pa = pd.read_parquet(paths.pseudo_actions)
    # head-relative ablation: written alongside, valid whenever the wrist tag is visible (no world needed)
    ph = pd.read_parquet(paths.pseudo_actions_head)
    assert len(ph) == N and (ph["reference_frame"] == "head").all() and ph["left_valid"].sum() == N - 2
    wp = pd.read_parquet(paths.wrist_pose)
    np.testing.assert_allclose(ph["right_x"].to_numpy(), wp["right_wrist_cam_x"].to_numpy(), atol=1e-9)
    assert s["head_pose_valid_fraction"] >= s["world_pose_valid_fraction"]
    bf = pd.read_parquet(paths.bimanual_features)
    assert len(pa) == N and len(bf) == N
    assert set(pa.columns) >= {"left_dx", "right_drz", "left_grasp", "right_valid", "action_valid", "left_x_filtered", "timestamp_ns"}
    assert pa["left_valid"].sum() == N - 2 and np.isnan(pa["left_grasp"]).all()  # no hands processed -> grasp NaN
    assert s["action_valid_fraction"] == 0.0  # grasp required for a complete action
    # delta consistency on the right wrist: ||dx,dy,dz|| matches the synthetic motion (~1 cm/frame)
    d = np.linalg.norm(pa[["right_dx", "right_dy", "right_dz"]].to_numpy()[:-1], axis=1)
    assert np.nanmedian(d) < 0.02 and np.nanmax(d) < 0.03
    assert bf["both_valid"].sum() == N - 2 and np.isfinite(bf["left_right_distance"][0])
    # QA: without hands -> video_only when both hands are required; action_labelled when not
    r = run_qa(paths, QAThresholds())
    assert r.tier == "video_only" and "hands_processed" in [k for k, c in r.checks.items() if not c["pass"]]
    r2 = run_qa(paths, QAThresholds(require_both_hands=False, min_wrist_valid_fraction=0.9, min_duration_s=0.5))
    assert r2.tier == "action_labelled", r2.checks
    meta = json.loads(paths.metadata.read_text())
    assert meta["dataset_tier"] == "action_labelled" and paths.qa_report.exists()
    # strict wrist availability -> the 2 missing left frames (91.7 %) drop it to video_only, video kept
    r3 = run_qa(paths, QAThresholds(require_both_hands=False, min_wrist_valid_fraction=0.95, min_duration_s=0.5))
    assert r3.tier == "video_only" and paths.video.exists()
