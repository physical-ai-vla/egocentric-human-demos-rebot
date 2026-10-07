"""Head-frame-only mode: no world tags on the table (they can be added later, but not retroactively)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ego_collector.actions.generate import generate_actions
from ego_collector.hands.grasp import GraspCalibration
from ego_collector.qa.episode import QAThresholds, run_qa
from ego_collector.tracking.detector import OpenCVTagDetector
from ego_collector.tracking.process import TagProcessingConfig, process_tags
from ego_collector.tracking.wrist_pose import WristExtrinsics
from synth import INTR
from test_process_tags import N, make_episode


def test_process_without_world_map_gives_head_frame_actions(tmp_path):
    paths, gt = make_episode(tmp_path)  # the video still contains world tags; we simply do not use them
    paths.update_metadata(frame_count=N, dropped_frames=0, duration_s=(N - 1) / 30, fps_measured=30.0, fps_target=30)
    s = process_tags(paths, intr=INTR, world_map=None, wrists=WristExtrinsics.default("tag36h11"), config=TagProcessingConfig(backend="opencv"), detector=OpenCVTagDetector("tag36h11"), progress=False)
    assert s["world_tags"] is False and s["world_pose_valid_fraction"] == 0.0
    wp = pd.read_parquet(paths.wrist_pose)
    assert not wp["world_pose_valid"].any()
    assert wp["right_tag_visible"].all() and np.isnan(wp["right_wrist_x"]).all() and np.isfinite(wp["right_wrist_cam_x"]).all()
    tags = pd.read_parquet(paths.apriltags)
    assert set(tags["role"]) == {"unknown", "left", "right"}  # 100..103 detected but unassigned
    a = generate_actions(paths, grasp_cal=GraspCalibration(), frames=("head",))
    assert a["head_pose_valid_fraction"] > 0.9 and paths.pseudo_actions_head.exists() and not paths.pseudo_actions.exists()
    r = run_qa(paths, QAThresholds(require_world=False, require_both_hands=False, min_wrist_valid_fraction=0.9, min_duration_s=0.5))
    assert r.tier == "action_labelled", r.checks
    assert any("head-frame-only" in w for w in r.warnings)
    r2 = run_qa(paths, QAThresholds(require_world=True, require_both_hands=False, min_wrist_valid_fraction=0.9, min_duration_s=0.5))
    assert r2.tier == "action_labelled"  # metadata says world_tags=False -> world checks turn soft automatically
