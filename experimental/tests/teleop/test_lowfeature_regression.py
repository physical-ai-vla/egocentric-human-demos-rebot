"""Low-feature regression (the sparse 12 s synthetic scene): the VIO must degrade gracefully — DEGRADED/LOST, never a runaway
trajectory — and the downstream chain must HOLD. Runs the real OpenVINS through ov_bridge; skipped when the bridge is not built."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from handumi_collector.pose.estimator import BackendUnavailable

pytestmark = pytest.mark.slow


def _bridge_available():
    try:
        from ego_teleop.tracking.backends.openvins import import_bridge; import_bridge(); return True
    except BackendUnavailable: return False


@pytest.mark.skipif(not _bridge_available(), reason="ov_bridge not built (~/vio/ov_bridge/build.sh)")
def test_sparse_scene_degrades_gracefully_and_downstream_holds(tmp_path):
    from ego_teleop.tools.m1_vio import run, synth_cal_dir
    from handumi_collector.pose.episode_io import derived_dir, read_table
    from handumi_collector.pose.se3 import poses7_to_T
    from ego_teleop.tracking.interfaces import WristPose, TrackingHealth
    from ego_teleop.tracking.wrist_pose_provider import TrackingSupervisor
    from ego_teleop.retarget.arm_relative_se3 import RelativeSE3Retargeter
    from ego_teleop.transforms.frames import HumanRobotFrameMapper
    from handumi_collector.pose.estimator import PoseEstimate, TrackingState
    from ego_teleop.tracking.wrist_pose_provider import EstimatorWristPoseProvider
    ep = tmp_path / "sparse12"
    subprocess.run([sys.executable, "-m", "handumi_collector.tools.synth_episode", str(ep), "--seconds", "12"], check=True, capture_output=True)
    cal = synth_cal_dir(ep, "right", ep / "_cal")
    res = run(ep, "right", backend="openvins", downscale=2, pose_config=None, overrides={}, cal_dir=cal)
    cam = read_table(derived_dir(ep, "openvins") / "right_camera_pose")
    states = cam["tracking_state"].value_counts().to_dict()
    valid = cam["valid"].to_numpy(bool); P = cam[["x", "y", "z"]].to_numpy(float)[valid]
    # graceful: mostly not-tracking, and whatever WAS reported valid stays in a sane volume (no runaway)
    assert states.get("tracking", 0) < 0.5 * len(cam), states
    assert len(P) == 0 or np.all(np.linalg.norm(P, axis=1) < 2.0), "runaway trajectory reported as valid"
    # downstream chain: replay the estimates through the provider + retargeter -> arm target held while LOST
    class Replay:
        info = type("I", (), dict(name="replay"))()
        def get_quality(self): return {}
    prov = EstimatorWristPoseProvider(Replay(), T_H_C=np.eye(4), supervisor=TrackingSupervisor())
    rt = RelativeSE3Retargeter(HumanRobotFrameMapper()); engaged = False; held = 0; moved = 0; R0 = np.eye(4)
    p7 = cam[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy(float); p7 = np.where(np.isfinite(p7), p7, [0, 0, 0, 0, 0, 0, 1]); Ts = poses7_to_T(p7)
    for i, row in cam.iterrows():
        est = PoseEstimate(int(row.t_ns), int(row.frame_index), Ts[i] if row.valid else None, TrackingState(row.tracking_state), confidence=row.confidence, num_features=row.num_features)
        wp = prov.ingest_estimate(est)
        if not engaged and wp.valid: rt.engage(wp, R0); engaged = True; continue
        if engaged:
            tgt = rt.update(wp, R0); held += tgt.held; moved += (not tgt.held)
            assert np.linalg.norm(tgt.T_RB_RE_target[:3, 3]) < 2.5
    assert held > moved, (held, moved)
