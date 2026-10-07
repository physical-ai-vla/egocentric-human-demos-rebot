"""The POC arm pose source, wired into the canonical pose pipeline (POC section 3: one new provider, no second stack).

    RgbdFrame -> RgbdPalmPoseEstimator -> PoseEstimate -> [TrackingSupervisor] -> WristPose -> relative SE(3) -> reBot
                 (hand3d/palm_pose.py)                     ^^^^^^^^^^^^^^^^^^ unchanged, shared with OpenVINS

`EstimatorWristPoseProvider` already turns any `PoseEstimator` into a `WristPoseProvider`, so the POC needs no new
abstraction: only a factory that plugs the RGB-D palm estimator in with `T_H_C = I` (the palm pose IS the human
control frame here — there is no wrist camera to transform away) and with its OWN `TrackingSupervisor` settings,
because palm-from-depth and fisheye VIO do not fail on the same scale.

`wrist_pose_live_*` keeps its meaning (OpenVINS wrist VIO). This source is `rgbd_hand_pose_live_*`, source
`rgbd_hand_palm`, `camera_mode: fixed`. Nothing here modifies the canonical provider or its storage contract."""
from __future__ import annotations
import numpy as np
from handumi_collector.pose.estimator import PoseEstimate
from ..hand3d.palm_pose import PalmPoseConfig, RgbdPalmPoseEstimator
from .interfaces import WristPose
from .wrist_pose_provider import EstimatorWristPoseProvider, TrackingSupervisor, TrackingSupervisorConfig

from ..hand3d.palm_pose import POSE_SOURCE  # noqa: F401  "fixed_rgbd_hand" — one tag, defined with the source
STREAM_PREFIX = "rgbd_hand_pose_live"        # never `wrist_pose_live_*`: a different sensor, a different failure mode
RELATIVE_STREAM_PREFIX = "rgbd_relative_pose_live"


class RgbdHandWristPoseProvider(EstimatorWristPoseProvider):
    """`EstimatorWristPoseProvider` that keeps the palm evidence on the pose it hands downstream.

    The base class deliberately carries only the metrics every VIO backend shares; the arm-pose health, the reason it
    was graded that way and the palm landmark count are what a POC log needs to explain a HOLD afterwards, so they are
    re-attached here. The pose itself, and every policy decision about it, still come from the base class."""

    def ingest_estimate(self, est: PoseEstimate) -> WristPose:
        wp = super().ingest_estimate(est)
        wp.extra.update(est.extra)
        return wp


def build_rgbd_hand_pose_provider(*, hand_provider, palm: PalmPoseConfig | None = None,
                                  tracking: TrackingSupervisorConfig | None = None,
                                  hand_pose_cfg=None) -> RgbdHandWristPoseProvider:
    """-> a `WristPoseProvider` over a fixed RGB-D camera. Feed it with `push_image(t_ns, frame_index, rgbd_frame)`.

    `hand_provider` is the existing `RgbdHandPoseProvider` (unchanged): this branch reuses the whole hand front-end —
    detection, handedness, identity, robust depth, deprojection — and adds only the palm-pose read-out."""
    est = RgbdPalmPoseEstimator(hand_provider, palm or PalmPoseConfig(), hand_pose_cfg=hand_pose_cfg)
    return RgbdHandWristPoseProvider(est, T_H_C=np.eye(4), supervisor=TrackingSupervisor(tracking))
