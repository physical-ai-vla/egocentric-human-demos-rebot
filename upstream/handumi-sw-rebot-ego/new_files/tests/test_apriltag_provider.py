"""End-to-end: synthetic head camera -> AprilTagTrackingProvider thread -> tracker-eval CSV."""

from __future__ import annotations

import threading
import time

import numpy as np

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.cameras.base import CameraDevice, CameraSample
from handumi.eval.tracker_eval import CaptureBuilder, static_jitter
from handumi.robots.utils import IDENTITY_POSE7
from handumi.tracking.apriltag import AprilTagDetector, AprilTagTrackingProvider
from test_apriltag import HANDS_WORLD, HEAD_WORLD, INTR_PIN, _config, _render, _scene, _world_map  # sibling test module


class FakeCamera(CameraDevice):
    """Static head-camera scene at ~60 Hz with monotonic timestamps."""

    def __init__(self, image: np.ndarray) -> None:
        self._image = np.repeat(image[:, :, None], 3, axis=2)
        self._lock = threading.Lock()
        self._t0 = time.monotonic_ns()

    @property
    def output_width(self) -> int:
        return self._image.shape[1]

    @property
    def output_height(self) -> int:
        return self._image.shape[0]

    def connect(self) -> None:
        return None

    def async_read(self) -> np.ndarray:
        return self._image

    def sample_at(self, target_time_ns=None) -> CameraSample:
        seq = (time.monotonic_ns() - self._t0) // 16_000_000
        return CameraSample(image=self._image, capture_time_ns=self._t0 + seq * 16_000_000, sequence=int(seq))

    def disconnect(self) -> None:
        return None


def test_provider_streams_world_frame_samples_and_feeds_tracker_eval():
    cfg, world = _config(), _world_map()
    det = AprilTagDetector(cfg.dictionary)
    img = _render(_scene(cfg, world, np.linalg.inv(HEAD_WORLD), HANDS_WORLD), det, INTR_PIN)
    provider = AprilTagTrackingProvider(
        camera=FakeCamera(img),
        intrinsics=INTR_PIN,
        config=cfg,
        calibration=ControllerTcpCalibration(left=IDENTITY_POSE7.copy(), right=IDENTITY_POSE7.copy()),
        world_map=world,
    )
    provider.start()
    builder = CaptureBuilder()
    last_seq = None
    deadline = time.monotonic() + 4.0
    try:
        while time.monotonic() < deadline and len(builder) < 30:
            s = provider.latest()
            if s.streaming and s.sequence != last_seq:
                last_seq = s.sequence
                builder.add(
                    t_ns=s.aligned_time_ns, mark=0, streaming=True, clock_synced=s.clock_synced,
                    left_tracked=s.left_tracked, right_tracked=s.right_tracked,
                    left_controller=s.left_controller_pose, right_controller=s.right_controller_pose,
                    left_tcp=s.left_tcp_pose, right_tcp=s.right_tcp_pose,
                )
            time.sleep(0.004)
    finally:
        provider.stop()
    cap = builder.build()
    assert len(cap) >= 15, f"only {len(cap)} samples (err={provider.last_error})"
    assert cap.tracked["left"].all() and cap.tracked["right"].all()
    assert abs(np.median(cap.tcp["left"][:, 2]) - HANDS_WORLD["left"][2, 3]) < 0.01  # world-frame height
    rep = static_jitter(cap, min_samples=10)
    assert rep.status == "PASS" and rep.sides["left"].translation_rms_mm < 0.5
    image, result = provider.latest_frame()
    assert image is not None and result is not None and result.world is not None and result.world.solve is not None
    time.sleep(0.3)
    stale = provider.latest()
    assert not stale.left_tracked and not stale.streaming and not stale.hmd_tracked
