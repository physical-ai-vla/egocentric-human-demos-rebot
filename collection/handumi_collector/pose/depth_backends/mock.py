"""`mock` — replays a prescribed pose sequence. For unit tests and for exercising run/QA/overlay without a tracker.
It is NEVER selectable by accident from a real pilot: it refuses to run unless `poses` or `ground_truth` is given."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ..depth_hand_tracker import DepthBackendInfo, DepthHandPoseTracker, RigidPoseEstimate, TrackerUnavailable, TrackingState
from ..se3 import pose7_to_T


class MockDepthTracker(DepthHandPoseTracker):
    INFO = DepthBackendInfo(name="mock", needs_mesh=False, needs_init_roi=False, metric_scale=True, can_reacquire=True,
                            runs_on="cpu", provides=("confidence",), notes="Replays given poses. Test-only.")

    def __init__(self, *, poses=None, ground_truth: str | Path | None = None, side: str = "left",
                 noise_mm: float = 0.0, noise_deg: float = 0.0, drop_frames: tuple = (), seed: int = 0, **_ignored) -> None:
        super().__init__()
        if poses is None and ground_truth is None:
            raise TrackerUnavailable("`mock` needs poses=<(N,7) array> or ground_truth=<ground_truth.json>")
        if poses is None:
            gt = json.loads(Path(ground_truth).read_text())
            poses = gt["poses7_T_depthcam_body"]
        self._poses = np.asarray(poses, np.float64).reshape(-1, 7)
        self._drop = set(int(i) for i in drop_frames)
        self._rng = np.random.default_rng(seed)
        self._noise_mm, self._noise_deg = float(noise_mm), float(noise_deg)
        self.side = side
        self.reset()

    def reset(self) -> None:
        self._i = 0
        self._last = None

    def _emit(self, t_ns: int, frame_index: int) -> RigidPoseEstimate:
        i = frame_index if frame_index >= 0 else self._i
        self._i = i + 1
        if i in self._drop or i >= len(self._poses):
            est = RigidPoseEstimate(timestamp_ns=int(t_ns), side=self.side or "", frame_index=int(i), tracking_valid=False,
                                    tracking_state=TrackingState.LOST)
        else:
            T = pose7_to_T(self._poses[i])
            if self._noise_mm or self._noise_deg:
                from scipy.spatial.transform import Rotation
                T = T.copy()
                T[:3, 3] += self._rng.normal(0, self._noise_mm * 1e-3, 3)
                T[:3, :3] = Rotation.from_rotvec(np.radians(self._rng.normal(0, self._noise_deg, 3))).as_matrix() @ T[:3, :3]
            est = RigidPoseEstimate.from_T(T, timestamp_ns=t_ns, side=self.side or "", frame_index=i,
                                           state=TrackingState.TRACKING, confidence=1.0)
        self._last = est
        return est

    def initialize(self, rgb, depth, camera_intrinsics, side, initial_mask=None, initial_bbox=None, *,
                   timestamp_ns: int = 0, frame_index: int = 0) -> RigidPoseEstimate:
        self.side = side
        self.intrinsics = camera_intrinsics
        self.reset()
        self.side = side
        return self._emit(timestamp_ns, frame_index)

    def track(self, rgb, depth, timestamp_ns: int, *, frame_index: int = -1) -> RigidPoseEstimate:
        return self._emit(timestamp_ns, frame_index)
