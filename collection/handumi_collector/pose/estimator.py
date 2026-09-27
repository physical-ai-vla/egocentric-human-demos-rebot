"""Backend-independent pose estimator interface. Every VIO / VO backend (mock, opencv_vo, orbslam3, dpvo, ...) speaks
this and nothing backend-specific leaks into the derived dataset schema (PoseEstimate is the schema)."""
from __future__ import annotations
import enum
from dataclasses import dataclass, field, asdict
import numpy as np


class TrackingState(str, enum.Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    DEGRADED = "degraded"
    LOST = "lost"
    STATIC = "static"            # segmented VIO: the IMU says the camera is not moving; pose held, rotation from the gyro

    @property
    def valid(self) -> bool:
        return self in (TrackingState.TRACKING, TrackingState.DEGRADED, TrackingState.STATIC)


@dataclass
class PoseEstimate:
    """One camera pose. Metrics a backend does not provide stay None — never faked."""
    timestamp_ns: int
    frame_index: int
    T_world_camera: np.ndarray | None          # 4x4 in the backend's (session) world; None when not valid
    tracking_state: TrackingState = TrackingState.UNINITIALIZED
    confidence: float | None = None            # backend-specific [0,1]
    num_features: int | None = None
    reprojection_error: float | None = None    # px
    imu_residual_deg: float | None = None      # backend-internal IMU/visual rotation residual, if it has one
    pose_jump_score: float | None = None       # backend-internal jump score, if it has one
    extra: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.T_world_camera is not None and self.tracking_state.valid

    def position(self) -> np.ndarray | None:
        return None if self.T_world_camera is None else self.T_world_camera[:3, 3].copy()

    def to_row(self) -> dict:
        from .se3 import T_to_pose7
        p = T_to_pose7(self.T_world_camera) if self.T_world_camera is not None else [np.nan] * 7
        return dict(t_ns=int(self.timestamp_ns), frame_index=int(self.frame_index), valid=bool(self.valid),
                    tracking_state=self.tracking_state.value, x=p[0], y=p[1], z=p[2], qx=p[3], qy=p[4], qz=p[5], qw=p[6],
                    confidence=self.confidence, num_features=self.num_features, reprojection_error=self.reprojection_error,
                    imu_residual_deg=self.imu_residual_deg, pose_jump_score=self.pose_jump_score)


@dataclass(frozen=True)
class BackendInfo:
    name: str
    uses_imu: bool                 # visual-inertial (True) or visual-only (False → IMU used for independent QA only)
    metric_scale: bool             # True when translations are metres (VI or stereo/depth); False for monocular VO
    online_capable: bool           # can run frame-by-frame during a session (continuous_session_tracking)
    provides: tuple[str, ...] = () # PoseEstimate metric names this backend actually fills
    notes: str = ""


class BackendUnavailable(RuntimeError):
    """Raised by a backend that needs software / binaries / calibration that are not present. Callers report, never fake."""


class PoseEstimator:
    """Streaming interface. Feed images (and IMU when `info.uses_imu`) in timestamp order; each push_image returns the
    estimate for that frame. Timestamps are ns on ONE common clock (the caller applies camera<->IMU offsets)."""
    info: BackendInfo

    def initialize(self, *, intrinsics: dict | None = None, T_camera_imu: np.ndarray | None = None, imu_noise: dict | None = None,
                   image_size: tuple[int, int] | None = None) -> None:
        raise NotImplementedError

    def push_imu(self, t_ns: int, gyro_rad_s, accel_m_s2) -> None:
        """Ignored by visual-only backends."""

    def push_image(self, t_ns: int, frame_index: int, image_bgr: np.ndarray) -> PoseEstimate:
        raise NotImplementedError

    def get_pose(self) -> PoseEstimate | None:
        """Latest estimate (online use)."""
        return getattr(self, "_last", None)

    def get_quality(self) -> dict:
        """Backend-specific health summary (counts, map size, ...). Free-form, for logs / UI."""
        return {}

    def reset(self) -> None:
        raise NotImplementedError

    def finish(self) -> list[PoseEstimate] | None:
        """Offline backends that refine / smooth after the last frame return the final trajectory here; streaming
        backends return None (their push_image results stand)."""
        return None
