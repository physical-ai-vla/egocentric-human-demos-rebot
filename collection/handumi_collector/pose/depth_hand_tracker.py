"""Backend-independent RGB-D rigid-body 6DoF tracker for the HandUMI controller body (ActiveUMI Stage A).

    RGB + depth + intrinsics + HandUMI rigid CAD mesh  ->  T_depthcam_body  (per side, per frame)

This is the ONLY interface the rest of the pipeline knows. Backend-specific numbers (FoundationPose scores, ICP fitness,
...) are carried in the canonical fields below or in `extra` for logs — they are never baked into the derived dataset
schema, so a backend swap does not change the dataset.

What this module deliberately does NOT do:
  * it does not estimate the jaw/finger opening  (grip comes from the Feetech encoder, authoritative)
  * it does not track moving HandUMI parts       (the mesh must be the rigid body only — see tracking_mesh.py)
  * it does not hide failures                    (uncertain => tracking_valid=False; never a copied last pose,
                                                  never a silent interpolation — see depth_run.py)

Frames:  D = the depth camera frame the pose is expressed in (head Orbbec, depth aligned to colour)
         B = the HandUMI rigid body frame defined by the tracking mesh
         pose = T_D_B (4x4), stored as position_xyz_m + quaternion_xyzw (never Euler; RPY is UI/debug only)."""
from __future__ import annotations
import enum
import importlib
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .rgbd_io import CameraIntrinsics
from .se3 import T_to_pose7, pose7_to_T

SIDES = ("left", "right")


class TrackingState(str, enum.Enum):
    UNINITIALIZED = "uninitialized"
    TRACKING = "tracking"
    DEGRADED = "degraded"      # pose produced but below the backend's own confidence bar -> valid, flagged
    LOST = "lost"              # no pose this frame
    REACQUIRED = "reacquired"  # first good frame after a LOST run

    @property
    def valid(self) -> bool:
        return self in (TrackingState.TRACKING, TrackingState.DEGRADED, TrackingState.REACQUIRED)


@dataclass
class RigidPoseEstimate:
    """One frame's rigid-body pose. Metrics a backend does not provide stay None — never faked."""
    timestamp_ns: int
    side: str
    frame_index: int = -1
    position_xyz_m: np.ndarray | None = None          # (3,) in the depth-camera frame D
    quaternion_xyzw: np.ndarray | None = None         # (4,) xyzw, qw >= 0
    tracking_valid: bool = False
    confidence: float | None = None                   # [0,1], backend-normalised
    reprojection_error_px: float | None = None
    depth_residual_mm: float | None = None            # model-vs-measured depth agreement (backend's inlier RMSE)
    tracking_state: TrackingState = TrackingState.UNINITIALIZED
    n_scene_points: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def T_depthcam_body(self) -> np.ndarray | None:
        if self.position_xyz_m is None or self.quaternion_xyzw is None:
            return None
        return pose7_to_T(np.concatenate([np.asarray(self.position_xyz_m, np.float64).reshape(3),
                                          np.asarray(self.quaternion_xyzw, np.float64).reshape(4)]))

    @classmethod
    def from_T(cls, T: np.ndarray | None, *, timestamp_ns: int, side: str, frame_index: int = -1,
               state: TrackingState = TrackingState.TRACKING, **kw) -> "RigidPoseEstimate":
        if T is None:
            return cls(timestamp_ns=int(timestamp_ns), side=side, frame_index=int(frame_index), tracking_valid=False,
                       tracking_state=TrackingState.LOST, **kw)
        p = T_to_pose7(T)
        return cls(timestamp_ns=int(timestamp_ns), side=side, frame_index=int(frame_index), position_xyz_m=p[:3],
                   quaternion_xyzw=p[3:7], tracking_valid=state.valid, tracking_state=state, **kw)

    def to_row(self) -> dict:
        p = self.position_xyz_m if self.position_xyz_m is not None else [np.nan] * 3
        q = self.quaternion_xyzw if self.quaternion_xyzw is not None else [np.nan] * 4
        return dict(frame_index=int(self.frame_index), t_ns=int(self.timestamp_ns), side=self.side,
                    x=float(p[0]), y=float(p[1]), z=float(p[2]),
                    qx=float(q[0]), qy=float(q[1]), qz=float(q[2]), qw=float(q[3]),
                    tracking_valid=bool(self.tracking_valid), tracking_state=self.tracking_state.value,
                    confidence=self.confidence, reprojection_error_px=self.reprojection_error_px,
                    depth_residual_mm=self.depth_residual_mm, n_scene_points=self.n_scene_points)


@dataclass(frozen=True)
class DepthBackendInfo:
    name: str
    needs_mesh: bool                  # requires the rigid CAD mesh (all model-based trackers do)
    needs_init_roi: bool              # requires an initial bbox/mask on the first frame
    metric_scale: bool                # True for every depth-based tracker
    can_reacquire: bool               # can recover on its own after a full loss (no operator re-init)
    runs_on: str = "cpu"              # "cpu" | "cuda"
    provides: tuple[str, ...] = ()    # RigidPoseEstimate fields this backend actually fills
    notes: str = ""


class TrackerUnavailable(RuntimeError):
    """Backend software / weights / hardware missing. Callers report it; they never substitute a different backend silently."""


class DepthHandPoseTracker:
    """Streaming, single-side. One instance per hand — a tracker never sees both hands, so it cannot swap their identity
    (§10); the two instances share only the camera frame D, which is what makes the inter-hand transform meaningful."""
    INFO: DepthBackendInfo

    def __init__(self, **options: Any) -> None:
        self.options = dict(options)
        self.side: str | None = None
        self.intrinsics: CameraIntrinsics | None = None
        self._last: RigidPoseEstimate | None = None

    # ---------------------------------------------------------------- lifecycle
    def initialize(self, rgb: np.ndarray, depth: np.ndarray, camera_intrinsics: CameraIntrinsics, side: str,
                   initial_mask: np.ndarray | None = None, initial_bbox: tuple[int, int, int, int] | None = None,
                   *, timestamp_ns: int = 0, frame_index: int = 0) -> RigidPoseEstimate:
        """Register the mesh into the first frame. `initial_bbox` is (x, y, w, h) in colour pixels; `initial_mask` is a
        bool/uint8 image. Returns the first pose (or a LOST estimate when registration fails — never an exception for a
        merely difficult frame)."""
        raise NotImplementedError

    def track(self, rgb: np.ndarray, depth: np.ndarray, timestamp_ns: int, *, frame_index: int = -1) -> RigidPoseEstimate:
        raise NotImplementedError

    def reset(self) -> None:
        """Drop all tracking state; the next call must be initialize()."""
        raise NotImplementedError

    def get_status(self) -> dict:
        """Free-form health summary for logs / UI. Not part of the dataset schema."""
        return dict(backend=self.INFO.name, side=self.side,
                    state=(self._last.tracking_state.value if self._last else TrackingState.UNINITIALIZED.value))

    def get_pose(self) -> RigidPoseEstimate | None:
        return self._last


# ------------------------------------------------------------------------------------------------ registry
_REGISTRY: dict[str, str] = {
    "mock": "handumi_collector.pose.depth_backends.mock:MockDepthTracker",
    "icp": "handumi_collector.pose.depth_backends.icp:IcpDepthTracker",
    "foundationpose": "handumi_collector.pose.depth_backends.foundationpose:FoundationPoseTracker",
}


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def make_tracker(name: str, **options) -> DepthHandPoseTracker:
    if name not in _REGISTRY:
        raise KeyError(f"unknown depth tracker backend {name!r}; known: {available_backends()}")
    mod, cls = _REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)(**options)


def backend_info(name: str) -> DepthBackendInfo:
    mod, cls = _REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls).INFO
