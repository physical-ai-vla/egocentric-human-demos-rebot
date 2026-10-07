"""Backend-independent tracking interfaces. Everything downstream (retargeting, safety, robot, recorder) consumes ONLY
these types; which VIO backend produced a WristPose is invisible past this boundary (M6 swaps the backend, nothing else)."""
from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from handumi_collector.pose.estimator import TrackingState
from ..transforms.se3 import make_T, T_to_pose7, pose7_to_T


class TrackingHealth(str, enum.Enum):
    """Command-policy view of wrist tracking (section 9/11 of the spec)."""
    OK = "TRACKING_OK"                # normal commands
    DEGRADED = "TRACKING_DEGRADED"    # reduce Cartesian speed
    LOST = "TRACKING_LOST"            # freeze arm translation, hold current EEF target


@dataclass
class WristPose:
    timestamp_ns: int                                   # host monotonic (camera clock after camera<->IMU offset)
    position_xyz_m: np.ndarray                          # T_W_H translation
    quaternion_xyzw: np.ndarray                         # T_W_H rotation
    linear_velocity_xyz: np.ndarray                     # m/s in W (finite difference of valid poses; zeros if unknown)
    angular_velocity_xyz: np.ndarray                    # rad/s in H (body)
    tracking_state: TrackingState                       # raw backend state
    health: TrackingHealth                              # supervised policy state (age + backend state + confidence)
    confidence: float | None = None
    source: str = ""                                    # backend name
    extra: dict = field(default_factory=dict)           # num_features, reprojection_error, ...

    @property
    def valid(self) -> bool:
        return self.health != TrackingHealth.LOST

    def T(self) -> np.ndarray:
        """T_W_H (4x4)."""
        return pose7_to_T(np.concatenate([self.position_xyz_m, self.quaternion_xyzw]))

    @classmethod
    def from_T(cls, timestamp_ns: int, T_W_H: np.ndarray, **kw) -> "WristPose":
        p = T_to_pose7(T_W_H)
        kw.setdefault("linear_velocity_xyz", np.zeros(3)); kw.setdefault("angular_velocity_xyz", np.zeros(3))
        kw.setdefault("tracking_state", TrackingState.TRACKING); kw.setdefault("health", TrackingHealth.OK)
        return cls(int(timestamp_ns), p[:3].copy(), p[3:].copy(), **kw)

    def to_row(self) -> dict:
        p, q, v, w = self.position_xyz_m, self.quaternion_xyzw, self.linear_velocity_xyz, self.angular_velocity_xyz
        return dict(t_ns=int(self.timestamp_ns), x=p[0], y=p[1], z=p[2], qx=q[0], qy=q[1], qz=q[2], qw=q[3],
                    vx=v[0], vy=v[1], vz=v[2], wx=w[0], wy=w[1], wz=w[2], tracking_state=self.tracking_state.value,
                    health=self.health.value, confidence=self.confidence, source=self.source,
                    num_features=self.extra.get("num_features"), reprojection_error=self.extra.get("reprojection_error"))


class WristPoseProvider(Protocol):
    def get_pose(self, now_ns: int | None = None) -> WristPose | None: ...


class HandHealth(str, enum.Enum):
    OK = "HAND_OK"          # fresh landmarks
    HOLD = "HAND_HOLD"      # landmarks lost < hold window: keep previous Aero target
    LOST = "HAND_LOST"      # lost longer: move to safe relaxed pose


@dataclass
class HandFeatures:
    timestamp_ns: int
    landmarks_wrist_rel: np.ndarray | None        # (21,3) wrist-relative, palm-normalized (unitless); None if not visible
    u7: np.ndarray | None                         # canonical normalized hand action in [0,1], order = AERO_CHANNELS
    confidence: float
    health: HandHealth
    side: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def visible(self) -> bool: return self.landmarks_wrist_rel is not None and self.u7 is not None

    def to_row(self) -> dict:
        row = dict(t_ns=int(self.timestamp_ns), visible=bool(self.visible), confidence=self.confidence, health=self.health.value, side=self.side)
        u = self.u7 if self.u7 is not None else [np.nan] * 7
        for i, n in enumerate(AERO_CHANNELS): row[f"u_{n}"] = float(u[i])
        lm = self.landmarks_wrist_rel
        for j in range(21):
            for k, ax in enumerate("xyz"): row[f"lm{j}_{ax}"] = float(lm[j, k]) if lm is not None else np.nan
        return row


class HandFeatureProvider(Protocol):
    def get_features(self, now_ns: int | None = None) -> HandFeatures | None: ...


# Canonical Aero active-channel order used EVERYWHERE (spec section 9). Equals ego_collector.hands3d.aero.COMPACT_NAMES.
AERO_CHANNELS = ("thumb_cmc_abduction", "thumb_cmc_flexion", "thumb_tendon", "index", "middle", "ring", "pinky")
