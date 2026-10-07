"""Canonical human-hand-pose representation and the provider boundary (spec sections 5, 11, 16).

`HandPoseProvider` is the swap point of the whole design: `MediaPipeHandPoseProvider` (the official monocular
baseline, B0) and `RgbdHandPoseProvider` (head RGB-D metric, B1) both produce a `HandPoseEstimate`, and every
retargeting backend consumes only that. Nothing downstream may ask which provider it came from except for logging.

Three representations are kept on every frame and all three are recorded (spec sections 11/23), because the human
pose — not the Aero target — is the canonical raw data: if Aero is replaced by another dexterous hand, the episodes
stay usable.

    landmarks_2d          (21,2) px in the UNMIRRORED sensor frame
    landmarks_camera_3d   (21,3) metres in the camera frame (NaN where invalid; metric only when `metric` is True)
    landmarks_local_3d    (21,3) wrist-origin palm-aligned, official Aero convention (aero_mocap.to_palm_local)

Hand-pose health is deliberately a different enum from wrist-VIO `TrackingHealth`: the two branches fail
independently and one must never be read as the other (spec section 16)."""
from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from . import aero_mocap as M


class HandPoseHealth(str, enum.Enum):
    INITIALIZING = "HAND_INITIALIZING"   # provider running, no accepted hand yet (or recovering, not yet stable)
    OK = "HAND_OK"                       # enough valid metric landmarks, stable identity, continuous
    DEGRADED = "HAND_DEGRADED"           # partial depth holes / transient occlusion / high uncertainty
    LOST = "HAND_LOST"                   # no hand, ambiguous identity, or a pose discontinuity


@dataclass
class HandPoseEstimate:
    timestamp_ns: int                            # host monotonic at frame capture — the teleop clock
    side: str                                    # "left" | "right", AFTER the mirror convention is applied
    landmarks_2d: np.ndarray                     # (21,2) px
    landmarks_camera_3d: np.ndarray              # (21,3) m
    landmarks_local_3d: np.ndarray               # (21,3) palm-local
    valid: np.ndarray                            # (21,) bool — per-landmark metric validity
    depth_m: np.ndarray                          # (21,) m (NaN where invalid)
    depth_confidence: np.ndarray                 # (21,) [0,1]
    detector_confidence: float                   # MediaPipe hand score
    health: HandPoseHealth
    source: str                                  # provider id: "mediapipe_mono" | "head_rgbd"
    metric: bool                                 # True when landmarks_camera_3d came from a depth sensor
    palm_scale_m: float = float("nan")           # hand-size proxy; NaN on a non-metric provider
    filled: np.ndarray | None = None             # (21,) bool — landmark reconstructed from the monocular shape prior
                                                 # because depth was missing. NEVER also marked valid: gates and health
                                                 # measure real depth coverage, retargeting gets complete geometry.
    handedness_confidence: float = float("nan")
    source_timestamp_ns: int | None = None       # device clock, when the camera exposes one
    pose_done_ns: int | None = None              # host monotonic when this estimate was finished (latency accounting)
    extra: dict = field(default_factory=dict)

    @property
    def n_filled(self) -> int: return 0 if self.filled is None else int(np.count_nonzero(self.filled))

    @property
    def n_valid(self) -> int: return int(np.count_nonzero(self.valid))

    @property
    def n_valid_tips(self) -> int: return int(np.count_nonzero(self.valid[[4, 8, 12, 16, 20]]))

    @property
    def usable(self) -> bool: return self.health in (HandPoseHealth.OK, HandPoseHealth.DEGRADED)

    def mocap25_local(self) -> np.ndarray:
        """(25,3) HandMocap keypoints in the palm-local frame — exactly what the official DexPilot node consumes."""
        return M.expand_to_mocap25(self.landmarks_local_3d)

    def latency_ms(self) -> float:
        return float("nan") if self.pose_done_ns is None else (self.pose_done_ns - self.timestamp_ns) / 1e6

    def to_row(self) -> dict:
        r = dict(t_ns=int(self.timestamp_ns), side=self.side, health=self.health.value, source=self.source,
                 metric=bool(self.metric), detector_confidence=float(self.detector_confidence),
                 handedness_confidence=float(self.handedness_confidence), palm_scale_m=float(self.palm_scale_m),
                 n_valid=self.n_valid, n_valid_tips=self.n_valid_tips, n_filled=self.n_filled,
                 source_t_ns=self.source_timestamp_ns, pose_done_ns=self.pose_done_ns, latency_ms=self.latency_ms())
        for j in range(M.N_LANDMARKS):
            r[f"lm{j}_u"] = float(self.landmarks_2d[j, 0]); r[f"lm{j}_v"] = float(self.landmarks_2d[j, 1])
            for k, ax in enumerate("xyz"):
                r[f"lm{j}_cam_{ax}"] = float(self.landmarks_camera_3d[j, k])
                r[f"lm{j}_loc_{ax}"] = float(self.landmarks_local_3d[j, k])
            r[f"lm{j}_valid"] = bool(self.valid[j]); r[f"lm{j}_depth_m"] = float(self.depth_m[j])
            r[f"lm{j}_filled"] = bool(False if self.filled is None else self.filled[j])
            r[f"lm{j}_depth_conf"] = float(self.depth_confidence[j])
        return r

    @classmethod
    def empty(cls, t_ns: int, side: str, source: str, health: HandPoseHealth = HandPoseHealth.LOST, **kw) -> "HandPoseEstimate":
        nan21 = lambda c: np.full((M.N_LANDMARKS, c), np.nan)
        return cls(int(t_ns), side, nan21(2), nan21(3), nan21(3), np.zeros(M.N_LANDMARKS, bool),
                   np.full(M.N_LANDMARKS, np.nan), np.zeros(M.N_LANDMARKS), 0.0, health, source, False, **kw)


class HandPoseProvider(Protocol):
    """The drop-in boundary of spec section 5. `frame` is provider-specific (a BGR image, or an RgbdFrame)."""

    def get_hand_pose(self, frame) -> HandPoseEstimate | None: ...
