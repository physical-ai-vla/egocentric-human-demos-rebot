"""UMI-style relative SE(3) arm retargeting with clutch/recenter (spec sections 5-9).

    engage / release:   T_H0 <- current human wrist pose (T_W_H),  T_R0 <- current robot EEF pose (T_RB_RE)
    every tick:         Delta_H = inv(T_H0) · T_W_H(t)          (human motion in the anchor wrist frame)
                        Delta_R = HumanRobotFrameMapper(Delta_H) (basis change + scale; the only axis logic)
                        T_target = T_R0 · Delta_R               (robot motion in the anchor EEF frame)

The VIO world origin is irrelevant (only local consistency matters) and VIO is never reset here — only the teleop
anchors are. TRACKING_LOST => hold the last target (no IMU-only XYZ). CLUTCHED => target frozen regardless of motion."""
from __future__ import annotations
import enum
from dataclasses import dataclass, field
import numpy as np
from ..tracking.interfaces import TrackingHealth, WristPose
from ..transforms.frames import HumanRobotFrameMapper
from ..transforms.se3 import inv_T, local_delta


class ArmRetargetState(str, enum.Enum):
    DISENGAGED = "disengaged"   # no anchors: target = current robot pose
    ENGAGED = "engaged"         # following
    CLUTCHED = "clutched"       # frozen by operator


@dataclass
class ArmTarget:
    T_RB_RE_target: np.ndarray
    state: ArmRetargetState
    held: bool                      # True when the target was NOT updated this tick (clutch / tracking lost / disengaged)
    hold_reason: str = ""
    speed_factor: float = 1.0       # 1.0 normal, <1 under TRACKING_DEGRADED (consumed by the safety limiter)
    delta_H: np.ndarray | None = None      # inv(T_H0) T_H(t)   (logged raw, robot-agnostic)
    delta_R: np.ndarray | None = None      # mapped delta

    def action6(self, T_prev_target: np.ndarray | None) -> np.ndarray:
        """Per-tick arm action a_t = [dxyz, rotvec] of inv(T_prev_target) · T_target (EEF-local), zeros if none."""
        return np.zeros(6) if T_prev_target is None else local_delta(T_prev_target, self.T_RB_RE_target)


@dataclass
class RelativeSE3Retargeter:
    mapper: HumanRobotFrameMapper
    degraded_speed_factor: float = 0.3
    state: ArmRetargetState = ArmRetargetState.DISENGAGED
    T_H0: np.ndarray | None = None
    T_R0: np.ndarray | None = None
    _last_target: np.ndarray | None = field(default=None, repr=False)

    # ---- anchors -------------------------------------------------------------------------------------------
    def engage(self, wrist: WristPose, T_RB_RE: np.ndarray) -> None:
        if not wrist.valid: raise RuntimeError("cannot engage: wrist tracking not valid")
        self.T_H0 = wrist.T(); self.T_R0 = np.asarray(T_RB_RE, np.float64).copy(); self._last_target = self.T_R0.copy()
        self.state = ArmRetargetState.ENGAGED

    def clutch(self) -> None:
        if self.state == ArmRetargetState.ENGAGED: self.state = ArmRetargetState.CLUTCHED

    def release(self, wrist: WristPose, T_RB_RE: np.ndarray) -> None:
        """Clutch release = re-anchor at the current human AND robot poses => zero jump by construction."""
        self.engage(wrist, T_RB_RE)

    def disengage(self) -> None:
        self.state = ArmRetargetState.DISENGAGED; self.T_H0 = self.T_R0 = None; self._last_target = None

    # ---- per tick ------------------------------------------------------------------------------------------
    def update(self, wrist: WristPose | None, T_RB_RE_current: np.ndarray) -> ArmTarget:
        cur = np.asarray(T_RB_RE_current, np.float64)
        if self.state == ArmRetargetState.DISENGAGED:
            return ArmTarget(cur.copy(), self.state, True, "disengaged")
        assert self._last_target is not None
        if self.state == ArmRetargetState.CLUTCHED:
            return ArmTarget(self._last_target.copy(), self.state, True, "clutch")
        if wrist is None or wrist.health == TrackingHealth.LOST:
            return ArmTarget(self._last_target.copy(), self.state, True, "tracking_lost", speed_factor=0.0)
        delta_H = inv_T(self.T_H0) @ wrist.T()
        delta_R = self.mapper.map_delta(delta_H)
        T_target = self.T_R0 @ delta_R
        self._last_target = T_target
        sf = self.degraded_speed_factor if wrist.health == TrackingHealth.DEGRADED else 1.0
        return ArmTarget(T_target.copy(), self.state, False, "", sf, delta_H, delta_R)

    @property
    def last_target(self) -> np.ndarray | None: return None if self._last_target is None else self._last_target.copy()
