"""VIO_STABLE gate: HOMING / recording / teleop anchoring may start only after the VIO has been initialised AND has produced
`min_stable_s` of continuous TRACKING_OK poses while the wrist is (nearly) still. This is how initialization becomes an explicit
lifecycle state instead of a QA exception: the HOME still-window is recorded AFTER this gate, so the existing HOME QA keeps its
meaning unchanged."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .interfaces import TrackingHealth, WristPose


@dataclass
class VioStableConfig:
    min_stable_s: float = 2.0            # continuous OK required
    max_lin_vel_m_s: float = 0.05        # "still enough" while proving stability
    max_ang_vel_deg_s: float = 10.0
    max_pos_drift_m: float = 0.02        # position excursion inside the stable window (a drifting still pose is not stable)


@dataclass
class VioStableGate:
    cfg: VioStableConfig = field(default_factory=VioStableConfig)
    _t0: int | None = None
    _p0: np.ndarray | None = None
    stable: bool = False
    reason: str = "no poses yet"

    def reset(self) -> None: self._t0 = self._p0 = None; self.stable = False; self.reason = "reset"

    def update(self, wp: WristPose | None) -> bool:
        if wp is None or wp.health != TrackingHealth.OK:
            self._t0 = self._p0 = None; self.stable = False; self.reason = "not TRACKING_OK" if wp else "no pose"; return False
        c = self.cfg
        if np.linalg.norm(wp.linear_velocity_xyz) > c.max_lin_vel_m_s or np.degrees(np.linalg.norm(wp.angular_velocity_xyz)) > c.max_ang_vel_deg_s:
            self._t0 = self._p0 = None; self.stable = False; self.reason = "moving: hold the wrist still"; return False
        if self._t0 is None: self._t0 = wp.timestamp_ns; self._p0 = wp.position_xyz_m.copy()
        if np.linalg.norm(wp.position_xyz_m - self._p0) > c.max_pos_drift_m:
            self._t0 = wp.timestamp_ns; self._p0 = wp.position_xyz_m.copy(); self.stable = False; self.reason = "position drifting while still"; return False
        held = (wp.timestamp_ns - self._t0) / 1e9
        self.stable = held >= c.min_stable_s; self.reason = "stable" if self.stable else f"stabilising {held:.1f}/{c.min_stable_s:.1f}s"
        return self.stable
