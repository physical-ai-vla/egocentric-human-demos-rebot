"""Real-time filters (2026-10-14, spec 16 + 27).

Two kinds, deliberately different:
  * MEASUREMENT filters (OneEuro on the human position): they clean a sensor signal and always advance -- the human did move,
    whether or not the robot followed.
  * ROBOT-TARGET filters (TransactionalPoseFilter): their state is part of "what the robot executed". propose() is pure;
    commit() only after the command was sent AND accepted; on any HOLD the state stays at the last executed value (spec 27)."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from ego_teleop.transforms.se3 import make_T


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff_hz); return 1.0 / (1.0 + tau / dt)


@dataclass
class OneEuro:
    """One-Euro filter on a vector (Casiez 2012): low cutoff at rest (no jitter), higher cutoff with speed (low lag)."""
    min_cutoff_hz: float = 1.0
    beta: float = 0.5
    d_cutoff_hz: float = 1.0
    _x: np.ndarray | None = None
    _dx: np.ndarray | None = None
    _t_ns: int | None = None

    def reset(self, x=None, t_ns: int | None = None) -> None:
        self._x = None if x is None else np.asarray(x, np.float64).copy(); self._dx = None; self._t_ns = t_ns

    def update(self, x, t_ns: int) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if not np.all(np.isfinite(x)): raise ValueError("OneEuro: non-finite measurement (caller must treat it as a dropout)")
        if self._x is None or self._t_ns is None or t_ns <= self._t_ns:
            if self._x is None or self._t_ns is None: self._x, self._dx, self._t_ns = x.copy(), np.zeros_like(x), int(t_ns)
            return self._x.copy()
        dt = (t_ns - self._t_ns) / 1e9; self._t_ns = int(t_ns)
        dx = (x - self._x) / dt; ad = _alpha(self.d_cutoff_hz, dt)
        self._dx = dx if self._dx is None else self._dx + ad * (dx - self._dx)
        cutoff = self.min_cutoff_hz + self.beta * float(np.linalg.norm(self._dx))
        self._x = self._x + _alpha(cutoff, dt) * (x - self._x)
        return self._x.copy()


@dataclass
class TransactionalPoseFilter:
    """EMA position + geodesic SLERP rotation toward the target; state = the last EXECUTED filtered pose."""
    alpha_pos: float = 0.35
    alpha_rot: float = 0.3
    # 10-16 One-Euro style (VR/mocap teleop): alpha from the target's speed -- heavy smoothing when slow (jitter), light when fast (lag).
    # mode fixed = the alphas above; one_euro: cutoff = min_cutoff_hz + beta * speed (m/s for position, rad/s for rotation), dt = 1/rate
    mode: str = "fixed"
    min_cutoff_hz: float = 1.0           # position min cutoff (one_euro)
    min_cutoff_rot_hz: float = 1.8       # rotation min cutoff (one_euro)
    beta_pos: float = 30.0               # per m/s   (= 0.03 per mm/s)
    beta_rot: float = 3.0                # per rad/s (~ 0.05 per deg/s)
    d_cutoff_hz: float = 1.0             # derivative low-pass (Casiez 2012)
    deadband_m: float = 0.0              # 10-16: |target - filtered| below this -> position held (stationary jitter)
    deadband_deg: float = 0.0
    predict_s: float = 0.0               # 10-16: extrapolate the raw target by its filtered velocity (perceived latency)
    dt_s: float = 1.0 / 30
    _T: np.ndarray | None = field(default=None, repr=False)
    _raw: np.ndarray | None = field(default=None, repr=False)      # previous RAW target (for the speed estimate)
    _spd: tuple = field(default=(0.0, 0.0), repr=False)            # filtered (lin m/s, ang rad/s)
    _vel: tuple = field(default=(None, None), repr=False)          # filtered velocity vectors (m/s, rad/s in the world frame)
    _pending: tuple | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for a in (self.alpha_pos, self.alpha_rot):
            if not 0.0 < a <= 1.0: raise ValueError("filter alphas must be in (0, 1]")

    def reset(self, T) -> None: self._T = np.asarray(T, np.float64).copy(); self._raw = None; self._spd = (0.0, 0.0); self._vel = (None, None); self._pending = None

    @property
    def value(self) -> np.ndarray | None: return None if self._T is None else self._T.copy()

    def propose(self, T_target) -> np.ndarray:
        """Pure: the filtered pose this tick would produce. Nothing stored."""
        if self._T is None: raise RuntimeError("TransactionalPoseFilter.reset() before propose()")
        T1 = np.asarray(T_target, np.float64)
        R0 = Rotation.from_matrix(self._T[:3, :3]); rv = (R0.inv() * Rotation.from_matrix(T1[:3, :3])).as_rotvec(); dp = T1[:3, 3] - self._T[:3, 3]
        if self.mode == "one_euro":
            # speed of the RAW target (not of the lag), low-passed at d_cutoff -> cutoff = min_cutoff + beta * speed (One-Euro)
            if self._raw is None: vl = np.zeros(3); va = np.zeros(3)
            else:
                vl = (T1[:3, 3] - self._raw[:3, 3]) / self.dt_s
                va = (Rotation.from_matrix(T1[:3, :3]) * Rotation.from_matrix(self._raw[:3, :3]).inv()).as_rotvec() / self.dt_s   # world-frame angular velocity
            ad = _alpha(self.d_cutoff_hz, self.dt_s)
            fl = vl if self._vel[0] is None else self._vel[0] + ad * (vl - self._vel[0]); fa = va if self._vel[1] is None else self._vel[1] + ad * (va - self._vel[1])
            sl, sa = float(np.linalg.norm(fl)), float(np.linalg.norm(fa))
            self._pending = (T1.copy(), (sl, sa), (fl, fa))             # applied in commit() only (transactional)
            ap = _alpha(self.min_cutoff_hz + self.beta_pos * sl, self.dt_s); ar = _alpha(self.min_cutoff_rot_hz + self.beta_rot * sa, self.dt_s)
            if self.predict_s > 0:                                       # short-horizon prediction of the TARGET (not of the filtered state)
                Tp = T1.copy(); Tp[:3, 3] = T1[:3, 3] + fl * self.predict_s; Tp[:3, :3] = (Rotation.from_rotvec(fa * self.predict_s) * Rotation.from_matrix(T1[:3, :3])).as_matrix()
                T1 = Tp; rv = (R0.inv() * Rotation.from_matrix(T1[:3, :3])).as_rotvec(); dp = T1[:3, 3] - self._T[:3, 3]
            if float(np.linalg.norm(dp)) < self.deadband_m: ap = 0.0
            if float(np.degrees(np.linalg.norm(rv))) < self.deadband_deg: ar = 0.0
        else: ap, ar = self.alpha_pos, self.alpha_rot
        return make_T((R0 * Rotation.from_rotvec(ar * rv)).as_matrix(), self._T[:3, 3] + ap * dp)

    def commit(self, T_executed) -> None:
        """The command derived from propose() was executed: it becomes the filter state."""
        self._T = np.asarray(T_executed, np.float64).copy()
        if self._pending is not None: self._raw, self._spd, self._vel = self._pending; self._pending = None
