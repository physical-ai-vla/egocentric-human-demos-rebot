"""Safety layer between the retargeted EEF target and the robot (spec section 7; invariant F: safety is below ML/tracking).

    T_target -> workspace check -> velocity limit -> acceleration limit -> (IK + joint limits in the robot client) -> command

Everything is in the robot base frame RB and in SI units. `speed_factor` (from tracking health) scales the velocity
limits; 0 means "do not move". A stale robot observation trips the communication watchdog => hold."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from ..transforms.se3 import make_T


@dataclass
class WorkspaceBox:
    min_xyz: tuple = (-0.1, -0.5, 0.0)
    max_xyz: tuple = (0.7, 0.5, 0.6)

    def contains(self, p) -> bool:
        p = np.asarray(p, np.float64); return bool(np.all(p >= self.min_xyz) and np.all(p <= self.max_xyz))

    def clamp(self, p) -> np.ndarray:
        return np.clip(np.asarray(p, np.float64), self.min_xyz, self.max_xyz)


@dataclass
class SafetyConfig:
    workspace: WorkspaceBox = field(default_factory=WorkspaceBox)
    workspace_mode: str = "clamp"          # clamp (project onto the box) | hold (refuse the tick, keep previous command)
    max_lin_vel_m_s: float = 0.15          # start VERY low on first hardware tests (spec M2/M3)
    max_ang_vel_deg_s: float = 45.0
    max_lin_acc_m_s2: float = 1.0
    robot_obs_max_age_ms: float = 200.0    # communication watchdog on the robot observation


@dataclass
class SafetyVerdict:
    T_cmd: np.ndarray
    ok: bool                       # False => the command equals the previous command (hold)
    reasons: list[str]
    clamped: bool = False
    lin_speed_m_s: float = 0.0
    ang_speed_deg_s: float = 0.0


class CartesianLimiter:
    """Moves the commanded pose toward the target with bounded linear/angular speed and bounded linear acceleration."""

    def __init__(self, cfg: SafetyConfig) -> None:
        self.cfg = cfg; self._v_prev = np.zeros(3)

    def reset(self) -> None: self._v_prev = np.zeros(3)

    def step(self, T_prev_cmd: np.ndarray, T_target: np.ndarray, dt_s: float, speed_factor: float = 1.0) -> tuple[np.ndarray, float, float]:
        dt = max(float(dt_s), 1e-4); sf = float(np.clip(speed_factor, 0.0, 1.0))
        p0, p1 = T_prev_cmd[:3, 3], T_target[:3, 3]
        dp = p1 - p0; dist = np.linalg.norm(dp)
        v_max = self.cfg.max_lin_vel_m_s * sf
        v_des = dp / dt if dist > 0 else np.zeros(3)
        if np.linalg.norm(v_des) > v_max: v_des = v_des / np.linalg.norm(v_des) * v_max
        dv = v_des - self._v_prev; a_max = self.cfg.max_lin_acc_m_s2 * dt
        if np.linalg.norm(dv) > a_max and sf > 0: v_des = self._v_prev + dv / np.linalg.norm(dv) * a_max
        if sf == 0: v_des = np.zeros(3)
        p = p0 + v_des * dt; self._v_prev = v_des
        R0, R1 = Rotation.from_matrix(T_prev_cmd[:3, :3]), Rotation.from_matrix(T_target[:3, :3])
        rv = (R0.inv() * R1).as_rotvec(); ang = np.linalg.norm(rv)
        w_max = np.radians(self.cfg.max_ang_vel_deg_s) * sf * dt
        if ang > w_max: rv = rv / ang * w_max if ang > 0 else rv
        R = (R0 * Rotation.from_rotvec(rv)).as_matrix()
        return make_T(R, p), float(np.linalg.norm(v_des)), float(np.degrees(np.linalg.norm(rv) / dt))


class ArmSafetyPipeline:
    def __init__(self, cfg: SafetyConfig | None = None) -> None:
        self.cfg = cfg or SafetyConfig(); self.limiter = CartesianLimiter(self.cfg); self._prev_cmd: np.ndarray | None = None

    def reset(self, T_current: np.ndarray) -> None:
        self._prev_cmd = np.asarray(T_current, np.float64).copy(); self.limiter.reset()

    def check(self, T_target: np.ndarray, *, dt_s: float, speed_factor: float = 1.0, robot_obs_age_ms: float = 0.0) -> SafetyVerdict:
        if self._prev_cmd is None: raise RuntimeError("ArmSafetyPipeline.reset(T_current) before the first check")
        reasons: list[str] = []; T_t = np.asarray(T_target, np.float64).copy(); clamped = False
        if robot_obs_age_ms > self.cfg.robot_obs_max_age_ms:
            reasons.append(f"robot_obs_stale:{robot_obs_age_ms:.0f}ms"); return SafetyVerdict(self._prev_cmd.copy(), False, reasons)
        if not self.cfg.workspace.contains(T_t[:3, 3]):
            if self.cfg.workspace_mode == "hold":
                reasons.append("workspace_violation"); return SafetyVerdict(self._prev_cmd.copy(), False, reasons)
            T_t[:3, 3] = self.cfg.workspace.clamp(T_t[:3, 3]); clamped = True; reasons.append("workspace_clamped")
        T_cmd, v, w = self.limiter.step(self._prev_cmd, T_t, dt_s, speed_factor)
        if not self.cfg.workspace.contains(T_cmd[:3, 3]): T_cmd[:3, 3] = self.cfg.workspace.clamp(T_cmd[:3, 3]); clamped = True
        self._prev_cmd = T_cmd
        return SafetyVerdict(T_cmd.copy(), True, reasons, clamped, v, w)

    @property
    def last_command(self) -> np.ndarray | None: return None if self._prev_cmd is None else self._prev_cmd.copy()
