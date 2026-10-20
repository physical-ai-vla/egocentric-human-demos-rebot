"""Per-arm Cartesian limits (2026-10-14, spec 16, 18, 27). Pure checks + the TRANSACTIONAL limiter.

    candidate T  ->  NaN reject  ->  target-jump reject (tracking glitch)  ->  workspace (reject=HOLD, or clamp)
                 ->  velocity / acceleration limiter (ego_teleop CartesianLimiter.propose: pure)
State (last executed command + limiter velocity) changes only in ArmLimits.commit(); hold() keeps it (spec 27).
Every reject / clamp is returned as a reason string so the recorder logs it."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from ego_teleop.robot.safety import CartesianLimiter, SafetyConfig, WorkspaceBox


@dataclass
class LimitVerdict:
    ok: bool
    T_cmd: np.ndarray                    # candidate (ok) or the last executed command (not ok)
    reasons: list[str]
    v_next: np.ndarray | None = None
    lin_speed_m_s: float = 0.0
    ang_speed_deg_s: float = 0.0
    clamped: bool = False


@dataclass
class ArmLimits:
    workspace: WorkspaceBox
    mode: str = "hold"
    max_lin_vel_m_s: float = 0.08
    max_ang_vel_deg_s: float = 25.0
    max_lin_acc_m_s2: float = 0.8
    max_target_jump_m: float = 0.05
    max_target_jump_deg: float = 20.0
    T_exec: np.ndarray | None = None
    _prev_target: np.ndarray | None = None
    limiter: CartesianLimiter = field(init=False)

    def __post_init__(self) -> None:
        self.limiter = CartesianLimiter(SafetyConfig(workspace=self.workspace, max_lin_vel_m_s=self.max_lin_vel_m_s,
                                                     max_ang_vel_deg_s=self.max_ang_vel_deg_s, max_lin_acc_m_s2=self.max_lin_acc_m_s2))

    @classmethod
    def from_cfg(cls, c, side: str) -> "ArmLimits":
        w = c.workspace[side]; L = c.limits
        return cls(WorkspaceBox(tuple(w["min_xyz"]), tuple(w["max_xyz"])), c.workspace["mode"], L["max_lin_vel_m_s"], L["max_ang_vel_deg_s"],
                   L["max_lin_acc_m_s2"], L["max_target_jump_m"], L["max_target_jump_deg"])

    def reset(self, T_exec: np.ndarray) -> None:
        self.T_exec = np.asarray(T_exec, np.float64).copy(); self._prev_target = None; self.limiter.reset()

    def rebase_target(self) -> None:
        """Forget the previous raw target (after re-anchor): the jump check restarts from the next target."""
        self._prev_target = None

    def propose(self, T_target: np.ndarray, dt_s: float, speed_factor: float = 1.0) -> LimitVerdict:
        if self.T_exec is None: raise RuntimeError("ArmLimits.reset(T_exec) first")
        T = np.asarray(T_target, np.float64); hold = lambda r: LimitVerdict(False, self.T_exec.copy(), r)
        if not np.all(np.isfinite(T)): return hold(["nan_target"])
        if self._prev_target is not None:
            dp = float(np.linalg.norm(T[:3, 3] - self._prev_target[:3, 3]))
            dr = float(np.degrees(Rotation.from_matrix(self._prev_target[:3, :3].T @ T[:3, :3]).magnitude()))
            self._prev_target = T.copy()            # the jump test compares consecutive RAW targets (measurement glitch)
            if dp > self.max_target_jump_m or dr > self.max_target_jump_deg: return hold([f"target_jump:{dp*1e3:.0f}mm/{dr:.0f}deg"])
        else: self._prev_target = T.copy()
        reasons: list[str] = []; clamped = False; T = T.copy()
        if not self.workspace.contains(T[:3, 3]):
            if self.mode == "hold": return hold([f"workspace_reject:{np.round(T[:3, 3], 3).tolist()}"])
            T[:3, 3] = self.workspace.clamp(T[:3, 3]); clamped = True; reasons.append("workspace_clamped")
        T_cmd, v_next, v, w = self.limiter.propose(self.T_exec, T, dt_s, speed_factor)
        if not np.all(np.isfinite(T_cmd)): return hold(["nan_command"])
        return LimitVerdict(True, T_cmd, reasons, v_next, v, w, clamped)

    def commit(self, verdict: LimitVerdict, T_executed: np.ndarray | None = None) -> None:
        if not verdict.ok or verdict.v_next is None: raise ValueError("commit() of a refused verdict; call hold()")
        self.T_exec = np.asarray(verdict.T_cmd if T_executed is None else T_executed, np.float64).copy(); self.limiter.set_velocity(verdict.v_next)

    def hold(self) -> None:
        """Nothing executed: last executed command unchanged, velocity zero (the robot stopped there)."""
        self.limiter.reset()
