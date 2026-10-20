"""Per-arm IK + feasibility (2026-10-14, spec 20). Reuses the existing solver and gate -- nothing re-implemented:
    PyrokiIkSolver (handumi rebot_b601 continuity IK, seed = last EXECUTED q; runs from ~/xvla-mac/bin/python)
    FeasibilityGate (finite, residual, MIT-bridge joint limits, per-command joint step = branch-jump guard)
ToyIk is for tests/replay without pyroki: q6 = [x, y, z, rotvec] of the TCP itself (numbers meaningless, plumbing only)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from ego_teleop.robot.feasibility import FeasibilityConfig, FeasibilityGate, FeasibilityVerdict
from ego_teleop.robot.rebot_client import JointLimits, bridge_joint_limits_rad
from ego_teleop.transforms.se3 import make_T


class ToyIk:
    """FK(q) = make_T(Exp(q[3:]), q[:3]); IK exact unless the target is in `unreachable(T)`."""
    def __init__(self, unreachable=None) -> None: self.unreachable = unreachable or (lambda T: False)
    def sync(self, joints_deg_all) -> None: pass
    def fk(self, q) -> np.ndarray: q = np.asarray(q, np.float64); return make_T(Rotation.from_rotvec(q[3:]).as_matrix(), q[:3])
    def ik(self, q_cur, T) -> tuple[np.ndarray, float, float]:
        q = np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()]
        return (q, 0.5, 30.0) if self.unreachable(T) else (q, 0.0, 0.0)


def toy_limits() -> JointLimits:
    return JointLimits(lower_rad=(-2.0, -2.0, -2.0, -np.pi, -np.pi, -np.pi), upper_rad=(2.0, 2.0, 2.0, np.pi, np.pi, np.pi), max_step_rad=0.05)


@dataclass
class ArmIk:
    solver: object
    gate: FeasibilityGate

    @classmethod
    def from_cfg(cls, c, solver, limits: JointLimits | None = None) -> "ArmIk":
        k = c.ik
        if limits is None:
            lo, hi = bridge_joint_limits_rad(k["limit_margin_deg"])
            limits = JointLimits(lower_rad=tuple(lo), upper_rad=tuple(hi), max_step_rad=k["max_joint_step_rad"])
        return cls(solver, FeasibilityGate(FeasibilityConfig(k["max_pos_err_m"], k["max_rot_err_deg"], limits)))

    def solve(self, q_exec: np.ndarray, T_target: np.ndarray) -> FeasibilityVerdict:
        try: q, pe, re = self.solver.ik(np.asarray(q_exec, np.float64), np.asarray(T_target, np.float64))
        except Exception as e: return FeasibilityVerdict(False, [f"ik_error:{type(e).__name__}"], None, float("nan"), float("nan"), float("nan"))
        return self.gate.check(q_exec, q, pe, re)

    def fk(self, q) -> np.ndarray: return self.solver.fk(np.asarray(q, np.float64))


def pyroki_pair(overrides: dict | None = None) -> dict:
    """{'left': PyrokiIkSolver, 'right': PyrokiIkSolver} (needs handumi -> run from ~/xvla-mac/bin/python).
    overrides: ik.solver_overrides from the YAML (rest_weight, max_joint_delta_deg)."""
    from ego_teleop.robot.rebot_client import PyrokiIkSolver
    ov = dict(overrides or {})
    if "max_joint_delta_deg" in ov: ov["max_joint_delta"] = float(np.radians(ov.pop("max_joint_delta_deg")))
    return {s: PyrokiIkSolver.for_rebot_b601(s, ik_overrides=ov or None) for s in ("left", "right")}


def ik_pair(c) -> dict:
    """Solver per side from the YAML: ik.backend = pyroki (handumi, default so far) | pink (Pinocchio QP, ik_pink.py)."""
    if c.ik.get("backend", "pyroki") == "pink":
        from .ik_pink import pink_pair
        return pink_pair(c.ik.get("pink"))
    return pyroki_pair(c.ik.get("solver_overrides"))
