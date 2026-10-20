"""Pink (Pinocchio differential IK, QP per step) solver for reBot B601 (2026-10-15, user: "pink로 해줘봐").
Drop-in for PyrokiIkSolver: sync / fk / ik(q_cur, T) -> (q6, pos_err_m, rot_err_deg), q_full + idx6.

Same URDF and TCP frame as the handumi rebot_b601 embodiment (handumi-sw/assets/rebot_b601, ee link <side>_tcp), joint values
in the bridge's follower-rad convention (PyrokiIkSolver.sync maps them 1:1 as well). Per arm, a reduced model with the other
arm and the jaws locked at their observed values.

ik(): seeded with the last EXECUTED q; iterate Pink's QP (FrameTask on the TCP + a weak PostureTask toward the seed =
continuity, ConfigurationLimit = URDF limits), integrating `dt` per iteration, and box-project the cumulative step to
+-max_joint_delta per joint (same role as pyroki's limit_joint_delta). Stops when the TCP error is under tol."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

URDF = Path.home() / "handumi-sw/assets/rebot_b601/rebot_b601_bimanual.urdf"
HOME6 = {"left": [-0.3, -0.05, -0.25, 0.55, -0.1, -0.4], "right": [0.3, -0.05, -0.25, 0.55, 0.1, 0.4]}   # handumi rebot_b601 home_q


@dataclass
class PinkParams:
    position_cost: float = 1.0          # per m
    orientation_cost: float = 0.3       # per rad (pyroki 125/20 ~ 0.16; a little stiffer here)
    posture_cost: float = 1e-3          # pull toward the seed (continuity), tiny so it never fights the target
    lm_damping: float = 1e-4            # Levenberg-Marquardt damping of the frame task
    damping: float = 1e-8               # QP regulariser
    dt: float = 0.1                     # integration step per iteration
    iters: int = 40
    tol_pos_m: float = 5e-4
    tol_rot_deg: float = 0.3
    max_joint_delta_deg: float = 2.8         # must stay under ik.max_joint_step_rad (0.05 rad) or the gate rejects
    solver: str = "quadprog"


class PinkIkSolver:
    def __init__(self, side: str, params: PinkParams | None = None, urdf: Path = URDF) -> None:
        import pinocchio as pin
        import pink
        from pink.tasks import FrameTask, PostureTask
        from pink.limits import ConfigurationLimit
        self.pin, self.pink, self.side, self.p = pin, pink, side, params or PinkParams()
        self.full = pin.buildModelFromUrdf(str(urdf))
        names = [self.full.names[i] for i in range(1, self.full.njoints)]
        self.arm_names = {s: [f"{s}_joint{k}" for k in range(1, 7)] for s in ("left", "right")}
        self.qidx = {s: [self.full.joints[self.full.getJointId(n)].idx_q for n in self.arm_names[s]] for s in ("left", "right")}
        self.q_full_pin = pin.neutral(self.full)
        for s in ("left", "right"): self.q_full_pin[self.qidx[s]] = HOME6[s]
        for n in names:
            if "gripper" in n: self.q_full_pin[self.full.joints[self.full.getJointId(n)].idx_q] = 0.0715 / 2
        self.lock_ids = [self.full.getJointId(n) for n in names if n not in self.arm_names[side]]
        self._build()
        # PyrokiIkSolver-compatible layout used by callers (live_teleop seeds MockBridge with q_full[idx6])
        self.q_full = np.r_[HOME6["left"], HOME6["right"]].astype(np.float64); self.svc_idx = {"left": list(range(6)), "right": list(range(6, 12))}
        self.idx6 = self.svc_idx[side]
        self.FrameTask, self.PostureTask, self.ConfigurationLimit = FrameTask, PostureTask, ConfigurationLimit

    def _build(self) -> None:
        pin = self.pin
        self.model = pin.buildReducedModel(self.full, self.lock_ids, self.q_full_pin)
        self.data = self.model.createData(); self.fid = self.model.getFrameId(f"{self.side}_tcp")
        self.ridx = [self.model.joints[self.model.getJointId(n)].idx_q for n in self.arm_names[self.side]]   # reduced q order -> joint k

    def sync(self, joints_deg_all) -> None:
        """Other arm (+ jaws) follow the observed robot -> rebuild the reduced model only when the other arm actually moved."""
        from ego_teleop.robot.rebot_client import SIDE_SLICE
        other = "left" if self.side == "right" else "right"
        q_o = np.radians(np.asarray(joints_deg_all[SIDE_SLICE[other]], np.float64))
        for s in ("left", "right"): self.q_full[self.svc_idx[s]] = np.radians(np.asarray(joints_deg_all[SIDE_SLICE[s]], np.float64))
        if np.max(np.abs(self.q_full_pin[self.qidx[other]] - q_o)) > 1e-3:
            self.q_full_pin[self.qidx[other]] = q_o; self._build()        # the right arm's kinematics do not depend on it, but keep it exact

    def _q(self, q6) -> np.ndarray:
        q = self.pin.neutral(self.model); q[self.ridx] = np.asarray(q6, np.float64); return q

    def fk(self, q6) -> np.ndarray:
        pin = self.pin; q = self._q(q6)
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.fid].homogeneous.copy()

    def ik(self, q_current_rad, T_target) -> tuple[np.ndarray, float, float]:
        pin, pink, P = self.pin, self.pink, self.p
        q0 = self._q(q_current_rad); q = q0.copy()
        task = self.FrameTask(f"{self.side}_tcp", position_cost=P.position_cost, orientation_cost=P.orientation_cost, lm_damping=P.lm_damping)
        post = self.PostureTask(cost=P.posture_cost); post.set_target(q0)
        task.set_target(pin.SE3(np.asarray(T_target[:3, :3], np.float64), np.asarray(T_target[:3, 3], np.float64)))
        lim = [self.ConfigurationLimit(self.model)]; dmax = np.radians(P.max_joint_delta_deg)
        for _ in range(P.iters):
            conf = pink.Configuration(self.model, self.data, q)
            pe, re = pose_errors(conf.get_transform_frame_to_world(f"{self.side}_tcp").homogeneous, T_target)
            if pe < P.tol_pos_m and re < P.tol_rot_deg: break
            v = pink.solve_ik(conf, [task, post], P.dt, solver=P.solver, damping=P.damping, limits=lim, safety_break=False)
            q = pin.integrate(self.model, q, v * P.dt)
            q = q0 + np.clip(q - q0, -dmax, dmax)                          # per-command joint step bound (continuity)
        q6 = q[self.ridx].copy(); pe, re = pose_errors(self.fk(q6), T_target)
        return q6, pe, re


def pose_errors(T, T_ref) -> tuple[float, float]:
    from ego_teleop.transforms.se3 import rotation_angle_deg
    return float(np.linalg.norm(T[:3, 3] - T_ref[:3, 3])), float(rotation_angle_deg(T[:3, :3], T_ref[:3, :3]))


def pink_pair(params: dict | None = None) -> dict:
    P = PinkParams(**(params or {}))
    return {s: PinkIkSolver(s, P) for s in ("left", "right")}
