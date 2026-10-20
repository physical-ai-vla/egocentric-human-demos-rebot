"""[2026-10-16] reBot B601 side of the ego -> robot robotization: start poses, task frame, IK + feasibility checker.

Reuses the deployment stack, so the projected trajectory is what the robot loop would accept:
  laundry_system/robot/rebot_adapter.py  Kinematics (pink on reBot_B601_DM_dualarm.urdf, contract TCP = camera-aligned, origin =
                                          physical jaw tip) and PinkIKSolver (bridge joint limits, 3 deg clamp rule)
  laundry_system/deployment/run_rebot.py rebot_safety(): workspace boxes / table / arm separation / IK residual / joint speed,
                                          copied below (SAFETY) and checked against it in tests/test_robotize.py
  laundry_system/deployment/rebot_ui.py  READY: jaw tips L (0.28, 0.22, 0.15), R (0.28, -0.22, 0.15) m, contract x pitched 55 deg down
Every number is in the robot base frame (x forward, y left, z up), for the PHYSICAL JAW TIP.
"""
import pathlib, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path.home() / "laundry_system")); sys.path.insert(0, str(pathlib.Path.home() / "laundry_dit_training/datasets"))
from .core import make_T, task_frame  # noqa: E402

READY = dict(left=(0.28, 0.22, 0.15), right=(0.28, -0.22, 0.15), pitch_deg=55.0)
SAFETY = dict(workspace_min={"left": (0.05, -0.10, -0.03), "right": (0.05, -0.50, -0.03)},
              workspace_max={"left": (0.60, 0.50, 0.50), "right": (0.60, 0.10, 0.50)},
              table_z_m=-0.054, table_margin_m=0.010, min_bimanual_dist_m=0.10, max_joint_speed_rps=1.6,
              max_ik_pos_err_m=0.006, max_ik_rot_err_rad=0.06)


def ready_pose(side, pitch_deg=READY["pitch_deg"]):
    p = np.radians(pitch_deg); x = np.array([np.cos(p), 0, -np.sin(p)]); y = np.array([0, -1.0, 0]); z = np.cross(x, y)
    return make_T(np.stack([x, y, z], 1), READY[side])


def robot_task_frame():
    return task_frame(READY["left"], READY["right"], SAFETY["table_z_m"])


def c922_R_map(tilt_deg=None):
    """human (HandUMI camera_tcp_v2) TCP axes -> robot contract TCP axes = the virtual-C922 tilt the converter uses for the wrists"""
    from wrist_c922 import default_tilt_deg, tcp_tilt_T
    return np.asarray(tcp_tilt_T(default_tilt_deg() if tilt_deg is None else tilt_deg))[:3, :3]


class ReBotChecker:
    """checker(T_L, T_R, state) -> (ok, info, state') for core.project; state = (q_L, q_R) of the last accepted step"""

    def __init__(self, dt_s, iters=60):
        from robot.rebot_adapter import Kinematics, PinkIKSolver, SEEED_LIM_DEG
        self.kin = Kinematics(); self.ik = PinkIKSolver(self.kin, iters=iters); self.dt = dt_s
        self.lo, self.hi = np.radians(SEEED_LIM_DEG[:, 0]), np.radians(SEEED_LIM_DEG[:, 1])
        self.wmin = {k: np.array(v) for k, v in SAFETY["workspace_min"].items()}; self.wmax = {k: np.array(v) for k, v in SAFETY["workspace_max"].items()}
        qL, qR = np.zeros(6), np.zeros(6); TL, TR = ready_pose("left"), ready_pose("right")
        for _ in range(3): qL, qR = self.kin.solve(TL, TR, qL, qR, iters=300)
        self.q_ready = (qL, qR); self.n_calls = 0

    def __call__(self, T_L, T_R, state):
        self.n_calls += 1; qs = self.q_ready if state is None else state
        r = self.ik.solve(T_L, T_R, qs[0], qs[1])
        info = dict(reason=None, pos_err_m=float(r.pos_err_m), rot_err_rad=float(r.rot_err_rad))
        if not r.ok: info["reason"] = f"ik:{r.reason}"; return False, info, state
        q_L, q_R = np.asarray(r.q_L), np.asarray(r.q_R)
        FL, FR = self.kin.fk_contract(q_L, q_R)
        pe = max(np.linalg.norm(FL[:3, 3] - T_L[:3, 3]), np.linalg.norm(FR[:3, 3] - T_R[:3, 3]))
        re = max(float(np.arccos(np.clip((np.trace(F[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1))) for F, T in ((FL, T_L), (FR, T_R)))
        margin = float(np.degrees(min(np.min(np.minimum(q - self.lo, self.hi - q)) for q in (q_L, q_R))))
        info.update(pos_err_m=float(pe), rot_err_rad=float(re), joint_margin_deg=margin, tip_z_min=float(min(FL[2, 3], FR[2, 3])),
                    bimanual_m=float(np.linalg.norm(FL[:3, 3] - FR[:3, 3])))
        bad = []
        if pe > SAFETY["max_ik_pos_err_m"] or re > SAFETY["max_ik_rot_err_rad"]: bad.append("ik_residual")
        if info["tip_z_min"] < SAFETY["table_z_m"] + SAFETY["table_margin_m"]: bad.append("table")
        for k, F in (("left", FL), ("right", FR)):
            if np.any(F[:3, 3] < self.wmin[k]) or np.any(F[:3, 3] > self.wmax[k]): bad.append(f"workspace_{k}")
        if info["bimanual_m"] < SAFETY["min_bimanual_dist_m"]: bad.append("bimanual_dist")
        if state is not None:
            dq = np.max(np.abs(np.concatenate([q_L - qs[0], q_R - qs[1]]))); info["joint_step_rad"] = float(dq)
            if dq > SAFETY["max_joint_speed_rps"] * self.dt: bad.append("joint_step")
        if bad: info["reason"] = ",".join(bad); return False, info, state
        return True, info, (q_L, q_R)
