"""[2026-10-16] Ego (UMI human) TCP trajectory -> reBot TCP trajectory: the robot-independent SE(3) part of the robotization spec.

Never copies the human's absolute pose. Every mode re-applies human motion to a robot reference:
  Mode A  start-relative, per arm:    T_R(t) = T_R0 @ dT(t),        dT(t) = inv(T_H0) @ T_H(t)
  Mode B  task-relative, both arms:   human poses -> human task frame -> robot task frame (one rigid map for both arms, so the
                                      L<->R relative pose survives exactly when scale = 1 and orientation = FULL)
  Mode C  Mode B + feasibility projection (IK / workspace / table / joint step), done by `project()` with a robot checker.

Frames. W = the ego pose world (any frame with +z = table normal up, z = 0 on the table). Task frame (human):
origin = midpoint of the two TCP start positions dropped onto the table, x = forward, y = left, z = up, where "left" is the
direction from the right-hand start to the left-hand start (task anchor = the two start X tapes). The robot task frame is
built the same way from the robot's two start TCPs and its table height. Robot base axes are x forward, y left, z up.

Translation scaling S = diag(sx, sy, sz) acts on displacements expressed in TASK axes (forward / lateral / vertical); fixed per
robot calibration, never fitted per episode. Orientation policies act on the world/task-frame delta rotation of each arm:
  FULL           R(t) = D(t) @ R_ref                 D(t) = R_task_H(t) @ R_task_H0^T (task-frame delta)
  YAW_DOMINANT   R(t) = Rz(yaw(D)) @ clamp(tilt(D), max_tilt) @ R_ref
  TASK_ALIGNED   R(t) = Rz(yaw(D)) @ R_ref           (tilt dropped: gripper keeps its reference pitch / roll)
then the human->robot TCP axes map R_map is applied on the right (R(t) @ R_map) unless it was already applied upstream.
All rotations are SO(3) (no Euler smoothing anywhere).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

MODES = ("A", "B", "C")
ORIENT = ("FULL", "YAW_DOMINANT", "TASK_ALIGNED")


# ------------------------------------------------------------------------------------------------ SE(3) helpers
def make_T(R, p):
    R = np.asarray(R, float); p = np.asarray(p, float); T = np.zeros(R.shape[:-2] + (4, 4)); T[..., :3, :3] = R; T[..., :3, 3] = p; T[..., 3, 3] = 1
    return T


def inv_T(T):
    T = np.asarray(T, float); R = np.swapaxes(T[..., :3, :3], -1, -2); return make_T(R, -np.einsum("...ij,...j->...i", R, T[..., :3, 3]))


def rot_angle_deg(R):
    return np.degrees(Rot.from_matrix(np.asarray(R)).magnitude())


def Rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def swing_twist_z(R):
    """R = Rz(yaw) @ tilt, tilt = the rotation that moves z (swing); -> (yaw rad, tilt matrix)"""
    q = Rot.from_matrix(R).as_quat()                         # xyzw
    tw = np.array([0, 0, q[2], q[3]]); n = np.linalg.norm(tw)
    if n < 1e-9: return 0.0, R
    yaw = 2 * np.arctan2(tw[2], tw[3]); return float(yaw), Rz(yaw).T @ R


def clamp_rotation(R, max_deg):
    rv = Rot.from_matrix(R).as_rotvec(); a = np.linalg.norm(rv)
    return R if a <= np.radians(max_deg) else Rot.from_rotvec(rv * np.radians(max_deg) / a).as_matrix()


def interp_T(Ta, Tb, alpha):
    """SE(3) interpolation: position linear, rotation SLERP"""
    R = Slerp([0, 1], Rot.from_matrix(np.stack([Ta[:3, :3], Tb[:3, :3]])))([alpha]).as_matrix()[0]
    return make_T(R, (1 - alpha) * Ta[:3, 3] + alpha * Tb[:3, 3])


# ------------------------------------------------------------------------------------------------ task frame
def task_frame(p_left0, p_right0, table_z=0.0):
    """T_W_task from the two TCP start positions: origin = their midpoint on the table, y = right->left (horizontal), z = up"""
    y = np.asarray(p_left0, float) - np.asarray(p_right0, float); y[2] = 0; y /= np.linalg.norm(y)
    z = np.array([0, 0, 1.0]); x = np.cross(y, z)
    o = 0.5 * (np.asarray(p_left0, float) + np.asarray(p_right0, float)); o[2] = table_z
    return make_T(np.stack([x, y, z], 1), o)


# ------------------------------------------------------------------------------------------------ config
@dataclass
class RobotizeCfg:
    mode: str = "C"
    orientation: str = "FULL"
    scale_xyz: tuple = (1.0, 1.0, 1.0)            # task axes: forward, lateral, vertical (fixed per robot calibration)
    translation_frame: str = "auto"               # "start" (dp in the human start TCP frame, spec section 5) | "task" | "auto"
    max_tilt_deg: float = 20.0                    # YAW_DOMINANT tilt clamp
    R_map: np.ndarray = field(default_factory=lambda: np.eye(3))   # human TCP axes -> robot TCP axes (right-multiplied)
    grip_a: float = 1.0
    grip_b: float = 0.0
    alpha_decay: float = 0.8                      # line search: alpha *= decay while infeasible
    alpha_min: float = 0.02

    def tframe(self):
        return ("start" if self.mode == "A" else "task") if self.translation_frame == "auto" else self.translation_frame


def map_gripper(g, cfg):
    return np.clip(cfg.grip_a * np.asarray(g, float) + cfg.grip_b, 0.0, 1.0)


# ------------------------------------------------------------------------------------------------ robotization
def orient(D, R_ref, cfg):
    """D = task-frame delta rotation of the human TCP since the start; -> (robot rotation before R_map, clamp angle deg)"""
    if cfg.orientation == "FULL": return D @ R_ref, 0.0
    yaw, tilt = swing_twist_z(D)
    if cfg.orientation == "TASK_ALIGNED": return Rz(yaw) @ R_ref, float(rot_angle_deg(tilt))
    tc = clamp_rotation(tilt, cfg.max_tilt_deg); return Rz(yaw) @ tc @ R_ref, float(rot_angle_deg(tilt @ tc.T))


def robotize(T_H, T_R0, T_Wtask_H, T_Btask_R, cfg):
    """T_H: dict side -> (N,4,4) human TCP in W; T_R0: dict side -> robot start TCP (base frame, robot TCP axes).
    -> dict(raw={side: (N,4,4) robot target}, task={side: human in task frame}, delta={side: start-relative}, clamp={side: deg})"""
    S = np.diag(cfg.scale_xyz); out = dict(raw={}, task={}, delta={}, clamp={})
    Ti_H = inv_T(T_Wtask_H)
    for s, TH in T_H.items():
        Tt = np.einsum("ij,njk->nik", Ti_H, TH)                                   # human in its task frame
        d = np.einsum("ij,njk->nik", inv_T(Tt[0]), Tt)                            # start-relative (body frame of the start TCP)
        out["task"][s], out["delta"][s] = Tt, d
        R0t = Tt[0, :3, :3]; Dt = np.einsum("nij,kj->nik", Tt[:, :3, :3], R0t)    # human task-frame delta rotation R(t) R(0)^T
        R_task = T_Btask_R[:3, :3]                                                 # robot task axes in the robot base
        Db = np.einsum("ij,njk,lk->nil", R_task, Dt, R_task)                       # the same delta expressed in robot base axes
        if cfg.mode == "A":
            ref = T_R0[s]                                                          # robot start TCP in base axes
            R_ref = ref[:3, :3] @ cfg.R_map.T                                     # reference rotation in HUMAN TCP axes
            # A: the delta rotation is body-frame (start TCP) per the spec; FULL keeps it exactly, other policies use the
            # world-frame (z-up) decomposition of the same human rotation
            if cfg.orientation == "FULL":
                Rr = np.einsum("ij,njk->nik", R_ref, d[:, :3, :3]); clamp = np.zeros(len(d))
            else:
                Rr, clamp = zip(*[orient(D, R_ref, cfg) for D in Db]); Rr = np.stack(Rr); clamp = np.array(clamp)
            if cfg.tframe() == "start": dp = np.einsum("ij,nj->ni", R_ref @ S, d[:, :3, 3])   # start-frame dp, applied via the robot start
            else: dp = (Tt[:, :3, 3] - Tt[0, :3, 3]) @ (R_task @ S).T                         # task-axes dp rotated into the base
            p = ref[:3, 3] + dp
        else:                                                                       # B / C: task-relative
            R_ref = R_task @ R0t                                                   # human start orientation carried into the robot task
            Rr, clamp = zip(*[orient(D, R_ref, cfg) for D in Db]); Rr = np.stack(Rr); clamp = np.array(clamp)
            if cfg.tframe() == "start":
                p = T_Btask_R[:3, 3] + R_task @ Tt[0, :3, 3] + np.einsum("ij,nj->ni", R_ref @ S, d[:, :3, 3])
            else:
                p = (T_Btask_R[:3, 3] + R_task @ Tt[0, :3, 3]) + (Tt[:, :3, 3] - Tt[0, :3, 3]) @ (R_task @ S).T
        out["raw"][s] = make_T(np.einsum("nij,jk->nik", Rr, cfg.R_map), p); out["clamp"][s] = clamp
    return out


# ------------------------------------------------------------------------------------------------ feasibility projection
def project(raw, checker, cfg, valid=None):
    """Mode C line search. raw: dict side -> (N,4,4) desired; checker(T_L, T_R, state) -> (ok, info, new_state).
    Both arms share ONE alpha per step (keeps the L<->R relation); target = interp(prev_valid, desired, alpha).
    -> dict(feasible={side}, alpha (N,), ok (N,), info list)"""
    n = len(raw["left"]); feas = {s: np.empty_like(raw[s]) for s in raw}; alpha = np.zeros(n); okv = np.zeros(n, bool); infos = []
    prev = {s: raw[s][0] for s in raw}; state = None
    for i in range(n):
        if valid is not None and not valid[i]:
            for s in raw: feas[s][i] = prev[s]
            infos.append({"skipped": "invalid source"}); continue
        a = 1.0; first = None
        while True:
            T = {s: raw[s][i] if a >= 1.0 else interp_T(prev[s], raw[s][i], a) for s in raw}
            ok, info, st = checker(T["left"], T["right"], state)
            if first is None: first = info
            if ok or a <= cfg.alpha_min: break
            a *= cfg.alpha_decay
        if ok:
            prev, state = T, st; alpha[i], okv[i] = a, True
        else:                                     # no feasible step toward the target: HOLD the last accepted pose (alpha 0);
            okv[i] = state is not None            # that pose is executable, unless nothing was ever accepted
            a = 0.0
        info = dict(info); info["first_fail"] = None if a >= 1.0 else first.get("reason"); info["held"] = not ok; infos.append(info)
        for s in raw: feas[s][i] = prev[s]
    return dict(feasible=feas, alpha=alpha, ok=okv, info=infos)


# ------------------------------------------------------------------------------------------------ validation metrics
def amplitude(T):
    p = T[:, :3, 3]; d = np.linalg.norm(p - p[0], axis=1)
    return dict(max_disp_m=float(d.max()), path_m=float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), pos_std_m=float(np.linalg.norm(p.std(0))))


def rel_pose_err(TL_a, TR_a, TL_b, TR_b):
    A = np.einsum("nij,njk->nik", inv_T(TL_a), TR_a); B = np.einsum("nij,njk->nik", inv_T(TL_b), TR_b)
    dp = np.linalg.norm(A[:, :3, 3] - B[:, :3, 3], axis=1); dr = rot_angle_deg(np.einsum("nji,njk->nik", A[:, :3, :3], B[:, :3, :3]))
    return dict(pos_p50_mm=float(np.median(dp) * 1e3), pos_p95_mm=float(np.percentile(dp, 95) * 1e3), rot_p50_deg=float(np.median(dr)), rot_p95_deg=float(np.percentile(dr, 95)))


def orientation_change_deg(T):
    return float(rot_angle_deg(np.einsum("nji,jk->nik", T[:, :3, :3], T[0, :3, :3])).max())
