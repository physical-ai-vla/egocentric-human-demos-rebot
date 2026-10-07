"""[2026-09-29] IK_BACKEND=pink: Pinocchio/Pink QP differential IK, iterated to convergence (vs the PyRoki single LM solve).

Model: the same URDF as rebot_fk_torch (reBot_B601_DM_dualarm.urdf), gripper joints locked at 0.
TCP frames `left_tcp` / `right_tcp` = {side}_gripper_link @ T(+0.07 m along x), exactly rebot_fk_torch's TCP (== dataset TCP),
so targets are given in the DATASET frame directly (no R_DS_TO_FK). Checked against rebot_fk_torch by selftest().
Tasks (fixed before any rollout): FrameTask per arm, position_cost 1.0 (per m), orientation_cost 0.5 (per rad), lm_damping 1e-3;
PostureTask toward the SEED (measured q), cost 1e-3 (weak tie-break only); ConfigurationLimit (URDF limits).
Solver: quadprog, dt 0.05, up to 200 iterations, stop when both arms < 0.5 mm and < 0.5 deg.
q12 order = [L j1..j6, R j1..j6] (eef_kin arm joint order, follower frame).
"""
import os
import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
from pink.limits import ConfigurationLimit

URDF = os.path.expanduser(os.environ.get("REBOT_URDF", "~/holobrain-mac-model/reBot_B601_DM_dualarm.urdf"))
TIP_X = 0.07


class PinkIK:
    def __init__(self, arm_joint_names):
        full = pin.buildModelFromUrdf(URDF)
        keep = set(arm_joint_names)
        lock = [full.getJointId(n) for n in full.names[1:] if n not in keep]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full)) if lock else full
        for side in ("left", "right"):
            fid = self.model.getFrameId(f"{side}_gripper_link")
            fr = self.model.frames[fid]
            self.model.addFrame(pin.Frame(f"{side}_tcp", fr.parentJoint, fid, fr.placement * pin.SE3(np.eye(3), np.array([TIP_X, 0, 0])), pin.FrameType.OP_FRAME))
        self.data = self.model.createData()
        # q index of each of our 12 arm joints in the reduced model (all revolute, nq == nv == 12)
        self.idx = [self.model.joints[self.model.getJointId(n)].idx_q for n in arm_joint_names]
        self.names = list(arm_joint_names)
        assert self.model.nq == 12 and sorted(self.idx) == list(range(12)), (self.model.nq, self.idx)
        self.tasks = {s: FrameTask(f"{s}_tcp", position_cost=1.0, orientation_cost=0.5, lm_damping=1e-3) for s in ("left", "right")}
        self.posture = PostureTask(cost=1e-3)
        self.limits = [ConfigurationLimit(self.model)]

    def _to_pin(self, q12):
        q = np.zeros(12); q[self.idx] = q12; return q

    def _from_pin(self, q):
        return np.asarray(q)[self.idx].copy()

    def fk(self, q12):
        q = self._to_pin(q12); pin.framesForwardKinematics(self.model, self.data, q)
        return [self.data.oMf[self.model.getFrameId(f"{s}_tcp")].homogeneous.copy() for s in ("left", "right")]

    def _yaw_only_target(self, R_cur, R_tgt):
        """[2026-09-30] ori='yaw': keep only the BASE-z part of the orientation error. World-frame error w = log3(R_tgt R_cur^T);
        the task target becomes exp(w_z e_z) R_cur, so rotation about base x/y (roll/pitch) is free (only the posture tie-break
        acts on it). Well defined for a vertical approach axis too (unlike an Euler yaw of the tool heading)."""
        w = pin.log3(R_tgt @ R_cur.T)
        return pin.exp3(np.array([0.0, 0.0, w[2]])) @ R_cur

    @staticmethod
    def _elev(R):
        return np.arcsin(np.clip(R[2, 0], -1.0, 1.0))     # elevation of the tool x (approach) axis above the base xy plane

    def _pitch_only_target(self, R_cur, R_tgt):
        """[2026-09-30 user: roll + yaw excluded] ori='pitch': match only the approach-axis elevation (tool x vs base xy plane).
        Target = the minimal rotation that tilts the CURRENT approach axis to the target elevation, keeping its heading, applied
        to R_cur -> heading (yaw) and rotation about the approach axis (roll) stay free (posture tie-break only)."""
        x_c = R_cur[:, 0]; e_t = self._elev(R_tgt)
        h = np.arctan2(x_c[1], x_c[0]) if np.hypot(x_c[0], x_c[1]) > 1e-6 else np.arctan2(R_tgt[1, 0], R_tgt[0, 0])
        x_d = np.array([np.cos(e_t) * np.cos(h), np.cos(e_t) * np.sin(h), np.sin(e_t)])
        axis = np.cross(x_c, x_d); s = np.linalg.norm(axis); c = float(np.clip(x_c @ x_d, -1.0, 1.0))
        if s < 1e-9:
            return R_cur
        return pin.exp3(axis / s * np.arctan2(s, c)) @ R_cur

    @staticmethod
    def _yaw_hold_target(R_cur, R_tgt):
        """[2026-09-30 user: yaw hard-hold] roll/pitch from the policy, yaw held at the CURRENT pose. Swing-twist of the
        world-frame CHANGE D = R_tgt R_cur^T about base z: D = S * T_z; drop T_z -> R' = S R_cur. D is a small waypoint delta,
        so it stays far from the swing-twist singularity (twist taken about base z relative to the BASE frame is near-singular
        at the usual TCP orientation: sqrt(w^2+z^2) median 0.21, min 0.0 on 649 logged poses)."""
        q = pin.Quaternion(R_tgt @ R_cur.T); q.normalize()
        n = np.hypot(q.w, q.z)
        if n < 1e-9:                                            # 180 deg horizontal flip: no defined twist, keep the target
            return R_tgt
        t = pin.Quaternion(q.w / n, 0.0, 0.0, q.z / n)
        # [2026-09-30 review] swing = q * t^-1 (D = S T). Both orders leave z-twist 0.12 deg p95 on 654 poses, but t^-1 * q kept
        # the policy's tilt worse (swing err p95 2.18 vs 0.70 deg).
        return (q * t.inverse()).matrix() @ R_cur

    def solve(self, q12_seed, targets, iters=200, dt=0.05, posture_q12=None, posture_cost=1e-3, ori="full", hold_q12=None, lock=()):
        """targets: [T_left, T_right] 4x4 in the dataset/base frame. Returns q12, per-arm position error (mm), iterations.
        [2026-09-29] posture_q12 / posture_cost: tie-break toward a given joint pose (default: the seed, cost 1e-3).
        [2026-09-30] ori: 'full' (default, unchanged) | 'yaw' = position + rotation about base z only (roll/pitch excluded)
                     | 'pitch' = position + approach-axis elevation only (roll/yaw excluded)
                     | 'pitch_hold' = [user: "yaw still moves"] roll/yaw HELD at the seed (measured) pose, pitch from the target:
                       the target rotation becomes the seed TCP rotation tilted to the target elevation, then solved as 'full'
                     | 'yaw_hold' = roll/pitch from the target, yaw (base-z twist) held at the current pose, solved as 'full'.
        hold_q12: the pose whose orientation is HELD (default: the seed). pinkdq seeds at q_ref, so it must pass the measured q.
        [2026-09-30 user: "yaw still moves"] lock: joint-name substrings (e.g. ("wrist_yaw",)) HARD-locked at hold_q12 (default the
        seed): their URDF limits are narrowed to that value for this solve, so the other joints do the IK."""
        assert ori in ("full", "yaw", "pitch", "pitch_hold", "yaw_hold"), ori
        q12_seed = np.array(q12_seed, dtype=float, copy=True)
        limits = self.limits
        if lock:
            qh = q12_seed if hold_q12 is None else np.asarray(hold_q12, float)
            li = [i for i, n in enumerate(self.names) if any(k in n for k in lock)]
            q12_seed[li] = qh[li]
            m = pin.Model(self.model)
            for i in li:
                m.lowerPositionLimit[self.idx[i]] = qh[i] - 1e-6; m.upperPositionLimit[self.idx[i]] = qh[i] + 1e-6
            limits = [ConfigurationLimit(m)]
        q = self._to_pin(q12_seed)
        if ori in ("pitch_hold", "yaw_hold"):
            f = self._pitch_only_target if ori == "pitch_hold" else self._yaw_hold_target
            held = []
            for R0, T in zip(self.fk(q12_seed if hold_q12 is None else hold_q12), targets):
                Th = np.array(T, dtype=float, copy=True); Th[:3, :3] = f(R0[:3, :3], T[:3, :3]); held.append(Th)
            targets, ori = held, "full"
        proj = {"yaw": self._yaw_only_target, "pitch": self._pitch_only_target}.get(ori)
        conf = pink.Configuration(self.model, self.data, q)
        for s, T in zip(("left", "right"), targets):
            self.tasks[s].set_target(pin.SE3(T[:3, :3], T[:3, 3]))
        self.posture.cost = posture_cost
        self.posture.set_target(q.copy() if posture_q12 is None else self._to_pin(posture_q12))
        tasks = [self.tasks["left"], self.tasks["right"], self.posture]
        for it in range(iters):
            if proj is not None:   # re-project every iteration: the free rotation components drift and change R_cur
                for s, T in zip(("left", "right"), targets):
                    R_cur = conf.get_transform_frame_to_world(f"{s}_tcp").rotation
                    self.tasks[s].set_target(pin.SE3(proj(R_cur, T[:3, :3]), T[:3, 3]))
            v = pink.solve_ik(conf, tasks, dt, solver="quadprog", limits=limits, safety_break=False)
            conf.integrate_inplace(v, dt)
            err = [T[:3, 3] - conf.get_transform_frame_to_world(f"{s}_tcp").translation for s, T in zip(("left", "right"), targets)]
            if ori == "pitch":
                rer = [abs(self._elev(T[:3, :3]) - self._elev(conf.get_transform_frame_to_world(f"{s}_tcp").rotation)) for s, T in zip(("left", "right"), targets)]
            elif ori == "yaw":
                rer = [abs(pin.log3(T[:3, :3] @ conf.get_transform_frame_to_world(f"{s}_tcp").rotation.T)[2]) for s, T in zip(("left", "right"), targets)]
            else:
                rer = [np.linalg.norm(pin.log3(conf.get_transform_frame_to_world(f"{s}_tcp").rotation.T @ T[:3, :3])) for s, T in zip(("left", "right"), targets)]
            if max(np.linalg.norm(e) for e in err) < 5e-4 and max(rer) < np.radians(0.5):
                break
        q12 = self._from_pin(conf.q)
        return q12, np.array([np.linalg.norm(e) * 1000 for e in err]), it + 1


    def diff_step(self, q12, targets, ori_weight=0.5, damping=1e-2, lock=()):
        """[2026-10-01 user: IK from the CURRENT pose, not solved to the target] ONE damped-least-squares Jacobian step at the measured
        q: dq = J^T (J J^T + lambda^2 I)^-1 e, e = [p_tgt - p_cur ; ori_weight * log3(R_tgt R_cur^T)] per arm (base frame,
        LOCAL_WORLD_ALIGNED Jacobian of {side}_tcp). No iteration to convergence: the arm moves along the local linearisation of
        the requested displacement. lock: joint-name substrings whose columns are zeroed (held at q). Joint limits clip the result.
        Returns q12, per-arm position error of the RESULT vs the target (mm) by FK, and the requested displacement (mm)."""
        q = self._to_pin(q12); m, d = self.model, self.data
        pin.computeJointJacobians(m, d, q); pin.framesForwardKinematics(m, d, q)
        Js, es, req = [], [], []
        for s, T in zip(("left", "right"), targets):
            fid = m.getFrameId(f"{s}_tcp"); M = d.oMf[fid]
            J = pin.getFrameJacobian(m, d, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)        # (6, nq): [v; w] in base axes
            dp = T[:3, 3] - M.translation; dw = pin.log3(T[:3, :3] @ M.rotation.T)
            Js.append(np.vstack([J[:3], ori_weight * J[3:]])); es.append(np.concatenate([dp, ori_weight * dw])); req.append(np.linalg.norm(dp) * 1000)
        J = np.vstack(Js); e = np.concatenate(es)
        if lock:
            for i, n in enumerate(self.names):
                if any(k in n for k in lock): J[:, self.idx[i]] = 0.0
        # limit-aware: a joint the step would push past its URDF limit is set AT the limit, its column removed and its
        # contribution taken out of e, and the rest re-solved -- all on the SAME linearisation (still one Jacobian, no iteration
        # toward the target). Without this, clipping after the fact broke the solution near J2/J3 = 0 (rest pose).
        lo, hi = m.lowerPositionLimit, m.upperPositionLimit; fixed = np.zeros(m.nq, bool); dq_fixed = np.zeros(m.nq)
        for _ in range(m.nq):
            Jf = J.copy(); Jf[:, fixed] = 0.0
            dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + damping ** 2 * np.eye(J.shape[0]), e - J @ dq_fixed) + dq_fixed
            over = ~fixed & ((q + dq > hi) | (q + dq < lo))
            if not over.any(): break
            fixed |= over; dq_fixed[over] = np.clip(q + dq, lo, hi)[over] - q[over]
        qn = np.clip(q + dq, lo, hi)
        q12n = self._from_pin(qn); Tn = self.fk(q12n)
        perr = np.array([np.linalg.norm(Tn[i][:3, 3] - targets[i][:3, 3]) * 1000 for i in range(2)])
        return q12n, perr, np.array(req)


def selftest():
    import sys, torch
    sys.path.insert(0, os.path.expanduser("~/c8/c8old")); import rebot_fk_torch
    import eef_kin
    k = eef_kin.Kin(); names = [k._names[j] for j in k._arm["left"]] + [k._names[j] for j in k._arm["right"]]
    ik = PinkIK(names); fkt = rebot_fk_torch.ReBotFKTorch(dtype=torch.float64)
    rng = np.random.default_rng(0); ep, er = [], []
    for _ in range(50):
        q = rng.uniform(-0.6, 0.6, 12); q[[1, 2, 7, 8]] = -np.abs(q[[1, 2, 7, 8]])
        a = ik.fk(q); b = fkt.tcp(torch.tensor(q)[None]).numpy()[0]
        for r in (0, 1):
            ep.append(np.linalg.norm(a[r][:3, 3] - b[r][:3, 3]) * 1000); er.append(np.degrees(np.linalg.norm(pin.log3(a[r][:3, :3].T @ b[r][:3, :3]))))
    print(f"Pink TCP vs rebot_fk_torch TCP: pos max {max(ep):.4f} mm, rot max {max(er):.4f} deg")
    return ik, names


if __name__ == "__main__":
    selftest()
