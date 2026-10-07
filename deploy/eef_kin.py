"""[2026-09-18] FK and continuity IK for the EEF-delta deployment path, on the same reBot URDF/TCP model used to build the labels.
Thin wrapper over handumi.robots.registry so the eval UI needs no other project on the path.
  fk_pose7(q12)                       -> (L pose7, R pose7) in the robot base frame
  solve_chunk(q12_now, pos, quat)     -> (q12 per step, FK error mm per step/arm, reachable flag), warm-started step by step so the
                                         solution stays on one IK branch (the same continuity rule the offline retarget uses).
"""
import dataclasses, math, numpy as np

IK = dict(max_joint_delta=0.35, ori_weight=0.6)      # same clamp/weight as the frozen offline retarget


class Kin:
    def __init__(self, ik_cfg: dict | None = None):
        from handumi.robots.registry import load_embodiment
        cfg = dict(IK); cfg.update(ik_cfg or {})
        self.rt = load_embodiment("rebot_b601")
        w = dataclasses.replace(self.rt.config.ik_weights, max_joint_delta=(None if cfg["max_joint_delta"] is None else float(cfg["max_joint_delta"])), ori_weight=float(cfg["ori_weight"]))
        self.solver = self.rt.solver_cls(config=w)
        self.home_q = self.rt.home_q().astype(np.float32); self._names = list(self.solver.joint_names); self._arm = {}
        for s in ("left", "right"):
            arm = self.rt.config.arms[s]; g = {gj.name for gj in arm.gripper_joints}
            self._arm[s] = [self._names.index(n) for n in arm.joint_names if n not in g]

    def _svc(self, q12):
        out = [0.0] * 14
        out[0:6] = [float(x) for x in q12[:6]]; out[7:13] = [float(x) for x in q12[6:]]
        return out

    def _q(self, svc):
        q = np.array(self.home_q, np.float64)
        for k, j in enumerate(self._arm["left"]): q[j] = svc[k]
        for k, j in enumerate(self._arm["right"]): q[j] = svc[7 + k]
        return q

    def fk_pose7(self, q12):
        l7, r7 = self.solver.fk_pose7(self._q(self._svc(q12)).astype(np.float32))
        return np.asarray(l7, np.float64), np.asarray(r7, np.float64)

    def solve_chunk(self, q12_now, pos, quat):
        """pos (T,2,3), quat (T,2,4 xyzw) desired TCP poses -> q12 (T,12), fkerr mm (T,2), ok (T,2)."""
        T = len(pos); q12 = np.zeros((T, 12)); fkerr = np.zeros((T, 2)); ok = np.ones((T, 2), bool)
        q = self._q(self._svc(q12_now))
        for t in range(T):
            lp = pos[t, 0].astype(np.float32); lq = quat[t, 0].astype(np.float32)
            rp = pos[t, 1].astype(np.float32); rq = quat[t, 1].astype(np.float32)
            try:
                q = np.asarray(self.solver.ik(q.astype(np.float32), left_pose=(lp, lq), right_pose=(rp, rq)), np.float64)
            except Exception:
                ok[t] = False
                q12[t] = q12[t - 1] if t else np.asarray(self._svc(q12_now))[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
                fkerr[t] = 1e6; continue
            svc = [0.0] * 14
            for k, j in enumerate(self._arm["left"]): svc[k] = q[j]
            for k, j in enumerate(self._arm["right"]): svc[7 + k] = q[j]
            q12[t] = np.array(svc)[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
            l7, r7 = self.solver.fk_pose7(q.astype(np.float32))
            fkerr[t, 0] = np.linalg.norm(np.asarray(l7)[:3] - pos[t, 0]) * 1e3
            fkerr[t, 1] = np.linalg.norm(np.asarray(r7)[:3] - pos[t, 1]) * 1e3
        return q12, fkerr, ok
