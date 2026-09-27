#!/usr/bin/env python3
"""[2026-09-23] Track C pipeline core: legacy HandUMI camera pose -> v2 TCP -> robot-start anchored retarget ->
calibrated continuity IK -> per-row status. Contract: TRACK_C_PSEUDO_JOINT_CONTRACT.md.

Frames (C1 gate + OPEN-1, never re-derived here):
    T_hu_k     = T_cam_k @ X                          X = handumi_camera_tcp_v2      (npz tcp_pose is NOT read)
    rel_ds_k   = F^-1 @ inv(T_hu_0) @ T_hu_k @ F      F = Rx(180 deg)  (OPEN-1 evidence)
    T_tgt_k    = T_fk_R0 @ C^-1 @ rel_ds_k @ C        C = Ry(+90 deg)  (C1 T1: R_fk = R_ds @ C)

IK never holds silently: a failed row keeps its target, gets a status and reason, and the last good q is kept
in a separate field. Nothing is interpolated.
"""
import dataclasses, json, pathlib, sys
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

HOME = pathlib.Path.home()
sys.path[:0] = [str(HOME / "holobrain-mac-model"), str(HOME / "holobrain-mac-model/umi_pkg")]
import eef_kin

CAM_TCP_V2 = HOME / "ego_collector/configs/calibration/camera_tcp/handumi_camera_tcp_v2.yaml"
TRAIN_START_DEG14 = [-4.5, -2.0, -8.3, 30.4, -6.9, -5.2, 0.0, 4.2, -14.8, -1.1, 6.0, 3.2, -3.2, 0.0]
Q_R0 = np.radians([TRAIN_START_DEG14[i] for i in range(14) if i not in (6, 13)])
ARM_J = list(range(0, 6)) + list(range(8, 14))           # solver full-q indices of the 12 arm joints
# In-distribution anchor: per-arm median of R150 MEASURED joints over moving frames (|dq| > 0.3 deg/frame at
# 30 fps), 2026-09-23. TRAIN_START is the episode-start home pose; joint2/3 sit 2-8 deg from their 0 upper
# limit there, so human segments (which start mid-task) anchored at it get pinned on the limit.
Q_ACTIVE_DEG12 = [-35.5, -82.7, -55.0, 38.8, -6.9, -12.3, 37.1, -86.6, -55.3, 44.1, 1.9, -11.9]
ANCHORS = {"train_start": Q_R0, "r150_active_median": np.radians(Q_ACTIVE_DEG12)}
LIMIT_TOL_DEG = 0.1                                      # a solver sitting ON its bound (float32) is not a violation


def _h(R=None, t=None):
    T = np.eye(4)
    if R is not None: T[:3, :3] = R
    if t is not None: T[:3, 3] = t
    return T


C = _h(Rot.from_rotvec([0, np.pi / 2, 0]).as_matrix())
F = _h(Rot.from_rotvec([np.pi, 0, 0]).as_matrix())
_v2 = yaml.safe_load(CAM_TCP_V2.read_text())
X = _h(np.asarray(_v2["rotation"]["R_camera_tcp"], float), _v2["translation"]["t_camera_tcp_m"])



def provenance():
    return dict(pseudo_q_scope="local_robot_start_anchored", absolute_workspace_pose="synthetic",
                cross_arm_absolute_geometry="unavailable",
                role="observation-side pseudo proprioception, NOT robot action ground truth",
                C="Ry(+90deg), R_fk = R_ds @ C (c1_frame_regression T1)",
                F="Rx(180deg): R150 FK + wrist-video evidence (OPEN1_EVIDENCE.md); physical inspection PENDING; do not flip without it",
                anchor_meaning="r150_active_median is a representative reBot manipulation configuration for generating pseudo q, NOT a reconstruction of the human workspace",
                pseudo_q_meaning="plausible continuous joint trajectory if this local human TCP motion started from a typical R150 manipulation configuration",
                X_file=str(CAM_TCP_V2), X=X.tolist(), anchors_deg={k: np.degrees(v).round(2).tolist() for k, v in ANCHORS.items()}, limit_tol_deg=LIMIT_TOL_DEG, tcp_pose_used=False)


# ------------------------------------------------------------------ C2 legacy camera pose -> v2 TCP
def load_hu(npz_path):
    """-> dict(frame_idx, T_hu (n,4,4), seg_id, lost) ; T_hu is v2 TCP in that side's SLAM frame."""
    z = np.load(npz_path)
    valid = np.linalg.norm(z["cam_quat"], axis=1) > 0.5          # lost frames carry a zero quaternion
    Tc = np.tile(np.eye(4), (len(z["cam_pos"]), 1, 1))
    Tc[valid, :3, :3] = Rot.from_quat(z["cam_quat"][valid]).as_matrix(); Tc[valid, :3, 3] = z["cam_pos"][valid]
    seg = np.where(valid, z["seg_id"], -1)
    return dict(frame_idx=z["frame_idx"], T_hu=Tc @ X, seg_id=seg, lost=z["lost"], pose_valid=valid,
                n_seg_but_no_pose=int(((z["seg_id"] >= 0) & ~valid).sum()))


def segments(seg_id, min_len=2):
    """contiguous runs of one seg_id >= 0 -> list of (start, end_exclusive)."""
    out, s = [], None
    for i in range(len(seg_id) + 1):
        cur = seg_id[i] if i < len(seg_id) else -2
        if s is not None and (cur != seg_id[s]):
            if i - s >= min_len: out.append((s, i))
            s = None
        if s is None and cur >= 0: s = i
    return out


# ------------------------------------------------------------------ C3 robot-start anchored retarget
class Retarget:
    def __init__(self, kin, q_anchor=None):
        q_anchor = ANCHORS["r150_active_median"] if q_anchor is None else q_anchor
        self.q_anchor = np.asarray(q_anchor, float)
        self.T_R0 = [_h(Rot.from_quat(p[3:]).as_matrix(), p[:3]) for p in kin.fk_pose7(self.q_anchor)]

    def targets(self, T_hu_seg, arm):
        rel = np.linalg.inv(T_hu_seg[0])[None] @ T_hu_seg
        rel_ds = np.linalg.inv(F)[None] @ rel @ F[None]
        return self.T_R0[arm][None] @ np.linalg.inv(C)[None] @ rel_ds @ C[None]

    def hold(self, n, arm):
        return np.repeat(self.T_R0[arm][None], n, axis=0)


# ------------------------------------------------------------------ C4 calibrated continuity IK (FROZEN 2026-09-23)
# Decisions (user, 2026-09-23): anchor = r150_active_median, ori_weight = 20 (OPEN-4 sweep, reports/
# open4_ori_weight_r150_active_median.json). dq <= 0.6 is NOT a selection rule for observation-side pseudo q:
# 0.6 < |dq| <= 1.0 is DQ_WARN (soft, row still counts as success), > 1.0 is DQ_FAIL (hard).
HARD = ("IK_FAIL", "LIMIT", "DQ_FAIL", "BRANCH_SWITCH", "NOT_CONVERGED")
SOFT = ("DQ_WARN",)
SUCCESS = ("OK",) + SOFT


@dataclasses.dataclass
class IKCfg:
    ori_weight: float = 20.0
    anchor: str = "r150_active_median"
    max_joint_delta: float = 0.35      # per solver call (solver's own clamp)
    iters: int = 10
    conv_dq: float = 1e-4              # stop iterating when max|dq| of a call falls below this
    dq_warn: float = 0.6               # rad per stored step (15 Hz): soft
    dq_fail: float = 1.0               # rad per stored step: hard
    branch_dq: float = 0.3             # BRANCH_SWITCH: max|dq| above this while the target barely moved
    branch_dp_mm: float = 5.0
    branch_dth_deg: float = 5.0
    pos_ok_mm: float = 15.0            # NOT_CONVERGED per row (matches the episode p95 gate)
    rot_ok_deg: float = 5.0


class ContinuityIK:
    def __init__(self, cfg: IKCfg):
        self.cfg = cfg
        self.kin = eef_kin.Kin(dict(ori_weight=cfg.ori_weight, max_joint_delta=cfg.max_joint_delta))
        lim = self.kin.solver.robot.joints
        self.lo = np.asarray(lim.lower_limits, float)[ARM_J]; self.hi = np.asarray(lim.upper_limits, float)[ARM_J]
        self.q_anchor = ANCHORS[cfg.anchor]

    def _q12(self, qfull):
        svc = [0.0] * 14
        for k, j in enumerate(self.kin._arm["left"]): svc[k] = qfull[j]
        for k, j in enumerate(self.kin._arm["right"]): svc[7 + k] = qfull[j]
        return np.array(svc)[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]

    def solve(self, TL, TR, q12_start=None):
        """TL, TR (n,4,4) targets in the FK frame -> per-row arrays. Each arm seeds from ITS last successful q;
        a failed row keeps its target, gets a status + reason, and never becomes a seed."""
        q12_start = self.q_anchor if q12_start is None else q12_start
        n = len(TL); c = self.cfg
        q = np.zeros((n, 12)); q_last_good = np.zeros((n, 12)); iters = np.zeros(n, int)
        pos_err = np.full((n, 2), np.nan); rot_err = np.full((n, 2), np.nan); dq = np.full((n, 2), np.nan)
        status = np.zeros((n, 2), "<U14"); status[:] = "OK"; reason = [""] * n
        good = np.array(q12_start, float); seed = self.kin._q(self.kin._svc(good))
        tgt = (TL, TR)
        for t in range(n):
            lp, lq = TL[t, :3, 3], Rot.from_matrix(TL[t, :3, :3]).as_quat()
            rp, rq = TR[t, :3, 3], Rot.from_matrix(TR[t, :3, :3]).as_quat()
            qf = seed.copy(); q_last_good[t] = good
            try:
                for it in range(c.iters):
                    qn = np.asarray(self.kin.solver.ik(qf.astype(np.float32), left_pose=(lp.astype(np.float32), lq.astype(np.float32)),
                                                       right_pose=(rp.astype(np.float32), rq.astype(np.float32))), np.float64)
                    step = np.abs(qn - qf).max(); qf = qn; iters[t] = it + 1
                    if step < c.conv_dq: break
            except Exception as e:                           # no hold: the row is a failure with a reason
                status[t] = "IK_FAIL"; reason[t] = f"solver exception: {type(e).__name__}: {e}"[:200]
                q[t] = np.nan; continue
            q12 = self._q12(qf); q[t] = q12
            l7, r7 = self.kin.solver.fk_pose7(qf.astype(np.float32))
            for a, (p7, Tt) in enumerate(((l7, TL[t]), (r7, TR[t]))):
                p7 = np.asarray(p7, float)
                pos_err[t, a] = np.linalg.norm(p7[:3] - Tt[:3, 3]) * 1e3
                Rf = Rot.from_quat(p7[3:]).as_matrix()
                rot_err[t, a] = np.degrees(np.linalg.norm(Rot.from_matrix(Rf.T @ Tt[:3, :3]).as_rotvec()))
            why = []
            tol = np.radians(LIMIT_TOL_DEG)
            for a, sl in ((0, slice(0, 6)), (1, slice(6, 12))):
                d = float(np.abs(q12[sl] - good[sl]).max()); dq[t, a] = d
                Tp = tgt[a][t - 1] if t else tgt[a][t]
                dp_t = np.linalg.norm(tgt[a][t, :3, 3] - Tp[:3, 3]) * 1e3
                dth_t = np.degrees(np.linalg.norm(Rot.from_matrix(Tp[:3, :3].T @ tgt[a][t, :3, :3]).as_rotvec()))
                if (q12[sl] < self.lo[sl] - tol).any() or (q12[sl] > self.hi[sl] + tol).any():
                    status[t, a] = "LIMIT"; why.append(f"arm{a} joint limit")
                elif d > c.dq_fail:
                    status[t, a] = "DQ_FAIL"; why.append(f"arm{a} |dq|={d:.3f}>{c.dq_fail}")
                elif d > c.branch_dq and dp_t < c.branch_dp_mm and dth_t < c.branch_dth_deg:
                    status[t, a] = "BRANCH_SWITCH"; why.append(f"arm{a} |dq|={d:.3f} for target dp {dp_t:.1f}mm dth {dth_t:.1f}deg")
                elif pos_err[t, a] > c.pos_ok_mm or rot_err[t, a] > c.rot_ok_deg:
                    status[t, a] = "NOT_CONVERGED"; why.append(f"arm{a} {pos_err[t,a]:.1f}mm/{rot_err[t,a]:.1f}deg")
                elif d > c.dq_warn:
                    status[t, a] = "DQ_WARN"; why.append(f"arm{a} |dq|={d:.3f} (warn)")
                if status[t, a] in SUCCESS:
                    good[sl] = q12[sl]
            reason[t] = "; ".join(why)
            seed = self.kin._q(self.kin._svc(good))          # next row seeds from each arm's last SUCCESSFUL q
        return dict(q=q, q_last_good=q_last_good, pos_err=pos_err, rot_err=rot_err, dq=dq, status=status,
                    reason=np.array(reason, dtype=object), iters=iters,
                    at_limit=np.stack([((np.abs(q[:, s_] - self.lo[s_]) < np.radians(LIMIT_TOL_DEG)) |
                                        (np.abs(q[:, s_] - self.hi[s_]) < np.radians(LIMIT_TOL_DEG))).any(1)
                                       for s_ in (slice(0, 6), slice(6, 12))], 1))
