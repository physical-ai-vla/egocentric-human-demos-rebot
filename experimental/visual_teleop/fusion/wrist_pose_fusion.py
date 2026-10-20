"""Real-time human hand pose for one side (2026-10-14, spec 7, 11, 13-15, 26). No offline smoother in this loop.

    orientation  R_W_H = R_W_G(psi) . R_G_I(AHRS, IMU rate) . R_I_H(mount)
                 roll/pitch: gravity (AHRS).  yaw: gyro propagation; psi is set ONCE at init from the known init pose
                 and then only nudged by vision/tag observations that agree with the prediction (weak anchor, spec 14).
    position     fixed RGB-D wrist keypoint in W, One-Euro filtered; the virtual TCP = wrist + R_W_H . r_H (lever arm), or
                 the RGB-D thumb/index midpoint when configured. The IMU is NEVER integrated to position.

Per-tick status (spec 26):
    TRACKING  fresh position + healthy IMU orientation
    WARN      short position dropout (held, not extrapolated) | IMU invalid but vision orientation valid |
              IMU-only for longer than imu_only_max_s | low position confidence
    HOLD      position lost beyond hold_s | no orientation source | not initialised
"""
from __future__ import annotations
import enum
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from ego_teleop.transforms.se3 import make_T
from .filters import OneEuro


class HandStatus(str, enum.Enum):
    TRACKING = "TRACKING"; WARN = "WARN"; HOLD = "HOLD"


def Rz(psi: float) -> np.ndarray: return Rotation.from_rotvec([0, 0, psi]).as_matrix()


def yaw_of(R: np.ndarray) -> float:
    """Heading of a rotation about W +Z: the angle of its x axis projected on the XY plane (robust away from x || Z)."""
    x = R[:, 0]
    if np.hypot(x[0], x[1]) < 0.2: x = -R[:, 2] if abs(R[2, 2]) < 0.98 else R[:, 1]   # x nearly vertical -> use another axis
    return float(np.arctan2(x[1], x[0]))


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(Ra.T @ Rb).magnitude()))


@dataclass
class OrientationFusion:
    R_I_H: np.ndarray
    yaw_gain: float = 0.02
    reject_deg: float = 15.0
    imu_only_max_s: float | None = None     # None = no IMU-only warning (no vision yaw source configured)
    psi: float | None = None                 # R_W_G yaw; None until align()
    n_vis_accepted: int = 0
    n_vis_rejected: int = 0
    last_vis_ns: int | None = None
    init_tilt_deg: float | None = None

    def align(self, R_G_I: np.ndarray, R_W_H_known: np.ndarray, t_ns: int) -> float:
        """Init: the operator holds the hand in the known init orientation. Solves psi (yaw only -- G and W share +Z) and
        returns the residual TILT (deg) that a yaw cannot explain: large => wrong mount R_I_H or wrong init pose."""
        R_pred_noyaw = R_G_I @ self.R_I_H
        self.psi = yaw_of(R_W_H_known) - yaw_of(R_pred_noyaw)
        self.init_tilt_deg = angle_deg(Rz(self.psi) @ R_pred_noyaw, R_W_H_known); self.last_vis_ns = int(t_ns)
        return self.init_tilt_deg

    def align_from_motion(self, R_G_I_seq, p_W_seq, t_ns: int, *, acc_lin_I=None, min_travel_m: float = 0.06, max_misalign_deg: float = 25.0,
                          min_axis_cos: float = 0.8) -> dict:
        """Init yaw WITHOUT a jig (2026-10-14): the operator pushes the hand ~10 cm forward ALONG THE FINGERS (H z) and back.
        The RGB-D displacement (W, metric) and the IMU finger direction (G) are the same physical direction, so their
        horizontal headings differ exactly by psi. Gates: travel >= min_travel_m, finger axis not vertical, and after
        solving, the 3-D angle between the push and the predicted finger axis <= max_misalign_deg (else the push was not
        along the fingers, or R_I_H is wrong). Returns a report; psi is set only when ok."""
        # A heading match alone cannot see a push that was NOT along the fingers (psi just absorbs it). The IMU can: the
        # gravity-free linear acceleration of the push, in the hand frame, must lie along H z. acc_lin_I = accel - R_G_I^T g.
        axis_cos = None; imu_rep = {}
        if acc_lin_I is not None:
            A = np.asarray(acc_lin_I, np.float64); A = A[np.all(np.isfinite(A), axis=1)]
            if len(A) < 20: return dict(ok=False, reason="too few IMU samples for the push-axis check")
            _, sv, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False); u = Vt[0]; u_H = self.R_I_H.T @ u
            axis_cos = float(abs(u_H[2])); ratio = float(sv[0] / max(sv[1], 1e-9)); peak = float(np.max(np.linalg.norm(A - A.mean(0), axis=1)))
            imu_rep = dict(axis_cos=axis_cos, push_axis_H=np.round(u_H, 2).tolist(), axis_dominance=round(ratio, 2), peak_acc_m_s2=round(peak, 2))
            if ratio < 1.3 or peak < 1.0:      # 10-15: a gentle push is buried in gravity leakage -> the axis is not judgeable (1.5 -> 1.3: user pushes failed at 1.25)
                return dict(ok=False, reason=f"IMU: push too weak / unclear (peak {peak:.1f} m/s2, dominance {ratio:.1f}) -- push faster", **imu_rep)
            if axis_cos < min_axis_cos:
                return dict(ok=False, reason=f"IMU: push not along the jaws/fingers (|cos| {axis_cos:.2f} < {min_axis_cos}; axis in H {np.round(u_H, 2).tolist()}, "
                            "H z = along the jaws)", **imu_rep)
        P = np.asarray(p_W_seq, np.float64); Rs = np.asarray(R_G_I_seq, np.float64)
        ok_p = np.all(np.isfinite(P), axis=1)
        if ok_p.sum() < 5: return dict(ok=False, reason="too few RGB-D positions")
        P, Rs = P[ok_p], Rs[ok_p]
        k = int(np.argmax(np.linalg.norm(P - P[0], axis=1))); d = P[k] - P[0]; travel = float(np.linalg.norm(d))
        if travel < min_travel_m: return dict(ok=False, reason=f"travel {travel*1e3:.0f} mm < {min_travel_m*1e3:.0f} mm", travel_m=travel)
        z_G = (Rs[0] @ self.R_I_H)[:, 2]
        if np.hypot(z_G[0], z_G[1]) < 0.5: return dict(ok=False, reason="finger axis too vertical to define a heading", travel_m=travel)
        if np.hypot(d[0], d[1]) < 0.5 * travel: return dict(ok=False, reason="push mostly vertical", travel_m=travel)
        psi = float(np.arctan2(d[1], d[0]) - np.arctan2(z_G[1], z_G[0]))
        z_W = Rz(psi) @ z_G; mis = float(np.degrees(np.arccos(np.clip(z_W @ d / travel, -1, 1))))
        # 10-15: psi aligns the HORIZONTAL headings exactly, so `mis` only measures the pitch mismatch between the push and the jaw
        # axis (UMI slightly nosed down, keypoint depth noise). When the IMU already proved the push ran along the jaws, allow 45 deg.
        if axis_cos is not None and axis_cos >= 0.9: max_misalign_deg = max(max_misalign_deg, 45.0)
        rep = dict(ok=mis <= max_misalign_deg, psi=psi, travel_m=travel, misalign_deg=mis, **imu_rep)
        if not rep["ok"]: rep["reason"] = f"push vs finger axis {mis:.0f} deg > {max_misalign_deg:.0f} (not along the fingers, or R_I_H wrong)"; return rep
        self.psi = psi; self.last_vis_ns = int(t_ns); self.init_tilt_deg = None; return rep

    def predict(self, R_G_I: np.ndarray) -> np.ndarray:
        if self.psi is None: raise RuntimeError("OrientationFusion.align() first")
        return Rz(self.psi) @ R_G_I @ self.R_I_H

    def observe_vision(self, R_W_H_obs: np.ndarray, R_G_I: np.ndarray, t_ns: int) -> bool:
        """Weak absolute anchor (AprilTag / vision). Rejected if it disagrees with the IMU prediction; otherwise corrects
        a small fraction of the YAW error only (roll/pitch belong to gravity)."""
        if self.psi is None or not np.all(np.isfinite(R_W_H_obs)): return False
        R_pred = self.predict(R_G_I)
        if angle_deg(R_pred, R_W_H_obs) > self.reject_deg: self.n_vis_rejected += 1; return False
        err = (yaw_of(R_W_H_obs) - yaw_of(R_pred) + np.pi) % (2 * np.pi) - np.pi
        self.psi += self.yaw_gain * err; self.n_vis_accepted += 1; self.last_vis_ns = int(t_ns); return True

    def imu_only_s(self, t_ns: int) -> float: return 0.0 if self.last_vis_ns is None else (t_ns - self.last_vis_ns) / 1e9


@dataclass
class HandEstimate:
    side: str
    t_ns: int
    status: HandStatus
    reasons: list[str]
    T_W_H: np.ndarray | None                 # human virtual TCP (None => HOLD)
    wrist_W: np.ndarray | None
    R_W_H: np.ndarray | None
    position_conf: float
    orientation_source: str                  # imu | vision | none
    position_age_ms: float

    @property
    def commandable(self) -> bool: return self.status != HandStatus.HOLD and self.T_W_H is not None


@dataclass
class HandFusion:
    side: str
    orient: OrientationFusion
    tcp_offset_H: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position_source: str = "wrist_plus_offset"
    hold_s: float = 0.3
    min_position_conf: float = 0.5
    warn_position_conf: float = 0.7
    stale_warn_ms: float = 80.0             # position older than this (but < hold_s) => WARN, held not extrapolated
    one_euro: OneEuro = field(default_factory=OneEuro)
    _p: np.ndarray | None = None
    _p_t: int | None = None
    _p_conf: float = 0.0
    _tips_mid: np.ndarray | None = None

    @classmethod
    def from_cfg(cls, side: str, c, R_I_H: np.ndarray) -> "HandFusion":
        f = c.fusion; o = OrientationFusion(R_I_H, f["yaw_correction_gain"], f["tag_reject_deg"], f["orientation"]["imu_only_max_s"])
        P = f["position"]
        return cls(side, o, tcp_offset_H=np.asarray(P["tcp_offset_H_m"][side], np.float64), position_source=P["source"], hold_s=P["hold_s"],
                   min_position_conf=c.tracking["min_position_conf"], warn_position_conf=c.tracking["warn_position_conf"],
                   stale_warn_ms=P["stale_warn_ms"], one_euro=OneEuro(**P["one_euro"]))

    def observe_position(self, wrist_W, conf: float, t_ns: int, tips_mid_W=None) -> bool:
        """RGB-D measurement (already in W). Low-confidence / non-finite = dropout (not used, not filtered)."""
        p = None if wrist_W is None else np.asarray(wrist_W, np.float64)
        if p is None or not np.all(np.isfinite(p)) or conf < self.min_position_conf: return False
        self._p = self.one_euro.update(p, int(t_ns)); self._p_t = int(t_ns); self._p_conf = float(conf)
        self._tips_mid = None if tips_mid_W is None or not np.all(np.isfinite(tips_mid_W)) else np.asarray(tips_mid_W, np.float64)
        return True

    # ---- tag orientation as a weak yaw anchor (10-15) -------------------------------------------------------------
    R_tag_H: np.ndarray | None = None
    _tag_learn: list = field(default_factory=list)
    tag_learn_n: int = 30

    def observe_tag_orientation(self, R_W_tag: np.ndarray, imu, t_ns: int) -> str:
        """After INIT (psi set): learn R_tag_H = R_W_tag^T R_W_H(IMU) from the first tag_learn_n samples (chordal mean), freeze it;
        afterwards every tag sample is a weak yaw observation through the orientation gate. -> 'learning' | 'used' | 'rejected' | 'skip'."""
        if self.orient.psi is None or imu is None or not imu.healthy: return "skip"
        R_pred = self.orient.predict(imu.R_G_I)
        if self.R_tag_H is None:
            self._tag_learn.append(np.asarray(R_W_tag).T @ R_pred)
            if len(self._tag_learn) >= self.tag_learn_n:
                from ..imu.ahrs import normalize_rotation
                self.R_tag_H = normalize_rotation(np.mean(self._tag_learn, axis=0)); self._tag_learn = []
            return "learning"
        return "used" if self.orient.observe_vision(np.asarray(R_W_tag) @ self.R_tag_H, imu.R_G_I, t_ns) else "rejected"

    def reset_tag_learning(self) -> None: self.R_tag_H = None; self._tag_learn = []

    def estimate(self, t_ns: int, imu=None, R_W_H_vision=None) -> HandEstimate:
        """imu: ImuOrientation (or None); R_W_H_vision: optional full vision orientation used when the IMU is invalid."""
        reasons: list[str] = []; status = HandStatus.TRACKING
        # orientation
        R = None; src = "none"
        if imu is not None and self.orient.psi is not None:
            if imu.healthy: R = self.orient.predict(imu.R_G_I); src = "imu"
            else: reasons.append(imu.reason or "imu_unhealthy")
        elif self.orient.psi is None: reasons.append("not_initialised")
        else: reasons.append("imu_missing")
        if R is None and R_W_H_vision is not None and np.all(np.isfinite(R_W_H_vision)):
            R = np.asarray(R_W_H_vision, np.float64); src = "vision"; status = HandStatus.WARN; reasons.append("imu_invalid_vision_orientation")
        if R is None: return HandEstimate(self.side, t_ns, HandStatus.HOLD, reasons + ["no_orientation"], None, None, None, 0.0, "none", np.inf)
        if src == "imu" and self.orient.imu_only_max_s is not None and self.orient.imu_only_s(t_ns) > self.orient.imu_only_max_s: status = HandStatus.WARN; reasons.append("imu_only_yaw_drift")
        # position
        age_ms = np.inf if self._p_t is None else (t_ns - self._p_t) / 1e6
        if self._p is None or age_ms > self.hold_s * 1e3:
            return HandEstimate(self.side, t_ns, HandStatus.HOLD, reasons + [f"position_lost:{age_ms:.0f}ms"], None, None, R, 0.0, src, age_ms)
        if age_ms > self.stale_warn_ms: status = HandStatus.WARN; reasons.append(f"position_held:{age_ms:.0f}ms")
        if self._p_conf < self.warn_position_conf: status = HandStatus.WARN; reasons.append(f"position_conf:{self._p_conf:.2f}")
        if self.position_source == "fingertip_midpoint_rgbd" and self._tips_mid is not None: p_tcp = self._tips_mid
        else: p_tcp = self._p + R @ self.tcp_offset_H
        return HandEstimate(self.side, t_ns, status, reasons, make_T(R, p_tcp), self._p.copy(), R, self._p_conf, src, age_ms)
