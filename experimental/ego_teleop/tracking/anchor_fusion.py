"""The metric anchor: chest RGB-D absolute XYZ meets wrist visual-inertial 6DoF (multi-sensor spec sections 6, 8, 16).

The two branches do not measure the same thing in the same frame, and pretending they do is the whole trap:

    wrist VI   T_W_H(t)      wrist control frame H in the VI world W. Fast, smooth, 6 DoF, DRIFTS.
    chest RGB-D p_D_palm(t)  operator's palm centroid in the chest camera frame D. Metric, absolute, NO drift, slow,
                             noisy, occludable, and observed in a completely different frame.

So before a single correction can be computed, two unknowns have to be solved for:

    T_W_D    where the chest camera sits in the VI world (V1: the torso is treated as the teleop frame, section 1.1)
    r        the PALM CENTROID in the wrist control frame H -- the lever arm of section 8. Without it, a pure wrist
             ROTATION moves the palm centroid by up to |r| (5-10 cm) while the wrist control point does not move at
             all, the residual reads that as VI drift, and the anchor drags the robot around every time the operator
             turns their hand. This is the single most expensive mistake available in this file.

Both come out of one alternating least-squares fit over a sliding window of paired samples:

    predict    p̂_W(t) = p_VI(t) + R_VI(t) · r                    (where the palm should be, per VI)
    rotation   R_WD   = Kabsch( p_D_palm(t) -> p̂_W(t) )          (scale FIXED at 1: both sides are already metric)
    offsets    t_WD, r  jointly, in closed form, from  [ I  −R_VI(t) ] (t_WD; r) = p_VI(t) − R_WD · p_D_palm(t)
    residual   e(t)   = T_W_D · p_D_palm(t) − p̂_W(t)             (VI drift, in metres, in W)

t_WD and r are solved TOGETHER, not one after the other: a constant part of the lever arm is indistinguishable from a
translation of the camera, so alternating two separate means crawls toward the answer (measured: still 1.3 cm out
after 15 iterations) while the joint 6-unknown solve converges in three.

`r` is only observable once the wrist has actually rotated inside the window, and the rotation must not be locked to
the translation -- if the hand always rolls the same way it moves, "the palm swung around the wrist" and "the camera
is rotated slightly differently" fit the same data. Both conditions are checked: `min_rotation_span_deg`, and the
posterior sigma of r from the normal equations against `max_lever_arm_sigma_m`. When either fails, r KEEPS its last
converged value instead of being refitted to a coincidence. The 60 s protocol separates a translation window from a
`rotation_only` window precisely so that this estimate is well posed; once measured it can be frozen in the config
(`lever_arm_m`).

The correction itself is deliberately the simplest thing that can work (spec section 6: no learned fusion, no new EKF):

    C_target ← (1−α)·C_target + α·e         gated on RGB-D confidence, on the fit, and on an outlier bound
    C(t)     → slewed toward C_target at max_correction_rate_m_s
    p_fused  = p_VI + C                     R_fused = R_VI  (orientation is VI's job, section 7)

THE RATE LIMIT IS A ROBOT VELOCITY. Under relative-SE(3) teleoperation the robot follows increments of the fused
pose, so dC/dt is injected straight into the arm: 0.01 m/s of correction is 1 cm/s of arm motion nobody asked for.
Large corrections are therefore deferred to the clutch (section 13), where the arm is frozen and re-anchors after."""
from __future__ import annotations
import enum
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from .interfaces import TrackingHealth


# ---- rigid fit -------------------------------------------------------------------------------------------------
def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimising |R·src_i + t − dst_i|² (no scale: both sides are metric)."""
    src = np.asarray(src, np.float64).reshape(-1, 3); dst = np.asarray(dst, np.float64).reshape(-1, 3)
    if len(src) < 3 or len(src) != len(dst): raise ValueError("kabsch needs >= 3 matched points")
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cd - R @ cs


def solve_offsets(P_D, P_V, R_V, R_WD) -> tuple[np.ndarray, np.ndarray, float]:
    """Given the rotation, solve the camera translation and the lever arm TOGETHER (closed form) and say how well the
    data actually determined the lever arm.

        minimise  Σ | R_WD·p_D,i + t − R_VI,i·r − p_VI,i |²          unknowns (t, r)

    Normal equations, using R_VIᵀR_VI = I:   [[ n·I, −ΣR_i ], [ −ΣR_iᵀ, n·I ]] (t; r) = ( Σb_i ; −ΣR_iᵀ b_i ),
    with b_i = p_VI,i − R_WD·p_D,i. Sigma is sqrt of the largest diagonal of σ²(AᵀA)⁻¹ over the r block: it is large
    exactly when the wrist did not rotate independently of where it moved, which is when r is not identifiable."""
    P_D = np.asarray(P_D, np.float64); P_V = np.asarray(P_V, np.float64); R_V = np.asarray(R_V, np.float64)
    n = len(P_D)
    b = P_V - P_D @ np.asarray(R_WD, np.float64).T
    Rsum = R_V.sum(0)
    A = np.block([[n * np.eye(3), -Rsum], [-Rsum.T, n * np.eye(3)]])
    rhs = np.concatenate([b.sum(0), -np.einsum("nji,nj->i", R_V, b)])
    A = A + 1e-9 * np.eye(6)                       # a pure no-rotation window makes this exactly singular
    x = np.linalg.solve(A, rhs)
    t, r = x[:3], x[3:]
    res = b + np.einsum("nij,j->ni", R_V, r) - t
    dof = max(3 * n - 6, 1)
    var = float((res ** 2).sum()) / dof
    cov = np.linalg.inv(A) * var
    return t, r, float(np.sqrt(max(np.diag(cov)[3:].max(), 0.0)))


@dataclass
class AnchorAlignmentConfig:
    window_s: float = 20.0                 # sliding window of paired samples the fit runs on
    min_pairs: int = 60
    min_span_m: float = 0.12               # translation spread required before a fit means anything
    min_rotation_span_deg: float = 20.0    # rotation spread required before the lever arm is observable
    refit_every_s: float = 1.0
    max_rms_m: float = 0.03                # a fit worse than this is not accepted as an alignment
    estimate_lever_arm: bool = True
    lever_arm_m: tuple | None = None       # fixed r (calibrated offline); disables the online estimate
    max_lever_arm_m: float = 0.15          # anatomy: the palm centroid is never further than this from the wrist
    max_lever_arm_sigma_m: float = 0.01    # posterior sigma above this = the window cannot see r; keep the old value
    als_iters: int = 5
    freeze_after_align: bool = True        # STOP refitting once the fit is COMPLETE (aligned + lever arm settled). A sliding refit quietly absorbs slow VI
                                           # drift into T_W_D -- the residual goes to zero, C stops correcting and the
                                           # drift survives in the fused pose. The anchor can only see drift that the
                                           # alignment is not free to explain. V1 pays for that with a chest camera
                                           # assumed static in the VI world for the length of a take; a real torso
                                           # shift shows up as a persistent large residual and triggers a re-align.
    realign_residual_m: float = 0.20       # persistent residual above this ...
    realign_after_s: float = 2.0           # ... for this long -> re-open the fit (logged as a re-alignment)

    def __post_init__(self) -> None:
        if self.lever_arm_m is not None: self.lever_arm_m = tuple(float(v) for v in self.lever_arm_m)


@dataclass
class AlignmentState:
    aligned: bool = False
    reason: str = "no pairs yet"
    T_W_D: np.ndarray = field(default_factory=lambda: np.eye(4))
    lever_arm_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lever_arm_observable: bool = False
    rms_m: float = float("nan")
    lever_arm_sigma_m: float = float("nan")
    n_realign: int = 0
    n_pairs: int = 0
    span_m: float = 0.0
    rotation_span_deg: float = 0.0
    n_fits: int = 0


class AnchorAlignment:
    """Sliding-window estimate of T_W_D and of the palm lever arm r, from (RGB-D palm, VI pose) pairs."""

    def __init__(self, cfg: AnchorAlignmentConfig | None = None) -> None:
        self.cfg = cfg or AnchorAlignmentConfig()
        self.state = AlignmentState()
        if self.cfg.lever_arm_m is not None:
            self.state.lever_arm_m = np.asarray(self.cfg.lever_arm_m, np.float64)
            self.state.lever_arm_observable = True
        self._pairs: deque = deque()        # (t_ns, p_D(3), p_VI(3), R_VI(3,3))
        self._last_fit_ns: int | None = None
        self._high_residual_since_ns: int | None = None
        self._refit_requested = False

    # ---- data ----
    def add_pair(self, t_ns: int, p_rgbd_D, T_vi: np.ndarray) -> None:
        T = np.asarray(T_vi, np.float64)
        self._pairs.append((int(t_ns), np.asarray(p_rgbd_D, np.float64).reshape(3), T[:3, 3].copy(), T[:3, :3].copy()))
        cutoff = int(t_ns) - int(self.cfg.window_s * 1e9)
        while self._pairs and self._pairs[0][0] < cutoff: self._pairs.popleft()

    def force_refit(self) -> None:
        """Re-open the alignment: after a VI reset / relocalization, or when the operator re-seats the chest camera."""
        self._refit_requested = True

    def note_residual(self, t_ns: int, e) -> None:
        """Watchdog: a residual that stays absurd is not drift, it is a broken alignment (torso shift, wrong hand)."""
        c = self.cfg
        if float(np.linalg.norm(e)) <= c.realign_residual_m:
            self._high_residual_since_ns = None; return
        if self._high_residual_since_ns is None: self._high_residual_since_ns = int(t_ns)
        elif (t_ns - self._high_residual_since_ns) > c.realign_after_s * 1e9:
            self._refit_requested = True; self._high_residual_since_ns = None
            self.state.n_realign += 1

    @property
    def complete(self) -> bool:
        """Aligned AND the lever arm is settled — either fixed by config, not wanted, or actually observed.

        Freezing on `aligned` alone would lock in whatever was fitted during the first window with enough travel,
        which in the standard protocol is the translation window, where r is invisible. The lever arm would then be
        frozen at zero before the rotation window ever ran."""
        c, s = self.cfg, self.state
        return bool(s.aligned and (c.lever_arm_m is not None or not c.estimate_lever_arm or s.lever_arm_observable))

    def maybe_fit(self, t_ns: int) -> AlignmentState:
        if self.complete and self.cfg.freeze_after_align and not self._refit_requested:
            return self.state
        if self._last_fit_ns is not None and (t_ns - self._last_fit_ns) < self.cfg.refit_every_s * 1e9:
            return self.state
        self._last_fit_ns = int(t_ns); self._refit_requested = False
        return self.fit()

    # ---- the fit ----
    def fit(self) -> AlignmentState:
        c, s = self.cfg, self.state
        s.n_pairs = len(self._pairs)
        if s.n_pairs < c.min_pairs:
            s.aligned = False; s.reason = f"{s.n_pairs}/{c.min_pairs} pairs"; return s
        P_D = np.stack([p[1] for p in self._pairs]); P_V = np.stack([p[2] for p in self._pairs])
        R_V = np.stack([p[3] for p in self._pairs])
        # span is measured on the WRIST travel, not on the palm observation: a hand that only rotates swings the
        # palm centroid several centimetres while the control point stays put, and that must not look like motion.
        s.span_m = float(np.linalg.norm(P_V.max(0) - P_V.min(0)))
        rots = Rotation.from_matrix(R_V)
        s.rotation_span_deg = float(np.degrees((rots * rots[0].inv()).magnitude().max()))
        if s.span_m < c.min_span_m:
            s.aligned = False; s.reason = f"wrist travel {s.span_m*100:.1f} cm < {c.min_span_m*100:.0f} cm"; return s

        r = np.asarray(c.lever_arm_m, np.float64) if c.lever_arm_m is not None else s.lever_arm_m.copy()
        estimate_r = c.estimate_lever_arm and c.lever_arm_m is None and s.rotation_span_deg >= c.min_rotation_span_deg
        R_WD = s.T_W_D[:3, :3]; t_WD = s.T_W_D[:3, 3]
        sigma_r = float("nan")
        for _ in range(max(1, c.als_iters)):
            pred = P_V + np.einsum("nij,j->ni", R_V, r)               # p̂_W = p_VI + R_VI r
            R_WD, _ = kabsch(P_D, pred)
            t_WD, r_new, sigma_r = solve_offsets(P_D, P_V, R_V, R_WD)
            if estimate_r:
                n = float(np.linalg.norm(r_new))
                r = r_new if n <= c.max_lever_arm_m else r_new / n * c.max_lever_arm_m
            else:                                                      # r is fixed: t_WD alone absorbs the offset
                t_WD = (P_V + np.einsum("nij,j->ni", R_V, r) - P_D @ R_WD.T).mean(0)
        pred = P_V + np.einsum("nij,j->ni", R_V, r)
        res = (P_D @ R_WD.T + t_WD) - pred
        rms = float(np.sqrt((res ** 2).sum(1).mean()))
        s.n_fits += 1; s.rms_m = rms; s.lever_arm_sigma_m = sigma_r
        if estimate_r and not (sigma_r <= c.max_lever_arm_sigma_m):
            estimate_r = False                                         # the window cannot see r: keep the old value
            r = np.asarray(c.lever_arm_m, np.float64) if c.lever_arm_m is not None else s.lever_arm_m.copy()
            t_WD = (P_V + np.einsum("nij,j->ni", R_V, r) - P_D @ R_WD.T).mean(0)
            res = (P_D @ R_WD.T + t_WD) - (P_V + np.einsum("nij,j->ni", R_V, r))
            rms = float(np.sqrt((res ** 2).sum(1).mean())); s.rms_m = rms
        if rms > c.max_rms_m:
            s.aligned = False; s.reason = f"fit rms {rms*1000:.0f} mm > {c.max_rms_m*1000:.0f} mm"; return s
        T = np.eye(4); T[:3, :3] = R_WD; T[:3, 3] = t_WD
        s.T_W_D = T
        if estimate_r: s.lever_arm_m = r; s.lever_arm_observable = True
        s.aligned = True; s.reason = "aligned"
        return s

    # ---- use ----
    def anchor_in_world(self, p_rgbd_D) -> np.ndarray:
        """The RGB-D palm observation expressed in the VI world W."""
        p = np.asarray(p_rgbd_D, np.float64).reshape(3)
        return self.state.T_W_D[:3, :3] @ p + self.state.T_W_D[:3, 3]

    def predicted_palm(self, T_vi: np.ndarray) -> np.ndarray:
        """Where the palm centroid should be in W according to VI (lever arm included)."""
        T = np.asarray(T_vi, np.float64)
        return T[:3, 3] + T[:3, :3] @ self.state.lever_arm_m

    def residual(self, p_rgbd_D, T_vi: np.ndarray) -> np.ndarray:
        """e(t) = anchor − prediction, in W. This is VI drift, not hand motion."""
        return self.anchor_in_world(p_rgbd_D) - self.predicted_palm(T_vi)

    def to_dict(self) -> dict:
        s = self.state
        return dict(aligned=bool(s.aligned), reason=s.reason, rms_m=s.rms_m, n_pairs=s.n_pairs, n_fits=s.n_fits,
                    lever_arm_sigma_m=s.lever_arm_sigma_m, n_realign=s.n_realign,
                    span_m=s.span_m, rotation_span_deg=s.rotation_span_deg,
                    lever_arm_m=s.lever_arm_m.round(6).tolist(), lever_arm_observable=bool(s.lever_arm_observable),
                    T_W_D=s.T_W_D.round(6).tolist())


# ---- the correction --------------------------------------------------------------------------------------------
@dataclass
class AnchorFusionConfig:
    alpha: float = 0.02                    # EMA gain per ACCEPTED RGB-D observation (30 Hz -> ~1.7 s time constant)
    max_correction_m: float = 0.25         # hard bound on |C|; beyond this the two sensors disagree about the world
    max_correction_rate_m_s: float = 0.01  # THIS IS ROBOT VELOCITY. Keep it an order below the operator's motion.
    min_rgbd_confidence: float = 0.5
    max_residual_m: float = 0.15           # a bigger residual is an outlier/occlusion, not drift -> rejected
    deadband_m: float = 0.005              # do not chase depth noise
    defer_large_to_clutch: bool = True     # section 13: a big re-anchor mostly waits for the arm to be frozen
    large_correction_m: float = 0.05       # pending correction above this counts as "large"
    deferred_rate_fraction: float = 0.25   # what fraction of the rate limit a LARGE pending correction may still use
                                           # while commanding. 0.0 freezes it until the next clutch -- correct only if
                                           # the operator reliably clutches; otherwise the anchor stops working exactly
                                           # when it is needed most, so the default is a slower creep, not a stall.


@dataclass
class TranslationAnchorFusion:
    """C(t): the slow, bounded, rate-limited translation correction that turns drifting VI into anchored metric XYZ."""
    cfg: AnchorFusionConfig = field(default_factory=AnchorFusionConfig)
    C: np.ndarray = field(default_factory=lambda: np.zeros(3))
    C_target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_observed: int = 0
    n_rejected_conf: int = 0
    n_rejected_outlier: int = 0
    n_deadband: int = 0
    last_residual_m: float = float("nan")
    _t_prev: int | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.C = np.zeros(3); self.C_target = np.zeros(3); self._t_prev = None
        self.n_observed = self.n_rejected_conf = self.n_rejected_outlier = self.n_deadband = 0
        self.last_residual_m = float("nan")

    @property
    def pending_m(self) -> float:
        return float(np.linalg.norm(self.C_target - self.C))

    def observe(self, e: np.ndarray, confidence: float | None) -> bool:
        """One RGB-D residual. -> True when it moved the target."""
        c = self.cfg
        e = np.asarray(e, np.float64).reshape(3)
        n = float(np.linalg.norm(e)); self.last_residual_m = n
        if confidence is not None and confidence < c.min_rgbd_confidence:
            self.n_rejected_conf += 1; return False
        if not np.isfinite(n) or n > c.max_residual_m:
            self.n_rejected_outlier += 1; return False
        self.n_observed += 1
        if n < c.deadband_m: self.n_deadband += 1; return False
        tgt = (1.0 - c.alpha) * self.C_target + c.alpha * e
        m = float(np.linalg.norm(tgt))
        if m > c.max_correction_m: tgt = tgt / m * c.max_correction_m
        self.C_target = tgt
        return True

    def step(self, t_ns: int, *, clutched: bool = False) -> np.ndarray:
        """Slew C toward C_target. While the arm is frozen (clutch) the whole correction is applied at once."""
        c = self.cfg
        dt = 0.0 if self._t_prev is None else max(0.0, (int(t_ns) - self._t_prev) / 1e9)
        self._t_prev = int(t_ns)
        d = self.C_target - self.C
        n = float(np.linalg.norm(d))
        if n < 1e-9: return self.C.copy()
        if clutched:
            self.C = self.C_target.copy(); return self.C.copy()
        rate = c.max_correction_rate_m_s
        if c.defer_large_to_clutch and n > c.large_correction_m: rate *= c.deferred_rate_fraction
        step = rate * dt
        self.C = self.C + d / n * min(n, step)
        return self.C.copy()

    def stats(self) -> dict:
        return dict(c_x=float(self.C[0]), c_y=float(self.C[1]), c_z=float(self.C[2]),
                    correction_m=float(np.linalg.norm(self.C)), correction_pending_m=self.pending_m,
                    residual_m=self.last_residual_m, anchor_observed=self.n_observed,
                    anchor_rejected_conf=self.n_rejected_conf, anchor_rejected_outlier=self.n_rejected_outlier,
                    anchor_deadband=self.n_deadband)


# ---- the unified state (spec section 14) -----------------------------------------------------------------------
class FusionState(str, enum.Enum):
    INITIALIZING = "INITIALIZING"
    TRACKING_OK = "TRACKING_OK"
    DEGRADED = "DEGRADED"
    LOST = "LOST"


@dataclass
class FusionStateConfig:
    rgbd_grace_s: float = 2.0           # VI OK + RGB-D lost: how long before the unanchored pose is called DEGRADED
    max_unanchored_s: float = 30.0      # VI OK, never anchored: drifting VI is honest DEGRADED after this
    vi_lost_translation_only: bool = False   # VI LOST + RGB-D OK: default HOLD. True = follow RGB-D translation at
                                             # degraded speed with the last VI orientation frozen (section 14).


# (FusionState, TrackingHealth) — TrackingHealth is what the existing command policy consumes; INITIALIZING must
# never command, so it maps to LOST (= hold), which is exactly what the coordinator does with it.
def fusion_health(*, vi: TrackingHealth | None, rgbd: TrackingHealth | None, has_pose: bool, anchored: bool,
                  unanchored_s: float, rgbd_age_s: float, cfg: FusionStateConfig) -> tuple[FusionState, TrackingHealth, str]:
    if not has_pose:
        return FusionState.INITIALIZING, TrackingHealth.LOST, "no fused pose yet"
    vi_ok = vi == TrackingHealth.OK
    vi_deg = vi == TrackingHealth.DEGRADED
    rgbd_ok = rgbd in (TrackingHealth.OK, TrackingHealth.DEGRADED)
    if vi is None or vi == TrackingHealth.LOST:
        if rgbd_ok and cfg.vi_lost_translation_only:
            return FusionState.DEGRADED, TrackingHealth.DEGRADED, "vi_lost:rgbd_translation_only"
        return FusionState.LOST, TrackingHealth.LOST, "vi_lost" if rgbd_ok else "both_lost"
    if vi_deg:
        return FusionState.DEGRADED, TrackingHealth.DEGRADED, "vi_degraded" + ("" if rgbd_ok else ":rgbd_lost")
    if vi_ok and rgbd_ok and anchored:
        return FusionState.TRACKING_OK, TrackingHealth.OK, "fused"
    if vi_ok and not anchored:
        if unanchored_s > cfg.max_unanchored_s:
            return FusionState.DEGRADED, TrackingHealth.DEGRADED, "unanchored_vi"
        return FusionState.TRACKING_OK, TrackingHealth.OK, "vi_only:aligning"
    # anchored, but the RGB-D anchor has dropped out: run on VI for a while, then declare it
    if rgbd_age_s > cfg.rgbd_grace_s:
        return FusionState.DEGRADED, TrackingHealth.DEGRADED, f"rgbd_lost:{rgbd_age_s:.1f}s"
    return FusionState.TRACKING_OK, TrackingHealth.OK, "vi_only:rgbd_dropout"
