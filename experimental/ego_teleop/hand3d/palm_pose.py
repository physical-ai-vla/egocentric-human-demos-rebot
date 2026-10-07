"""RGB-D palm pose — a TEMPORARY arm pose source for the POC, while the wrist VIO hardware is not built yet.

The production arm architecture is unchanged and is NOT what this module implements:

    wrist fisheye + IMU -> OpenVINS -> WristPoseProvider -> relative SE(3) -> reBot        (canonical, untouched)

This module adds a second, throw-away *pose source* with the same output contract, so the arm mapping / safety / IK
pipeline can be exercised before the IMU arrives:

    FIXED RGB-D camera -> MediaPipe 2D -> aligned depth -> metric 3D hand -> palm pose -> (same relative SE(3) stack)

Two properties of this source are load-bearing and are asserted, not assumed:

  * **The camera must not move.** `T_world_camera` here is `T_camera_palm`: the camera frame IS the world frame. On a
    head-mounted camera every head rotation would appear as hand motion and the robot would follow the operator's
    head. `camera_mode: fixed` is recorded in every log for that reason.
  * **Hand health is not arm health** (spec section 16, extended). `HAND_OK` is about fingertips — it is what the Aero
    branch needs. The arm needs only a palm origin (and, in 6-DoF, a palm basis): a hand whose fingertip depth is half
    missing can still carry a perfectly good arm pose. `ArmPoseHealth` is therefore graded separately here, from the
    palm landmarks alone, and never read off `HandPoseHealth`.

What this module deliberately does NOT do: no jump/hysteresis/age policy (that is `TrackingSupervisor`, reused
unchanged one level up), no axis swaps or scale (that is `HumanRobotFrameMapper`, the only place), no filtering
(spec/POC section 7: measure the raw jitter first, add a deadband only if the numbers demand it)."""
from __future__ import annotations
import enum
from dataclasses import dataclass
import numpy as np
from handumi_collector.pose.estimator import BackendInfo, PoseEstimate, PoseEstimator, TrackingState
from ..transforms.se3 import make_T
from . import aero_mocap as M
from .interfaces import HandPoseEstimate, HandPoseHealth
from .supervisor import HandPoseStatus, HandPoseSupervisor, HandPoseSupervisorConfig

# The palm: wrist + the four finger MCPs. These move with the hand as a near-rigid body — finger flexion moves the
# phalanges, not the knuckles — which is what makes their centroid a usable arm-root position (POC section 4).
PALM_LANDMARKS = (M.WRIST, M.INDEX_MCP, M.MIDDLE_MCP, M.RING_MCP, M.PINKY_MCP)
# The landmarks `aero_mocap.palm_frame` needs for the canonical palm basis.
PALM_FRAME_LANDMARKS = (M.WRIST, M.INDEX_MCP, M.MIDDLE_MCP, M.RING_MCP)
# The two palm spans the geometry-sanity check watches (plan section 6B). Both are near-constant for a given operator:
# the hand is a rigid body between these landmarks, so a jump in either is a reconstruction error, not a human motion.
SPAN_INDEX_PINKY = (M.INDEX_MCP, M.PINKY_MCP)
SPAN_WRIST_MIDDLE = (M.WRIST, M.MIDDLE_MCP)


POSE_SOURCE = "fixed_rgbd_hand"     # provenance tag: fixed external RGB-D camera, no IMU, causal


class ArmPoseHealth(str, enum.Enum):
    """Validity of the *arm-root* pose. Separate from `HandPoseHealth` (fingers) and from `TrackingHealth` (policy)."""
    OK = "ARM_POSE_OK"
    DEGRADED = "ARM_POSE_DEGRADED"     # thin but usable palm evidence -> downstream slows down (speed_factor)
    LOST = "ARM_POSE_LOST"             # no usable palm origin -> the arm target is held, never guessed


@dataclass
class PalmPoseConfig:
    side: str = "right"
    origin: str = "palm_centroid"        # palm_centroid (robust, POC default) | wrist (single canonical landmark)
    orientation: str = "fixed"           # fixed = phase 1 (3-DoF: R = I, robot keeps its ENGAGE orientation)
                                         # palm  = phase 2 (6-DoF: the canonical aero_mocap palm basis)
    min_palm_landmarks: int = 3          # of 5, with REAL measured depth (a shape-prior `filled` landmark never counts)
    min_palm_landmarks_ok: int = 5       # fewer than this (but >= min_palm_landmarks) -> ARM_POSE_DEGRADED. Defaults
                                         # to all five so every OK frame shares one origin definition: the median
                                         # shifts when the landmark set changes, and that shift must be declared.
    max_palm_depth_spread_m: float | None = None   # optional: palm depth scatter above this -> ARM_POSE_LOST for the
                                           # frame. None (default) = measure only. There is no per-landmark outlier
                                           # threshold on purpose: the palm spans 5-9 cm by anatomy and that anatomy
                                           # lands entirely on the depth axis when the hand points at the camera, so
                                           # any fixed threshold either rejects healthy palms or catches nothing.
                                           # Gross flying pixels are already removed upstream by the depth sampler
                                           # (`DepthSamplerConfig.max_spread_m`, `reject_depth_outliers`), and the
                                           # median below absorbs what survives that.
    min_origin_depth_m: float = 0.20     # the arm-root band; tighter than the depth sampler's own range on purpose
    max_origin_depth_m: float = 1.00
    hand_degraded_is_arm_degraded: bool = True   # HAND_DEGRADED (finger depth holes) -> at most ARM_POSE_DEGRADED
    require_real_depth_for_rotation: bool = True  # 6-DoF: refuse a palm basis built on shape-prior landmarks

    # ---- geometry sanity (plan section 6). EVERY threshold defaults to None = measure, never reject: the thresholds
    # are supposed to come from measured V0 data, and a guessed one would silently eat real motion. V0 reports the
    # distributions; you set the numbers here afterwards.
    reference_palm_scale_m: float | None = None   # s_op, the operator's palm scale. None -> estimated online as the
                                                  # median of the first `scale_reference_frames` OK frames.
    scale_reference_frames: int = 30
    max_palm_scale_deviation: float | None = None  # |scale/s_op - 1| above this -> ARM_POSE_DEGRADED (reliability down)
    max_palm_scale_jump: float | None = None       # |dscale|/s_op between frames above this -> ARM_POSE_LOST
    max_span_jump_m: float | None = None           # jump in either palm span between frames -> ARM_POSE_LOST
    # NOTE: maximum frame-to-frame palm MOTION (plan section 6C) is NOT re-implemented here. That is
    # TrackingSupervisor.max_jump_m / max_jump_deg, reused unchanged one level up — one jump policy, not two.

    def __post_init__(self) -> None:
        if self.origin not in ("palm_centroid", "wrist"): raise ValueError(f"origin must be palm_centroid|wrist, got {self.origin!r}")
        if self.orientation not in ("fixed", "palm"): raise ValueError(f"orientation must be fixed|palm, got {self.orientation!r}")
        if self.side not in ("left", "right"): raise ValueError(f"side must be left|right, got {self.side!r}")


@dataclass
class PalmPose:
    """One palm pose in the (fixed) camera frame, with the evidence it was built from."""
    health: ArmPoseHealth
    reason: str = ""
    position_m: np.ndarray | None = None       # palm origin in the camera frame
    R_palm: np.ndarray | None = None           # canonical palm basis, ALWAYS computed when possible — even in
                                               # `orientation: fixed`, so phase-1 runs still measure phase-2 noise
    n_palm_landmarks: int = 0
    palm_depth_spread_m: float = float("nan")  # max |z - median z| over the kept palm landmarks (evidence quality)
    wrist_offset_m: float = float("nan")       # |centroid - wrist|, the constant the relative mapping cancels
    palm_scale_m: float = float("nan")         # mean of the two spans below; NaN unless both are really measured
    span_index_pinky_m: float = float("nan")
    span_wrist_middle_m: float = float("nan")
    palm_scale_ratio: float = float("nan")     # palm_scale_m / s_op — 1.0 means the hand is the size it was at start

    def T_camera_palm(self, orientation: str) -> np.ndarray | None:
        """The emitted pose. `fixed` emits identity rotation so the relative SE(3) stack produces a zero rotation
        delta and the robot holds its ENGAGE orientation — phase 1 without a second code path (POC section 4)."""
        if self.position_m is None: return None
        R = np.eye(3) if orientation == "fixed" else self.R_palm
        return None if R is None else make_T(R, self.position_m)


def palm_origin_camera(est: HandPoseEstimate, cfg: PalmPoseConfig) -> tuple[np.ndarray | None, int, float, str]:
    """-> (origin_m, n_landmarks_used, spread_m, reason). Only landmarks with REAL measured depth are used.

    `wrist` is the canonical single landmark (`aero_mocap.palm_frame`'s origin); `palm_centroid` is the robust variant
    the POC defaults to: the component-wise MEDIAN of the valid palm landmarks. A median needs no threshold to guess
    and one bad landmark out of five moves it by millimetres, where it would drag a mean by centimetres.

    The median is not the geometric palm centre, and it shifts when the set of valid landmarks changes — which is why
    `min_palm_landmarks_ok` defaults to all five: frames with a complete palm are ARM_POSE_OK and share one origin
    definition, and a frame that loses a landmark is declared DEGRADED rather than quietly moving the arm."""
    P = np.asarray(est.landmarks_camera_3d, np.float64)
    ok = np.asarray(est.valid, bool) & np.isfinite(P).all(1)          # `valid` is real depth only: filled stays False
    idx = [j for j in PALM_LANDMARKS if ok[j]]
    if cfg.origin == "wrist":
        if not ok[M.WRIST]: return None, len(idx), float("nan"), "wrist_landmark_invalid"
        return P[M.WRIST].copy(), 1, 0.0, ""
    if len(idx) < cfg.min_palm_landmarks:
        return None, len(idx), float("nan"), f"palm_landmarks:{len(idx)}<{cfg.min_palm_landmarks}"
    Q = P[idx]
    origin = np.median(Q, axis=0)
    spread = float(np.max(np.abs(Q[:, 2] - origin[2])))
    if cfg.max_palm_depth_spread_m is not None and spread > cfg.max_palm_depth_spread_m:
        return None, len(idx), spread, f"palm_depth_scatter:{spread:.3f}m"
    return origin, len(idx), spread, ""


def palm_rotation_camera(est: HandPoseEstimate, cfg: PalmPoseConfig) -> tuple[np.ndarray | None, str]:
    """The CANONICAL palm basis (`aero_mocap.palm_frame`) evaluated on the metric camera-frame landmarks.

    Deliberately not a second axis convention: the Aero branch, the recorded `human_hand` stream and this arm pose all
    mean the same thing by "palm frame" (x: index->ring MCP, z: wrist->middle MCP, y = z x x, x re-orthogonalised)."""
    if cfg.require_real_depth_for_rotation and not all(bool(est.valid[j]) for j in PALM_FRAME_LANDMARKS):
        return None, "palm_basis_needs_real_depth"
    try:
        R, _ = M.palm_frame(est.landmarks_camera_3d, cfg.side)
    except ValueError as e:
        return None, f"palm_basis:{e}"
    return R, ""


def palm_spans_m(est: HandPoseEstimate) -> tuple[float, float]:
    """(index MCP <-> pinky MCP, wrist <-> middle MCP) in metres, NaN unless BOTH endpoints have real depth.

    Measured from the camera-frame landmarks rather than from `HandPoseEstimate.palm_scale_m`, which is computed over
    the completed (shape-prior-filled) geometry: a span that is partly invented cannot testify about reconstruction
    quality, so it is reported as NaN instead."""
    P = np.asarray(est.landmarks_camera_3d, np.float64); v = np.asarray(est.valid, bool)
    def span(a, b):
        return float(np.linalg.norm(P[a] - P[b])) if (v[a] and v[b] and np.isfinite(P[[a, b]]).all()) else float("nan")
    return span(*SPAN_INDEX_PINKY), span(*SPAN_WRIST_MIDDLE)


def palm_pose_from_estimate(est: HandPoseEstimate | None, cfg: PalmPoseConfig,
                            *, hand_status: HandPoseStatus | None = None) -> PalmPose:
    """Grade the ARM-root pose from the PROVIDER's estimate, with the hand supervisor as an advisor only.

    The distinction is the point of plan section 8. `HandPoseSupervisor` answers "may this drive the Aero fingers?",
    and it says no when the fingertips are missing — five lost fingertips is HAND_LOST. The arm asks a different
    question, and a palm with all five landmarks measured answers it perfectly well. So the hand verdict may only
    *lower* the arm to DEGRADED; the vetoes that do produce ARM_POSE_LOST are the ones the arm actually depends on:
    the provider reconstructed no hand at all, the source is not metric, there is no palm origin, or the origin is
    outside the usable depth band. Frame-to-frame jumps are not judged here either — that is `TrackingSupervisor`."""
    if est is None or est.health is HandPoseHealth.LOST:
        reason = "no_hand" if est is None else (est.extra.get("reason") or "provider_lost")
        return PalmPose(ArmPoseHealth.LOST, reason)
    if not est.metric:
        return PalmPose(ArmPoseHealth.LOST, "non_metric_provider")   # a monocular hand has no metres to give the arm

    p, n, spread, why = palm_origin_camera(est, cfg)
    if p is None:
        return PalmPose(ArmPoseHealth.LOST, why or "no_palm_origin", n_palm_landmarks=n, palm_depth_spread_m=spread)
    z = float(p[2])
    if not (cfg.min_origin_depth_m <= z <= cfg.max_origin_depth_m):
        return PalmPose(ArmPoseHealth.LOST, f"origin_depth:{z:.3f}m", None, None, n, spread)

    R, r_why = palm_rotation_camera(est, cfg)
    wrist = np.asarray(est.landmarks_camera_3d[M.WRIST], np.float64)
    off = float(np.linalg.norm(p - wrist)) if np.isfinite(wrist).all() else float("nan")
    s_ip, s_wm = palm_spans_m(est)
    pose = PalmPose(ArmPoseHealth.OK, "", p, R, n, spread, off, float(np.mean([s_ip, s_wm])), s_ip, s_wm)

    if cfg.orientation == "palm" and R is None:
        return PalmPose(ArmPoseHealth.LOST, r_why or "no_palm_basis", None, None, n, spread, off)
    if n < cfg.min_palm_landmarks_ok:
        pose.health, pose.reason = ArmPoseHealth.DEGRADED, f"palm_landmarks:{n}"
    elif (cfg.hand_degraded_is_arm_degraded and hand_status is not None
          and hand_status.health is not HandPoseHealth.OK):
        pose.health, pose.reason = ArmPoseHealth.DEGRADED, f"{hand_status.health.value.lower()}:{hand_status.reason}"
    return pose


class PalmGeometryMonitor:
    """Hand-scale consistency and palm-geometry sanity across frames (plan section 6A/6B).

    The operator's palm is a fixed size. `s_op` is therefore measured once (median of the first OK frames, or pinned
    in the config) and then only ever used to *judge* a frame — never to rescale a command. A gain that follows the
    measured palm scale would turn depth noise into robot motion, which is the opposite of what this is for.

    Every threshold is optional. With all of them None the monitor is pure measurement, which is the V0 default: the
    numbers it records are what the thresholds should later be set from."""

    def __init__(self, cfg: PalmPoseConfig) -> None:
        self.cfg = cfg
        self.reference_palm_scale_m: float | None = cfg.reference_palm_scale_m
        self._samples: list[float] = []
        self._prev: tuple[float, float, float] | None = None      # (scale, index-pinky span, wrist-middle span)
        self.rejects = 0

    def update(self, pose: PalmPose) -> PalmPose:
        c = self.cfg
        scale, s_ip, s_wm = pose.palm_scale_m, pose.span_index_pinky_m, pose.span_wrist_middle_m
        if pose.health is not ArmPoseHealth.LOST and np.isfinite(scale):
            if self.reference_palm_scale_m is None:
                self._samples.append(scale)
                if len(self._samples) >= c.scale_reference_frames:
                    self.reference_palm_scale_m = float(np.median(self._samples))
            ref = self.reference_palm_scale_m
            if ref and ref > 1e-6:
                pose.palm_scale_ratio = scale / ref
                if c.max_palm_scale_deviation is not None and abs(pose.palm_scale_ratio - 1.0) > c.max_palm_scale_deviation:
                    pose.health = ArmPoseHealth.DEGRADED          # implausible size -> trust it less, do not drop it
                    pose.reason = f"palm_scale_deviation:{pose.palm_scale_ratio:.2f}"
                if c.max_palm_scale_jump is not None and self._prev is not None and np.isfinite(self._prev[0]):
                    if abs(scale - self._prev[0]) / ref > c.max_palm_scale_jump:
                        pose = self._reject(pose, f"palm_scale_jump:{abs(scale - self._prev[0])*1000:.0f}mm")
            if c.max_span_jump_m is not None and self._prev is not None and pose.health is not ArmPoseHealth.LOST:
                for name, now, before in (("index_pinky", s_ip, self._prev[1]), ("wrist_middle", s_wm, self._prev[2])):
                    if np.isfinite(now) and np.isfinite(before) and abs(now - before) > c.max_span_jump_m:
                        pose = self._reject(pose, f"span_jump_{name}:{abs(now - before)*1000:.0f}mm"); break
        self._prev = (scale, s_ip, s_wm)
        return pose

    def _reject(self, pose: PalmPose, reason: str) -> PalmPose:
        """Impossible geometry: the reconstruction is wrong, so the pose is dropped rather than sent as a motion."""
        self.rejects += 1
        pose.health, pose.reason, pose.position_m, pose.R_palm = ArmPoseHealth.LOST, reason, None, None
        return pose

    def stats(self) -> dict:
        return dict(reference_palm_scale_m=self.reference_palm_scale_m, geometry_rejects=self.rejects,
                    reference_samples=len(self._samples))


_STATE = {ArmPoseHealth.OK: TrackingState.TRACKING, ArmPoseHealth.DEGRADED: TrackingState.DEGRADED,
          ArmPoseHealth.LOST: TrackingState.LOST}


class RgbdPalmPoseEstimator(PoseEstimator):
    """`PoseEstimator` over a fixed RGB-D camera: RgbdFrame in, palm `PoseEstimate` out.

    Implementing the existing estimator interface is the whole point — `EstimatorWristPoseProvider`,
    `TrackingSupervisor`, `VioStableGate`, `RelativeSE3Retargeter`, `ArmSafetyPipeline` and the reBot client then all
    run unmodified on top of it, and downstream cannot tell (nor ask) which source produced the pose.

    `T_world_camera` is `T_camera_palm` — the world IS the fixed camera. `info.uses_imu` is False and `info.name`
    ("rgbd_hand_palm") reaches every log line, so this can never be mistaken for an OpenVINS wrist pose."""

    info = BackendInfo(
        name="rgbd_hand_palm", uses_imu=False, metric_scale=True, online_capable=True,
        provides=("confidence", "num_features"),
        notes="POC arm pose source: operator palm pose in a FIXED RGB-D camera frame (world == camera). NOT wrist "
              "VIO; invalid the moment the camera moves. Replaced by wrist fisheye + IMU + OpenVINS in production.")

    def __init__(self, provider, cfg: PalmPoseConfig | None = None, *,
                 supervisor: HandPoseSupervisor | None = None, hand_pose_cfg: HandPoseSupervisorConfig | None = None) -> None:
        self.provider = provider
        self.cfg = cfg or PalmPoseConfig()
        self.hand_sup = supervisor or HandPoseSupervisor(hand_pose_cfg)
        self.geometry = PalmGeometryMonitor(self.cfg)
        self.counters = {h.value: 0 for h in ArmPoseHealth}
        self.reasons: dict[str, int] = {}
        self._n = 0
        self.last_hand: HandPoseStatus | None = None
        self.last_palm: PalmPose | None = None
        self._last: PoseEstimate | None = None

    def initialize(self, **_kw) -> None:
        """No-op: intrinsics, depth scale and the depth<->colour alignment travel with every `RgbdFrame.calib`."""

    def reset(self) -> None:
        self.hand_sup = HandPoseSupervisor(self.hand_sup.cfg)
        self.geometry = PalmGeometryMonitor(self.cfg)
        self.counters = {h.value: 0 for h in ArmPoseHealth}; self.reasons.clear(); self._n = 0
        self.last_hand = self.last_palm = self._last = None

    def push_image(self, t_ns: int, frame_index: int, frame) -> PoseEstimate:
        """`frame` is an `RgbdFrame` (colour + aligned depth), not a bare image — depth is the whole point."""
        if not hasattr(frame, "depth_m"):
            raise TypeError("RgbdPalmPoseEstimator needs an RgbdFrame (colour + aligned depth), got a plain image")
        est = self.provider.get_hand_pose(frame)
        return self.ingest_hand(int(t_ns), int(frame_index), est)

    def ingest_hand(self, t_ns: int, frame_index: int, est: HandPoseEstimate | None) -> PoseEstimate:
        """The same step on an already-computed `HandPoseEstimate` — used by offline replay and by the tests."""
        status = self.hand_sup.update(est, int(t_ns))       # the HAND branch verdict — advisory for the arm
        palm = self.geometry.update(palm_pose_from_estimate(est, self.cfg, hand_status=status))
        self.last_hand, self.last_palm = status, palm
        self.counters[palm.health.value] += 1
        if palm.reason: self.reasons[palm.reason] = self.reasons.get(palm.reason, 0) + 1
        self._n += 1

        T = palm.T_camera_palm(self.cfg.orientation)
        extra = dict(arm_pose_health=palm.health.value, arm_pose_reason=palm.reason,
                     hand_health=status.health.value, hand_reason=status.reason,
                     n_palm_landmarks=palm.n_palm_landmarks, palm_depth_spread_m=palm.palm_depth_spread_m,
                     wrist_offset_m=palm.wrist_offset_m, palm_scale_m=palm.palm_scale_m,
                     span_index_pinky_m=palm.span_index_pinky_m, span_wrist_middle_m=palm.span_wrist_middle_m,
                     palm_scale_ratio=palm.palm_scale_ratio,
                     origin=self.cfg.origin, orientation=self.cfg.orientation,
                     camera_mode="fixed", imu_used=False, pose_source=POSE_SOURCE)
        if palm.R_palm is not None:
            # always carried, including in `fixed` mode: phase 1 measures phase 2's orientation noise for free
            extra["R_palm"] = palm.R_palm.tolist()
        pe = PoseEstimate(int(t_ns), int(frame_index), T, _STATE[palm.health] if T is not None else TrackingState.LOST,
                          confidence=float(est.detector_confidence) if est is not None else None,
                          num_features=int(est.n_valid) if est is not None else None, extra=extra)
        self._last = pe
        return pe

    def get_quality(self) -> dict:
        return dict(backend=self.info.name, frames=self._n, arm_pose=dict(self.counters), reasons=dict(self.reasons),
                    hand=dict(self.hand_sup.counters), camera_mode="fixed", imu_used=False, pose_source=POSE_SOURCE,
                    orientation=self.cfg.orientation, origin=self.cfg.origin, geometry=self.geometry.stats())
