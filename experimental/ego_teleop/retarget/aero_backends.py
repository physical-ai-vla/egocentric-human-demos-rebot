"""Swappable Aero retargeting backends (spec sections 13-15) — the second half of the human -> Aero path.

    HandPoseEstimate  ->  AeroRetargetingBackend  ->  AeroTarget (16 joints, rad)  ->  aero_open_sdk
                             ^ DexPilotAeroRetargeter   (PRIMARY, official dex_retargeting + Aero URDF)
                             ^ Semantic7DAeroRetargeter (the existing tendon-semantic mapping: baseline / debug / fallback)

Both consume ONLY the canonical human representation, so a future learned retargeter drops in the same way and the
recorded human pose stays reusable if Aero is replaced (invariant, spec section 32).

Why DexPilot and not a joint-angle copy: human and Aero morphology differ, so the thing worth preserving is the
task-space relationships (thumb<->index, thumb<->middle, tips<->palm), which is exactly what DexPilot optimises. The
existing semantic-7D path stays because it is interpretable, needs no optimiser, and is the B2 arm of the comparison.

`DexPilotAeroRetargeter` reproduces `dex_retargeting_node.py` bit for bit — same config, same two undocumented
alignment hacks, same per-actuator scale/offset, same clip — with ROS removed. See docs/ego_teleop/AERO_A0_AUDIT.md."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import numpy as np
from ego_collector.hands3d.aero import default_urdf
from ..hand3d import aero_mocap as M
from ..hand3d.interfaces import HandPoseEstimate

FINGER_TIPS = ("thumb", "index", "middle", "ring", "pinky")

# Per-actuator (scale, offset_rad) from dex_retargeting_node.py, applied to the reindexed 16-joint vector.
OFFICIAL_SCALE_FACTORS = (
    (1.5, np.deg2rad(0.0)),     # thumb CMC abduction        -> joint 0
    (2.0, np.deg2rad(-60.0)),   # thumb CMC flexion          -> joint 1
    (3.5, np.deg2rad(0.0)),     # thumb MCP/IP tendon        -> joints 2,3
    (1.15, np.deg2rad(-10.0)),  # index tendon               -> joints 4-6
    (1.15, np.deg2rad(-10.0)),  # middle tendon              -> joints 7-9
    (1.15, np.deg2rad(-10.0)),  # ring tendon                -> joints 10-12
    (1.2, np.deg2rad(-10.0)),   # pinky tendon               -> joints 13-15
)


@dataclass
class AeroTarget:
    """One Aero command in the official 16-joint space (radians, AERO_JOINT_NAMES order)."""
    timestamp_ns: int                      # inherited from the hand pose it came from (the teleop clock)
    joints_rad: np.ndarray                 # (16,) clipped to the Aero joint box
    method: str                            # "dexpilot" | "position" | "vector" | "semantic7d"
    pose_source: str                       # which HandPoseProvider produced the input ("head_rgbd" / "mediapipe_mono")
    source: str = "retarget"               # retarget | hold | rate_limited
    raw_rad: np.ndarray | None = None      # optimiser output before scale/offset/clip (diagnostics)
    diagnostics: dict = field(default_factory=dict)
    retarget_done_ns: int | None = None     # host monotonic after the optimiser — latency accounting (spec section 22)

    @property
    def joints_deg(self) -> np.ndarray: return np.degrees(self.joints_rad)

    def to_row(self) -> dict:
        r = dict(t_ns=int(self.timestamp_ns), method=self.method, pose_source=self.pose_source, source=self.source,
                 retarget_done_ns=self.retarget_done_ns,
                 retarget_ms=float("nan") if self.retarget_done_ns is None else (self.retarget_done_ns - self.timestamp_ns) / 1e6)
        for i, n in enumerate(M.AERO_JOINT_NAMES):
            r[f"q_{n}_rad"] = float(self.joints_rad[i])
            r[f"raw_{n}_rad"] = float("nan") if self.raw_rad is None else float(self.raw_rad[i])
        for k, v in self.diagnostics.items():
            if isinstance(v, (int, float, bool, str)) or v is None: r[f"diag_{k}"] = v
        return r


class AeroRetargetingBackend(Protocol):
    method: str
    def retarget(self, hand_pose: HandPoseEstimate) -> AeroTarget: ...


# ---------------------------------------------------------------------------------------------------------------
# DexPilot / dex_retargeting (PRIMARY)
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class DexPilotConfig:
    side: str = "right"
    method: str = "dexpilot"               # dexpilot | vector | position
    scaling_factor: float = 1.2
    low_pass_alpha: float = 0.9
    urdf_path: str | Path | None = None     # default: assets/aero_hand_open/aero_hand_open_{side}.urdf
    apply_scale_factors: bool = True        # the official per-actuator scale/offset
    base_offset_m: tuple[float, float, float] = (0.0, 0.01, 0.0)   # official hack 1 (align to the hand base link)
    pinky_lift: tuple[float, float] = (0.12, 0.02)                 # official hack 2 (z threshold, z bump)
    apply_hacks: bool = True
    rescale_palm_to_m: float | None = None  # optional: normalise hand size before retargeting (None = off, as upstream)


class DexPilotAeroRetargeter:
    """Official DexPilot path, ROS removed. Deterministic for a given input and warm-start state."""

    def __init__(self, cfg: DexPilotConfig | None = None) -> None:
        from dex_retargeting.retargeting_config import RetargetingConfig
        self.cfg = cfg or DexPilotConfig()
        self.method = self.cfg.method
        self.side = self.cfg.side
        self._r = RetargetingConfig.from_dict(self.make_config()).build()
        self.joint_names = [f"{self.side}_{n}" for n in M.AERO_JOINT_NAMES]
        self._reindex = [self._r.joint_names.index(n) if n in self._r.joint_names else None for n in self.joint_names]
        self._lower = np.deg2rad(np.array(M.AERO_JOINT_LOWER_DEG))
        self._upper = np.deg2rad(np.array(M.AERO_JOINT_UPPER_DEG))

    # ---- config (verbatim from dex_retargeting_node.make_config) -----------------------------------------
    def make_config(self) -> dict:
        c, s = self.cfg, self.cfg.side
        urdf = str(c.urdf_path or default_urdf(s))
        tips = [f"{s}_{f}_tip_link" for f in FINGER_TIPS]
        if c.method == "position":
            return dict(type="position", urdf_path=urdf, target_link_names=tips,
                        target_link_human_indices=list(M.MOCAP25_TIPS), scaling_factor=c.scaling_factor)
        if c.method == "vector":
            return dict(type="vector", urdf_path=urdf, target_origin_link_names=[f"{s}_base_link"] * 5,
                        target_task_link_names=tips, target_link_human_indices=[[0] * 5, list(M.MOCAP25_TIPS)],
                        scaling_factor=c.scaling_factor, low_pass_alpha=c.low_pass_alpha)
        if c.method == "dexpilot":
            return dict(type="dexpilot", urdf_path=urdf, wrist_link_name=f"{s}_base_link", finger_tip_link_names=tips,
                        target_link_human_indices=[[9, 14, 19, 24, 14, 19, 24, 19, 24, 24, 0, 0, 0, 0, 0],
                                                   [4, 4, 4, 4, 9, 9, 9, 14, 14, 19, 4, 9, 14, 19, 24]],
                        scaling_factor=c.scaling_factor, low_pass_alpha=c.low_pass_alpha)
        raise ValueError(f"unknown retargeting method {c.method!r}")

    # ---- runtime -----------------------------------------------------------------------------------------
    def _reference(self, pose25: np.ndarray) -> np.ndarray:
        idx = np.asarray(self._r.optimizer.target_link_human_indices)
        if self._r.optimizer.retargeting_type == "POSITION": return pose25[idx, :]
        return pose25[idx[1, :], :] - pose25[idx[0, :], :]

    def apply_scale_factors(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, np.float64).copy()
        sf = OFFICIAL_SCALE_FACTORS
        q[0] = q[0] * sf[0][0] + sf[0][1]
        q[1] = q[1] * sf[1][0] + sf[1][1]
        q[2:4] = q[2:4] * sf[2][0] + sf[2][1]
        for k, (lo, hi) in enumerate(((4, 7), (7, 10), (10, 13), (13, 16))):
            q[lo:hi] = q[lo:hi] * sf[3 + k][0] + sf[3 + k][1]
        return q

    def prepare(self, hand_pose: HandPoseEstimate) -> np.ndarray:
        """Palm-local 21 landmarks -> the 25-keypoint array the official node feeds the optimiser (hacks included)."""
        local = hand_pose.landmarks_local_3d
        if self.cfg.rescale_palm_to_m is not None and np.isfinite(hand_pose.palm_scale_m) and hand_pose.palm_scale_m > 1e-6:
            local = local * (self.cfg.rescale_palm_to_m / hand_pose.palm_scale_m)
        pose25 = M.expand_to_mocap25(local)
        if self.cfg.apply_hacks:
            pose25 = pose25 + np.asarray(self.cfg.base_offset_m, np.float64)
            z_thr, z_bump = self.cfg.pinky_lift
            if pose25[24][2] > z_thr: pose25[24][2] += z_bump
        return pose25

    def retarget(self, hand_pose: HandPoseEstimate) -> AeroTarget:
        if hand_pose.side != self.side:
            raise ValueError(f"{self.side} retargeter got a {hand_pose.side} hand pose")
        pose25 = self.prepare(hand_pose)
        if not np.all(np.isfinite(pose25)):
            raise ValueError("retargeting needs complete geometry: palm-local landmarks contain NaN "
                             "(the provider must fill or the supervisor must mark the frame LOST)")
        q_opt = np.asarray(self._r.retarget(self._reference(pose25)), np.float64)
        raw = np.array([0.0 if i is None else q_opt[i] for i in self._reindex])
        q = self.apply_scale_factors(raw) if self.cfg.apply_scale_factors else raw.copy()
        q = np.clip(q, self._lower, self._upper)
        return AeroTarget(hand_pose.timestamp_ns, q, self.method, hand_pose.source, "retarget", raw,
                          dict(palm_scale_m=float(hand_pose.palm_scale_m), n_valid=hand_pose.n_valid,
                               n_filled=hand_pose.n_filled, metric=bool(hand_pose.metric),
                               saturated=int(np.count_nonzero((q <= self._lower + 1e-9) | (q >= self._upper - 1e-9)))),
                          retarget_done_ns=time.monotonic_ns())


# ---------------------------------------------------------------------------------------------------------------
# Semantic 7D (existing baseline / debug path / fallback — kept, per spec section 13)
# ---------------------------------------------------------------------------------------------------------------
class Semantic7DAeroRetargeter:
    """Wraps the existing `AeroRetargeter` (u7 -> compact-7 deg -> 16 joints) behind the same backend interface.

    Not a replacement for DexPilot: it copies tendon-weighted human joint angles, which is directionally wrong for
    thumb opposition (hence `thumb_gap_servo` offline). It stays because it is cheap, has no optimiser to fail, and
    is the B2 arm of the comparison."""

    method = "semantic7d"

    def __init__(self, retargeter=None, *, side: str = "right") -> None:
        from .aero_retarget import AeroRetargeter
        from ego_collector.hands3d.aero import compact_to_full16_deg
        self._r = retargeter or AeroRetargeter()
        self._to16 = compact_to_full16_deg
        self.side = side

    def retarget(self, hand_pose: HandPoseEstimate) -> AeroTarget:
        from .hand_features import features_from_landmarks
        feat = features_from_landmarks(hand_pose.timestamp_ns, hand_pose.landmarks_local_3d, hand_pose.detector_confidence, side=hand_pose.side)
        cmd = self._r.retarget(hand_pose.timestamp_ns, feat.u7)
        q = np.radians(self._to16(cmd.compact_deg))
        return AeroTarget(hand_pose.timestamp_ns, q, self.method, hand_pose.source, "retarget", None,
                          dict(u7=";".join(f"{v:.3f}" for v in feat.u7), palm_scale_m=float(hand_pose.palm_scale_m)),
                          retarget_done_ns=time.monotonic_ns())


# ---------------------------------------------------------------------------------------------------------------
# Command safety (spec section 17-18): hold on loss, rate limit, joint bounds, watchdog
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class AeroLimiterConfig:
    max_joint_rate_rad_s: float = 6.0      # ~344 deg/s; the first hardware runs should lower this
    max_step_rad: float = 0.25             # hard per-command step, independent of dt (jump guard)
    command_timeout_ms: float = 200.0      # no new target for this long -> watchdog: hold and say so
    resume_ramp_s: float = 0.0             # >0: after a hold, ease back in over this long instead of jumping


class AeroCommandLimiter:
    """Turns retargeted targets into what actually goes to the hand. HAND_LOST holds the LAST COMMANDED target —
    never an auto-open, which would drop whatever is being carried (spec section 17)."""

    def __init__(self, cfg: AeroLimiterConfig | None = None) -> None:
        self.cfg = cfg or AeroLimiterConfig()
        self._prev: np.ndarray | None = None
        self._prev_t: int | None = None
        self._lower = np.deg2rad(np.array(M.AERO_JOINT_LOWER_DEG))
        self._upper = np.deg2rad(np.array(M.AERO_JOINT_UPPER_DEG))
        self.holds = 0
        self.rate_limited = 0

    def step(self, target: AeroTarget | None, t_ns: int) -> AeroTarget | None:
        """`target=None` means "no new command this tick" (hand lost / initializing / clutched)."""
        if target is None:
            if self._prev is None: return None
            self.holds += 1
            age = float("nan") if self._prev_t is None else (int(t_ns) - self._prev_t) / 1e6
            return AeroTarget(int(t_ns), self._prev.copy(), "hold", "", "hold",
                              diagnostics=dict(held_ms=age, watchdog=bool(age == age and age > self.cfg.command_timeout_ms)))
        q = np.clip(np.asarray(target.joints_rad, np.float64), self._lower, self._upper)
        src = target.source
        if self._prev is not None:
            prev_t = int(t_ns) if self._prev_t is None else self._prev_t   # `or` here would treat t_ns == 0 as unset
            dt = max((int(t_ns) - prev_t) / 1e9, 1e-3)
            step = min(self.cfg.max_joint_rate_rad_s * dt, self.cfg.max_step_rad)
            d = np.clip(q - self._prev, -step, step)
            if np.any(np.abs(q - self._prev) > step + 1e-12): self.rate_limited += 1; src = "rate_limited"
            q = self._prev + d
        self._prev, self._prev_t = q.copy(), int(t_ns)
        out = AeroTarget(int(t_ns), q, target.method, target.pose_source, src, target.raw_rad,
                         dict(target.diagnostics), target.retarget_done_ns)
        return out

    def reset(self) -> None:
        self._prev = None; self._prev_t = None
