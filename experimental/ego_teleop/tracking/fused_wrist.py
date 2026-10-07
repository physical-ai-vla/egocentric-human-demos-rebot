"""`FusedWristPoseProvider` — chest RGB-D + wrist Arducam + wrist IMU -> one continuous metric wrist pose.

    chest RGB-D  ──► RgbdHandWristPoseProvider ──► p_D_palm ──┐
                                                              ├─► AnchorAlignment ─► e(t) ─► C(t) ─┐
    Arducam ─┐                                                │                                    │
             ├─► PoseEstimator (VIO/VI-SLAM) ─► T_map ─► LocalPoseContinuity ─► T_local ────────────┼─► T_fused
    IMU  ────┘                                                                                      │
                                                                                                    ▼
                                                                          WristPose (source `fused_wrist`)
                                                                          -> RelativeSE(3) -> safety -> reBot

This class satisfies the EXISTING `WristPoseProvider` protocol and nothing downstream changes: the retargeter, the
frame mapper, the safety pipeline, the coordinator and the reBot client are the same objects the wrist-VIO and the
RGB-D POC paths already run through (spec section 11).

Three properties are load-bearing:

  * **Only the local pose leaves this class.** Loop closures and map re-optimisations are absorbed by
    `LocalPoseContinuity`; `T_map_wrist` is logged and never commanded (section 3).
  * **Orientation is VI's.** The chest view's palm orientation is computed and logged for comparison, never used as
    the control rotation (section 7) — foreshortening and finger articulation make it the weaker source.
  * **Nothing is extrapolated.** When both branches are LOST the last fused pose is held with health LOST (the
    coordinator then freezes the arm target). A zero pose is never emitted (section 14).

Ablation (section 21) runs through this same class by `mode`: `fused` (C), `vi_only` (C ≡ 0), `rgbd_only` (the RGB-D
palm pose is passed through as the control pose). A/B/C therefore differ in one config value, not in a code path."""
from __future__ import annotations
import enum
import threading
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from handumi_collector.pose.estimator import TrackingState
from ..transforms.se3 import make_T
from .anchor_fusion import (AnchorAlignment, AnchorAlignmentConfig, AnchorFusionConfig, FusionState, FusionStateConfig,
                            TranslationAnchorFusion, fusion_health)
from .interfaces import TrackingHealth, WristPose
from .local_pose import LocalPoseConfig, LocalPoseContinuity
from .wrist_pose_provider import TrackingSupervisor, TrackingSupervisorConfig

SOURCE = "fused_wrist"
STREAM_PREFIX = "fused_wrist_pose_live"     # never `wrist_pose_live_*` (wrist VIO alone) — a different sensor set


class FusionMode(str, enum.Enum):
    FUSED = "fused"          # C: chest RGB-D anchors the wrist VI translation
    VI_ONLY = "vi_only"      # ablation B: Arducam + IMU alone
    RGBD_ONLY = "rgbd_only"  # ablation A: chest RGB-D palm pose alone (the existing POC source)


@dataclass
class FusedWristConfig:
    side: str = "right"
    mode: str = "fused"
    max_pair_dt_ms: float = 20.0          # RGB-D sample and VI pose further apart than this are not one observation
    vi_ring_s: float = 2.0
    supervisor: TrackingSupervisorConfig = field(default_factory=TrackingSupervisorConfig)  # AGE policy for the fused pose
    local: LocalPoseConfig = field(default_factory=LocalPoseConfig)
    alignment: AnchorAlignmentConfig = field(default_factory=AnchorAlignmentConfig)
    fusion: AnchorFusionConfig = field(default_factory=AnchorFusionConfig)
    state: FusionStateConfig = field(default_factory=FusionStateConfig)

    def __post_init__(self) -> None:
        FusionMode(self.mode)             # raises on a typo rather than silently running the wrong ablation
        if self.side not in ("left", "right"): raise ValueError(f"side must be left|right, got {self.side!r}")


class FusedWristPoseProvider:
    """Feed it with `push_vi` / `push_rgbd` (sensor threads), or hand it the two providers and call `poll(now_ns)`."""

    def __init__(self, cfg: FusedWristConfig | None = None, *, vi_provider=None, rgbd_provider=None) -> None:
        self.cfg = cfg or FusedWristConfig()
        self.mode = FusionMode(self.cfg.mode)
        self.vi, self.rgbd = vi_provider, rgbd_provider
        self.local = LocalPoseContinuity(self.cfg.local)
        self.align = AnchorAlignment(self.cfg.alignment)
        self.corr = TranslationAnchorFusion(self.cfg.fusion)
        self.sup = TrackingSupervisor(self.cfg.supervisor)
        self._ring: deque = deque()                    # (t_ns, T_local) VI history for pairing
        self._vi_last: WristPose | None = None
        self._vi_T_local: np.ndarray | None = None
        self._rgbd_last: WristPose | None = None
        self._rgbd_ok_ns: int | None = None            # last RGB-D sample that was usable as an anchor
        self._last_e: np.ndarray | None = None         # last anchor residual e = anchor − VI prediction
        self._first_ns: int | None = None
        self._unanchored_since_ns: int | None = None    # when the anchor was last unavailable (None = anchored now)
        self._last_out: WristPose | None = None
        self._prev_out: tuple[int, np.ndarray] | None = None   # (t_ns, T) of the previous emitted pose
        self._last_vw: tuple[np.ndarray, np.ndarray] = (np.zeros(3), np.zeros(3))
        self._clutched = False
        self._n_vi = self._n_rgbd = self._n_pairs = self._n_pair_dt_rejected = 0
        # live, the two branches are pushed from their own sensor threads while the control loop reads: one lock, held
        # only for the (short) state updates, is what makes that safe without giving either sensor a second clock
        self._lock = threading.RLock()

    # ---- operator state (section 13) ---------------------------------------------------------------------
    def set_clutched(self, clutched: bool) -> None:
        """The arm is frozen: a pending anchor correction may be applied in full, with no robot motion."""
        self._clutched = bool(clutched)

    # ---- feeding -----------------------------------------------------------------------------------------
    def push_vi(self, wp: WristPose | None) -> None:
        if wp is None: return
        with self._lock:
            self._push_vi(wp)

    def _push_vi(self, wp: WristPose) -> None:
        self._n_vi += 1
        self._vi_last = wp
        if self._first_ns is None: self._first_ns = wp.timestamp_ns; self._unanchored_since_ns = int(wp.timestamp_ns)
        if wp.health == TrackingHealth.LOST or not np.all(np.isfinite(wp.position_xyz_m)): return
        T_local = self.local.update(wp.T(), extra=wp.extra)
        self._vi_T_local = T_local
        self._ring.append((int(wp.timestamp_ns), T_local.copy()))
        cutoff = int(wp.timestamp_ns) - int(self.cfg.vi_ring_s * 1e9)
        while self._ring and self._ring[0][0] < cutoff: self._ring.popleft()

    def push_rgbd(self, wp: WristPose | None) -> None:
        if wp is None: return
        with self._lock:
            self._push_rgbd(wp)

    def _push_rgbd(self, wp: WristPose) -> None:
        self._n_rgbd += 1
        self._rgbd_last = wp
        if self._first_ns is None: self._first_ns = wp.timestamp_ns; self._unanchored_since_ns = int(wp.timestamp_ns)
        if self.mode is FusionMode.RGBD_ONLY:
            if wp.health != TrackingHealth.LOST and np.all(np.isfinite(wp.position_xyz_m)): self._rgbd_ok_ns = int(wp.timestamp_ns)
            return
        if wp.health == TrackingHealth.LOST or not np.all(np.isfinite(wp.position_xyz_m)): return
        self._rgbd_ok_ns = int(wp.timestamp_ns)
        T_vi = self._vi_at(int(wp.timestamp_ns))
        if T_vi is None: return
        self._n_pairs += 1
        self.align.add_pair(int(wp.timestamp_ns), wp.position_xyz_m, T_vi)
        st = self.align.maybe_fit(int(wp.timestamp_ns))
        if not st.aligned:
            if self._unanchored_since_ns is None: self._unanchored_since_ns = int(wp.timestamp_ns)
            return
        self._unanchored_since_ns = None
        if self.mode is FusionMode.FUSED:
            e = self.align.residual(wp.position_xyz_m, T_vi)
            self._last_e = e
            self.align.note_residual(int(wp.timestamp_ns), e)     # a residual that stays absurd = a broken alignment
            self.corr.observe(e, wp.confidence)

    def poll(self, now_ns: int) -> WristPose | None:
        """Pull both branches (when they were handed in) and emit. Sensor-thread setups call push_* instead."""
        if self.vi is not None:
            wp = self.vi.get_pose(now_ns)
            if wp is not None and (self._vi_last is None or wp.timestamp_ns != self._vi_last.timestamp_ns): self.push_vi(wp)
            elif wp is not None: self._vi_last = wp                       # same sample, refreshed age grading
        if self.rgbd is not None:
            wr = self.rgbd.get_pose(now_ns)
            if wr is not None and (self._rgbd_last is None or wr.timestamp_ns != self._rgbd_last.timestamp_ns): self.push_rgbd(wr)
            elif wr is not None: self._rgbd_last = wr
        return self.get_pose(now_ns)

    # ---- output ------------------------------------------------------------------------------------------
    def _vi_at(self, t_ns: int) -> np.ndarray | None:
        """VI local pose nearest `t_ns` within max_pair_dt_ms (never interpolated past the tolerance)."""
        if not self._ring: return None
        i = min(range(len(self._ring)), key=lambda k: abs(self._ring[k][0] - t_ns))
        if abs(self._ring[i][0] - t_ns) > self.cfg.max_pair_dt_ms * 1e6:
            self._n_pair_dt_rejected += 1; return None
        return self._ring[i][1]

    def get_pose(self, now_ns: int | None = None) -> WristPose | None:
        with self._lock:
            return self._get_pose(now_ns)

    def _get_pose(self, now_ns: int | None = None) -> WristPose | None:
        vi, rg = self._vi_last, self._rgbd_last
        if vi is None and rg is None: return self._last_out
        t_now = int(now_ns) if now_ns is not None else int(max(vi.timestamp_ns if vi else 0, rg.timestamp_ns if rg else 0))

        if self.mode is FusionMode.RGBD_ONLY:
            base = rg
            T_base = None if rg is None or not np.all(np.isfinite(rg.position_xyz_m)) else rg.T()
        else:
            base = vi
            T_base = None if self._vi_T_local is None else self._vi_T_local

        C = self.corr.step(t_now, clutched=self._clutched) if self.mode is FusionMode.FUSED else np.zeros(3)

        anchored = bool(self.align.state.aligned) and self.mode is FusionMode.FUSED
        unanchored_s = 0.0 if self._unanchored_since_ns is None else max(0.0, (t_now - self._unanchored_since_ns) / 1e9)
        rgbd_age_s = float("inf") if self._rgbd_ok_ns is None else max(0.0, (t_now - self._rgbd_ok_ns) / 1e9)
        vi_health = None if vi is None else self.sup.age_health(vi.health, max(0.0, (t_now - vi.timestamp_ns) / 1e6))
        rgbd_health = None if rg is None else rg.health
        if self.mode is FusionMode.RGBD_ONLY:
            # ablation A has no VI branch at all: the RGB-D pose IS the tracker, graded by its own supervisor
            vi_health = None if rg is None else self.sup.age_health(rg.health, max(0.0, (t_now - rg.timestamp_ns) / 1e6))
            anchored, unanchored_s, rgbd_age_s = True, 0.0, 0.0
        elif self.mode is FusionMode.VI_ONLY:
            # ablation B is unanchored BY DESIGN: the missing anchor is the experiment, not a fault to grade on
            rgbd_health, anchored, unanchored_s, rgbd_age_s = None, True, 0.0, 0.0

        state, health, reason = fusion_health(vi=vi_health, rgbd=rgbd_health, has_pose=T_base is not None or self._last_out is not None,
                                              anchored=anchored, unanchored_s=unanchored_s, rgbd_age_s=rgbd_age_s, cfg=self.cfg.state)
        if self.mode is FusionMode.VI_ONLY and state is FusionState.TRACKING_OK: reason = "vi_only"   # not a dropout

        if T_base is None or health == TrackingHealth.LOST:
            if self._last_out is None:
                return None                                            # nothing yet: the coordinator reads that as LOST
            hold = self._last_out
            out = WristPose(t_now, hold.position_xyz_m.copy(), hold.quaternion_xyzw.copy(), np.zeros(3), np.zeros(3),
                            hold.tracking_state, TrackingHealth.LOST, hold.confidence, SOURCE,
                            self._extra(state, reason, vi, rg, C, held=True))
            self._last_out = out
            return out

        T = make_T(T_base[:3, :3], T_base[:3, 3] + C)
        t_pose = int(base.timestamp_ns)
        # the fusion clock runs faster than either sensor, so most ticks re-emit the same sample. Velocity is a
        # property of the SAMPLE, not of the tick: recomputing it as zero on every repeat would make a moving hand
        # read as still to the stability gate half the time.
        v, w = self._last_vw
        if self._prev_out is not None and t_pose > self._prev_out[0]:
            dt = (t_pose - self._prev_out[0]) / 1e9
            v = (T[:3, 3] - self._prev_out[1][:3, 3]) / dt
            w = Rotation.from_matrix(self._prev_out[1][:3, :3].T @ T[:3, :3]).as_rotvec() / dt
            self._prev_out = (t_pose, T.copy()); self._last_vw = (v, w)
        elif self._prev_out is None:
            self._prev_out = (t_pose, T.copy())
        out = WristPose.from_T(t_pose, T, linear_velocity_xyz=v, angular_velocity_xyz=w,
                               tracking_state=base.tracking_state if base is not None else TrackingState.TRACKING,
                               health=health, confidence=base.confidence if base is not None else None,
                               source=SOURCE, extra=self._extra(state, reason, vi, rg, C, held=False))
        self._last_out = out
        return out

    # ---- logging (section 18) ----------------------------------------------------------------------------
    def _extra(self, state: FusionState, reason: str, vi: WristPose | None, rg: WristPose | None, C: np.ndarray, *, held: bool) -> dict:
        a = self.align.state
        e = dict(fusion_state=state.value, fusion_reason=reason, fusion_mode=self.mode.value, held=bool(held),
                 vi_health=vi.health.value if vi else "", rgbd_health=rg.health.value if rg else "",
                 vi_confidence=None if vi is None else vi.confidence, rgbd_confidence=None if rg is None else rg.confidence,
                 vi_t_ns=None if vi is None else int(vi.timestamp_ns), rgbd_t_ns=None if rg is None else int(rg.timestamp_ns),
                 anchored=bool(a.aligned), align_reason=a.reason, align_rms_m=a.rms_m, align_pairs=a.n_pairs,
                 align_span_m=a.span_m, align_rotation_span_deg=a.rotation_span_deg,
                 lever_arm_m=float(np.linalg.norm(a.lever_arm_m)), lever_arm_observable=bool(a.lever_arm_observable),
                 pairs=self._n_pairs, pairs_rejected_dt=self._n_pair_dt_rejected, n_vi=self._n_vi, n_rgbd=self._n_rgbd,
                 clutched=bool(self._clutched))
        e.update(self.corr.stats()); e.update(self.local.stats())
        # |e| is how much drift the anchor SEES; |e − C| is how much of it is still in the pose the robot gets. The
        # second is the one that says whether fusion is working — the first stays large by definition while it does.
        e["residual_corrected_m"] = float("nan") if self._last_e is None else float(np.linalg.norm(self._last_e - C))
        if vi is not None and np.all(np.isfinite(vi.position_xyz_m)):
            e.update(vi_map_x=float(vi.position_xyz_m[0]), vi_map_y=float(vi.position_xyz_m[1]), vi_map_z=float(vi.position_xyz_m[2]))
        if vi is not None:
            # The three poses the first-hardware-run gate is read off, in one row (section 3):
            #   visual_raw_*  the backend's answer BEFORE its global step   (only backends that report it)
            #   vi_map_*      the pose on the wire, after the global step
            #   vi_*          T_local_control — the only one the robot follows, and the one that may not jump
            # plus the per-frame diagnostics that say WHY a step happened. Without visual_raw a loop closure and a
            # tracker losing the hand produce the same row.
            for k in ("visual_raw_x", "visual_raw_y", "visual_raw_z", "d_visual_raw_m", "map_update", "map_update_m",
                      "keyframe", "server_ms", "net_ms", "cap_to_send_ms", "link_ms", "server_state"):
                if k in vi.extra: e[f"vi_{k}" if k.startswith(("map_", "server", "net", "cap", "link", "key")) else k] = vi.extra[k]
        if self._vi_T_local is not None:
            p = self._vi_T_local[:3, 3]; e.update(vi_x=float(p[0]), vi_y=float(p[1]), vi_z=float(p[2]))
        if rg is not None and np.all(np.isfinite(rg.position_xyz_m)):
            p = rg.position_xyz_m
            e.update(rgbd_x=float(p[0]), rgbd_y=float(p[1]), rgbd_z=float(p[2]))
            if a.aligned:
                q = self.align.anchor_in_world(p)
                e.update(rgbd_w_x=float(q[0]), rgbd_w_y=float(q[1]), rgbd_w_z=float(q[2]))
            for k in ("arm_pose_health", "arm_pose_reason", "hand_health", "n_palm_landmarks", "palm_scale_ratio"):
                if k in rg.extra: e[f"rgbd_{k}"] = rg.extra[k]
        return e

    def stats(self) -> dict:
        return dict(mode=self.mode.value, side=self.cfg.side, vi_samples=self._n_vi, rgbd_samples=self._n_rgbd,
                    pairs=self._n_pairs, pairs_rejected_dt=self._n_pair_dt_rejected,
                    alignment=self.align.to_dict(), correction=self.corr.stats(), local_pose=self.local.stats())


def build_fused_wrist_pose_provider(cfg: FusedWristConfig | None = None, *, vi_provider=None, rgbd_provider=None) -> FusedWristPoseProvider:
    return FusedWristPoseProvider(cfg, vi_provider=vi_provider, rgbd_provider=rgbd_provider)
