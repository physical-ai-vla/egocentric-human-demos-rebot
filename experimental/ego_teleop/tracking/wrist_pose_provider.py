"""WristPoseProvider on top of any handumi_collector.pose PoseEstimator (tag-free VIO, mock, ...).

    camera frames (host ns) ─┐
    IMU samples (host ns)  ──┴─> [camera<->IMU time offset] -> PoseEstimator -> T_W_C -> T_W_H = T_W_C · inv(T_H_C)
                                                                                  └-> TrackingSupervisor -> health

The supervisor is the ONLY place that turns backend state / age / confidence into TRACKING_OK|DEGRADED|LOST. It never
extrapolates position: when frames stop arriving the last pose is reported with health LOST after `lost_after_ms`
(IMU-only dead reckoning of XYZ is deliberately impossible through this interface — invariant A)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from handumi_collector.pose.estimator import PoseEstimate, PoseEstimator, TrackingState
from ..transforms.se3 import inv_T
from .interfaces import TrackingHealth, WristPose


@dataclass
class TrackingSupervisorConfig:
    degraded_after_ms: float = 80.0        # no valid pose newer than this -> DEGRADED
    lost_after_ms: float = 250.0           # ... than this -> LOST (arm target frozen)
    min_confidence_ok: float = 0.5         # backend confidence below -> DEGRADED (if backend provides it)
    min_features_ok: int = 80              # feature count below -> DEGRADED (if provided)
    recover_after_n_ok: int = 3            # consecutive good estimates required to leave LOST (hysteresis, avoids flicker)
    max_jump_m: float = 0.15               # frame-to-frame translation jump larger than this -> LOST for that estimate
    max_jump_deg: float = 25.0
    # An ANNOUNCED map correction is not a tracking failure and must not be graded as a jump. Without this, a loop
    # closure bigger than max_jump_m is graded LOST, `FusedWristPoseProvider._push_vi` returns early, and
    # `LocalPoseContinuity` — the one thing that would have made it continuous — never sees it. The pose then has an
    # unabsorbed offset in it for the rest of the session. The keys are the same ones the absorber reads, and
    # `LocalPoseConfig.max_absorb_m` is what stops a backend flagging its way past every sanity check.
    map_update_keys: tuple = ("map_update", "loop_closure", "global_correction")


class TrackingSupervisor:
    def __init__(self, cfg: TrackingSupervisorConfig | None = None) -> None:
        self.cfg = cfg or TrackingSupervisorConfig(); self._ok_streak = 0; self._was_lost = True

    def grade_estimate(self, est: PoseEstimate, prev_T: np.ndarray | None, dt_s: float) -> TrackingHealth:
        """Health of one fresh estimate (age = 0)."""
        c = self.cfg
        if not est.valid: self._ok_streak = 0; self._was_lost = True; return TrackingHealth.LOST
        announced = any(bool((est.extra or {}).get(k)) for k in c.map_update_keys)
        if prev_T is not None and dt_s > 0 and not announced:
            d = np.linalg.norm(est.T_world_camera[:3, 3] - prev_T[:3, 3])
            ang = np.degrees(Rotation.from_matrix(prev_T[:3, :3].T @ est.T_world_camera[:3, :3]).magnitude())
            if d > c.max_jump_m or ang > c.max_jump_deg: self._ok_streak = 0; self._was_lost = True; return TrackingHealth.LOST
        degraded = est.tracking_state == TrackingState.DEGRADED
        if est.confidence is not None and est.confidence < c.min_confidence_ok: degraded = True
        if est.num_features is not None and est.num_features < c.min_features_ok: degraded = True
        self._ok_streak += 1                                   # consecutive VALID (non-jump) estimates
        if self._was_lost:
            if self._ok_streak >= c.recover_after_n_ok: self._was_lost = False
            else: return TrackingHealth.LOST                   # hysteresis: stay LOST until the stream proves itself
        return TrackingHealth.DEGRADED if degraded else TrackingHealth.OK

    def age_health(self, base: TrackingHealth, age_ms: float) -> TrackingHealth:
        if age_ms > self.cfg.lost_after_ms: return TrackingHealth.LOST
        if age_ms > self.cfg.degraded_after_ms and base == TrackingHealth.OK: return TrackingHealth.DEGRADED
        return base


class EstimatorWristPoseProvider:
    def __init__(self, estimator: PoseEstimator, *, T_H_C: np.ndarray, supervisor: TrackingSupervisor | None = None,
                 camera_imu_offset_ns: int = 0, source: str | None = None) -> None:
        """T_H_C: camera pose in the human wrist control frame (calibration wrist_camera_<side>). camera_imu_offset_ns:
        imu_host_ns + offset ≈ camera_host_ns (from timing.estimate_camera_imu_offset); applied to IMU samples here so the
        estimator sees ONE clock."""
        self.est = estimator; self.T_C_H = inv_T(T_H_C); self.sup = supervisor or TrackingSupervisor()
        self.offset_ns = int(camera_imu_offset_ns); self.source = source or getattr(getattr(estimator, "info", None), "name", type(estimator).__name__)
        self._prev: tuple[int, np.ndarray] | None = None       # (t_ns, T_W_C) of the last VALID estimate
        self._last: WristPose | None = None
        self._n_frames = 0; self._n_imu = 0

    # ---- feeding (called by sensor threads / offline replay, timestamps in host ns) ----
    def push_imu(self, t_host_ns: int, gyro_rad_s, accel_m_s2) -> None:
        self._n_imu += 1; self.est.push_imu(int(t_host_ns) + self.offset_ns, gyro_rad_s, accel_m_s2)

    def push_image(self, t_host_ns: int, frame_index: int, image_bgr) -> WristPose:
        self._n_frames += 1
        est = self.est.push_image(int(t_host_ns), int(frame_index), image_bgr)
        return self.ingest_estimate(est)

    def ingest_estimate(self, est: PoseEstimate) -> WristPose:
        """Feed a PoseEstimate produced elsewhere (e.g. drained from handumi_collector.pose.live.LivePoseRunner.buffer)."""
        prev_T = self._prev[1] if self._prev else None
        dt = (est.timestamp_ns - self._prev[0]) / 1e9 if self._prev else 0.0
        health = self.sup.grade_estimate(est, prev_T, dt)
        # The backend's own `extra` MUST survive to here. `LocalPoseContinuity` decides whether a discontinuity is an
        # announced map correction or a tracking failure by looking for `map_update` / `loop_closure` /
        # `global_correction` in exactly this dict (fused_wrist.yaml provider.local.map_update_keys). Rebuilding it
        # from scratch, as this did, silently disabled loop-closure absorption for every backend that announces one:
        # the flag was set, sent, parsed, and then dropped one layer above the only code that reads it.
        extra = dict(est.extra or {})
        extra.update(num_features=est.num_features, reprojection_error=est.reprojection_error)
        if health == TrackingHealth.LOST or est.T_world_camera is None:
            if est.T_world_camera is not None: self._prev = (est.timestamp_ns, est.T_world_camera.copy())   # next jump check against the NEW position
            if self._last is not None:      # hold last pose, zero velocity, LOST
                wp = WristPose(est.timestamp_ns, self._last.position_xyz_m.copy(), self._last.quaternion_xyzw.copy(), np.zeros(3), np.zeros(3),
                               est.tracking_state, TrackingHealth.LOST, est.confidence, self.source, extra)
            else:
                wp = WristPose(est.timestamp_ns, np.full(3, np.nan), np.array([0, 0, 0, 1.0]), np.zeros(3), np.zeros(3),
                               est.tracking_state, TrackingHealth.LOST, est.confidence, self.source, extra)
            self._last = wp; return wp
        T_W_C = est.T_world_camera; T_W_H = T_W_C @ self.T_C_H
        v = np.zeros(3); w = np.zeros(3)
        if prev_T is not None and dt > 1e-6:
            prev_H = prev_T @ self.T_C_H
            v = (T_W_H[:3, 3] - prev_H[:3, 3]) / dt
            w = Rotation.from_matrix(prev_H[:3, :3].T @ T_W_H[:3, :3]).as_rotvec() / dt
        wp = WristPose.from_T(est.timestamp_ns, T_W_H, linear_velocity_xyz=v, angular_velocity_xyz=w, tracking_state=est.tracking_state,
                              health=health, confidence=est.confidence, source=self.source, extra=extra)
        self._prev = (est.timestamp_ns, T_W_C.copy()); self._last = wp
        return wp

    # ---- consumer side (coordinator, 30 Hz) ----
    def get_pose(self, now_ns: int | None = None) -> WristPose | None:
        if self._last is None: return None
        if now_ns is None: return self._last
        age_ms = (int(now_ns) - self._last.timestamp_ns) / 1e6
        h = self.sup.age_health(self._last.health, age_ms)
        if h == self._last.health: return self._last
        wp = WristPose(self._last.timestamp_ns, self._last.position_xyz_m, self._last.quaternion_xyzw,
                       np.zeros(3) if h == TrackingHealth.LOST else self._last.linear_velocity_xyz,
                       np.zeros(3) if h == TrackingHealth.LOST else self._last.angular_velocity_xyz,
                       self._last.tracking_state, h, self._last.confidence, self.source, dict(self._last.extra, age_ms=age_ms))
        return wp

    def stats(self) -> dict:
        return dict(frames=self._n_frames, imu=self._n_imu, backend=self.source, quality=self.est.get_quality())
