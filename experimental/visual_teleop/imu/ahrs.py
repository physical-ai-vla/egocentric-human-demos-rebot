"""Streaming Mahony AHRS for one wrist IMU (2026-10-14, spec section 11/13). Same law as the batch
scripts/handumi_canonical_track.py:mahony (gyro + accelerometer-gravity PI correction, accel gated when |a| != g), but
sample-by-sample so it can run in the live loop at the IMU rate.

Output R_G_I: IMU body -> gravity frame G (+Z up). Roll/pitch are observable (gravity); YAW IS NOT (pure gyro integration,
drifts) -- the world yaw is set at teleop init and corrected only by vision (fusion/wrist_pose_fusion.py).
The IMU is never integrated to position (spec 11).

Integrate on the DEVICE clock (true dt between samples); host stamps carry USB batching (memory handumi-vio / imu_orientation.py).
Gyro bias: estimated from the first `bias_still_s` of quiet data; until then `ready` is False."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation

G0 = 9.80665


def _exp_so3(w: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(w).as_matrix()


def normalize_rotation(R: np.ndarray) -> np.ndarray:
    """Nearest rotation (SVD); the matrix-form of quaternion normalisation."""
    U, _, Vt = np.linalg.svd(R); M = U @ Vt
    if np.linalg.det(M) < 0: U[:, -1] *= -1; M = U @ Vt
    return M


@dataclass
class AhrsConfig:
    kp: float = 1.0
    ki: float = 0.01
    acc_tol_m_s2: float = 0.6        # use the accelerometer only when | |a| - g_rest | < tol (not accelerating)
    gyro_gate_rad_s: float = 2.0     # ... and the wrist is not spinning fast
    bias_still_s: float = 1.0
    max_dt_s: float = 0.05           # a gap longer than this is clamped (and counted) -- never integrate a USB stall blindly
    still_gyro_std: float = 0.05     # rad/s, bias window acceptance

    @classmethod
    def from_dict(cls, d: dict | None) -> "AhrsConfig":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


@dataclass
class MahonyAhrs:
    cfg: AhrsConfig = field(default_factory=AhrsConfig)
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    g_rest: float = G0
    ready: bool = False
    n: int = 0
    n_acc_used: int = 0
    n_gaps: int = 0
    last_device_us: int | None = None
    _integ: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _init_buf: list = field(default_factory=list)

    def update(self, device_us: int, gyro, acc) -> np.ndarray | None:
        """Feed one sample. Returns R_G_I once ready, else None. Non-finite samples are dropped (counted as gaps)."""
        g = np.asarray(gyro, np.float64); a = np.asarray(acc, np.float64)
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(a))): self.n_gaps += 1; return self.R.copy() if self.ready else None
        if not self.ready: return self._init(int(device_us), g, a)
        dt = (int(device_us) - self.last_device_us) * 1e-6 if self.last_device_us is not None else 0.0
        self.last_device_us = int(device_us)
        if dt <= 0: return self.R.copy()
        if dt > self.cfg.max_dt_s: self.n_gaps += 1; dt = self.cfg.max_dt_s
        w = g - self.bias; an = np.linalg.norm(a)
        if an > 1e-6 and abs(an - self.g_rest) < self.cfg.acc_tol_m_s2 and np.linalg.norm(w) < self.cfg.gyro_gate_rad_s:
            e = np.cross(a / an, self.R[2])          # measured up (body) x predicted up (body = third row of R_G_I)
            self._integ += self.cfg.ki * e * dt; w = w + self.cfg.kp * e + self._integ; self.n_acc_used += 1
        self.R = self.R @ _exp_so3(w * dt); self.n += 1
        if self.n % 100 == 0: self.R = normalize_rotation(self.R)
        return self.R.copy()

    def _init(self, t_us: int, g: np.ndarray, a: np.ndarray) -> None:
        self._init_buf.append((t_us, g, a)); self.last_device_us = t_us
        t0 = self._init_buf[0][0]
        if (t_us - t0) * 1e-6 < self.cfg.bias_still_s: return None
        G = np.array([x[1] for x in self._init_buf]); A = np.array([x[2] for x in self._init_buf])
        if np.linalg.norm(G.std(0)) > self.cfg.still_gyro_std:            # not still: slide the window
            self._init_buf = self._init_buf[len(self._init_buf) // 2:]; return None
        self.bias = G.mean(0); self.g_rest = float(np.linalg.norm(A, axis=1).mean())
        up = A.mean(0) / np.linalg.norm(A.mean(0))
        self.R = Rotation.align_vectors([[0, 0, 1.0]], [up])[0].as_matrix()      # body up -> G +Z, yaw = 0
        self.ready = True; self._init_buf = []
        return self.R.copy()

    @property
    def acc_correction_share(self) -> float: return self.n_acc_used / max(self.n, 1)

    def tilt_deg(self) -> float:
        """Angle between body +Z and G +Z (diagnostic)."""
        return float(np.degrees(np.arccos(np.clip(self.R[2, 2], -1, 1))))
