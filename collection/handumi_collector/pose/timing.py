"""Timestamp pipeline for VIO input.

Two clocks meet here: camera frames carry host monotonic `capture_ns`; IMU samples carry the Teensy `device_timestamp_us`
(precise) and a jittery `host_receive_ns`. We map device->host with a robust affine fit (USB latency is one-sided, so the
fit is anchored on the low-latency envelope) and then apply the *measured* camera<->IMU offset from pose.yaml. The offset
is never assumed to be zero: `offsets.measured: false` is surfaced as a QA warning. `estimate_camera_imu_offset` measures it
by cross-correlating visual angular speed (from any pose backend) against gyro magnitude."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ClockFit:
    slope_ns_per_us: float       # ~1000
    offset_ns: float
    residual_std_ms: float
    n: int
    method: str = "robust_lower_envelope"

    def to_host_ns(self, device_us) -> np.ndarray:
        return (np.asarray(device_us, np.float64) * self.slope_ns_per_us + self.offset_ns).astype(np.int64)

    def to_dict(self) -> dict:
        return dict(slope_ns_per_us=self.slope_ns_per_us, offset_ns=self.offset_ns, residual_std_ms=round(self.residual_std_ms, 4), n=self.n,
                    drift_ppm=round((self.slope_ns_per_us / 1000.0 - 1.0) * 1e6, 2), method=self.method)


def fit_device_to_host(device_us: np.ndarray, host_ns: np.ndarray, *, envelope_percentile: float = 5.0) -> ClockFit:
    """host = slope*device + offset. Least squares for the slope (long baseline), intercept re-anchored to the low-latency
    envelope (percentile of residuals) because receive latency only ever adds delay."""
    d = np.asarray(device_us, np.float64); h = np.asarray(host_ns, np.float64)
    if len(d) < 2: raise ValueError("need >= 2 IMU samples for a clock fit")
    d0, h0 = d[0], h[0]
    A = np.vstack([d - d0, np.ones_like(d)]).T
    slope, icpt = np.linalg.lstsq(A, h - h0, rcond=None)[0]
    if len(d) < 50: slope = 1000.0 if abs(slope - 1000.0) > 50 else slope     # too short to trust a slope fit → nominal µs→ns
    res = (h - h0) - (slope * (d - d0) + icpt)
    icpt2 = icpt + np.percentile(res, envelope_percentile)
    offset = h0 + icpt2 - slope * d0
    return ClockFit(float(slope), float(offset), float(res.std() / 1e6), int(len(d)))


def visual_angular_speed(t_ns: np.ndarray, Rs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|rotvec(R_i^T R_{i+1})| / dt at midpoints, rad/s; input rotations are camera (or any rigid) frame in world."""
    t = np.asarray(t_ns, np.float64) / 1e9
    if len(t) < 2: return np.zeros(0), np.zeros(0)
    rel = Rotation.from_matrix(Rs[:-1]).inv() * Rotation.from_matrix(Rs[1:])
    dt = np.diff(t)
    ok = dt > 0
    return (0.5 * (t[:-1] + t[1:]))[ok] * 1e9, (rel.magnitude()[ok] / dt[ok])


def estimate_camera_imu_offset(t_cam_ns, Rs_cam, t_imu_ns, gyro, *, max_offset_ms: float = 100.0, step_ms: float = 0.5, gyro_bias=None) -> dict:
    """Camera<->IMU time offset by matching *the same quantity* on both sides: per frame-interval rotation magnitude.
    visual: |log(R_i^T R_{i+1})|;  IMU: |∫_{t_i - s}^{t_{i+1} - s} (ω - bias) dt| from the cumulative gyro integral, for each candidate
    shift s. Score = normalised correlation; the peak is refined parabolically. Returns offset_ms such that
    imu_time + offset_ms ≈ camera_time (add it to IMU host times), plus confidence (= peak correlation) and sharpness.
    Frame-independent (magnitudes), so it works before T_camera_imu is calibrated."""
    tc = np.asarray(t_cam_ns, np.float64); Rs = np.asarray(Rs_cam)
    ti = np.asarray(t_imu_ns, np.float64); g = np.asarray(gyro, np.float64)
    if len(tc) < 10 or len(ti) < 20: return dict(offset_ms=None, confidence=0.0, reason="too few samples")
    order = np.argsort(ti); ti, g = ti[order], g[order]
    # bias: use the still-window estimate when given. NOT the median of the whole episode (biased for non-zero-mean motion);
    # an unremoved constant bias only adds a near-constant to every per-frame magnitude, which the correlation ignores
    if gyro_bias is not None: g = g - np.asarray(gyro_bias, np.float64)
    theta = (Rotation.from_matrix(Rs[:-1]).inv() * Rotation.from_matrix(Rs[1:])).magnitude()
    ok = np.diff(tc) > 0; theta = theta[ok]; ta, tb = tc[:-1][ok], tc[1:][ok]
    if theta.std() < 1e-6: return dict(offset_ms=None, confidence=0.0, reason="no rotation in the visual trajectory")
    dt = np.diff(ti) / 1e9
    C = np.zeros_like(g); C[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * dt[:, None], axis=0)      # cumulative vector integral (rad)
    def imu_mag(shift_ns):
        lo = ta - shift_ns; hi = tb - shift_ns
        inside = (lo >= ti[0]) & (hi <= ti[-1])
        if inside.sum() < 10: return None, inside
        Ca = np.column_stack([np.interp(lo[inside], ti, C[:, k]) for k in range(3)]); Cb = np.column_stack([np.interp(hi[inside], ti, C[:, k]) for k in range(3)])
        return np.linalg.norm(Cb - Ca, axis=1), inside
    shifts = np.arange(-max_offset_ms, max_offset_ms + step_ms, step_ms); scores = np.full(len(shifts), -np.inf)
    for k, s_ms in enumerate(shifts):
        phi, inside = imu_mag(s_ms * 1e6)
        if phi is None: continue
        th = theta[inside]
        if th.std() < 1e-9 or phi.std() < 1e-9: continue
        scores[k] = float(np.mean((th - th.mean()) / th.std() * (phi - phi.mean()) / phi.std()))
    k = int(np.argmax(scores)); best = scores[k]
    if not np.isfinite(best): return dict(offset_ms=None, confidence=0.0, reason="no overlap")
    off = shifts[k]
    if 0 < k < len(shifts) - 1 and np.isfinite(scores[k - 1]) and np.isfinite(scores[k + 1]):        # parabolic refinement
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]; den = (y0 - 2 * y1 + y2)
        if den < 0: off = shifts[k] + 0.5 * (y0 - y2) / den * step_ms
    far = scores[np.abs(shifts - shifts[k]) >= 20.0]
    sharpness = float(best - np.max(far)) if len(far) and np.isfinite(np.max(far)) else float(best)
    return dict(offset_ms=float(off), peak_corr=float(best), sharpness=sharpness, confidence=float(np.clip(best, 0.0, 1.0)), n_visual=int(len(theta)), n_imu=int(len(ti)),
                note="offset_ms: add to IMU host-clock times to land on the camera clock (pose.yaml offsets.<side>_camera_imu_offset_ms)")


def imu_host_times_ns(imu, clock: ClockFit, camera_imu_offset_ns: int) -> np.ndarray:
    """IMU sample times on the camera (host) clock, offset-corrected: t = fit(device_us) + offset."""
    return clock.to_host_ns(imu.device_us) + int(camera_imu_offset_ns)


def interleave(frame_times_ns: np.ndarray, imu_times_ns: np.ndarray):
    """Yield ('imu', i) / ('image', j) in time order with every IMU sample <= a frame's time delivered before that frame."""
    i = j = 0; n_i, n_f = len(imu_times_ns), len(frame_times_ns)
    while j < n_f:
        while i < n_i and imu_times_ns[i] <= frame_times_ns[j]:
            yield "imu", i; i += 1
        yield "image", j; j += 1
    while i < n_i:
        yield "imu", i; i += 1
