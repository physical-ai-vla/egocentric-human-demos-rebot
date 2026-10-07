"""IMU-side processing used by the pose pipeline: gyro bias, gyro integration (independent rotation check), and
still-window detection for the HOME protocol. Operates on raw sensor axes; T_camera_imu is applied by the caller."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def integrate_gyro(t_ns: np.ndarray, gyro: np.ndarray, t0_ns: int, t1_ns: int, bias: np.ndarray | None = None) -> np.ndarray:
    """Rotation R_imu(t0)->imu(t1) from raw gyro samples in [t0, t1] (body-frame composition, trapezoidal dt)."""
    t = np.asarray(t_ns, np.int64); g = np.asarray(gyro, np.float64)
    if bias is not None: g = g - np.asarray(bias, np.float64)
    m = (t >= t0_ns) & (t <= t1_ns)
    idx = np.nonzero(m)[0]
    if len(idx) < 2: return np.eye(3)
    ts = t[idx] / 1e9; gs = g[idx]
    R = Rotation.identity()
    for i in range(len(idx) - 1):
        dt = ts[i + 1] - ts[i]
        if dt <= 0: continue
        w = 0.5 * (gs[i] + gs[i + 1])
        R = R * Rotation.from_rotvec(w * dt)
    return R.as_matrix()


@dataclass
class StaticWindow:
    t0_ns: int
    t1_ns: int
    gyro_bias: np.ndarray          # mean raw gyro (rad/s)
    gyro_std_dps: float
    accel_mean: np.ndarray
    accel_norm_std: float
    n_samples: int

    @property
    def duration_s(self) -> float: return (self.t1_ns - self.t0_ns) / 1e9

    def to_dict(self) -> dict:
        return dict(t0_ns=int(self.t0_ns), t1_ns=int(self.t1_ns), duration_s=round(self.duration_s, 3), gyro_bias_dps=np.degrees(self.gyro_bias).round(4).tolist(),
                    gyro_std_dps=round(float(self.gyro_std_dps), 4), accel_mean=self.accel_mean.round(4).tolist(), accel_norm_std=round(float(self.accel_norm_std), 4),
                    n_samples=int(self.n_samples))


def window_stats(t_ns, gyro, accel, t0_ns, t1_ns) -> StaticWindow | None:
    m = (t_ns >= t0_ns) & (t_ns <= t1_ns)
    if m.sum() < 5: return None
    g, a = gyro[m], accel[m]
    return StaticWindow(int(t0_ns), int(t1_ns), g.mean(axis=0), float(np.degrees(np.linalg.norm(g.std(axis=0)))), a.mean(axis=0),
                        float(np.linalg.norm(a, axis=1).std()), int(m.sum()))


def still_mask(t_ns: np.ndarray, gyro: np.ndarray, accel: np.ndarray, *, gyro_thresh_dps: float, accel_std_thresh: float,
               win_s: float = 0.25) -> np.ndarray:
    """Per-sample stillness: in a sliding window, |gyro - window mean| small AND accel-norm std small. Bias-free by construction
    (uses local mean, so a constant gyro bias does not break detection)."""
    n = len(t_ns)
    if n == 0: return np.zeros(0, bool)
    dt = np.median(np.diff(t_ns)) / 1e9 if n > 1 else 1 / 400
    w = max(5, int(round(win_s / max(dt, 1e-4))))
    out = np.zeros(n, bool)
    an = np.linalg.norm(accel, axis=1)
    gthr = np.radians(gyro_thresh_dps)
    # centred rolling std via cumulative sums
    def roll_std(x):
        x = np.asarray(x, np.float64); c1 = np.cumsum(np.insert(x, 0, 0)); c2 = np.cumsum(np.insert(x * x, 0, 0))
        lo = np.clip(np.arange(n) - w // 2, 0, n); hi = np.clip(np.arange(n) + w // 2 + 1, 0, n); k = hi - lo
        mean = (c1[hi] - c1[lo]) / k; var = (c2[hi] - c2[lo]) / k - mean ** 2
        return np.sqrt(np.maximum(var, 0))
    gstd = np.sqrt(sum(roll_std(gyro[:, i]) ** 2 for i in range(3)))
    out = (gstd < gthr) & (roll_std(an) < accel_std_thresh)
    return out


def detect_home_windows(t_ns: np.ndarray, gyro: np.ndarray, accel: np.ndarray, cfg: dict, *, t_start_ns: int, t_stop_ns: int,
                        explicit_leave_ns: int | None = None, explicit_return_ns: int | None = None) -> tuple[StaticWindow | None, StaticWindow | None]:
    """HOME start window = first still run beginning at episode start; HOME end window = last still run ending at episode stop.
    Explicit operator marks (home_leave / home_return events) bound the search when present."""
    trim = int(float(cfg.get("edge_trim_s", 0.15)) * 1e9)
    min_d = int(float(cfg.get("min_duration_s", 0.8)) * 1e9); max_d = int(float(cfg.get("max_duration_s", 3.0)) * 1e9)
    if len(t_ns) < 10: return None, None
    m = still_mask(t_ns, gyro, accel, gyro_thresh_dps=float(cfg.get("gyro_thresh_dps", 3.0)), accel_std_thresh=float(cfg.get("accel_std_thresh_m_s2", 0.35)))
    runs = _runs(m)
    start = end = None
    # start: a still run that covers the episode start region
    s_lo = t_start_ns + trim; s_hi = (explicit_leave_ns if explicit_leave_ns else t_start_ns + max_d + trim)
    for a, b in runs:
        ta, tb = t_ns[a], t_ns[b - 1]
        if ta <= s_lo + int(0.3e9) and tb > ta:
            t0 = max(ta, s_lo); t1 = min(tb, s_hi, t0 + max_d)
            if t1 - t0 >= min_d: start = window_stats(t_ns, gyro, accel, t0, t1); break
    e_hi = t_stop_ns - trim; e_lo = (explicit_return_ns if explicit_return_ns else t_stop_ns - max_d - trim)
    for a, b in reversed(runs):
        ta, tb = t_ns[a], t_ns[b - 1]
        if tb >= e_hi - int(0.3e9) and tb > ta:
            t1 = min(tb, e_hi); t0 = max(ta, e_lo, t1 - max_d)
            if t1 - t0 >= min_d: end = window_stats(t_ns, gyro, accel, t0, t1); break
    return start, end


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs of True."""
    if len(mask) == 0: return []
    d = np.diff(mask.astype(np.int8)); starts = list(np.nonzero(d == 1)[0] + 1); ends = list(np.nonzero(d == -1)[0] + 1)
    if mask[0]: starts.insert(0, 0)
    if mask[-1]: ends.append(len(mask))
    return list(zip(starts, ends))
