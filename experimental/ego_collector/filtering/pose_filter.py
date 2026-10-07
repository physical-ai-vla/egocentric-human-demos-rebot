"""Conservative pose smoothing for *derived* data (raw poses are never overwritten).

Translation: One Euro filter (low lag, adaptive). Rotation: SLERP towards the new sample
with the same adaptive alpha. Gaps (invalid frames) reset the filter so nothing is invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ego_collector.tracking.transforms import normalize_quat


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2 * np.pi * max(cutoff_hz, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


@dataclass
class OneEuroConfig:
    # Deliberately light: at 30 Hz with a 0.3 m/s hand this adds < 1 frame of lag (human timing is kept).
    min_cutoff_hz: float = 4.0  # lower = smoother at rest
    beta: float = 2.0  # higher = less lag during fast motion
    d_cutoff_hz: float = 1.0
    rotation_cutoff_hz: float = 5.0


def one_euro_pose7(poses7: np.ndarray, valid: np.ndarray, timestamps_ns: np.ndarray, config: OneEuroConfig = OneEuroConfig()) -> np.ndarray:
    """Return filtered pose7 array (NaN where invalid). Filter state resets after each gap."""
    poses7 = np.asarray(poses7, dtype=np.float64)
    valid = np.asarray(valid, bool)
    t = np.asarray(timestamps_ns, dtype=np.float64) / 1e9
    out = np.full_like(poses7, np.nan)
    x_prev = dx_prev = None
    r_prev = None
    t_prev = None
    for i in range(len(poses7)):
        if not valid[i]:
            x_prev = dx_prev = r_prev = t_prev = None
            continue
        x = poses7[i, :3]
        r = Rotation.from_quat(normalize_quat(poses7[i, 3:7]))
        if x_prev is None or t_prev is None:
            out[i] = np.concatenate([x, r.as_quat()])
            x_prev, dx_prev, r_prev, t_prev = x.copy(), np.zeros(3), r, t[i]
            continue
        dt = max(t[i] - t_prev, 1e-3)
        dx = (x - x_prev) / dt
        a_d = _alpha(config.d_cutoff_hz, dt)
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = config.min_cutoff_hz + config.beta * np.linalg.norm(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * x_prev
        a_r = _alpha(config.rotation_cutoff_hz + config.beta * np.linalg.norm(dx_hat), dt)
        rel = r_prev.inv() * r
        r_hat = r_prev * Rotation.from_rotvec(rel.as_rotvec() * a_r)
        out[i] = np.concatenate([x_hat, normalize_quat(r_hat.as_quat())])
        x_prev, dx_prev, r_prev, t_prev = x_hat, dx_hat, r_hat, t[i]
    return out


def interpolate_short_gaps(poses7: np.ndarray, valid: np.ndarray, timestamps_ns: np.ndarray, *, max_gap_frames: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Fill gaps of <= max_gap_frames by lerp/slerp between neighbours. Returns (poses, filled_mask)."""
    from ego_collector.tracking.transforms import slerp_pose7

    poses7 = np.asarray(poses7, dtype=np.float64).copy()
    valid = np.asarray(valid, bool).copy()
    filled = np.zeros(len(valid), bool)
    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap_frames:
            for k in range(a + 1, b):
                alpha = (timestamps_ns[k] - timestamps_ns[a]) / max(timestamps_ns[b] - timestamps_ns[a], 1)
                poses7[k] = slerp_pose7(poses7[a], poses7[b], float(alpha))
                filled[k] = True
    return poses7, filled


__all__ = ["OneEuroConfig", "interpolate_short_gaps", "one_euro_pose7"]
