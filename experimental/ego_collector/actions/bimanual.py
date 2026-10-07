"""Left/right relative features: T_left_right, distance, relative xyz/rotation, relative velocity."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ego_collector.tracking.transforms import pose7_to_T, relative_T


def bimanual_features(timestamps_ns: np.ndarray, left: np.ndarray, right: np.ndarray, left_valid: np.ndarray, right_valid: np.ndarray) -> dict[str, np.ndarray]:
    n = len(timestamps_ns)
    both = np.asarray(left_valid, bool) & np.asarray(right_valid, bool)
    rel_xyz = np.full((n, 3), np.nan)
    rel_rotvec = np.full((n, 3), np.nan)
    rel_quat = np.full((n, 4), np.nan)
    dist = np.full(n, np.nan)
    for i in np.flatnonzero(both):
        T = relative_T(pose7_to_T(left[i]), pose7_to_T(right[i]))
        rel_xyz[i] = T[:3, 3]
        r = Rotation.from_matrix(T[:3, :3])
        rel_rotvec[i] = r.as_rotvec()
        rel_quat[i] = r.as_quat()
        dist[i] = np.linalg.norm(np.asarray(right[i][:3]) - np.asarray(left[i][:3]))
    t = np.asarray(timestamps_ns, dtype=np.float64) / 1e9
    rel_vel = np.full((n, 3), np.nan)
    dist_rate = np.full(n, np.nan)
    for i in range(n - 1):
        if both[i] and both[i + 1] and t[i + 1] > t[i]:
            dt = t[i + 1] - t[i]
            rel_vel[i] = (rel_xyz[i + 1] - rel_xyz[i]) / dt
            dist_rate[i] = (dist[i + 1] - dist[i]) / dt
    cols = {"timestamp_ns": np.asarray(timestamps_ns, dtype=np.int64), "both_valid": both, "left_right_distance": dist, "left_right_distance_rate": dist_rate}
    for j, k in enumerate(("x", "y", "z")):
        cols[f"relative_{k}"] = rel_xyz[:, j]
        cols[f"relative_r{k}"] = rel_rotvec[:, j]
        cols[f"relative_v{k}"] = rel_vel[:, j]
    for j, k in enumerate(("qx", "qy", "qz", "qw")):
        cols[f"relative_{k}"] = rel_quat[:, j]
    return cols


__all__ = ["bimanual_features"]
