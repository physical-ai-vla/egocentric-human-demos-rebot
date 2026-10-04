"""Timestamp-safe pose / gripper sampling on a raw trajectory.

Position: linear.  Rotation: quaternion SLERP.  Gripper: linear.  A query time is VALID only if it falls on a valid
raw sample, or between two consecutive VALID raw samples at most ``max_gap_s`` apart.  Nothing is extrapolated,
padded or held: a query outside the valid raw support is invalid (pose = identity placeholder, gripper = NaN).
State history (t - UMI_DT) and action targets (t + k UMI_DT) are evaluated with ``PoseTrack.sample`` on the RAW
track, never on the 15 Hz rows (design decision D1).
"""
import numpy as np
from ..geometry.rotation6d import matrix_to_quaternion, quaternion_to_matrix, slerp
from ..geometry.transforms import make_T


class PoseTrack:
    def __init__(self, t_s, T, valid, gripper, max_gap_s=0.050):
        t_s = np.asarray(t_s, np.float64); assert np.all(np.diff(t_s) > 0), "raw timestamps must be strictly increasing"
        self.t, self.valid, self.max_gap = t_s, np.asarray(valid, bool), float(max_gap_s)
        T = np.asarray(T, np.float64); self.p = T[:, :3, 3]
        self.q = matrix_to_quaternion(T[:, :3, :3])
        self.g = np.asarray(gripper, np.float64)
        self.valid &= np.isfinite(self.p).all(1) & np.isfinite(self.g)

    def sample(self, times):
        """-> T (m,4,4), g (m,), ok (m,)"""
        x = np.atleast_1d(np.asarray(times, np.float64)); n = len(self.t)
        j = np.clip(np.searchsorted(self.t, x, side="right"), 1, n - 1); a = j - 1; b = j
        exact_a = np.isclose(x, self.t[a], rtol=0, atol=1e-9); exact_b = np.isclose(x, self.t[b], rtol=0, atol=1e-9)
        inside = (x >= self.t[0] - 1e-9) & (x <= self.t[-1] + 1e-9)
        ok_interp = self.valid[a] & self.valid[b] & (self.t[b] - self.t[a] <= self.max_gap + 1e-9)
        ok = inside & ((exact_a & self.valid[a]) | (exact_b & self.valid[b]) | ok_interp)
        w = np.clip((x - self.t[a]) / (self.t[b] - self.t[a]), 0.0, 1.0)
        w = np.where(exact_a, 0.0, np.where(exact_b, 1.0, w))
        p = (1 - w)[:, None] * self.p[a] + w[:, None] * self.p[b]
        q = slerp(self.q[a], self.q[b], w)
        g = (1 - w) * self.g[a] + w * self.g[b]
        T = make_T(quaternion_to_matrix(q), p)
        T[~ok] = np.eye(4); g = np.where(ok, g, np.nan)
        return T, g, ok


def canonical_times(t_start, t_end, fps=15.0):
    """row grid t_start + n / fps, n = 0.. (t_end inclusive within 1 us)"""
    n = int(np.floor((t_end - t_start) * fps + 1e-6)) + 1
    return t_start + np.arange(n) / fps


def resample_episode(timestamps, poses, gripper, target_fps=15, valid=None, max_gap_s=0.050, t_start=None):
    """spec API: resample one arm's raw track onto the canonical row grid.
    poses: (n,4,4) or dict(position=(n,3), quaternion=(n,4) wxyz).  Returns (times, T, g, ok)."""
    if isinstance(poses, dict):
        poses = make_T(quaternion_to_matrix(poses["quaternion"]), poses["position"])
    valid = np.ones(len(timestamps), bool) if valid is None else valid
    tr = PoseTrack(timestamps, poses, valid, gripper, max_gap_s)
    v = np.flatnonzero(tr.valid); t0 = tr.t[v[0]] if t_start is None else t_start
    times = canonical_times(t0, tr.t[v[-1]], target_fps)
    T, g, ok = tr.sample(times)
    return times, T, g, ok
