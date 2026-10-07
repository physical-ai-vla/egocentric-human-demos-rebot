"""Master-timeline resampling: every head C922 frame gets the nearest/interpolated wrist TCP pose and grip value.
Long gaps are never interpolated — a head frame farther than max_gap from valid samples is invalid."""
from __future__ import annotations
import numpy as np
from .se3 import interp_pose


def resample_poses(head_t_ns: np.ndarray, pose_t_ns: np.ndarray, Ts: np.ndarray, valid: np.ndarray, *, max_gap_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (Ts_head (H,4,4), valid_head (H,), gap_ns (H,)). Interpolates between the two neighbouring VALID poses when
    both are within max_gap; uses the single neighbour when only one is (nearest); invalid otherwise."""
    H = len(head_t_ns); out = np.tile(np.eye(4), (H, 1, 1)); ok = np.zeros(H, bool); gap = np.full(H, np.iinfo(np.int64).max, np.int64)
    vi = np.nonzero(np.asarray(valid, bool))[0]
    if len(vi) == 0: return out, ok, gap
    pt = np.asarray(pose_t_ns, np.int64)[vi]; pT = np.asarray(Ts)[vi]
    for h, t in enumerate(np.asarray(head_t_ns, np.int64)):
        k = int(np.searchsorted(pt, t))
        lo, hi = k - 1, k
        cands = []
        if lo >= 0: cands.append((abs(int(t - pt[lo])), lo))
        if hi < len(pt): cands.append((abs(int(pt[hi] - t)), hi))
        if not cands: continue
        near = min(cands)
        gap[h] = near[0]
        if near[0] > max_gap_ns: continue
        if lo >= 0 and hi < len(pt) and (t - pt[lo]) <= max_gap_ns and (pt[hi] - t) <= max_gap_ns and pt[hi] > pt[lo]:
            a = (t - pt[lo]) / (pt[hi] - pt[lo]); out[h] = interp_pose(pT[lo], pT[hi], a)
        else:
            out[h] = pT[near[1]]
        ok[h] = True
    return out, ok, gap


def resample_scalar(head_t_ns: np.ndarray, t_ns: np.ndarray, values: np.ndarray, *, max_gap_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-sample scalar (grip) with gap guard; NaN + invalid beyond max_gap."""
    H = len(head_t_ns); out = np.full(H, np.nan); ok = np.zeros(H, bool)
    t = np.asarray(t_ns, np.int64); v = np.asarray(values, np.float64)
    fin = np.isfinite(v); t, v = t[fin], v[fin]
    if len(t) == 0: return out, ok
    for h, th in enumerate(np.asarray(head_t_ns, np.int64)):
        k = int(np.searchsorted(t, th)); best = None
        for c in (k - 1, k):
            if 0 <= c < len(t):
                d = abs(int(t[c] - th))
                if best is None or d < best[0]: best = (d, c)
        if best and best[0] <= max_gap_ns: out[h] = v[best[1]]; ok[h] = True
    return out, ok
