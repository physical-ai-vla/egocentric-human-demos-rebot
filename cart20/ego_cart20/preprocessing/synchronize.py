"""Multi-stream synchronization on capture timestamps (seconds, one shared monotonic clock)."""
import numpy as np


def nearest_index(stream_t, query_t, tol_s):
    """index of the nearest stream sample for every query, -1 if |dt| > tol_s; also returns |dt|"""
    s = np.asarray(stream_t, np.float64); q = np.atleast_1d(np.asarray(query_t, np.float64))
    if len(s) == 0: return np.full(len(q), -1), np.full(len(q), np.inf)
    j = np.clip(np.searchsorted(s, q), 1, max(len(s) - 1, 1)) if len(s) > 1 else np.zeros(len(q), int)
    if len(s) > 1: j = np.where(np.abs(s[j - 1] - q) <= np.abs(s[j] - q), j - 1, j)
    dt = np.abs(s[j] - q)
    return np.where(dt <= tol_s, j, -1), dt


def task_start_time(tracks):
    """first raw time at which EVERY arm track has a valid sample there (sampled with its own interpolation rule):
    candidates are each track's valid raw timestamps, checked against all tracks"""
    cand = np.unique(np.concatenate([tr.t[tr.valid] for tr in tracks]))
    ok = np.ones(len(cand), bool)
    for tr in tracks: ok &= tr.sample(cand)[2]
    if not ok.any(): raise ValueError("no instant where all arms are valid")
    return float(cand[np.flatnonzero(ok)[0]])


def common_end_time(tracks):
    return float(min(tr.t[tr.valid][-1] for tr in tracks))
