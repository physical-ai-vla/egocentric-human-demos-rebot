"""Canonical model-facing action: ΔT_{t,k} = inv(T_TCP,t) · T_TCP,t+k  →  [dxyz_local, rotvec] (6), clamped at the episode
end exactly like robot `relative_ee_state` (robot-cockpit/convert/rebot_ee.py) and handumi-sw `local_delta`.
Per-hand relative motion is valid in each hand's own VIO world — no shared L/R frame is needed or assumed."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .se3 import inv_T


def relative_tcp(Ts: np.ndarray, lead: int, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(N,6) inv(T_t) T_{min(t+lead, N-1)} and (N,) validity (both endpoints valid). Clamped at the end like the robot."""
    Ts = np.asarray(Ts, np.float64).reshape(-1, 4, 4); N = len(Ts)
    out = np.zeros((N, 6)); ok = np.zeros(N, bool)
    v = np.ones(N, bool) if valid is None else np.asarray(valid, bool)
    for t in range(N):
        f = min(t + lead, N - 1)
        if not (v[t] and v[f]): continue
        d = inv_T(Ts[t]) @ Ts[f]
        out[t, :3] = d[:3, 3]; out[t, 3:] = Rotation.from_matrix(d[:3, :3]).as_rotvec(); ok[t] = True
    return out, ok


def relative_chunk(Ts: np.ndarray, t: int, horizon: int, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(horizon,6) inv(T_t) T_{t+1..t+horizon} (clamped), plus validity per step — the exporter's action chunk."""
    Ts = np.asarray(Ts, np.float64).reshape(-1, 4, 4); N = len(Ts)
    v = np.ones(N, bool) if valid is None else np.asarray(valid, bool)
    out = np.zeros((horizon, 6)); ok = np.zeros(horizon, bool)
    Ti = inv_T(Ts[t])
    for k in range(1, horizon + 1):
        f = min(t + k, N - 1)
        if not (v[t] and v[f]): continue
        d = Ti @ Ts[f]; out[k - 1, :3] = d[:3, 3]; out[k - 1, 3:] = Rotation.from_matrix(d[:3, :3]).as_rotvec(); ok[k - 1] = True
    return out, ok


def c_state_rows(T_left: np.ndarray, T_right: np.ndarray, grip_left: np.ndarray, grip_right: np.ndarray, lead: int,
                 valid_left: np.ndarray | None = None, valid_right: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(N,14) [L: dxyz rotvec grip(t+k) | R: ...] in the robot C_state layout (grip already in the exporter's convention),
    plus (N,) validity = both hands valid. Grip is taken at t+k (future) like relative_ee_state."""
    N = len(T_left); dl, okl = relative_tcp(T_left, lead, valid_left); dr, okr = relative_tcp(T_right, lead, valid_right)
    fut = np.minimum(np.arange(N) + lead, N - 1)
    out = np.zeros((N, 14), np.float32)
    out[:, 0:6] = dl; out[:, 6] = np.asarray(grip_left)[fut]; out[:, 7:13] = dr; out[:, 13] = np.asarray(grip_right)[fut]
    return out, okl & okr
