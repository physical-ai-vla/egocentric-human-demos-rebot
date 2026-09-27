"""SE(3) helpers for the pose pipeline. Convention: ``T_A_B`` = pose of frame B expressed in frame A (4x4).
pose7 = [x, y, z, qx, qy, qz, qw] (xyzw, qw >= 0) — identical to ego_collector.tracking.transforms and handumi-sw,
re-implemented here so the pose package never imports the legacy fiducial package."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

IDENTITY7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def normalize_quat(q) -> np.ndarray:
    q = np.asarray(q, np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12: return np.array([0.0, 0.0, 0.0, 1.0])
    q = q / n
    return -q if q[3] < 0 else q


def make_T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None: T[:3, :3] = np.asarray(R, np.float64)
    if t is not None: T[:3, 3] = np.asarray(t, np.float64).reshape(3)
    return T


def inv_T(T) -> np.ndarray:
    T = np.asarray(T, np.float64); R = T[:3, :3]
    return make_T(R.T, -R.T @ T[:3, 3])


def T_to_pose7(T) -> np.ndarray:
    T = np.asarray(T, np.float64)
    return np.concatenate([T[:3, 3], normalize_quat(Rotation.from_matrix(T[:3, :3]).as_quat())])


def pose7_to_T(p) -> np.ndarray:
    p = np.asarray(p, np.float64).reshape(7)
    return make_T(Rotation.from_quat(normalize_quat(p[3:7])).as_matrix(), p[:3])


def poses7_to_T(P) -> np.ndarray:
    P = np.asarray(P, np.float64).reshape(-1, 7)
    out = np.tile(np.eye(4), (len(P), 1, 1))
    if len(P):
        out[:, :3, :3] = Rotation.from_quat(P[:, 3:7]).as_matrix(); out[:, :3, 3] = P[:, :3]
    return out


def Ts_to_pose7(Ts) -> np.ndarray:
    Ts = np.asarray(Ts, np.float64).reshape(-1, 4, 4)
    if not len(Ts): return np.zeros((0, 7))
    q = Rotation.from_matrix(Ts[:, :3, :3]).as_quat()
    q[q[:, 3] < 0] *= -1
    return np.concatenate([Ts[:, :3, 3], q], axis=1)


def rotation_angle_deg(Ra, Rb=None) -> float:
    """Angle of Ra (if Rb None) or of Ra^T Rb, in degrees."""
    Ra = np.asarray(Ra, np.float64)[:3, :3]
    R = Ra if Rb is None else Ra.T @ np.asarray(Rb, np.float64)[:3, :3]
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def T_from_rotvec_t(rotvec, t) -> np.ndarray:
    return make_T(Rotation.from_rotvec(np.asarray(rotvec, np.float64).reshape(3)).as_matrix(), t)


def local_delta(T_a, T_b) -> np.ndarray:
    """[dxyz_local, rotvec] of inv(T_a) @ T_b — the C_state / HandUMI relative-EE definition."""
    d = inv_T(T_a) @ np.asarray(T_b, np.float64)
    return np.concatenate([d[:3, 3], Rotation.from_matrix(d[:3, :3]).as_rotvec()])


def mean_pose(Ts) -> np.ndarray:
    """Chordal mean of a small set of poses (translation mean + quaternion mean via scipy)."""
    Ts = np.asarray(Ts, np.float64).reshape(-1, 4, 4)
    if len(Ts) == 0: raise ValueError("mean_pose of empty set")
    R = Rotation.from_matrix(Ts[:, :3, :3]).mean().as_matrix()
    return make_T(R, Ts[:, :3, 3].mean(axis=0))


def interp_pose(T_a, T_b, alpha: float) -> np.ndarray:
    """Lerp translation, geodesic rotation interpolation between two poses at alpha in [0,1]."""
    T_a = np.asarray(T_a, np.float64); T_b = np.asarray(T_b, np.float64)
    ra, rb = Rotation.from_matrix(T_a[:3, :3]), Rotation.from_matrix(T_b[:3, :3])
    r = ra * Rotation.from_rotvec((ra.inv() * rb).as_rotvec() * float(alpha))
    return make_T(r.as_matrix(), (1 - alpha) * T_a[:3, 3] + alpha * T_b[:3, 3])
