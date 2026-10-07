"""SE(3) helpers. Convention: ``T_A_B`` is the pose of frame B expressed in frame A (4x4).

    T_world_leftTag = T_world_camera @ T_camera_leftTag

Pose7 = ``[x, y, z, qx, qy, qz, qw]`` (scipy/ROS order), quaternion normalized with qw >= 0.
Euler ``rpy`` = extrinsic roll(x) → pitch(y) → yaw(z) in radians (``R = Rz @ Ry @ Rx``).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

IDENTITY_POSE7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def rpy_to_matrix(rpy) -> np.ndarray:
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64)).as_matrix()


def matrix_to_rpy(R) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_euler("xyz")


def make_T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=np.float64)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def T_from_xyz_rpy(xyz, rpy) -> np.ndarray:
    return make_T(rpy_to_matrix(rpy), xyz)


def T_from_rvec_tvec(rvec, tvec) -> np.ndarray:
    R = Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix()
    return make_T(R, tvec)


def invert_T(T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def normalize_quat(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    q = q / n
    return -q if q[3] < 0 else q


def T_to_pose7(T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    q = normalize_quat(Rotation.from_matrix(T[:3, :3]).as_quat())
    return np.concatenate([T[:3, 3], q])


def pose7_to_T(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(7)
    return make_T(Rotation.from_quat(normalize_quat(p[3:7])).as_matrix(), p[:3])


def relative_T(T_world_a, T_world_b) -> np.ndarray:
    """``T_a_b = inv(T_world_a) @ T_world_b`` (b expressed in a)."""
    return invert_T(T_world_a) @ np.asarray(T_world_b, dtype=np.float64)


def delta_pose(T_t, T_t1) -> np.ndarray:
    """Per-step pseudo action ``[dx, dy, dz, drx, dry, drz]``.

    Translation delta in the *world* frame (``p(t+1) - p(t)``), rotation delta as
    the rotation vector of ``R(t)^T @ R(t+1)`` (body frame), following the spec.
    """
    T_t = np.asarray(T_t, dtype=np.float64)
    T_t1 = np.asarray(T_t1, dtype=np.float64)
    dp = T_t1[:3, 3] - T_t[:3, 3]
    dR = T_t[:3, :3].T @ T_t1[:3, :3]
    drot = Rotation.from_matrix(dR).as_rotvec()
    return np.concatenate([dp, drot])


def apply_delta(T_t, delta6) -> np.ndarray:
    """Inverse of :func:`delta_pose`."""
    T_t = np.asarray(T_t, dtype=np.float64)
    d = np.asarray(delta6, dtype=np.float64).reshape(6)
    R = T_t[:3, :3] @ Rotation.from_rotvec(d[3:6]).as_matrix()
    return make_T(R, T_t[:3, 3] + d[:3])


def rotation_angle_deg(T_a, T_b) -> float:
    Ra = np.asarray(T_a, dtype=np.float64)[:3, :3]
    Rb = np.asarray(T_b, dtype=np.float64)[:3, :3]
    return float(np.degrees(Rotation.from_matrix(Ra.T @ Rb).magnitude()))


def translation_distance(T_a, T_b) -> float:
    return float(np.linalg.norm(np.asarray(T_a)[:3, 3] - np.asarray(T_b)[:3, 3]))


def slerp_pose7(p_a, p_b, alpha: float) -> np.ndarray:
    """Interpolate two pose7 (lerp translation, slerp rotation)."""
    a = np.asarray(p_a, dtype=np.float64)
    b = np.asarray(p_b, dtype=np.float64)
    ra, rb = Rotation.from_quat(normalize_quat(a[3:])), Rotation.from_quat(normalize_quat(b[3:]))
    rel = ra.inv() * rb
    r = ra * Rotation.from_rotvec(rel.as_rotvec() * float(alpha))
    return np.concatenate([(1 - alpha) * a[:3] + alpha * b[:3], normalize_quat(r.as_quat())])


__all__ = [
    "IDENTITY_POSE7",
    "T_from_rvec_tvec",
    "T_from_xyz_rpy",
    "T_to_pose7",
    "apply_delta",
    "delta_pose",
    "invert_T",
    "make_T",
    "matrix_to_rpy",
    "normalize_quat",
    "pose7_to_T",
    "relative_T",
    "rotation_angle_deg",
    "rpy_to_matrix",
    "slerp_pose7",
    "translation_distance",
]
