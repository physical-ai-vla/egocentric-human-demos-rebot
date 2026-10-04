"""SE(3) helpers (4x4 homogeneous, batched over leading axes)."""
import numpy as np
from .rotation6d import matrix_to_rotation_6d, quaternion_to_matrix, rotation_6d_to_matrix


def make_T(R, p):
    R = np.asarray(R, np.float64); p = np.asarray(p, np.float64)
    T = np.zeros(R.shape[:-2] + (4, 4)); T[..., :3, :3] = R; T[..., :3, 3] = p; T[..., 3, 3] = 1.0
    return T


def pose_to_T(position, quaternion_wxyz):
    return make_T(quaternion_to_matrix(quaternion_wxyz), position)


def inverse(T):
    """closed-form SE(3) inverse"""
    T = np.asarray(T, np.float64); Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    return make_T(Rt, -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3]))


def relative(T_from, T_to):
    """inv(T_from) @ T_to"""
    return inverse(T_from) @ np.asarray(T_to, np.float64)


def translation(T):
    return np.asarray(T)[..., :3, 3]


def pose9(T):
    """[xyz 3 | rot6d 6] of a relative pose"""
    T = np.asarray(T)
    return np.concatenate([T[..., :3, 3], matrix_to_rotation_6d(T[..., :3, :3])], -1)


def pose9_to_T(v):
    v = np.asarray(v, np.float64)
    return make_T(rotation_6d_to_matrix(v[..., 3:9]), v[..., :3])
