"""Quaternion / SO(3) / rot6d utilities -- the ONE implementation used everywhere in ego_cart20.

rot6d = the first two ROWS of R, flattened [R00 R01 R02 R10 R11 R12] (umi.common.pose_util.mat_to_rot6d = mat[..., :2, :]),
the convention of the v4 RELCART20 state and of every REL16 action.  Identity -> [1, 0, 0, 0, 1, 0].
Quaternions are scalar-first [qw, qx, qy, qz].  All functions are batched over leading axes.
"""
import numpy as np


def quaternion_to_matrix(q):
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return R.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(R):
    """robust (Shepperd) conversion; returns qw >= 0"""
    R = np.asarray(R, np.float64); sh = R.shape[:-2]; M = R.reshape(-1, 3, 3); out = np.empty((len(M), 4))
    for i, m in enumerate(M):
        tr = np.trace(m)
        if tr > 0:
            s = 2 * np.sqrt(tr + 1); out[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]); out[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]); out[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]); out[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    out /= np.linalg.norm(out, axis=1, keepdims=True); out[out[:, 0] < 0] *= -1
    return out.reshape(sh + (4,))


def matrix_to_rotation_6d(R):
    R = np.asarray(R)
    return R[..., :2, :].reshape(R.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6):
    """Gram-Schmidt on the two stored rows (umi pose_util.rot6d_to_mat): rows b1, b2, b3 = b1 x b2"""
    d6 = np.asarray(d6, np.float64); a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, -1, keepdims=True) * b1; b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], -2)


def rotation_angle(R):
    """geodesic angle (rad) of rotation matrices"""
    c = (np.trace(np.asarray(R, np.float64), axis1=-2, axis2=-1) - 1) / 2
    return np.arccos(np.clip(c, -1.0, 1.0))


def slerp(q0, q1, w):
    """batched quaternion SLERP (shortest arc), q0/q1 (..., 4) wxyz, w (...)"""
    q0 = np.asarray(q0, np.float64); q1 = np.asarray(q1, np.float64); w = np.asarray(w, np.float64)[..., None]
    d = np.sum(q0 * q1, -1, keepdims=True); q1 = np.where(d < 0, -q1, q1); d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1)); s = np.sin(th); small = s < 1e-8
    a = np.where(small, 1 - w, np.sin((1 - w) * th) / np.where(small, 1, s))
    b = np.where(small, w, np.sin(w * th) / np.where(small, 1, s))
    q = a * q0 + b * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)
