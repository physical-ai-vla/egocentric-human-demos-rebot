"""[2026-10-12] UMI-only T-shirt spec: the two relative poses that must never be confused (spec section 37).

  inter_gripper_relative_pose(T_L, T_R)   T_LR(t) = T_L(t)^-1 T_R(t)          OBSERVATION / proprioception (same time, two grippers)
  temporal_action(T, k)                    dT(t, k) = T(t)^-1 T(t + k)          ACTION label (same gripper, two times)

Both take world-frame TCP poses (n, 4, 4). rot6d = ego_cart20 convention (first two ROWS of R, = umi pose_util).
resample_poses() puts a stream on a common timeline: translation linear, rotation SLERP (never component-wise).
"""
import numpy as np

from ..geometry.rotation6d import matrix_to_quaternion, matrix_to_rotation_6d, quaternion_to_matrix, slerp


def inv(T):
    T = np.asarray(T, np.float64); R = T[..., :3, :3]; p = T[..., :3, 3]
    out = np.zeros_like(T); out[..., :3, :3] = np.swapaxes(R, -1, -2); out[..., :3, 3] = -np.einsum("...ji,...j->...i", R, p); out[..., 3, 3] = 1.0
    return out


def pose9(T):
    """[xyz 3 | rot6d 6]"""
    T = np.asarray(T, np.float64); return np.concatenate([T[..., :3, 3], matrix_to_rotation_6d(T[..., :3, :3])], -1)


def inter_gripper_relative_pose(T_W_L, T_W_R):
    """OBSERVATION. T_LR = T_L^-1 T_R: where the right TCP is, expressed in the left TCP frame. (n,4,4) -> (n,4,4)"""
    return inv(T_W_L) @ np.asarray(T_W_R, np.float64)


def temporal_action(T_W, k):
    """ACTION. dT(t, k) = T(t)^-1 T(t+k) for every t with t+k in range. (n,4,4) -> (n-k,4,4)"""
    T_W = np.asarray(T_W, np.float64); return inv(T_W[:-k]) @ T_W[k:]


def temporal_action_chunk(T_W, t_idx, horizon):
    """ACTION chunk at row t_idx: [T(t)^-1 T(t+k) for k = 1..horizon] -> (horizon, 4, 4) (UMI-relative, NOT incremental)."""
    T_W = np.asarray(T_W, np.float64); return inv(T_W[t_idx])[None] @ T_W[t_idx + 1:t_idx + 1 + horizon]


def resample_poses(t_src, T_src, valid_src, t_dst, max_gap_s=0.1):
    """Common-timeline resampling: translation linear, rotation SLERP. A target time is valid only if its two bracketing
    source samples are valid and at most max_gap_s apart. -> (T_dst (m,4,4), valid_dst (m,))"""
    t_src = np.asarray(t_src, np.float64); T_src = np.asarray(T_src, np.float64); valid_src = np.asarray(valid_src, bool)
    t_dst = np.asarray(t_dst, np.float64); m = len(t_dst)
    j = np.clip(np.searchsorted(t_src, t_dst) - 1, 0, len(t_src) - 2)
    t0, t1 = t_src[j], t_src[j + 1]; w = np.clip((t_dst - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)
    ok = valid_src[j] & valid_src[j + 1] & ((t1 - t0) <= max_gap_s) & (t_dst >= t_src[0]) & (t_dst <= t_src[-1])
    q = matrix_to_quaternion(T_src[:, :3, :3]); out = np.tile(np.eye(4), (m, 1, 1))
    out[:, :3, 3] = (1 - w)[:, None] * T_src[j, :3, 3] + w[:, None] * T_src[j + 1, :3, 3]
    out[:, :3, :3] = quaternion_to_matrix(slerp(q[j], q[j + 1], w))
    return out, ok
