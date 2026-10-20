"""[2026-10-12] UMI-only spec section 38: inter-gripper / temporal-action / interpolation / rot6d tests."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ego_cart20.geometry.rotation6d import matrix_to_rotation_6d, rotation_6d_to_matrix
from ego_cart20.umi_fusion.relpose import (inter_gripper_relative_pose, inv, pose9, resample_poses, temporal_action,
                                           temporal_action_chunk)

RNG = np.random.default_rng(0)


def rand_T(n):
    T = np.tile(np.eye(4), (n, 1, 1)); T[:, :3, :3] = Rot.random(n, random_state=1).as_matrix(); T[:, :3, 3] = RNG.normal(0, 0.3, (n, 3)); return T


def test_inter_gripper_translation():
    L = np.eye(4)[None]; R = np.eye(4)[None].copy(); R[0, 0, 3] = 0.5
    np.testing.assert_allclose(inter_gripper_relative_pose(L, R)[0, :3, 3], [0.5, 0, 0], atol=1e-12)


def test_inter_gripper_in_left_frame():
    L = np.eye(4)[None].copy(); L[0, :3, :3] = Rot.from_euler("z", 90, degrees=True).as_matrix()   # left TCP yawed 90 deg
    R = np.eye(4)[None].copy(); R[0, 0, 3] = 0.5                                                      # right +0.5 m world X
    np.testing.assert_allclose(inter_gripper_relative_pose(L, R)[0, :3, 3], [0, -0.5, 0], atol=1e-12)


def test_global_frame_invariance():
    L, R = rand_T(50), rand_T(50); G = rand_T(1)[0]
    np.testing.assert_allclose(inter_gripper_relative_pose(G @ L, G @ R), inter_gripper_relative_pose(L, R), atol=1e-10)
    np.testing.assert_allclose(temporal_action(G @ L, 3), temporal_action(L, 3), atol=1e-10)


def test_inverse_consistency():
    L, R = rand_T(50), rand_T(50)
    np.testing.assert_allclose(inter_gripper_relative_pose(L, R) @ inter_gripper_relative_pose(R, L), np.tile(np.eye(4), (50, 1, 1)), atol=1e-10)


def test_temporal_action_is_umi_relative_not_incremental():
    T = rand_T(40); ch = temporal_action_chunk(T, 5, 8)
    for k in range(1, 9):
        np.testing.assert_allclose(ch[k - 1], inv(T[5]) @ T[5 + k], atol=1e-10)
    np.testing.assert_allclose(temporal_action(T, 4)[5], ch[3], atol=1e-10)


def test_rot6d_roundtrip_and_convention():
    Rm = Rot.random(100, random_state=2).as_matrix()
    np.testing.assert_allclose(rotation_6d_to_matrix(matrix_to_rotation_6d(Rm)), Rm, atol=1e-10)
    np.testing.assert_allclose(matrix_to_rotation_6d(np.eye(3)), [1, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(matrix_to_rotation_6d(Rm[0]), Rm[0][:2, :].ravel())            # ROWS (umi pose_util)
    assert pose9(rand_T(3)).shape == (3, 9)


def test_timestamp_interpolation_relative_pose():
    """L and R sampled on different clocks; on a common timeline the relative pose matches the analytic one."""
    def T_at(t, side):
        T = np.tile(np.eye(4), (len(t), 1, 1)); ang = (0.6 if side == "L" else -0.4) * t
        T[:, :3, :3] = Rot.from_rotvec(np.stack([np.zeros_like(t), np.zeros_like(t), ang], 1)).as_matrix()
        T[:, :3, 3] = np.stack([0.1 * t + (0 if side == "L" else 0.4), 0.05 * t, 0 * t], 1); return T
    tL = np.arange(0, 3, 1 / 30); tR = np.arange(0.013, 3, 1 / 30); tc = np.arange(0.1, 2.9, 1 / 15)
    TL, okL = resample_poses(tL, T_at(tL, "L"), np.ones(len(tL), bool), tc)
    TR, okR = resample_poses(tR, T_at(tR, "R"), np.ones(len(tR), bool), tc)
    assert okL.all() and okR.all()
    np.testing.assert_allclose(inter_gripper_relative_pose(TL, TR), inter_gripper_relative_pose(T_at(tc, "L"), T_at(tc, "R")), atol=1e-6)


def test_resample_slerp_not_componentwise():
    T = np.tile(np.eye(4), (2, 1, 1)); T[1, :3, :3] = Rot.from_euler("z", 170, degrees=True).as_matrix()
    out, ok = resample_poses([0.0, 1.0], T, [True, True], [0.5], max_gap_s=2.0)
    assert ok[0]; np.testing.assert_allclose(Rot.from_matrix(out[0, :3, :3]).as_euler("zyx", degrees=True)[0], 85.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(out[0, :3, :3]), 1.0, atol=1e-12)


def test_resample_gap_and_invalid():
    t = np.array([0.0, 0.033, 0.3, 0.333]); T = np.tile(np.eye(4), (4, 1, 1)); v = np.array([True, True, True, False])
    _, ok = resample_poses(t, T, v, [0.01, 0.2, 0.32], max_gap_s=0.1)
    assert ok.tolist() == [True, False, False]
