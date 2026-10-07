from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ego_collector.tracking.transforms import (
    T_from_xyz_rpy,
    T_to_pose7,
    apply_delta,
    delta_pose,
    invert_T,
    normalize_quat,
    pose7_to_T,
    relative_T,
    rotation_angle_deg,
    slerp_pose7,
)


def _rand_T(rng):
    return T_from_xyz_rpy(rng.normal(size=3), rng.uniform(-3, 3, size=3))


def test_compose_world_tag_from_world_camera_and_camera_tag():
    T_world_camera = T_from_xyz_rpy([0.1, -0.4, 0.6], [np.radians(-130), 0.05, 0.2])
    T_camera_tag = T_from_xyz_rpy([0.05, 0.02, 0.55], [np.radians(180), 0.1, -0.3])
    T_world_tag = T_world_camera @ T_camera_tag
    # point on the tag's +X axis lands where the chained transform says
    p_tag = np.array([0.03, 0, 0, 1.0])
    p_world_direct = T_world_tag @ p_tag
    p_world_chained = T_world_camera @ (T_camera_tag @ p_tag)
    np.testing.assert_allclose(p_world_direct, p_world_chained, atol=1e-12)
    # and inversion recovers the factors
    np.testing.assert_allclose(invert_T(T_world_camera) @ T_world_tag, T_camera_tag, atol=1e-12)


def test_inverse_and_pose7_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        T = _rand_T(rng)
        np.testing.assert_allclose(T @ invert_T(T), np.eye(4), atol=1e-12)
        p = T_to_pose7(T)
        assert abs(np.linalg.norm(p[3:]) - 1) < 1e-12 and p[6] >= 0
        np.testing.assert_allclose(pose7_to_T(p), T, atol=1e-12)


def test_normalize_quat_sign_and_zero():
    q = normalize_quat([0, 0, 0, -2.0])
    np.testing.assert_allclose(q, [0, 0, 0, 1])
    np.testing.assert_allclose(normalize_quat([0, 0, 0, 0]), [0, 0, 0, 1])


def test_delta_pose_roundtrip_and_semantics():
    rng = np.random.default_rng(1)
    for _ in range(20):
        Ta, Tb = _rand_T(rng), _rand_T(rng)
        d = delta_pose(Ta, Tb)
        np.testing.assert_allclose(apply_delta(Ta, d), Tb, atol=1e-9)
    # pure world-frame translation, body rotated: dx is world-frame
    Ta = T_from_xyz_rpy([0, 0, 0], [0, 0, np.pi / 2])
    Tb = T_from_xyz_rpy([0.1, 0, 0], [0, 0, np.pi / 2])
    np.testing.assert_allclose(delta_pose(Ta, Tb), [0.1, 0, 0, 0, 0, 0], atol=1e-12)
    # pure body-frame rotation about the body z axis
    Tb = T_from_xyz_rpy([0, 0, 0], [0, 0, np.pi / 2 + 0.3])
    np.testing.assert_allclose(delta_pose(Ta, Tb)[3:], [0, 0, 0.3], atol=1e-12)


def test_relative_left_right_and_angle():
    T_left = T_from_xyz_rpy([0, 0, 0], [0, 0, np.pi / 2])
    T_right = T_from_xyz_rpy([0.3, 0, 0], [0, 0, 0])
    T_lr = relative_T(T_left, T_right)
    np.testing.assert_allclose(T_lr[:3, 3], [0, -0.3, 0], atol=1e-12)  # right appears at -y in the left frame
    assert abs(rotation_angle_deg(T_left, T_right) - 90) < 1e-9


def test_slerp_midpoint():
    a = T_to_pose7(T_from_xyz_rpy([0, 0, 0], [0, 0, 0]))
    b = T_to_pose7(T_from_xyz_rpy([1, 0, 0], [0, 0, np.pi / 2]))
    m = slerp_pose7(a, b, 0.5)
    np.testing.assert_allclose(m[:3], [0.5, 0, 0])
    assert abs(Rotation.from_quat(m[3:]).as_euler("xyz")[2] - np.pi / 4) < 1e-9
