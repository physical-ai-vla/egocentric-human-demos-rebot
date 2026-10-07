from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from handumi.dataset.derived import (
    ContactProxyConfig,
    bimanual_relation,
    contact_proxy,
    derive_episode,
    gripper_events,
)

FPS = 30.0


def _width_profile():
    # open 80 mm (1 s) -> close to 30 mm over 1 s -> hold 2 s -> open to 80 over 1 s -> hold 1 s
    hold0 = np.full(30, 80.0)
    close = np.linspace(80, 30, 30)
    hold1 = np.full(60, 30.0)
    open_ = np.linspace(30, 80, 30)
    hold2 = np.full(30, 80.0)
    return np.concatenate([hold0, close, hold1, open_, hold2])


def test_gripper_events_detect_grasp_and_release():
    w = _width_profile()
    ev = gripper_events(w, FPS)
    gs = np.flatnonzero(ev["grasp_start"])
    rel = np.flatnonzero(ev["release"])
    assert len(gs) == 1 and len(rel) == 1
    assert 58 <= gs[0] <= 68  # just after closing ends at index 60
    assert 118 <= rel[0] <= 128  # opening starts at index 120
    assert ev["grasped"][gs[0] : rel[0]].all()
    assert not ev["grasped"][:55].any() and not ev["grasped"][135:].any()
    assert ev["closing"][35:55].all() and ev["opening"][125:145].all()


def test_contact_proxy_rises_on_current_and_flags_slip():
    w = _width_profile()
    ev = gripper_events(w, FPS)
    current = np.full(len(w), 100.0)  # free-motion baseline 100 mA
    gs = int(np.flatnonzero(ev["grasp_start"])[0])
    current[gs:120] = 300.0  # squeezing object
    current[100:120] = 120.0  # object slipped out at index 100 -> current collapses
    cp = contact_proxy(current, w, FPS, events=ev, config=ContactProxyConfig())
    assert abs(cp["current_baseline_ma"][0] - 100.0) < 1e-6
    assert cp["contact_estimate"][gs + 2] > 0.9
    assert cp["contact_estimate"][10] == 0.0  # open + free -> no contact
    slip = np.flatnonzero(cp["possible_slip"])
    assert len(slip) >= 1 and 100 <= slip[0] <= 104


def test_bimanual_relation_geometry():
    T = 10
    q = Rotation.from_euler("z", 90, degrees=True).as_quat()
    left = np.tile(np.array([0, 0, 0, *q]), (T, 1))
    right = np.tile(np.array([0.3, 0, 0, 0, 0, 0, 1.0]), (T, 1))
    right[:, 0] = 0.3 + 0.01 * np.arange(T)  # drifting apart at 0.3 m/s
    rel = bimanual_relation(left, right, FPS)
    # right is +x in world; left frame is rotated 90 deg about z => right appears at -y
    np.testing.assert_allclose(rel["relative_xyz"][0], [0, -0.3, 0], atol=1e-6)
    assert abs(rel["hand_distance"][0] - 0.3) < 1e-6
    assert abs(rel["hand_distance_rate"][5] - 0.3) < 1e-3
    np.testing.assert_allclose(np.abs(rel["relative_rotvec"][0]), [0, 0, np.pi / 2], atol=1e-6)


def test_derive_episode_columns_have_uniform_length():
    T = 180
    w = _width_profile()
    left = np.tile(np.array([0, 0, 0.1, 0, 0, 0, 1.0]), (T, 1))
    left[:, 0] = np.linspace(0, 0.2, T)
    right = np.tile(np.array([0.3, 0, 0.1, 0, 0, 0, 1.0]), (T, 1))
    cols = derive_episode(left_tcp=left, right_tcp=right, left_width_mm=w, right_width_mm=w[::-1].copy(), fps=FPS, left_current_ma=np.full(T, 90.0))
    assert all(len(v) == T for v in cols.values()), {k: len(v) for k, v in cols.items() if len(v) != T}
    assert "left.motor.contact_estimate" in cols and "right.motor.contact_estimate" not in cols
    assert abs(cols["left.delta.dx"][0] - 0.2 / (T - 1)) < 1e-6 and cols["left.delta.dx"][-1] == 0.0
