from __future__ import annotations

import numpy as np

from ego_collector.actions.bimanual import bimanual_features
from ego_collector.actions.pseudo_action import action14, pseudo_action_table, states_from_arrays
from ego_collector.filtering.pose_filter import OneEuroConfig, interpolate_short_gaps, one_euro_pose7
from ego_collector.hands.grasp import GraspCalibration, aperture_features
from ego_collector.tracking.transforms import T_from_xyz_rpy, T_to_pose7, apply_delta, pose7_to_T


def _traj(n=30, x0=0.0):
    ts = np.arange(n, dtype=np.int64) * 33_333_333
    poses = np.stack([T_to_pose7(T_from_xyz_rpy([x0 + 0.01 * i, 0.3, 0.1 + 0.005 * i], [0.5, 0.02 * i, 0.3])) for i in range(n)])
    return ts, poses


def test_pseudo_action_table_deltas_integrate_and_respect_validity():
    ts, left = _traj()
    _, right = _traj(x0=0.3)
    lv = np.ones(30, bool)
    lv[12] = False  # one missing left frame
    grasp = {"left": np.linspace(1, 0, 30), "right": np.full(30, 0.7)}
    states = states_from_arrays(ts, {"left": left, "right": right}, {"left": lv, "right": np.ones(30, bool)}, grasp)
    cols = pseudo_action_table(states)
    assert cols["left_valid"].sum() == 29 and np.isnan(cols["left_x"][12])
    # deltas around the gap are NaN, elsewhere they integrate to the next pose
    assert not cols["left_delta_valid"][11] and not cols["left_delta_valid"][12] and cols["left_delta_valid"][13]
    for i in (0, 5, 20):
        d = np.array([cols[f"left_{k}"][i] for k in ("dx", "dy", "dz", "drx", "dry", "drz")])
        np.testing.assert_allclose(apply_delta(pose7_to_T(left[i]), d), pose7_to_T(left[i + 1]), atol=1e-9)
    assert not cols["action_valid"][29] and cols["action_valid"][0]
    a = action14(cols)
    assert a.shape == (30, 14) and a[0, 6] == 1.0 and a[0, 13] == 0.7


def test_bimanual_features_distance_and_rate():
    ts, left = _traj()
    right = left.copy()
    right[:, 0] += 0.3 + 0.01 * np.arange(30)  # drifting apart at 0.3 m/s
    cols = bimanual_features(ts, left, right, np.ones(30, bool), np.ones(30, bool))
    assert abs(cols["left_right_distance"][0] - 0.3) < 1e-9
    assert abs(cols["left_right_distance_rate"][5] - 0.3) < 1e-6
    assert np.isnan(cols["left_right_distance_rate"][29])


def test_grasp_aperture_and_calibration(tmp_path):
    lm = np.zeros((21, 2))
    lm[4] = [100, 100]  # thumb tip
    lm[8] = [160, 100]  # index tip -> 60 px
    lm[5] = [120, 200]
    lm[17] = [200, 200]  # hand scale 80 px
    f = aperture_features(lm)
    assert f["aperture_raw_px"] == 60 and abs(f["aperture_norm"] - 0.75) < 1e-9
    cal = GraspCalibration.from_samples({"left": np.full(10, 1.5) + np.linspace(-0.1, 0.1, 10)}, {"left": np.full(10, 0.3)})
    assert 1.4 < cal.left_open <= 1.6 and abs(cal.left_closed - 0.3) < 1e-9 and cal.right_open == 1.6  # right untouched
    g = cal.grasp("left", np.array([0.3, 0.9, 1.6, np.nan]))
    assert g[0] == 0.0 and 0.4 < g[1] < 0.6 and g[2] == 1.0 and np.isnan(g[3])
    cal.save(tmp_path / "g.yaml")
    assert GraspCalibration.load(tmp_path / "g.yaml") == cal


def test_one_euro_filter_smooths_noise_and_resets_on_gaps():
    ts = np.arange(60, dtype=np.int64) * 33_333_333
    # slow hand (2 mm/frame = 6 cm/s) with 6 mm tag jitter: the filter must reduce error without lagging away
    clean = np.stack([T_to_pose7(T_from_xyz_rpy([0.002 * i, 0.3, 0.1], [0.5, 0.005 * i, 0.3])) for i in range(60)])
    rng = np.random.default_rng(0)
    noisy = clean.copy()
    noisy[:, :3] += rng.normal(0, 0.006, size=(60, 3))
    valid = np.ones(60, bool)
    valid[30] = False
    f = one_euro_pose7(noisy, valid, ts, OneEuroConfig())
    assert np.isnan(f[30]).all()
    err_noisy = np.linalg.norm(noisy[10:30, :3] - clean[10:30, :3], axis=1).mean()
    err_filt = np.linalg.norm(f[10:30, :3] - clean[10:30, :3], axis=1).mean()
    assert err_filt < err_noisy * 0.9, (err_filt, err_noisy)
    # after the gap the filter restarts from the raw sample
    np.testing.assert_allclose(f[31], np.concatenate([noisy[31, :3], noisy[31, 3:]]), atol=1e-9)


def test_interpolate_short_gaps_only():
    ts, poses = _traj(20)
    valid = np.ones(20, bool)
    valid[5:7] = False  # 2-frame gap -> filled
    valid[12:17] = False  # 5-frame gap -> left alone
    p, filled = interpolate_short_gaps(poses, valid, ts, max_gap_frames=3)
    assert filled[5] and filled[6] and not filled[13]
    np.testing.assert_allclose(p[5, :3], (poses[4, :3] + poses[7, :3]) / 2 + (poses[5, :3] - (poses[4, :3] + poses[7, :3]) / 2) * 0, atol=0.006)
