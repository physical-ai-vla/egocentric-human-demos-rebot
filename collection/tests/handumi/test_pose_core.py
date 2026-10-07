"""M2 pose subsystem unit tests: SE3, relative action semantics, IMU integration / still detection, timing, sync, QA."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from handumi_collector.config import PoseCfg
from handumi_collector.pose import se3
from handumi_collector.pose.imu import StaticWindow, detect_home_windows, integrate_gyro, still_mask
from handumi_collector.pose.qa import evaluate_side
from handumi_collector.pose.relative import c_state_rows, relative_chunk, relative_tcp
from handumi_collector.pose.sync import resample_poses, resample_scalar
from handumi_collector.pose.timing import estimate_camera_imu_offset, fit_device_to_host, interleave


def rand_T(rng):
    return se3.make_T(Rotation.random(random_state=rng.integers(1 << 31)).as_matrix(), rng.normal(size=3))


def test_se3_matches_legacy_conventions():
    from ego_collector.tracking import transforms as legacy      # pure-math module; the pose package itself must not import it
    rng = np.random.default_rng(1)
    for _ in range(20):
        T = rand_T(rng)
        assert np.allclose(se3.T_to_pose7(T), legacy.T_to_pose7(T))
        assert np.allclose(se3.pose7_to_T(se3.T_to_pose7(T)), T)
        assert np.allclose(se3.inv_T(T), legacy.invert_T(T))
    P = np.stack([se3.T_to_pose7(rand_T(rng)) for _ in range(5)])
    assert np.allclose(se3.Ts_to_pose7(se3.poses7_to_T(P)), P)


def test_relative_tcp_is_c_state_definition():
    rng = np.random.default_rng(2); Ts = np.stack([rand_T(rng) for _ in range(12)]); lead = 5
    d, ok = relative_tcp(Ts, lead)
    assert ok.all()
    for t in range(12):
        f = min(t + lead, 11); ref = np.linalg.inv(Ts[t]) @ Ts[f]
        assert np.allclose(d[t, :3], ref[:3, 3]); assert np.allclose(d[t, 3:], Rotation.from_matrix(ref[:3, :3]).as_rotvec())
    # local frame translation (not world) — differs from ego_collector.delta_pose's world dxyz
    assert not np.allclose(d[0, :3], Ts[lead][:3, 3] - Ts[0][:3, 3])
    chunk, cok = relative_chunk(Ts, 2, 30); assert chunk.shape == (30, 6) and cok.all() and np.allclose(chunk[lead - 1], d[2])
    valid = np.ones(12, bool); valid[7] = False
    d2, ok2 = relative_tcp(Ts, lead, valid); assert not ok2[2] and not ok2[7] and ok2[0]
    rows, rok = c_state_rows(Ts, Ts, np.linspace(0, 1, 12), np.zeros(12), lead)
    assert rows.shape == (12, 14) and rows[0, 6] == pytest.approx(np.linspace(0, 1, 12)[lead]) and rok.all()


def test_gyro_integration_constant_rate():
    hz = 400; t = np.arange(0, 2.0, 1 / hz); w = np.array([0.0, 0.0, np.radians(30)])
    gyro = np.tile(w, (len(t), 1)) + np.array([0.01, -0.02, 0.0])
    R = integrate_gyro((t * 1e9).astype(np.int64), gyro, 0, int(1e9), bias=np.array([0.01, -0.02, 0.0]))
    assert se3.rotation_angle_deg(R, Rotation.from_euler("z", 30, degrees=True).as_matrix()) < 0.05


def test_still_windows_detected_with_bias_and_explicit_marks():
    hz = 400; T = 8.0; t = np.arange(0, T, 1 / hz); rng = np.random.default_rng(0)
    moving = (t > 1.5) & (t < T - 1.5)
    gyro = np.radians(0.5) * np.ones((len(t), 3)) + rng.normal(0, np.radians(0.05), (len(t), 3))
    gyro[moving] += np.radians(40) * np.sin(2 * np.pi * t[moving, None] * np.array([0.7, 1.1, 0.4]))
    accel = np.tile([0, 0, 9.81], (len(t), 1)) + rng.normal(0, 0.02, (len(t), 3)); accel[moving, 0] += 2.0 * np.sin(5 * t[moving])
    m = still_mask(t * 1e9, gyro, accel, gyro_thresh_dps=3.0, accel_std_thresh=0.35)
    assert m[:400].mean() > 0.95 and m[int(2.5 * hz):int(5.5 * hz)].mean() < 0.05
    cfg = PoseCfg().static_window
    hs, he = detect_home_windows((t * 1e9).astype(np.int64), gyro, accel, cfg, t_start_ns=0, t_stop_ns=int(T * 1e9))
    assert hs is not None and he is not None and hs.duration_s > 0.8 and he.duration_s > 0.8
    assert np.allclose(np.degrees(hs.gyro_bias), 0.5, atol=0.05)          # bias recovered from the still start
    hs2, he2 = detect_home_windows((t * 1e9).astype(np.int64), gyro, accel, cfg, t_start_ns=0, t_stop_ns=int(T * 1e9),
                                   explicit_leave_ns=int(1.2e9), explicit_return_ns=int(7.0e9))
    assert hs2.t1_ns <= int(1.2e9) and he2.t0_ns >= int(7.0e9)
    # nothing still -> None, not a fake window
    assert detect_home_windows((t * 1e9).astype(np.int64), gyro + np.radians(50), accel + 3 * rng.normal(size=accel.shape), cfg, t_start_ns=0, t_stop_ns=int(T * 1e9)) == (None, None)


def test_clock_fit_and_camera_imu_offset_recovery():
    rng = np.random.default_rng(3); hz = 400; n = 400 * 6
    dev_us = 1_000_000 + (np.arange(n) * 2500).astype(np.int64)
    true_ns = 5_000_000_000 + dev_us * 1000 * 1.00001
    host = (true_ns + np.abs(rng.normal(0.4e6, 0.25e6, n))).astype(np.int64)      # one-sided USB latency
    fit = fit_device_to_host(dev_us, host)
    assert abs(fit.to_dict()["drift_ppm"] - 10) < 3 and np.abs(fit.to_host_ns(dev_us) - true_ns).max() < 0.4e6
    # offset recovery: camera rotations from a known angular-velocity profile, IMU shifted by +17 ms
    t_cam = np.arange(0, 6, 1 / 30); f = 0.6; ang = np.radians(60) * (1 - np.cos(2 * np.pi * f * t_cam)) / (2 * np.pi * f)   # exact integral of w(t)
    Rs = Rotation.from_euler("y", ang.reshape(-1, 1)).as_matrix()
    t_imu = np.arange(0, 6, 1 / hz); w = np.radians(60) * np.sin(2 * np.pi * f * t_imu)
    gyro = np.c_[np.zeros_like(w), w, np.zeros_like(w)] + np.radians(0.3)
    r = estimate_camera_imu_offset((t_cam * 1e9).astype(np.int64), Rs, ((t_imu - 0.017) * 1e9).astype(np.int64), gyro, max_offset_ms=60)
    assert r["offset_ms"] is not None and abs(r["offset_ms"] - 17.0) <= 2.0 and r["confidence"] > 0.8
    ev = list(interleave(np.array([100, 200]), np.array([50, 100, 150, 250])))
    assert ev == [("imu", 0), ("imu", 1), ("image", 0), ("imu", 2), ("image", 1), ("imu", 3)]


def test_resampling_never_bridges_long_gaps():
    pose_t = np.array([0, 33, 66, 99, 500, 533], np.int64) * 1_000_000
    Ts = np.stack([se3.make_T(None, [i, 0, 0]) for i in range(6)]); valid = np.ones(6, bool); valid[2] = False
    head_t = np.array([16, 50, 200, 300, 516], np.int64) * 1_000_000
    T, ok, gap = resample_poses(head_t, pose_t, Ts, valid, max_gap_ns=100_000_000)
    assert ok.tolist() == [True, True, False, False, True]
    assert T[0][0, 3] == pytest.approx(16 / 33, abs=1e-6)                      # interpolated between 0 and 1
    assert T[1][0, 3] == pytest.approx(1 + (50 - 33) / (99 - 33) * 2, abs=1e-6)  # index 2 invalid -> interpolates 1..3
    g, gok = resample_scalar(head_t, np.array([0, 40, 600]) * 1_000_000, np.array([0.1, 0.2, 0.9]), max_gap_ns=150_000_000)
    assert gok.tolist() == [True, True, False, False, True] and g[4] == pytest.approx(0.9)


def _synthetic_side(drift_mm=0.0, drift_deg=0.0, lost=(), jump=False, n=300, fps=30):
    t = (np.arange(n) * 1e9 / fps).astype(np.int64) + 10**9
    Ts = np.tile(np.eye(4), (n, 1, 1)); valid = np.ones(n, bool); states = ["tracking"] * n
    for i in range(n):
        u = np.clip((i / fps - 1.5) / 7.0, 0, 1); b = np.sin(np.pi * u) ** 2
        Ts[i] = se3.make_T(Rotation.from_euler("y", 30 * b, degrees=True).as_matrix(), [0.2 * b, 0, 0])
        if i / fps > 8.5:
            Ts[i] = se3.make_T(Rotation.from_euler("z", drift_deg, degrees=True).as_matrix(), [drift_mm / 1000, 0, 0])
    for a, b in lost:
        valid[a:b] = False; states[a:b] = ["lost"] * (b - a)
    if jump: Ts[150][:3, 3] += [0.5, 0, 0]
    hs = StaticWindow(t[0], t[0] + int(1.4e9), np.zeros(3), 0.05, np.array([0, 0, 9.81]), 0.02, 560)
    he = StaticWindow(t[-1] - int(1.4e9), t[-1], np.zeros(3), 0.05, np.array([0, 0, 9.81]), 0.02, 560)
    return t, Ts, valid, states, hs, he


def test_home_return_qa_thresholds_from_config():
    cfg = PoseCfg()
    for drift_mm, drift_deg, expect in ((0.5, 0.1, "PASS"), (15, 0.1, "WARN"), (0.5, 5, "WARN"), (40, 0.1, "REJECT"), (0.5, 9, "REJECT")):
        t, Ts, valid, states, hs, he = _synthetic_side(drift_mm, drift_deg)
        qa = evaluate_side(side="left", backend="x", metric_scale=True, t_ns=t, Ts=Ts, valid=valid, states=states, home_start=hs, home_end=he,
                           imu_t_ns=None, gyro=None, gyro_bias=None, R_camera_imu=None, cfg=cfg, fps=30)
        assert qa.verdict == expect, (drift_mm, drift_deg, qa.reasons)
        assert qa.return_translation_mm == pytest.approx(drift_mm, abs=0.05) and qa.return_rotation_deg == pytest.approx(drift_deg, abs=0.01)
    # non-metric backend: translation drift reported but not judged
    t, Ts, valid, states, hs, he = _synthetic_side(40, 0.1)
    qa = evaluate_side(side="left", backend="vo", metric_scale=False, t_ns=t, Ts=Ts, valid=valid, states=states, home_start=hs, home_end=he,
                       imu_t_ns=None, gyro=None, gyro_bias=None, R_camera_imu=None, cfg=cfg, fps=30)
    assert qa.verdict == "PASS" and any("not metric" in f for f in qa.flags)
    # custom thresholds are honoured (nothing hard-coded)
    strict = PoseCfg(home_return={"pass": {"translation_mm": 0.1, "rotation_deg": 0.01}, "warn": {"translation_mm": 0.2, "rotation_deg": 0.02}})
    t, Ts, valid, states, hs, he = _synthetic_side(0.5, 0.1)
    assert evaluate_side(side="left", backend="x", metric_scale=True, t_ns=t, Ts=Ts, valid=valid, states=states, home_start=hs, home_end=he,
                         imu_t_ns=None, gyro=None, gyro_bias=None, R_camera_imu=None, cfg=strict, fps=30).verdict == "REJECT"


def test_lost_jump_and_imu_consistency_qa():
    cfg = PoseCfg()
    t, Ts, valid, states, hs, he = _synthetic_side(lost=[(100, 110)])
    qa = evaluate_side(side="left", backend="x", metric_scale=True, t_ns=t, Ts=Ts, valid=valid, states=states, home_start=hs, home_end=he,
                       imu_t_ns=None, gyro=None, gyro_bias=None, R_camera_imu=None, cfg=cfg, fps=30)
    assert qa.lost_events == 1 and qa.verdict == "WARN"
    t, Ts, valid, states, hs, he = _synthetic_side(lost=[(100, 160)])
    assert evaluate_side(side="left", backend="x", metric_scale=True, t_ns=t, Ts=Ts, valid=valid, states=states, home_start=hs, home_end=he,
                         imu_t_ns=None, gyro=None, gyro_bias=None, R_camera_imu=None, cfg=cfg, fps=30).verdict == "REJECT"
    t, Ts, valid, states, hs, he = _synthetic_side(jump=True)
    qa = evaluate_side(side="left", backend="x", metric_scale=True, t_ns=t, Ts=Ts, valid=valid, states=states, home_start=hs, home_end=he,
                       imu_t_ns=None, gyro=None, gyro_bias=None, R_camera_imu=None, cfg=cfg, fps=30)
    assert qa.jumps_translation >= 1 and qa.verdict == "REJECT"
    # IMU consistency: gyro consistent with the trajectory -> PASS; gyro from a different motion -> REJECT
    t, Ts, valid, states, hs, he = _synthetic_side()
    ti = np.arange(t[0], t[-1], int(2.5e6)).astype(np.int64)
    def gyro_from(Ts_, t_):
        R = Rotation.from_matrix(Ts_[:, :3, :3]); tt = t_ / 1e9
        rel = (R[:-1].inv() * R[1:]).as_rotvec() / np.diff(tt)[:, None]
        return np.array([np.interp(ti / 1e9, 0.5 * (tt[:-1] + tt[1:]), rel[:, k]) for k in range(3)]).T
    good = gyro_from(Ts, t)
    qa = evaluate_side(side="left", backend="x", metric_scale=True, t_ns=t, Ts=Ts, valid=valid, states=states, home_start=hs, home_end=he,
                       imu_t_ns=ti, gyro=good, gyro_bias=np.zeros(3), R_camera_imu=np.eye(4), cfg=cfg, fps=30)
    assert qa.verdict == "PASS" and qa.imu_residual_deg_p95 < 1.0
    bad = good.copy(); bad[:, 0] += np.radians(90)
    qa = evaluate_side(side="left", backend="x", metric_scale=True, t_ns=t, Ts=Ts, valid=valid, states=states, home_start=hs, home_end=he,
                       imu_t_ns=ti, gyro=bad, gyro_bias=np.zeros(3), R_camera_imu=np.eye(4), cfg=cfg, fps=30)
    assert qa.verdict == "REJECT" and any("IMU/VIO" in r for r in qa.reasons)
