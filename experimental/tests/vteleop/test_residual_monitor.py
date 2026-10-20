"""RGB-D residual monitor (2026-10-15, FastUMI-style secondary check of the 6DoF tracker)."""
import numpy as np
from scipy.spatial.transform import Rotation
from visual_teleop.vision.residual_monitor import ResidualMonitor, fit_yaw_translation

MS = 1_000_000
A = np.eye(4); A[:3, :3] = Rotation.from_euler("z", 37, degrees=True).as_matrix(); A[:3, 3] = [0.8, -0.3, -0.9]   # tracker W -> camera W


def path(k): return np.array([0.4 + 0.08 * np.cos(0.05 * k), 0.08 * np.sin(0.05 * k), 0.3 + 0.01 * k / 100])


def rgbd(p, noise=0.0, rng=np.random.default_rng(0)): return A[:3, :3] @ p + A[:3, 3] + rng.normal(0, noise, 3)


def learned(**kw):
    m = ResidualMonitor(learn_n=40, **kw)
    for k in range(200):
        p = path(k); st = m.update(k * 10 * MS, p, (k * 10 * MS, rgbd(p, 0.002)))
    assert m.T_rgbd_trk is not None and m.fit_rms_m < 0.005
    return m


def test_fit_recovers_yaw_and_offset():
    P = np.array([path(k) for k in range(100)]); T, rms = fit_yaw_translation(P, np.array([rgbd(p) for p in P]))
    np.testing.assert_allclose(T, A, atol=1e-9); assert rms < 1e-9


def test_learning_needs_spread():
    m = ResidualMonitor(learn_n=10)
    for k in range(100): st = m.update(k * 10 * MS, np.array([0.4, 0, 0.3]), (k * 10 * MS, rgbd(np.array([0.4, 0, 0.3]))))
    assert st["state"] == "learning" and m.T_rgbd_trk is None


def test_ok_warn_hold_and_single_outlier():
    m = learned(warn_m=0.03, hold_m=0.08, hold_s=0.5); t = 3000 * MS; p = path(0)
    assert m.update(t, p, (t, rgbd(p)))["state"] == "ok"
    assert m.update(t, p, (t, rgbd(p) + [0.05, 0, 0]))["state"] == "warn"
    assert m.update(t + 10 * MS, p, (t + 10 * MS, rgbd(p) + [0.2, 0, 0]))["state"] == "warn"      # one outlier: not hold yet
    assert m.update(t + 20 * MS, p, (t + 20 * MS, rgbd(p)))["state"] == "ok"                       # recovered -> timer reset
    for k in range(60): st = m.update(t + (30 + k * 10) * MS, p, (t + (30 + k * 10) * MS, rgbd(p) + [0.2, 0, 0]))
    assert st["state"] == "hold"


def test_occlusion_gives_no_verdict():
    m = learned(); t = 3000 * MS
    assert m.update(t, path(0), None)["state"] == "no_rgbd"
    assert m.update(t, path(0), (t - 200 * MS, rgbd(path(0))))["state"] == "no_rgbd"                # stale RGB-D sample
