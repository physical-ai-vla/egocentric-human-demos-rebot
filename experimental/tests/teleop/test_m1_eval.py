import numpy as np
from scipy.spatial.transform import Rotation
from handumi_collector.pose.se3 import make_T
from ego_teleop.tracking.m1_eval import evaluate, runs

NS = 1_000_000_000


def traj(n=300, fps=30, still=1.0):
    t = np.arange(n) * (NS // fps); Ts = []
    for k in range(n):
        s = t[k] / NS
        if s < still or s > (n / fps) - still: Ts.append(np.eye(4)); continue
        u = (s - still) / ((n / fps) - 2 * still); b = np.sin(np.pi * u) ** 2
        Ts.append(make_T(Rotation.from_euler("y", np.radians(60) * b).as_matrix(), [0.2 * b, 0.1 * np.sin(2 * np.pi * u) * b, 0.05 * b]))
    return t, np.array(Ts)


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]


def test_perfect_estimate_passes_with_zero_errors():
    t, Ts = traj(); v = np.ones(len(t), bool)
    r = evaluate(t, Ts, v, gt_t_ns=t, gt_Ts=Ts, static_s=1.0)
    assert r.verdict == "PASS" and r.ate_rmse_mm < 1e-6 and r.rpe_trans_rmse_mm < 1e-6 and abs(r.scale_ratio - 1) < 1e-9
    assert r.return_to_start_mm < 1e-6 and r.static["drift_mm"] == 0 and r.lost_runs == 0


def test_world_offset_does_not_matter_but_scale_does():
    t, Ts = traj(); v = np.ones(len(t), bool)
    W = make_T(Rotation.from_euler("z", 1.3).as_matrix(), [5, -2, 1]); est = np.array([W @ T for T in Ts])
    r = evaluate(t, est, v, gt_t_ns=t, gt_Ts=Ts, static_s=1.0); assert r.verdict == "PASS" and r.ate_rmse_mm < 1e-6
    scaled = Ts.copy(); scaled[:, :3, 3] *= 0.7                                  # non-metric VO
    r2 = evaluate(t, scaled, v, gt_t_ns=t, gt_Ts=Ts, static_s=1.0)
    assert r2.verdict == "FAIL" and abs(r2.scale_ratio - 0.7) < 1e-6 and any("scale" in x for x in r2.reasons)


def test_lost_runs_recovery_and_static_drift():
    t, Ts = traj(); v = np.ones(len(t), bool); v[100:130] = False; v[:5] = False
    drift = Ts.copy(); drift[:31, 0, 3] += np.linspace(0, 0.05, 31)              # 5 cm drift across the whole 1 s "still" window
    r = evaluate(t, drift, v, static_s=1.0)
    assert r.lost_runs == 1 and r.init_frames == 5 and abs(r.longest_lost_s - 1.0) < 0.05 and r.recovery_times_s and abs(r.time_to_first_valid_s - 5 / 30) < 1e-6
    assert r.static["drift_mm"] > 40 and r.verdict == "FAIL" and any("static drift" in x for x in r.reasons)
