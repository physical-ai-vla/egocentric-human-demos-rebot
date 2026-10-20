"""C922 + IMU metric position solver (2026-10-15)."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from visual_teleop.vision.kpt_translation import project, solve_translation

K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]])
P = np.array([[0, 0, 0], [0.02, 0, -0.05], [-0.02, 0, -0.05], [0.04, 0.03, -0.13], [-0.04, 0.03, -0.13]])   # tip, jaw bases, body corners
R_CH = R.from_euler("xyz", [20, -35, 60], degrees=True).as_matrix(); t_true = np.array([0.12, -0.08, 0.95])


def test_exact_and_noisy():
    uv = project(K, R_CH, t_true, P); s = solve_translation(K, R_CH, P, uv)
    assert s.ok and np.allclose(s.t_CH, t_true, atol=1e-9) and s.rms_px < 1e-6
    rng = np.random.default_rng(0); errs = [np.linalg.norm(solve_translation(K, R_CH, P, uv + rng.normal(0, 1.5, uv.shape)).t_CH - t_true) for _ in range(200)]
    assert np.percentile(errs, 90) < 0.02                          # p90 ~1.4 cm at 1 m with 1.5 px noise, 5 points


def test_two_points_metric_one_point_not():
    uv = project(K, R_CH, t_true, P); uv1 = uv.copy(); uv1[2:] = np.nan
    s = solve_translation(K, R_CH, P, uv1); assert s.ok and np.allclose(s.t_CH, t_true, atol=1e-6)
    uv1[1] = np.nan; s = solve_translation(K, R_CH, P, uv1); assert not s.ok and "2 keypoints" in s.reason


def test_occluded_tip_still_solved_from_body():
    uv = project(K, R_CH, t_true, P); uv[0] = np.nan                  # jaw tip under the cloth
    s = solve_translation(K, R_CH, P, uv); assert s.ok and np.allclose(s.t_CH, t_true, atol=1e-6)


def test_outlier_dropped():
    uv = project(K, R_CH, t_true, P); uv[3] += [40, -25]              # one wrong detection
    s = solve_translation(K, R_CH, P, uv, names=["tip", "jl", "jr", "bl", "br"])
    assert s.dropped == ["bl"] and np.linalg.norm(s.t_CH - t_true) < 1e-6
