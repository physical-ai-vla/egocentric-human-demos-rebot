"""[2026-10-12] synthetic check of umi_fusion.fusion: two MASt3R segments in different Sim3 frames + a 1.2 s gap,
Orbbec body point with 2 mm noise, exact gyro. Recovers the camera pose, labels provenance, stays continuous at the gap."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ego_cart20.umi_fusion.fusion import SideInputs, fuse_side


def make(gap=(4.0, 5.2), seed=0):
    rng = np.random.default_rng(seed); fps = 30.0; t = np.arange(0, 10, 1 / fps); n = len(t)
    w_true = np.stack([0.6 * np.sin(0.7 * t), 0.4 * np.cos(0.5 * t), 0.3 * np.sin(1.1 * t + 1)], 1)       # rad/s camera frame
    R = [np.eye(3)]
    for k in range(n - 1): R.append(R[-1] @ Rot.from_rotvec(w_true[k] / fps).as_matrix())
    R = np.array(R); p = np.stack([0.3 + 0.15 * np.sin(0.4 * t), 0.2 * np.cos(0.3 * t), 0.1 + 0.05 * np.sin(0.9 * t)], 1)
    r_c = np.array([0.0, 0.055, 0.12])
    body = p + np.einsum("nij,j->ni", R, r_c) + rng.normal(0, 0.002, (n, 3))
    R_cb = Rot.from_euler("xyz", [10, -95, 5], degrees=True).as_matrix()
    gyro_t = np.arange(0, 10, 1 / 200.0); wk = np.array([np.interp(gyro_t, t, w_true[:, i]) for i in range(3)]).T
    gyro_w = wk @ R_cb                                                    # body-frame rate: w_b = R_cb^T w_c
    seg = np.where(t < gap[0], 0, np.where(t >= gap[1], 1, -1))
    p_seg = np.zeros((n, 3)); R_seg = np.tile(np.eye(3), (n, 1, 1))
    for s, (sig, rv, tr) in enumerate([(1.7, [0.2, -0.1, 0.3], [0.5, -0.2, 0.1]), (0.6, [-1.0, 0.4, 0.2], [-0.3, 0.9, 0.0])]):
        Rs = Rot.from_rotvec(rv).as_matrix(); m = seg == s
        R_seg[m] = Rs.T @ R[m]; p_seg[m] = ((Rs.T @ (p[m] - tr).T).T) / sig
    inp = SideInputs(t=t, seg=seg, p_seg=p_seg, R_seg=R_seg, body_t=t, body_p=body, body_ok=np.ones(n, bool),
                     gyro_t=gyro_t, gyro_w=gyro_w, R_cb=R_cb)
    return inp, R, p, r_c


def test_fusion_recovers_pose_and_bridges_gap():
    inp, R, p, r_c = make(); f = fuse_side(inp)
    np.testing.assert_allclose(f.r_c, r_c, atol=0.01)
    perr = np.linalg.norm(f.T_W_c[:, :3, 3] - p, axis=1) * 1e3
    rerr = np.degrees(Rot.from_matrix(np.einsum("nji,njk->nik", f.T_W_c[:, :3, :3], R)).magnitude())
    m3 = f.pose_source == "MAST3R_ORBBEC"; br = f.pose_source == "ORBBEC_IMU"
    assert m3.sum() > 200 and br.sum() == ((inp.t >= 4.0) & (inp.t < 5.2)).sum()
    assert np.percentile(perr[m3], 90) < 5 and np.percentile(rerr[m3], 90) < 1.0
    assert np.percentile(perr[br], 90) < 8 and rerr[br].max() < 2.0
    steps = np.linalg.norm(np.diff(f.T_W_c[:, :3, 3], axis=0), axis=1) * 1e3                 # continuity: no jump at the switch
    assert steps.max() < 20


def test_gap_confidence_by_anchor_distance():
    inp, *_ = make(gap=(2.0, 6.5)); f = fuse_side(inp); g = (inp.t >= 2.0) & (inp.t < 6.5)
    d = np.minimum(inp.t - 2.0 + 1 / 30, 6.5 - inp.t)                                    # distance to the nearest anchor
    assert (f.rotation_source[g] == "IMU").all()
    assert (f.confidence_rotation[g & (d <= 0.45)] == 1.0).all() and (f.confidence_rotation[g & (d >= 0.55)] == 0.5).all()


def test_gap_longer_than_max_is_invalid():
    inp, *_ = make(gap=(1.5, 8.0)); f = fuse_side(inp); g = (inp.t >= 1.5) & (inp.t < 8.0)
    assert (f.pose_source[g] == "INVALID").all() and (f.rotation_source[g] == "NONE").all()


def test_qa_coverage_union_and_longest_gap():
    from ego_cart20.umi_fusion.qa import coverage, episode_gate
    t = np.arange(0, 10, 0.1); m = t < 3; o = (t > 2) & ((t < 5) | (t > 6)); f = m | o
    c = coverage(t, m, o, f)
    assert c["both_valid"]["frac"] == round(float((m & o).mean()), 4) and c["either_valid"]["frac"] == round(float((m | o).mean()), 4)
    assert abs(c["either_valid"]["longest_gap_s"] - 1.1) < 0.11 and abs(c["mast3r_valid"]["longest_gap_s"] - 7.0) < 0.11
    assert episode_gate(c)["status"] in ("PASS", "WARN", "FAIL")
