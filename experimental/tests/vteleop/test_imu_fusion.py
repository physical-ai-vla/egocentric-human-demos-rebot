"""Spec 37: IMU quaternion normalisation, IMU -> world/robot frame conversion, camera/IMU timestamp offset, AHRS, tag gate."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from visual_teleop import config as vcfg
from visual_teleop.fusion.wrist_pose_fusion import OrientationFusion, HandFusion, HandStatus, yaw_of
from visual_teleop.imu.ahrs import MahonyAhrs, AhrsConfig, normalize_rotation
from visual_teleop.imu.imu_reader import ImuOrientation
from visual_teleop.imu.time_sync import estimate_side, grade

G = 9.80665


def test_normalize_rotation_projects_to_so3():
    R = Rotation.from_rotvec([0.3, -0.2, 0.9]).as_matrix() + 1e-3 * np.random.default_rng(0).normal(size=(3, 3))
    M = normalize_rotation(R); np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12); assert abs(np.linalg.det(M) - 1) < 1e-12


def _imu_stream(R_body_seq, dt=0.005, bias=(0.01, -0.02, 0.005)):
    """body orientations (G<-I) -> gyro (body) + accel (body, gravity only)."""
    out = []
    for k in range(len(R_body_seq) - 1):
        w = Rotation.from_matrix(R_body_seq[k].T @ R_body_seq[k + 1]).as_rotvec() / dt
        a = R_body_seq[k].T @ np.array([0, 0, G])
        out.append((int(k * dt * 1e6), w + np.array(bias), a))
    return out


def test_ahrs_bias_and_roll_pitch_converge_yaw_follows_gyro():
    tilt = Rotation.from_rotvec([0.4, -0.3, 0]).as_matrix()
    still = [tilt] * 400                                                                 # 2 s still -> bias + attitude
    turn = [Rotation.from_rotvec([0, 0, 0.5 * k / 200]).as_matrix() @ tilt for k in range(200)]   # yaw +0.5 rad about G z
    ahrs = MahonyAhrs(AhrsConfig(bias_still_s=1.0))
    for t_us, g, a in _imu_stream(still + turn): R = ahrs.update(t_us, g, a)
    assert ahrs.ready; np.testing.assert_allclose(ahrs.bias, [0.01, -0.02, 0.005], atol=1e-6)
    truth = turn[-2]
    # roll/pitch (the up vector) observable; yaw relative to the start follows the gyro
    np.testing.assert_allclose(R[2], truth[2], atol=0.01)
    dyaw = (yaw_of(R) - yaw_of(still[0] if False else ahrs_initial_R(tilt)))
    assert abs(((dyaw - 0.5 * 199 / 200) + np.pi) % (2 * np.pi) - np.pi) < 0.02


def ahrs_initial_R(tilt):
    up = tilt.T @ np.array([0, 0, 1.0])
    return Rotation.align_vectors([[0, 0, 1.0]], [up])[0].as_matrix()


def test_ahrs_rejects_nonfinite_and_refuses_before_still():
    ahrs = MahonyAhrs(AhrsConfig(bias_still_s=0.5))
    assert ahrs.update(0, [np.nan, 0, 0], [0, 0, G]) is None and ahrs.n_gaps == 1
    rng = np.random.default_rng(1)
    for k in range(200): ahrs.update(k * 5000, rng.normal(0, 0.5, 3), [0, 0, G])          # shaking: never "still"
    assert not ahrs.ready


def test_orientation_align_yaw_and_imu_to_world():
    R_I_H = Rotation.from_rotvec([0, np.pi / 2, 0]).as_matrix()          # mount: H = I rotated 90 deg about y
    of = OrientationFusion(R_I_H)
    R_G_I = Rotation.from_rotvec([0, 0, 0.7]).as_matrix()                # IMU heading arbitrary in G
    R_W_H_known = Rotation.from_rotvec([0, 0, -0.2]).as_matrix() @ Rotation.from_rotvec([0, np.pi / 2, 0]).as_matrix()
    tilt = of.align(R_G_I, R_W_H_known, 0); assert tilt < 1e-6
    np.testing.assert_allclose(of.predict(R_G_I), R_W_H_known, atol=1e-9)
    # a later wrist rotation of +30 deg about the wrist's own x shows up as the same body-frame delta
    R2 = R_G_I @ R_I_H @ Rotation.from_rotvec([np.radians(30), 0, 0]).as_matrix() @ R_I_H.T
    np.testing.assert_allclose(of.predict(R2), R_W_H_known @ Rotation.from_rotvec([np.radians(30), 0, 0]).as_matrix(), atol=1e-9)


def test_align_reports_tilt_when_mount_is_wrong():
    of = OrientationFusion(np.eye(3)); R_known = Rotation.from_rotvec([0.5, 0, 0]).as_matrix()
    assert of.align(np.eye(3), R_known, 0) > 25


def test_tag_gate_rejects_disagreeing_and_corrects_yaw_slowly():
    of = OrientationFusion(np.eye(3), yaw_gain=0.1, reject_deg=15); of.align(np.eye(3), np.eye(3), 0)
    assert not of.observe_vision(Rotation.from_rotvec([0, 0, np.radians(40)]).as_matrix(), np.eye(3), 1)
    assert of.n_vis_rejected == 1 and of.psi == 0
    assert of.observe_vision(Rotation.from_rotvec([0, 0, np.radians(10)]).as_matrix(), np.eye(3), 2)
    assert abs(of.psi - 0.1 * np.radians(10)) < 1e-9


def test_time_offset_recovered_and_graded():
    c = vcfg.load().time_sync; rng = np.random.default_rng(2)
    t_imu = np.arange(0, 6.0, 0.005)
    from scipy.ndimage import gaussian_filter1d                       # band-limited random wrist motion (~3 Hz), like a real wave
    w = gaussian_filter1d(rng.normal(0, 25.0, (len(t_imu), 3)), 12, axis=0) + rng.normal(0, 0.02, (len(t_imu), 3))
    R = [np.eye(3)]
    for k in range(len(t_imu) - 1): R.append(R[-1] @ Rotation.from_rotvec(w[k] * 0.005).as_matrix())
    R = np.array(R); true_off_ms = 23.0
    t_cam = np.arange(0.2, 5.8, 1 / 30); idx = np.searchsorted(t_imu, t_cam)
    # camera stamps are LATE by 23 ms: the frame at host time t shows the pose of IMU time t - 23 ms
    est = estimate_side("right", (t_cam + true_off_ms / 1e3) * 1e9, R[idx], t_imu * 1e9, w, c)
    assert est.verdict == "OK" and abs(est.offset_ms - true_off_ms) < 2.0, est
    assert grade("left", dict(offset_ms=None, reason="x"), c).verdict == "PAUSE"
    assert grade("left", dict(offset_ms=5.0, peak_corr=0.65, sharpness=0.2), c).verdict == "WARN"
    assert grade("left", dict(offset_ms=5.0, peak_corr=0.3, sharpness=0.2), c).verdict == "PAUSE"


def _imu(R, t, healthy=True): return ImuOrientation("left", t, R, np.zeros(3), healthy, "" if healthy else "imu_stale")


def test_hand_fusion_states_tracking_warn_hold():
    c = vcfg.load(); hf = HandFusion.from_cfg("left", c, np.eye(3))
    assert hf.estimate(0, _imu(np.eye(3), 0)).status == HandStatus.HOLD                   # not initialised
    hf.orient.align(np.eye(3), np.eye(3), 0)
    hf.observe_position([0.1, 0.2, 0.3], 0.9, 0)
    e = hf.estimate(10_000_000, _imu(np.eye(3), 10_000_000)); assert e.status == HandStatus.TRACKING and e.commandable
    e = hf.estimate(150_000_000, _imu(np.eye(3), 150_000_000)); assert e.status == HandStatus.WARN and e.commandable      # held position
    e = hf.estimate(400_000_000, _imu(np.eye(3), 400_000_000)); assert e.status == HandStatus.HOLD and not e.commandable  # > hold_s
    hf.observe_position([0.1, 0.2, 0.3], 0.9, 500_000_000)
    e = hf.estimate(510_000_000, _imu(np.eye(3), 0, healthy=False)); assert e.status == HandStatus.HOLD                   # no orientation
    e = hf.estimate(510_000_000, _imu(np.eye(3), 0, healthy=False), R_W_H_vision=np.eye(3)); assert e.status == HandStatus.WARN
    assert not hf.observe_position([np.nan, 0, 0], 0.9, 520_000_000)
    assert not hf.observe_position([0, 0, 0], 0.1, 520_000_000)                          # low confidence = dropout


def test_tcp_offset_follows_orientation():
    c = vcfg.load(overrides={"fusion": {"position": {"tcp_offset_H_m": {"left": [0.1, 0, 0], "right": [0, 0, 0]}}}})
    hf = HandFusion.from_cfg("left", c, np.eye(3)); hf.orient.align(np.eye(3), np.eye(3), 0); hf.observe_position([0, 0, 0], 1.0, 0)
    Rz90 = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    e = hf.estimate(1, _imu(Rz90, 1)); np.testing.assert_allclose(e.T_W_H[:3, 3], [0, 0.1, 0], atol=1e-9)


def test_align_from_motion_recovers_yaw_and_rejects_bad_push():
    R_I_H = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()                    # arbitrary (known) mount
    true_psi = 1.1; of = OrientationFusion(R_I_H)
    # hand: fingers (H z) horizontal, heading 0.4 rad in W, pitched down 10 deg; IMU G yaw differs from W by true_psi
    R_W_H = Rotation.from_euler("zy", [0.4, np.radians(80)]).as_matrix()            # z_H ~ horizontal
    R_G_I = Rotation.from_rotvec([0, 0, -true_psi]).as_matrix() @ R_W_H @ R_I_H.T
    z = R_W_H[:, 2]; P = np.array([[0.3, 0.1, 0.2]]) + np.linspace(0, 0.1, 20)[:, None] * z + np.random.default_rng(0).normal(0, 0.002, (20, 3))
    rep = of.align_from_motion([R_G_I] * 20, P, 0)
    assert rep["ok"] and abs(((rep["psi"] - true_psi + np.pi) % (2 * np.pi)) - np.pi) < 0.05, rep
    np.testing.assert_allclose(of.predict(R_G_I), R_W_H, atol=0.06)
    side = np.cross([0, 0, 1.0], z); side /= np.linalg.norm(side)
    P_side = P[0] + np.linspace(0, 0.1, 20)[:, None] * side
    # RGB-D alone CANNOT tell a sideways push: it "succeeds" with psi 90 deg off -- exactly why the IMU axis check exists
    rep2 = OrientationFusion(R_I_H).align_from_motion([R_G_I] * 20, P_side, 0)
    assert rep2["ok"] and abs(abs(((rep2["psi"] - true_psi + np.pi) % (2 * np.pi)) - np.pi) - np.pi / 2) < 0.1
    # with the push's linear acceleration (IMU frame) the sideways push is rejected and the finger push accepted
    prof = np.sin(np.linspace(0, 2 * np.pi, 80))[:, None]
    R_W_I = R_W_H @ R_I_H.T                                                        # W-frame push direction -> IMU frame
    acc_side = prof * (R_W_I.T @ side)[None] * 2.0; acc_fwd = prof * (R_W_I.T @ z)[None] * 2.0
    of3 = OrientationFusion(R_I_H); r3 = of3.align_from_motion([R_G_I] * 20, P_side, 0, acc_lin_I=acc_side)
    assert not r3["ok"] and "not along the jaws" in r3["reason"] and of3.psi is None
    assert OrientationFusion(R_I_H).align_from_motion([R_G_I] * 20, P, 0, acc_lin_I=acc_fwd)["ok"]
    assert not OrientationFusion(R_I_H).align_from_motion([R_G_I] * 20, P[:1] + np.linspace(0, 0.03, 20)[:, None] * z, 0)["ok"]   # too short


def test_tag_orientation_learned_then_corrects_yaw():
    c = vcfg.load(); hf = HandFusion.from_cfg("left", c, np.eye(3)); hf.orient.yaw_gain = 0.5
    R_G_I = Rotation.from_rotvec([0.1, -0.2, 0.4]).as_matrix(); hf.orient.align(R_G_I, R_G_I, 0)        # psi = 0
    R_tag_H_true = Rotation.from_rotvec([np.pi, 0, 0.3]).as_matrix()                                     # tag mounted upside down, twisted
    R_W_H = hf.orient.predict(R_G_I); R_W_tag = R_W_H @ R_tag_H_true.T
    for k in range(hf.tag_learn_n): assert hf.observe_tag_orientation(R_W_tag, _imu(R_G_I, k), k) == "learning"
    np.testing.assert_allclose(hf.R_tag_H, R_tag_H_true, atol=1e-9)
    # IMU yaw drifts by 6 deg; the tag still sees the truth -> psi pulled back toward it
    R_drift = Rotation.from_rotvec([0, 0, np.radians(6)]).as_matrix() @ R_G_I
    psi0 = hf.orient.psi
    for k in range(10): hf.observe_tag_orientation(R_W_tag, _imu(R_drift, 100 + k), 100 + k)
    assert hf.orient.psi < psi0 - np.radians(5)


def test_umi_tape_tag_fusion_learns_offset_and_bridges_hidden_tape():
    from types import SimpleNamespace as NS
    from visual_teleop.vision.umi_pose import UmiPositionFusion
    R = Rotation.from_rotvec([0.2, -0.4, 1.1]).as_matrix(); r_true = np.array([0.0, 0.03, 0.12])        # tag -> jaw tip in the UMI frame
    fu = UmiPositionFusion(learn_n=10); rng = np.random.default_rng(0)
    def obs(p_tag):
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p_tag
        return NS(p_W=p_tag + R @ r_true + rng.normal(0, 0.002, 3), conf=1.0), NS(T_W_tag=T)
    assert fu.update(None, obs(np.zeros(3))[1], R)[2] == "none"                                         # tag alone, nothing learned yet
    for k in range(10): tape, tag = obs(np.array([0.3, 0.1 * k / 10, 0.2])); p, c, s = fu.update(tape, tag, R); assert s == "tape"
    np.testing.assert_allclose(fu.r_H, r_true, atol=0.003)
    p_tag = np.array([0.5, -0.1, 0.25]); _, tag = obs(p_tag)
    p, c, s = fu.update(None, tag, R)                                                                   # jaws in the cloth: tape hidden
    assert s == "tag" and np.linalg.norm(p - (p_tag + R @ r_true)) < 0.004
    fu.reset(); assert fu.r_H is None and fu.update(None, tag, R)[2] == "none"
