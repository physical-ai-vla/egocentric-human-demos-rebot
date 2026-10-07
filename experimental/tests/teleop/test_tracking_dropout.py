import numpy as np
from handumi_collector.pose.backends.mock import MockBackend
from handumi_collector.pose.estimator import TrackingState
from ego_teleop.tracking.interfaces import TrackingHealth
from ego_teleop.tracking.wrist_pose_provider import EstimatorWristPoseProvider, TrackingSupervisor, TrackingSupervisorConfig
from ego_teleop.transforms.se3 import make_T, inv_T
from .conftest import T_of

NS = 1_000_000_000
def const_vel(v=(0.1, 0, 0)):
    return lambda t_ns: make_T(None, np.asarray(v) * (t_ns / NS))


def run(provider, backend, t0, t1, fps=30, img=None):
    out = []
    for i, t in enumerate(range(t0, t1, NS // fps)):
        out.append(provider.push_image(t, i, img))
    return out


def test_init_then_ok_with_hysteresis_and_velocity():
    be = MockBackend(const_vel(), init_frames=3)
    T_H_C = T_of((0, 0, 0.05), (0, 0, 90))
    p = EstimatorWristPoseProvider(be, T_H_C=T_H_C, supervisor=TrackingSupervisor(TrackingSupervisorConfig(recover_after_n_ok=3)))
    poses = run(p, be, 0, 2 * NS)
    healths = [w.health for w in poses]
    assert healths[:3] == [TrackingHealth.LOST] * 3               # backend INITIALIZING
    assert TrackingHealth.OK in healths and healths[-1] == TrackingHealth.OK
    first_ok = healths.index(TrackingHealth.OK); assert first_ok >= 3 + 2   # hysteresis: needs recover_after_n_ok good frames
    last = poses[-1]
    assert np.allclose(last.linear_velocity_xyz, [0.1, 0, 0], atol=1e-6)
    # T_W_H = T_W_C @ inv(T_H_C)
    T_W_C = const_vel()(last.timestamp_ns)
    assert np.allclose(last.T(), T_W_C @ inv_T(T_H_C))


def test_lost_window_holds_last_pose_zero_velocity_then_recovers():
    be = MockBackend(const_vel(), init_frames=1, lost_windows_ns=[(1 * NS, int(1.5 * NS))])
    p = EstimatorWristPoseProvider(be, T_H_C=np.eye(4))
    poses = run(p, be, 0, 3 * NS)
    lost = [w for w in poses if 1 * NS <= w.timestamp_ns <= 1.5 * NS]
    assert lost and all(w.health == TrackingHealth.LOST for w in lost)
    before = [w for w in poses if w.timestamp_ns < NS and w.health == TrackingHealth.OK][-1]
    for w in lost:
        assert np.allclose(w.position_xyz_m, before.position_xyz_m) and np.allclose(w.linear_velocity_xyz, 0)   # no IMU-only XYZ
    assert poses[-1].health == TrackingHealth.OK


def test_age_based_degraded_then_lost_without_new_frames():
    be = MockBackend(const_vel(), init_frames=1)
    p = EstimatorWristPoseProvider(be, T_H_C=np.eye(4), supervisor=TrackingSupervisor(TrackingSupervisorConfig(degraded_after_ms=80, lost_after_ms=250, recover_after_n_ok=1)))
    run(p, be, 0, NS)
    t_last = p.get_pose().timestamp_ns
    assert p.get_pose(t_last + 10_000_000).health == TrackingHealth.OK
    assert p.get_pose(t_last + 100_000_000).health == TrackingHealth.DEGRADED
    w = p.get_pose(t_last + 300_000_000); assert w.health == TrackingHealth.LOST and np.allclose(w.linear_velocity_xyz, 0)


def test_pose_jump_is_flagged_lost():
    def jumpy(t_ns):
        T = np.eye(4); T[0, 3] = 0.0 if t_ns < NS else 1.0; return T     # 1 m jump at t = 1 s
    be = MockBackend(jumpy, init_frames=1)
    p = EstimatorWristPoseProvider(be, T_H_C=np.eye(4), supervisor=TrackingSupervisor(TrackingSupervisorConfig(recover_after_n_ok=2)))
    poses = run(p, be, 0, 2 * NS)
    at_jump = [w for w in poses if w.timestamp_ns >= NS][0]
    assert at_jump.health == TrackingHealth.LOST
    assert poses[-1].health == TrackingHealth.OK


def test_low_confidence_is_degraded_and_imu_offset_applied():
    class LowConf(MockBackend):
        def push_image(self, t_ns, i, img):
            e = super().push_image(t_ns, i, img); e.confidence = 0.2; return e
    be = LowConf(const_vel(), init_frames=1)
    seen = []
    be.push_imu = lambda t, g, a: seen.append(t)
    p = EstimatorWristPoseProvider(be, T_H_C=np.eye(4), camera_imu_offset_ns=5_000_000, supervisor=TrackingSupervisor(TrackingSupervisorConfig(recover_after_n_ok=1)))
    p.push_imu(1000, np.zeros(3), np.zeros(3)); assert seen == [5_001_000]
    poses = run(p, be, 0, NS)
    assert poses[-1].health == TrackingHealth.DEGRADED
