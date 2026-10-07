import numpy as np
import pytest
from handumi_collector.devices.base import SampleBuffer, ImuSample, CameraFrame
from handumi_collector.pose.backends.mock import MockBackend
from ego_teleop.tracking.imu_tap import FanoutBuffer, install_imu_tap, LiveImuClock, LiveVioFeeder
from ego_teleop.tracking.vio_stable import VioStableGate, VioStableConfig
from ego_teleop.tracking.interfaces import WristPose, TrackingHealth
from ego_teleop.tracking.wrist_pose_provider import EstimatorWristPoseProvider, TrackingSupervisor, TrackingSupervisorConfig
from ego_teleop.tracking.m1_eval import parse_segments, segments_from_events, segment_metrics, segment_verdict
from ego_teleop.transforms.se3 import make_T
from .conftest import T_of

NS = 1_000_000_000


class FakeImu:
    class cfg: rate_hz = 400
    def __init__(self): self.buffer = SampleBuffer(2000); self.last_sample = None
    def handle(self, s): self.buffer.append(s); self.last_sample = s        # mirrors TeensyImu.handle


def imu(seq, t_us, host_ns): return ImuSample("right", seq, t_us, host_ns, 0, 0, 9.81, 0, 0, 0, 30.0)


def test_fanout_tap_receives_every_sample_and_recorder_drain_is_unaffected():
    dev = FakeImu(); dev.handle(imu(0, 0, 0))                     # pre-existing sample carried over
    tap = install_imu_tap(dev, buffer_s=1.0)
    for k in range(1, 401): dev.handle(imu(k, k * 2500, k * 2_500_000 + 400_000))
    assert isinstance(dev.buffer, FanoutBuffer) and len(dev.buffer) == 401 and len(tap) == 400
    recorder_got = dev.buffer.drain(); assert len(recorder_got) == 401 and len(tap) == 400   # drain of the primary leaves the tap alone
    tap2 = install_imu_tap(dev); dev.handle(imu(401, 401 * 2500, 401 * 2_500_000)); assert tap.total == 401 and len(tap) == 400 and len(tap2) == 1   # ring: maxlen 400, total counts all


def test_live_clock_lower_envelope_then_fit():
    c = LiveImuClock(min_for_slope=800)
    rng = np.random.default_rng(0)
    for k in range(300):
        c.update(k * 2500, k * 2_500_000 + 400_000 + int(abs(rng.normal(0, 200_000))))   # one-sided USB latency ~0.4 ms
    assert abs(c.to_host_ns(299 * 2500) - 299 * 2_500_000) < 600_000                     # envelope: within the latency floor
    for k in range(300, 1200): c.update(k * 2500, k * 2_500_000 + 400_000 + int(abs(rng.normal(0, 200_000))))
    assert abs(c.slope_ns_per_us - 1000.0) < 1.0 and abs(c.to_host_ns(1199 * 2500) - 1199 * 2_500_000) < 800_000


class FakeCam:
    def __init__(self): self._f = None
    def publish(self, idx, t): self._f = CameraFrame("right_wrist", idx, t, np.zeros((48, 64, 3), np.uint8))
    def latest(self): return self._f


def test_live_feeder_pushes_full_rate_imu_and_each_frame_once():
    be = MockBackend(lambda t: make_T(None, [0.1 * t / NS, 0, 0]), init_frames=1)
    prov = EstimatorWristPoseProvider(be, T_H_C=np.eye(4), supervisor=TrackingSupervisor(TrackingSupervisorConfig(recover_after_n_ok=1)))
    dev = FakeImu(); tap = install_imu_tap(dev); cam = FakeCam()
    feeder = LiveVioFeeder(prov, imu_tap=tap, camera=cam, downscale=2)
    t = 0
    for fi in range(10):
        for k in range(13): t += 2_500_000; dev.handle(imu(fi * 13 + k, t // 1000, t + 400_000))
        cam.publish(fi, t); feeder.step(); feeder.step()                                    # second step: same frame -> not re-pushed
    assert feeder.stats["frames"] == 10 and feeder.stats["imu"] + feeder.stats["imu_dropped_pre_clock"] == 130 and feeder.stats["imu"] >= 80
    assert prov.get_pose().health == TrackingHealth.OK and prov.stats()["imu"] == feeder.stats["imu"]


def test_vio_stable_gate_requires_still_ok_poses_for_min_time():
    g = VioStableGate(VioStableConfig(min_stable_s=1.0, max_lin_vel_m_s=0.05))
    def wp(t, vel=0.0, health=TrackingHealth.OK, x=0.0): return WristPose.from_T(t, T_of((x, 0, 0)), health=health, linear_velocity_xyz=np.array([vel, 0, 0]))
    assert not g.update(None) and "no pose" in g.reason
    assert not g.update(wp(0, health=TrackingHealth.LOST))
    assert not g.update(wp(0)) and not g.update(wp(int(0.5 * NS)))
    assert g.update(wp(NS)) and g.stable
    assert not g.update(wp(int(1.5 * NS), vel=0.3)) and "moving" in g.reason               # motion resets the gate
    assert not g.update(wp(2 * NS)) and not g.update(wp(int(2.5 * NS), x=0.05)) and "drifting" in g.reason


def test_segments_and_scale_from_known_displacement():
    t0 = 5 * NS; t = t0 + np.arange(0, 20 * 30) * (NS // 30); Ts = []
    for k, tk in enumerate(t):
        s = (tk - t0) / NS
        x = 0.0 if s < 10 else 0.2 * np.sin(np.pi * (s - 10) / 10) ** 2 * 1.0           # S1: 0 -> 20 cm -> 0
        Ts.append(make_T(None, [x, 0, 0]))
    Ts = np.array(Ts); v = np.ones(len(t), bool)
    segs = parse_segments("S0:0-10,S1:10-20", int(t0)); assert segs[1]["t0_ns"] == t0 + 10 * NS
    proto = dict(segments=dict(S0=dict(kind="stationary"), S1=dict(kind="translation", physical_pp_m=0.20)))
    m = segment_metrics(t, Ts, v, segs, proto)
    assert m[0]["kind"] == "stationary" and m[0]["drift_mm"] == 0 and abs(m[1]["scale_ratio"] - 1.0) < 0.02
    assert segment_verdict(m, proto)[0] == "PASS"
    Ts2 = Ts.copy(); Ts2[:, 0, 3] *= 0.6
    assert segment_verdict(segment_metrics(t, Ts2, v, segs, proto), proto)[0] == "FAIL"
    ev = [dict(t_ns=int(t0), kind="segment", detail=dict(name="S0")), dict(t_ns=int(t0 + 10 * NS), kind="segment", detail=dict(name="S1")), dict(t_ns=int(t0), kind="home_leave")]
    assert [s["name"] for s in segments_from_events(ev, int(t[-1]))] == ["S0", "S1"]
