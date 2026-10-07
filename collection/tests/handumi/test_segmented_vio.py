"""Segmented relative VIO. A fake inner backend replays a known camera trajectory in its OWN world frame (a different
random offset per instance, like a re-initialised OpenVINS); synthetic IMU says still / moving. What must hold:
static frames carry delta p = 0 and the gyro's rotation; moving frames carry the true relative motion; the stitch across a
segment boundary is continuous; each segment's backend saw the static PREFIX before its motion; a slow pure translation
(no gyro, no acceleration, moving picture) is NOT called static; and nothing is ever faked during a segment's init lag."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from handumi_collector.pose.backends.segmented import SegmentedEstimator, static_windows
from handumi_collector.pose.estimator import BackendInfo, PoseEstimate, PoseEstimator, TrackingState
from handumi_collector.pose.se3 import inv_T

NS = 1_000_000_000; HZ_IMU = 200; FPS = 30


def make_T(R=None, t=(0, 0, 0)):
    T = np.eye(4); T[:3, :3] = np.eye(3) if R is None else R; T[:3, 3] = t; return T


class Truth:
    """Camera trajectory: still 0-2 s, move +x at 0.2 m/s with a slow yaw 2-5 s, still 5-7 s, move -y 7-9 s, still 9-10 s."""
    def __init__(self): self.plan = [(0, 2, (0, 0, 0), 0.0), (2, 5, (0.2, 0, 0), 20.0), (5, 7, (0, 0, 0), 0.0), (7, 9, (0, -0.15, 0), 0.0), (9, 10, (0, 0, 0), 0.0)]
    def pose(self, t):
        p = np.zeros(3); yaw = 0.0
        for a, b, v, w in self.plan:
            d = np.clip(t - a, 0, b - a); p = p + np.asarray(v) * d; yaw += w * d
        return make_T(Rotation.from_euler("z", yaw, degrees=True).as_matrix(), p)
    def gyro(self, t):
        for a, b, v, w in self.plan:
            if a <= t < b: return np.array([0, 0, np.radians(w)])
        return np.zeros(3)
    def moving(self, t): return any(a <= t < b and (np.linalg.norm(v) > 0 or w) for a, b, v, w in self.plan)


class FakeInner(PoseEstimator):
    """Valid `lag` seconds after the first frame it sees, then reports the truth in its own randomly offset world."""
    info = BackendInfo("fake", uses_imu=True, metric_scale=True, online_capable=False)
    instances: list = []
    def __init__(self, truth, lag=0.4):
        self.truth, self.lag = truth, lag; self.first_t = None; self.imu = []; self.frames = []
        rng = np.random.default_rng(len(FakeInner.instances)); self.offset = make_T(Rotation.random(random_state=rng).as_matrix(), rng.normal(size=3))
        FakeInner.instances.append(self)
    def initialize(self, **kw): pass
    def push_imu(self, t_ns, g, a): self.imu.append(t_ns)
    def push_image(self, t_ns, fi, img):
        self.frames.append(t_ns); t = t_ns / NS
        if self.first_t is None: self.first_t = t
        if t - self.first_t < self.lag: return PoseEstimate(t_ns, fi, None, TrackingState.INITIALIZING)
        return PoseEstimate(t_ns, fi, self.offset @ self.truth.pose(t), TrackingState.TRACKING, confidence=1.0, num_features=50)
    def finish(self): return None


def synth_imu(truth, dur=10.0, tremor_deg_s=1.0, seed=0, motion_jitter=0.4):
    rng = np.random.default_rng(seed); t = np.arange(0, dur, 1 / HZ_IMU); tn = (t * NS).astype(np.int64)
    gyro = np.array([truth.gyro(x) for x in t]) + np.radians(tremor_deg_s) * rng.normal(size=(len(t), 3)) / np.sqrt(3)
    acc = np.tile([0, 0, 9.81], (len(t), 1)) + 0.03 * rng.normal(size=(len(t), 3))
    for i, x in enumerate(t):                      # a real hand accelerates unevenly while it moves; motion_jitter=0 = ideal constant velocity
        if truth.moving(x) and motion_jitter: acc[i] += motion_jitter * rng.normal(size=3)
    return tn, gyro, acc


def frames(dur=10.0):
    return (np.arange(0, dur, 1 / FPS) * NS).astype(np.int64)


def run(truth, cfg=None, imgs=None, lag=0.4, motion_jitter=0.4):
    FakeInner.instances = []
    tn, g, a = synth_imu(truth, motion_jitter=motion_jitter); tf = frames()
    est = SegmentedEstimator(lambda: FakeInner(truth, lag), imu_t_ns=tn, gyro=g, accel=a, T_camera_imu=np.eye(4), cfg=dict(cfg or {}, enabled=True))
    est.initialize(intrinsics=None, T_camera_imu=np.eye(4))
    out = []; j = 0
    blank = np.zeros((60, 80, 3), np.uint8)
    for k, t in enumerate(tf):
        while j < len(tn) and tn[j] <= t: est.push_imu(int(tn[j]), g[j], a[j]); j += 1
        out.append(est.push_image(int(t), k, blank if imgs is None else imgs(k)))
    est.finish(); return est, out, tf


def rel(T0, T1): return inv_T(T0) @ T1


def test_static_windows_need_gyro_and_accel_and_duration():
    truth = Truth(); tn, g, a = synth_imu(truth)
    w = static_windows(tn, g, a, dict(min_static_s=0.5, gyro_max_deg_s=4.0, accel_std_max=0.25, max_gap_ms=25.0, window_s=0.2))
    spans = [(round(x["t0_ns"] / NS, 1), round(x["t1_ns"] / NS, 1)) for x in w]
    assert len(spans) == 3 and spans[0][0] <= 0.2 and 1.8 <= spans[0][1] <= 2.2 and 4.8 <= spans[1][0] <= 5.3 and 6.8 <= spans[1][1] <= 7.2, spans
    # gyro alone is not enough: a still gyro with a shaking accelerometer is not static
    a2 = a.copy(); a2[:, 0] += 0.6 * np.sin(np.arange(len(a2)) * 3.0)
    assert static_windows(tn, g, a2, dict(min_static_s=0.5, gyro_max_deg_s=4.0, accel_std_max=0.25, max_gap_ms=25.0, window_s=0.2)) == []


def test_static_frames_hold_translation_and_moving_frames_carry_true_relative_motion():
    truth = Truth(); est, out, tf = run(truth)
    modes = [r["mode"] for r in est.trace]
    for e, m, t in zip(out, modes, tf / NS):
        if m == "static": assert e.tracking_state == TrackingState.STATIC and e.valid
    # consecutive valid pairs: static -> zero translation; moving (both valid) -> the truth's relative translation
    for i in range(1, len(out)):
        e0, e1 = out[i - 1], out[i]
        if not (e0.valid and e1.valid): continue
        d = rel(e0.T_world_camera, e1.T_world_camera); dtrue = rel(truth.pose(tf[i - 1] / NS), truth.pose(tf[i] / NS))
        if modes[i] == "static" and modes[i - 1] == "static":
            assert np.linalg.norm(d[:3, 3]) < 1e-9, "static must hold translation exactly"
            assert np.degrees(Rotation.from_matrix(d[:3, :3]).magnitude()) < 0.3
        elif modes[i] == "moving" and modes[i - 1] == "moving":
            assert np.allclose(d[:3, 3], dtrue[:3, 3], atol=2e-3), (i, d[:3, 3], dtrue[:3, 3])
            assert np.allclose(d[:3, :3], dtrue[:3, :3], atol=1e-3)
    # every moving stretch eventually became valid, and each one used a DIFFERENT inner backend (re-init), fed a prefix
    segs = [inst for inst in FakeInner.instances if inst.frames and any(True for _ in inst.frames) and inst.first_t is not None and len(inst.frames) > 15]
    assert len(segs) >= 2, "two moving stretches -> at least two inner backends"
    for inst in segs:
        first_frame_s = inst.frames[0] / NS
        assert any(abs(first_frame_s - (a - 2.0)) < 0.25 or first_frame_s < 0.1 for a, b, v, w in truth.plan if truth.moving(a + 0.01)), \
            f"segment backend started at {first_frame_s:.2f}s -- should be ~2 s before a motion start (static prefix)"
        assert min(inst.imu) <= inst.frames[0], "IMU prefix fed before the first frame"


def test_prefix_makes_the_segment_valid_from_its_first_moving_frame():
    """The point of the static prefix: the backend initialises BEFORE the motion, so nothing is lost at the edge."""
    truth = Truth(); est, out, tf = run(truth, lag=0.4)
    modes = [r["mode"] for r in est.trace]
    first_static = next(i for i, m in enumerate(modes) if m == "static"); first_moving = next(i for i, m in enumerate(modes) if m == "moving" and i > first_static)
    assert out[first_moving].valid and out[first_moving].tracking_state == TrackingState.TRACKING


def test_boundary_is_continuous_and_init_lag_is_reported_missing_not_held():
    truth = Truth(); est, out, tf = run(truth, lag=2.5)                        # lag longer than the 2 s prefix -> a real init gap
    modes = [r["mode"] for r in est.trace]
    # init lag: the first frames of a moving stretch are INITIALIZING (invalid), never a held pose
    first_static = next(i for i, m in enumerate(modes) if m == "static"); first_moving = next(i for i, m in enumerate(modes) if m == "moving" and i > first_static)
    assert out[first_moving].tracking_state == TrackingState.INITIALIZING and not out[first_moving].valid
    # across the static -> moving boundary, position is continuous (the anchor lands on the held pose; no jump)
    i_valid = next(i for i in range(first_moving, len(out)) if out[i].valid)
    last_static = max(i for i in range(first_moving) if out[i].valid)
    jump = np.linalg.norm(out[i_valid].T_world_camera[:3, 3] - out[last_static].T_world_camera[:3, 3])
    true_gap = np.linalg.norm(truth.pose(tf[i_valid] / NS)[:3, 3] - truth.pose(tf[last_static] / NS)[:3, 3])
    assert jump < 1e-6, f"stitch must land on the held pose (jump {jump*1e3:.1f} mm); the {true_gap*1e3:.0f} mm of motion during init lag is reported as missing frames, not invented"
    q = est.get_quality(); assert q["static_windows"] == 3 and q["segments_started"] >= 2 and q["frames_static"] > 0


def test_pure_slow_translation_with_moving_picture_is_not_static():
    """No rotation, constant velocity -> the IMU sees nothing; the picture does. The guard must keep it moving."""
    class Slide(Truth):
        def __init__(self): self.plan = [(0, 2, (0, 0, 0), 0.0), (2, 8, (0.05, 0, 0), 0.0), (8, 10, (0, 0, 0), 0.0)]
    truth = Slide(); rng = np.random.default_rng(1); tex = rng.integers(0, 255, (60, 400), np.uint8)
    def imgs(k):
        t = k / FPS; shift = int(0 if not truth.moving(t) else (t - 2.0) * 40) % 300     # ~40 px/s drift of a textured scene
        return np.repeat(tex[:, shift:shift + 80, None], 3, axis=2)
    est, out, tf = run(truth, imgs=imgs, motion_jitter=0.0)          # ideal constant velocity: the IMU sees NOTHING
    modes = np.array([r["mode"] for r in est.trace]); tt = tf / NS
    mid = (tt > 3.0) & (tt < 7.5)
    assert (modes[mid] == "moving").mean() > 0.9, f"slow pure translation was called static in {(modes[mid]=='static').mean():.0%} of frames"
    assert (modes[(tt > 0.3) & (tt < 1.8)] == "static").all(), "the truly still start is still static"
    assert est.get_quality()["frames_forced_moving"] > 0


def test_starved_segment_goes_lost_instead_of_inventing_motion():
    """The inner filter keeps emitting poses with zero visual features (IMU-only). After starve_frames the wrapper must stop
    trusting it: LOST until the next static window, and no jump when the next segment re-anchors."""
    class Starving(FakeInner):
        def push_image(self, t_ns, fi, img):
            e = super().push_image(t_ns, fi, img); t = t_ns / NS
            if e.valid and 3.0 <= t < 4.5:                       # 1.5 s without features in the middle of the first motion
                e.num_features = 0; e.T_world_camera = e.T_world_camera.copy(); e.T_world_camera[:3, 3] += 5.0   # garbage pose
            return e
    truth = Truth(); FakeInner.instances = []
    tn, g, a = synth_imu(truth); tf = frames()
    est = SegmentedEstimator(lambda: Starving(truth, 0.4), imu_t_ns=tn, gyro=g, accel=a, T_camera_imu=np.eye(4), cfg=dict(enabled=True, starve_frames=10, starve_min_features=1))
    est.initialize(intrinsics=None, T_camera_imu=np.eye(4)); out = []; j = 0; blank = np.zeros((60, 80, 3), np.uint8)
    for k, t in enumerate(tf):
        while j < len(tn) and tn[j] <= t: est.push_imu(int(tn[j]), g[j], a[j]); j += 1
        out.append(est.push_image(int(t), k, blank))
    tt = tf / NS
    starved = [e for e, t in zip(out, tt) if 3.4 <= t < 4.9]; assert all(not e.valid and e.tracking_state == TrackingState.LOST for e in starved)
    late = [e for e, t in zip(out, tt) if 4.6 <= t < 5.0]; assert all(not e.valid for e in late), "a dead segment stays dead until the next static window"
    assert any(e.valid for e, t in zip(out, tt) if 7.5 <= t < 9.0), "the next moving segment recovers with a fresh backend"
    for i in range(1, len(out)):
        if out[i].valid and out[i - 1].valid: assert np.linalg.norm(out[i].T_world_camera[:3, 3] - out[i - 1].T_world_camera[:3, 3]) < 0.05, "no invented jumps"
    assert est.get_quality()["starved_segments"] == 1
