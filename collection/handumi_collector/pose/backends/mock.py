"""Mock backend: replays a ground-truth camera trajectory (given as a callable t_ns -> 4x4 or an array) with optional
noise, drift and LOST segments. Ignores pixels. Exists for tests and for exercising the pipeline/QA/inspector/benchmark
without a real VIO; it is never selectable in production (`pose.yaml backend: mock` is refused by tools unless --allow-mock)."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from ..estimator import BackendInfo, PoseEstimate, PoseEstimator, TrackingState
from ..se3 import make_T


class MockBackend(PoseEstimator):
    INFO = BackendInfo("mock", uses_imu=True, metric_scale=True, online_capable=True, provides=("confidence", "num_features"),
                       notes="ground-truth replay for tests; never for data")
    info = INFO

    def __init__(self, trajectory=None, *, noise_pos_m: float = 0.0, noise_rot_deg: float = 0.0, drift_m_per_s: float = 0.0,
                 lost_windows_ns: list[tuple[int, int]] | None = None, init_frames: int = 3, seed: int = 0, fail_on_init: bool = False,
                 raise_on_frame: int | None = None, **_ignored) -> None:
        self.traj = trajectory if trajectory is not None else (lambda t_ns: np.eye(4))
        self.noise_pos, self.noise_rot, self.drift = noise_pos_m, np.radians(noise_rot_deg), drift_m_per_s
        self.lost = lost_windows_ns or []
        self.init_frames = init_frames
        self.rng = np.random.default_rng(seed)
        self.fail_on_init = fail_on_init; self.raise_on_frame = raise_on_frame
        self.reset()

    def initialize(self, **kw) -> None:
        if self.fail_on_init: raise RuntimeError("mock backend: initialize failure (test)")
        self._initialized = True

    def reset(self) -> None:
        self._n = 0; self._t0 = None; self._last = None; self._imu = 0; self._initialized = False

    def push_imu(self, t_ns: int, gyro_rad_s, accel_m_s2) -> None:
        self._imu += 1

    def push_image(self, t_ns: int, frame_index: int, image_bgr) -> PoseEstimate:
        if self.raise_on_frame is not None and frame_index == self.raise_on_frame: raise RuntimeError("mock backend: crash on frame (test)")
        self._n += 1
        if self._t0 is None: self._t0 = t_ns
        if self._n <= self.init_frames:
            est = PoseEstimate(t_ns, frame_index, None, TrackingState.INITIALIZING, confidence=0.0, num_features=0)
        elif any(a <= t_ns <= b for a, b in self.lost):
            est = PoseEstimate(t_ns, frame_index, None, TrackingState.LOST, confidence=0.0, num_features=0)
        else:
            T = np.array(self.traj(t_ns), np.float64).copy()
            dt = (t_ns - self._t0) / 1e9
            T[:3, 3] += np.array([self.drift * dt, 0, 0]) + self.rng.normal(0, self.noise_pos, 3)
            if self.noise_rot > 0:
                T[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(self.rng.normal(0, self.noise_rot, 3)).as_matrix()
            est = PoseEstimate(t_ns, frame_index, T, TrackingState.TRACKING, confidence=1.0, num_features=500)
        self._last = est
        return est

    def get_quality(self) -> dict: return dict(frames=self._n, imu_samples=self._imu)


def circle_trajectory(radius_m: float = 0.15, period_s: float = 6.0, height_m: float = 0.0, t0_ns: int = 0):
    """Test trajectory: camera moves on a horizontal circle while yawing; the same pose at t0 and t0+period (HOME return)."""
    def f(t_ns: int) -> np.ndarray:
        s = (t_ns - t0_ns) / 1e9; a = 2 * np.pi * s / period_s
        R = Rotation.from_euler("z", a).as_matrix()
        return make_T(R, [radius_m * np.cos(a) - radius_m, radius_m * np.sin(a), height_m])
    return f


def home_manip_home_trajectory(t_start_ns: int, t_stop_ns: int, *, still_s: float = 1.5, amp_m: float = 0.2, rot_deg: float = 40.0):
    """HOME (still) -> manipulation bump -> HOME (still, identical pose). Used to test HOME-return QA end-to-end."""
    def f(t_ns: int) -> np.ndarray:
        s = (t_ns - t_start_ns) / 1e9; total = (t_stop_ns - t_start_ns) / 1e9
        if s <= still_s or s >= total - still_s: return np.eye(4)
        u = (s - still_s) / max(total - 2 * still_s, 1e-6)          # 0..1 across the manipulation
        b = np.sin(np.pi * u) ** 2
        R = Rotation.from_euler("y", np.radians(rot_deg) * b).as_matrix()
        return make_T(R, [amp_m * b, 0.5 * amp_m * np.sin(2 * np.pi * u) * b, 0.3 * amp_m * b])
    return f
