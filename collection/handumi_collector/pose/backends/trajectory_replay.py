"""Trajectory-replay backend: an externally computed camera trajectory (MASt3R-Fusion result.txt, ...) replayed through the SAME
pose pipeline as every live backend (run_side -> canonicalise -> QA -> derived/pose_<name>/<side>_camera_pose.parquet -> m1 report),
so a bake-off candidate is judged with exactly the metrics and gates OpenVINS is judged with. No pose is invented: a frame whose
timestamp has no trajectory row within `max_gap_ms` is LOST (or INITIALIZING before the first row).

    m1_vio EPISODE --side right --backend mast3r --override traj_path=/path/result.txt

Formats
  mast3r_fusion   `stamp_s tx ty tz qx qy qz qw scale bax bay baz bgx bgy bgz frame_id kf_flag` per line (main.py); per-frame rows
                  (flag 0) and keyframe rows (flag 1) share a stamp -- the LAST row written for a stamp wins (keyframe-refined when
                  present). T_WC = camera pose in the estimator's gravity-unaligned world, metres once V-I init has converged.
  tum             `stamp_s tx ty tz qx qy qz qw`
Backends: `mast3r` (raw) and `mast3r_filtered` (filter="spike", see _spike_filter) -> derived/pose_mast3r / pose_mast3r_filtered."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..estimator import BackendInfo, PoseEstimate, PoseEstimator, TrackingState
from ..se3 import pose7_to_T


class TrajectoryReplayBackend(PoseEstimator):
    INFO = BackendInfo(name="trajectory_replay", uses_imu=True, metric_scale=True, online_capable=False, provides=("confidence",),
                       notes="replays an offline trajectory file; validity = a row within max_gap_ms of the frame time")
    info = INFO

    def __init__(self, traj_path: str, fmt: str = "mast3r_fusion", max_gap_ms: float = 40.0, filter: str | None = None, v_max_m_s: float = 2.0,
                 w_max_deg_s: float = 720.0, max_span_rows: int = 2, **_ignored) -> None:
        self.traj_path = Path(traj_path); self.fmt = fmt; self.max_gap_ns = int(float(max_gap_ms) * 1e6)
        if not self.traj_path.exists(): raise FileNotFoundError(f"trajectory file {self.traj_path}")
        self.t_ns, self.p7, self.scale = self._load(self.traj_path, fmt); self.filter_stats = {}
        if filter == "spike": self.t_ns, self.p7, self.scale = self._spike_filter(self.t_ns, self.p7, self.scale, float(v_max_m_s), float(w_max_deg_s), int(max_span_rows))
        elif filter: raise ValueError(f"unknown filter {filter!r}")
        self.reset()

    def _spike_filter(self, t_ns, p7, scale, v_max, w_max_deg, max_span):
        """Output-level post-filter (backend `mast3r_filtered`): an ISOLATED run of <= max_span rows that departs from its neighbours faster
        than v_max (translation) or w_max (rotation) on the way in AND comes back on the way out is a teleport spike -> replaced by linear /
        slerp interpolation between its two valid neighbours (gap <= 120 ms). Anything longer is left alone (no long-gap interpolation,
        no smoothing, no rescaling): a real excursion must stay visible to the downstream plausibility gates."""
        from scipy.spatial.transform import Rotation, Slerp
        t = t_ns / 1e9; P = p7[:, :3].copy(); Q = p7[:, 3:7].copy(); n = len(t); fixed = np.zeros(n, bool)
        dt = np.diff(t); v = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(dt, 1e-3)
        R = Rotation.from_quat(Q); w = np.degrees((R[1:] * R[:-1].inv()).magnitude()) / np.maximum(dt, 1e-3)
        jump = (v > v_max) | (w > w_max_deg)                                          # jump[i] = between row i and i+1
        i = 0
        while i < n - 1:
            if jump[i]:
                for span in range(1, max_span + 1):                                   # rows i+1 .. i+span are the spike if jump[i+span] closes it
                    j = i + span
                    if j < n - 1 and jump[j] and not any(jump[i + 1:j]) and (t[j + 1] - t[i]) <= 0.08 * (max_span + 2):   # rows are ~66 ms apart (15 fps replay)
                        a, b = i, j + 1; s = (t[a + 1:b] - t[a]) / (t[b] - t[a])
                        P[a + 1:b] = P[a] + s[:, None] * (P[b] - P[a])
                        Q[a + 1:b] = Slerp([0.0, 1.0], Rotation.from_quat(np.stack([Q[a], Q[b]])))(s).as_quat()
                        fixed[a + 1:b] = True; i = j; break
                else: i += 1
            else: i += 1
        self.filter_stats = dict(rows=int(n), spikes_fixed=int(fixed.sum()), jump_rows=int(jump.sum()), v_max_m_s=v_max, w_max_deg_s=w_max_deg)
        return t_ns, np.concatenate([P, Q], 1), scale

    @staticmethod
    def _load(path: Path, fmt: str):
        rows = np.loadtxt(path, ndmin=2)
        if rows.size == 0: raise ValueError(f"{path}: empty trajectory")
        if fmt == "mast3r_fusion":
            stamps, p7, scale = rows[:, 0], rows[:, 1:8], rows[:, 8]
        elif fmt == "tum":
            stamps, p7, scale = rows[:, 0], rows[:, 1:8], np.ones(len(rows))
        else: raise ValueError(f"unknown trajectory format {fmt!r}")
        t_ns = np.round(stamps * 1e9).astype(np.int64) if stamps.max() < 1e12 else stamps.astype(np.int64)
        last = {}                                                       # last row per stamp wins
        for i, t in enumerate(t_ns): last[int(t)] = i
        idx = np.array(sorted(last.values(), key=lambda i: t_ns[i]), int)
        return t_ns[idx], p7[idx], scale[idx]

    def initialize(self, **kw) -> None: self._initialized = True
    def reset(self) -> None: self._last = None; self._n = 0; self._hits = 0; self._initialized = False
    def push_imu(self, t_ns: int, gyro_rad_s, accel_m_s2) -> None: pass

    def push_image(self, t_ns: int, frame_index: int, image_bgr) -> PoseEstimate:
        self._n += 1
        j = int(np.searchsorted(self.t_ns, t_ns)); cands = [k for k in (j - 1, j) if 0 <= k < len(self.t_ns)]
        k = min(cands, key=lambda k: abs(int(self.t_ns[k]) - t_ns)) if cands else None
        if k is not None and abs(int(self.t_ns[k]) - t_ns) <= self.max_gap_ns and np.isfinite(self.p7[k]).all():
            self._hits += 1
            est = PoseEstimate(t_ns, frame_index, pose7_to_T(self.p7[k]), TrackingState.TRACKING, confidence=1.0, extra=dict(scale=float(self.scale[k])))
        elif t_ns < int(self.t_ns[0]):
            est = PoseEstimate(t_ns, frame_index, None, TrackingState.INITIALIZING, confidence=0.0)
        else:
            est = PoseEstimate(t_ns, frame_index, None, TrackingState.LOST, confidence=0.0)
        self._last = est; return est

    def get_quality(self) -> dict: return dict(frames=self._n, matched=self._hits, trajectory_rows=int(len(self.t_ns)), source=str(self.traj_path), **self.filter_stats)


class MASt3RFilteredBackend(TrajectoryReplayBackend):
    INFO = BackendInfo(name="mast3r_filtered", uses_imu=True, metric_scale=True, online_capable=False, provides=("confidence",), notes="MASt3R-Fusion replay + isolated teleport-spike post-filter")
    info = INFO

    def __init__(self, traj_path: str, **kw) -> None:
        kw.setdefault("filter", "spike"); super().__init__(traj_path, **kw)
