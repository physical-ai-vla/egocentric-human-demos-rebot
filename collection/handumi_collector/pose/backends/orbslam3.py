"""ORB-SLAM3 Monocular-Inertial adapter (Backend A, visual-inertial).

ORB-SLAM3 is a C++ system; no maintained Python binding is installed here, so this adapter works as an *external process*
backend in two modes:
  1. export  — write the episode's wrist stream as a EuRoC-style bundle (cam0/data/*.png + cam0/data.csv, imu0/data.csv,
               timestamps.txt) and an ORB-SLAM3 settings yaml (KannalaBrandt8 from fisheye_<side>_vNNN, T_b_c from
               camera_imu_<side>_vNNN, IMU noise), then run `mono_inertial_euroc` if `orbslam3.binary` is configured.
  2. import  — read the TUM-format trajectory ORB-SLAM3 writes (f_<name>.txt / CameraTrajectory.txt: t x y z qx qy qz qw,
               T_world_camera) back into PoseEstimates aligned to frame timestamps.
Not being able to run is reported as BackendUnavailable — never approximated."""
from __future__ import annotations
import csv
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from ..estimator import BackendInfo, BackendUnavailable, PoseEstimate, PoseEstimator, TrackingState
from ..se3 import pose7_to_T


class OrbSlam3Backend(PoseEstimator):
    INFO = BackendInfo("orbslam3", uses_imu=True, metric_scale=True, online_capable=False, provides=(),
                       notes="external process (EuRoC export -> mono_inertial_euroc -> TUM trajectory import); metrics per frame not exposed")
    info = INFO

    def __init__(self, *, binary: str | None = None, vocabulary: str | None = None, work_dir: str | None = None, trajectory_file: str | None = None,
                 keep_export: bool = False, **_ignored) -> None:
        self.binary = binary or os.environ.get("ORB_SLAM3_MONO_INERTIAL")
        self.vocab = vocabulary or os.environ.get("ORB_SLAM3_VOCAB")
        self.work_dir = Path(work_dir) if work_dir else None
        self.trajectory_file = Path(trajectory_file) if trajectory_file else None
        self.keep_export = keep_export
        self.reset()

    def reset(self) -> None:
        self._frames: list[tuple[int, int, np.ndarray]] = []; self._imu: list[tuple[int, np.ndarray, np.ndarray]] = []; self._last = None
        self._intr = None; self._T_c_i = None; self._noise = None; self._size = None

    def initialize(self, *, intrinsics=None, T_camera_imu=None, imu_noise=None, image_size=None) -> None:
        if self.trajectory_file is None:
            if not self.binary or not Path(self.binary).exists():
                raise BackendUnavailable("orbslam3: set pose.yaml backend_options.orbslam3.binary (or $ORB_SLAM3_MONO_INERTIAL) to the mono_inertial_euroc "
                                         "executable, or pass trajectory_file= to import a trajectory produced elsewhere")
            if not self.vocab or not Path(self.vocab).exists(): raise BackendUnavailable("orbslam3: ORBvoc.txt path missing (vocabulary / $ORB_SLAM3_VOCAB)")
            if intrinsics is None: raise BackendUnavailable("orbslam3: fisheye intrinsics (fisheye_<side>_vNNN.yaml) required")
            if T_camera_imu is None: raise BackendUnavailable("orbslam3: T_camera_imu (camera_imu_<side>_vNNN.yaml) required for visual-inertial")
        self._intr, self._T_c_i, self._noise, self._size = intrinsics, T_camera_imu, imu_noise or {}, image_size

    def push_imu(self, t_ns: int, gyro_rad_s, accel_m_s2) -> None:
        self._imu.append((int(t_ns), np.asarray(gyro_rad_s, np.float64), np.asarray(accel_m_s2, np.float64)))

    def push_image(self, t_ns: int, frame_index: int, image_bgr) -> PoseEstimate:
        self._frames.append((int(t_ns), int(frame_index), image_bgr))
        est = PoseEstimate(t_ns, frame_index, None, TrackingState.INITIALIZING); self._last = est
        return est

    # ------------------------------------------------------------------ batch
    def finish(self) -> list[PoseEstimate]:
        if self.trajectory_file is not None:
            traj = read_tum_trajectory(self.trajectory_file)
        else:
            wd = self.work_dir or Path.cwd() / "orbslam3_work"
            self.export_euroc(wd)
            settings = self.write_settings(wd / "settings.yaml")
            out = wd / "traj"; out.mkdir(exist_ok=True)
            cmd = [self.binary, self.vocab, str(settings), str(wd / "mav0"), str(wd / "timestamps.txt"), str(out / "run")]
            r = subprocess.run(cmd, capture_output=True, text=True)
            (wd / "orbslam3.log").write_text(r.stdout + "\n" + r.stderr)
            cands = sorted(out.glob("f_*.txt")) or sorted(Path.cwd().glob("f_run.txt")) or sorted(Path.cwd().glob("CameraTrajectory.txt"))
            if r.returncode != 0 or not cands: raise RuntimeError(f"orbslam3 failed (rc={r.returncode}); see {wd / 'orbslam3.log'}")
            traj = read_tum_trajectory(cands[0])
            if not self.keep_export: shutil.rmtree(wd / "mav0", ignore_errors=True)
        return align_trajectory_to_frames(traj, self._frames)

    def export_euroc(self, wd: Path) -> Path:
        import cv2
        cam = wd / "mav0" / "cam0" / "data"; imu = wd / "mav0" / "imu0"; cam.mkdir(parents=True, exist_ok=True); imu.mkdir(parents=True, exist_ok=True)
        with open(wd / "mav0" / "cam0" / "data.csv", "w", newline="") as f, open(wd / "timestamps.txt", "w") as ft:
            w = csv.writer(f); w.writerow(["#timestamp [ns]", "filename"])
            for t, _fi, img in self._frames:
                cv2.imwrite(str(cam / f"{t}.png"), img); w.writerow([t, f"{t}.png"]); ft.write(f"{t}\n")
        with open(imu / "data.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["#timestamp [ns]", "w_RS_S_x", "w_RS_S_y", "w_RS_S_z", "a_RS_S_x", "a_RS_S_y", "a_RS_S_z"])
            for t, g, a in self._imu: w.writerow([t, *g, *a])
        return wd

    def write_settings(self, path: Path) -> Path:
        K, D = np.asarray(self._intr["K"]), np.asarray(self._intr["D"]); w, h = self._size or self._intr["image_size"]
        n = self._noise
        T = np.asarray(self._T_c_i, np.float64)          # ORB-SLAM3 wants T_b_c (camera in IMU/body): inverse of ours
        Tbc = np.linalg.inv(T)
        fps = 30.0
        lines = ["%YAML:1.0", "File.version: \"1.0\"", "Camera.type: \"KannalaBrandt8\"",
                 f"Camera1.fx: {K[0,0]}", f"Camera1.fy: {K[1,1]}", f"Camera1.cx: {K[0,2]}", f"Camera1.cy: {K[1,2]}",
                 f"Camera1.k1: {D[0]}", f"Camera1.k2: {D[1]}", f"Camera1.k3: {D[2]}", f"Camera1.k4: {D[3]}",
                 f"Camera.width: {int(w)}", f"Camera.height: {int(h)}", f"Camera.fps: {fps}", "Camera.RGB: 1",
                 "IMU.T_b_c1: !!opencv-matrix", "   rows: 4", "   cols: 4", "   dt: f",
                 "   data: [" + ", ".join(f"{v:.9f}" for v in Tbc.reshape(-1)) + "]",
                 f"IMU.NoiseGyro: {n.get('gyro_noise_density', 1.7e-4)}", f"IMU.NoiseAcc: {n.get('accel_noise_density', 2.0e-3)}",
                 f"IMU.GyroWalk: {n.get('gyro_random_walk', 1.9e-5)}", f"IMU.AccWalk: {n.get('accel_random_walk', 3.0e-3)}",
                 f"IMU.Frequency: {n.get('rate_hz', 400.0)}",
                 "ORBextractor.nFeatures: 1500", "ORBextractor.scaleFactor: 1.2", "ORBextractor.nLevels: 8",
                 "ORBextractor.iniThFAST: 20", "ORBextractor.minThFAST: 7", "Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1",
                 "Viewer.GraphLineWidth: 0.9", "Viewer.PointSize: 2", "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3",
                 "Viewer.ViewpointX: 0", "Viewer.ViewpointY: -0.7", "Viewer.ViewpointZ: -3.5", "Viewer.ViewpointF: 500"]
        path.write_text("\n".join(lines) + "\n"); return path


def read_tum_trajectory(path: Path) -> list[tuple[int, np.ndarray]]:
    """TUM: `t x y z qx qy qz qw` (t in s or ns) → [(t_ns, T_world_camera)]."""
    out = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"): continue
        v = [float(x) for x in s.split()]
        if len(v) < 8: continue
        t = v[0]; t_ns = int(round(t)) if t > 1e12 else int(round(t * 1e9))
        out.append((t_ns, pose7_to_T(v[1:8])))
    return out


def align_trajectory_to_frames(traj: list[tuple[int, np.ndarray]], frames: list[tuple[int, int, object]], *, tol_ns: int = 2_000_000) -> list[PoseEstimate]:
    """Frames without a trajectory entry (ORB-SLAM3 drops frames while initialising or lost) are LOST/INITIALIZING, not interpolated."""
    ts = np.array([t for t, _ in traj], np.int64); Ts = [T for _, T in traj]
    out = []; seen_valid = False
    for t, fi, _img in frames:
        if len(ts) == 0: out.append(PoseEstimate(t, fi, None, TrackingState.LOST)); continue
        k = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1)); k = k if abs(ts[k] - t) <= abs(ts[max(k - 1, 0)] - t) else max(k - 1, 0)
        if abs(int(ts[k]) - t) <= tol_ns:
            out.append(PoseEstimate(t, fi, Ts[k], TrackingState.TRACKING)); seen_valid = True
        else:
            out.append(PoseEstimate(t, fi, None, TrackingState.LOST if seen_valid else TrackingState.INITIALIZING))
    return out
