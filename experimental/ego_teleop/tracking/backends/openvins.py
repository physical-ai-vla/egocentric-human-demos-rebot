"""OpenVINS (MSCKF, visual-inertial, metric, online) as a handumi_collector PoseEstimator — M1-A candidate.

OpenVINS is GPL-3.0 and is NOT vendored: it is built in ~/vio/open_vins and reached through the small pybind11 module
~/vio/ov_bridge (see build.sh there). This file only (1) writes an OpenVINS config set from OUR versioned calibration
(fisheye_<side> Kannala-Brandt -> `distortion_model: equidistant`; camera_imu_<side> T_camera_imu -> `T_imu_cam`), (2) feeds
IMU + grayscale frames with their ORIGINAL timestamps (one clock; the offline pipeline already shifted IMU onto the camera
clock, so timeshift_cam_imu starts at 0 and OpenVINS may refine it), and (3) turns the filter state into PoseEstimate:
T_world_camera = T_G_I · T_I_C (current extrinsic estimate). Absence of the bridge/calibration => BackendUnavailable, never fake."""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import yaml
from handumi_collector.pose.estimator import BackendInfo, BackendUnavailable, PoseEstimate, PoseEstimator, TrackingState
from handumi_collector.pose.se3 import inv_T

DEFAULT_BRIDGE_DIRS = (os.environ.get("OV_BRIDGE_DIR", ""), str(Path.home() / "vio/ov_bridge/build"))

# ICM42688P placeholders (datasheet x~4 inflation, the usual VIO practice) until an Allan-variance run exists. Units: Kalibr.
ICM42688P_NOISE = dict(gyroscope_noise_density=2.0e-4, gyroscope_random_walk=2.0e-5, accelerometer_noise_density=2.0e-3,
                       accelerometer_random_walk=3.0e-3, update_rate=400.0)

ESTIMATOR_DEFAULTS = dict(
    verbosity="WARNING", use_fej=True, integration="rk4", use_stereo=False, max_cameras=1,
    calib_cam_extrinsics=False, calib_cam_intrinsics=False, calib_cam_timeoffset=True, calib_imu_intrinsics=False, calib_imu_g_sensitivity=False,
    max_clones=11, max_slam=50, max_slam_in_update=25, max_msckf_in_update=40, dt_slam_delay=1, gravity_mag=9.81,
    feat_rep_msckf="GLOBAL_3D", feat_rep_slam="ANCHORED_MSCKF_INVERSE_DEPTH", feat_rep_aruco="ANCHORED_MSCKF_INVERSE_DEPTH",
    try_zupt=True, zupt_chi2_multipler=0, zupt_max_velocity=0.1, zupt_noise_multiplier=10, zupt_max_disparity=0.5, zupt_only_at_beginning=False,
    init_window_time=1.0, init_imu_thresh=1.0, init_max_disparity=10.0, init_max_features=50, init_dyn_use=False,
    init_dyn_mle_opt_calib=False, init_dyn_mle_max_iter=50, init_dyn_mle_max_time=0.05, init_dyn_mle_max_threads=6, init_dyn_num_pose=6,
    init_dyn_min_deg=10.0, init_dyn_inflation_ori=10, init_dyn_inflation_vel=100, init_dyn_inflation_bg=10, init_dyn_inflation_ba=100,
    init_dyn_min_rec_cond=1e-12, init_dyn_bias_g=[0.0, 0.0, 0.0], init_dyn_bias_a=[0.0, 0.0, 0.0],
    record_timing_information=False, record_timing_filepath="/tmp/ov_msckf_timing.txt", save_total_state=False,
    use_klt=True, num_pts=250, fast_threshold=15, grid_x=5, grid_y=5, min_px_dist=10, knn_ratio=0.70, track_frequency=30.0,
    downsample_cameras=False, num_opencv_threads=4, histogram_method="HISTOGRAM", use_aruco=False, num_aruco=1024, downsize_aruco=True,
    up_msckf_sigma_px=1, up_msckf_chi2_multipler=1, up_slam_sigma_px=1, up_slam_chi2_multipler=1, up_aruco_sigma_px=1, up_aruco_chi2_multipler=1,
    use_mask=False, relative_config_imu="kalibr_imu_chain.yaml", relative_config_imucam="kalibr_imucam_chain.yaml",
)


class _CvDumper(yaml.SafeDumper):
    """OpenCV's YAML reader wants block sequences indented UNDER their key (PyYAML's default 'indentless' style fails to parse)."""
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)

    def ignore_aliases(self, data):      # OpenCV's reader has no anchors/aliases; repeated identity matrices must be written out
        return True


# Calibration policy (2026-09-10): offline calibration (Kalibr, checkerboard/circle-grid) is PRIMARY; OpenVINS online calibration only
# refines and sanity-checks it. CALIBRATION mode optimises the extrinsic + time offset and traces them per frame so convergence can be
# compared ACROSS takes (tools.calib_converge); PRODUCTION mode loads the versioned values and keeps them fixed.
MODE_OVERRIDES = {
    "production":  dict(calib_cam_extrinsics=False, calib_cam_intrinsics=False, calib_cam_timeoffset=False),
    "calibration": dict(calib_cam_extrinsics=True, calib_cam_intrinsics=False, calib_cam_timeoffset=True),
}


def _yaml(obj: dict) -> str:
    """OpenVINS uses OpenCV's YAML reader: `%YAML:1.0` header, leaf lists in flow style, nested lists indented."""
    return "%YAML:1.0\n" + yaml.dump(obj, Dumper=_CvDumper, sort_keys=False, default_flow_style=None)


def write_openvins_config(out_dir: str | Path, *, intrinsics: dict, T_camera_imu: np.ndarray, imu_noise: dict | None = None,
                          downscale: int = 1, timeshift_cam_imu_s: float = 0.0, overrides: dict | None = None) -> Path:
    """Write estimator_config.yaml + kalibr_imucam_chain.yaml + kalibr_imu_chain.yaml; returns the estimator_config path.
    intrinsics: {model: kannala_brandt|pinhole, K 3x3, D (4,), image_size (w,h)} at FULL resolution; `downscale` rescales."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    K = np.asarray(intrinsics["K"], np.float64) / float(downscale); D = np.asarray(intrinsics.get("D", np.zeros(4)), np.float64).reshape(-1)[:4]
    w, h = (int(v) // int(downscale) for v in intrinsics["image_size"])
    model = intrinsics.get("model", "kannala_brandt")
    dist = "equidistant" if model in ("kannala_brandt", "equidistant", "fisheye") else "radtan"
    T_C_I = np.asarray(T_camera_imu, np.float64).reshape(4, 4)
    T_imu_cam = inv_T(T_C_I)                                   # OpenVINS: T_imu_cam = camera pose in IMU frame (T_CtoI)
    cam = {"cam0": dict(T_imu_cam=[[float(x) for x in row] for row in T_imu_cam], cam_overlaps=[], camera_model="pinhole", distortion_model=dist,
                        distortion_coeffs=[float(x) for x in (list(D) + [0.0] * 4)[:4]], intrinsics=[float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
                        resolution=[w, h], timeshift_cam_imu=float(timeshift_cam_imu_s), rostopic="/cam0/image_raw")}
    noise = dict(ICM42688P_NOISE); noise.update(imu_noise or {})
    eye3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    imu = {"imu0": dict(T_i_b=[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
                        accelerometer_noise_density=float(noise["accelerometer_noise_density"]), accelerometer_random_walk=float(noise["accelerometer_random_walk"]),
                        gyroscope_noise_density=float(noise["gyroscope_noise_density"]), gyroscope_random_walk=float(noise["gyroscope_random_walk"]),
                        rostopic="/imu0", time_offset=0.0, update_rate=float(noise.get("update_rate", 400.0)), model="kalibr",
                        Tw=eye3, R_IMUtoGYRO=eye3, Ta=eye3, R_IMUtoACC=eye3, Tg=[[0.0] * 3] * 3)}
    est = dict(ESTIMATOR_DEFAULTS); est.update(overrides or {})
    (out / "kalibr_imucam_chain.yaml").write_text(_yaml(cam)); (out / "kalibr_imu_chain.yaml").write_text(_yaml(imu))
    p = out / "estimator_config.yaml"; p.write_text(_yaml(est)); return p


def import_bridge(bridge_dir: str | None = None):
    for d in ([bridge_dir] if bridge_dir else []) + [x for x in DEFAULT_BRIDGE_DIRS if x]:
        if d and Path(d).exists() and d not in sys.path: sys.path.insert(0, d)
    try:
        import ov_bridge  # type: ignore
        return ov_bridge
    except ImportError as exc:
        raise BackendUnavailable(f"openvins: ov_bridge module not importable ({exc}); build it with ~/vio/ov_bridge/build.sh") from exc


class OpenVinsBackend(PoseEstimator):
    INFO = BackendInfo("openvins", uses_imu=True, metric_scale=True, online_capable=True, provides=("confidence", "num_features"),
                       notes="OpenVINS MSCKF via ~/vio/ov_bridge (GPL-3.0, external build); equidistant fisheye; timeshift refined online")
    info = INFO

    def __init__(self, *, downscale: int = 1, bridge_dir: str | None = None, work_dir: str | None = None, verbosity: str = "WARNING",
                 overrides: dict | None = None, mode: str = "production", min_features_degraded: int = 30, min_features_lost: int = 5,
                 lost_after_frames: int = 15, confidence_features: int = 120, **_ignored) -> None:
        if mode not in MODE_OVERRIDES: raise ValueError(f"mode must be one of {sorted(MODE_OVERRIDES)}")
        self.mode = mode; self.downscale = max(1, int(downscale)); self.bridge_dir = bridge_dir; self.work_dir = work_dir; self.verbosity = verbosity
        self.overrides = dict(MODE_OVERRIDES[mode]); self.overrides.update(overrides or {})      # explicit overrides win over the mode
        self.min_deg, self.min_lost, self.lost_after = min_features_degraded, min_features_lost, lost_after_frames
        self.conf_feats = confidence_features; self.reset()

    def reset(self) -> None:
        self._bridge = None; self._last = None; self._bad_streak = 0; self._prev_t = None; self._prev_p = None; self._config = None
        self._n_lost = 0; self._n_frames = 0; self._T_C_I0 = None; self.trace: list[dict] = []   # per-frame extrinsic / dt estimates

    def initialize(self, *, intrinsics=None, T_camera_imu=None, imu_noise=None, image_size=None) -> None:
        if intrinsics is None: raise BackendUnavailable("openvins: fisheye intrinsics required (configs/calibration/fisheye_<side>_vNNN.yaml)")
        if T_camera_imu is None: raise BackendUnavailable("openvins: T_camera_imu required (configs/calibration/camera_imu_<side>_vNNN.yaml) — visual-inertial needs the extrinsic")
        mod = import_bridge(self.bridge_dir)
        wd = Path(self.work_dir) if self.work_dir else Path(tempfile.mkdtemp(prefix="openvins_"))
        self._config = write_openvins_config(wd, intrinsics=intrinsics, T_camera_imu=T_camera_imu, imu_noise=imu_noise, downscale=self.downscale, overrides=self.overrides)
        self._T_C_I0 = np.asarray(T_camera_imu, np.float64).reshape(4, 4).copy()
        self._bridge = mod.Bridge(str(self._config), self.verbosity)

    def push_imu(self, t_ns: int, gyro_rad_s, accel_m_s2) -> None:
        if self._bridge is None: raise RuntimeError("openvins: initialize() first")
        self._bridge.feed_imu(int(t_ns) * 1e-9, np.asarray(gyro_rad_s, np.float64).reshape(3), np.asarray(accel_m_s2, np.float64).reshape(3))

    def push_image(self, t_ns: int, frame_index: int, image_bgr) -> PoseEstimate:
        if self._bridge is None: raise RuntimeError("openvins: initialize() first")
        import cv2
        img = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        inited = self._bridge.feed_image(int(t_ns) * 1e-9, np.ascontiguousarray(img)); self._n_frames += 1
        st = self._bridge.state()
        n_feat = int(st["n_msckf"]) + int(st["n_slam"])
        extra = dict(n_msckf=int(st["n_msckf"]), n_slam=int(st["n_slam"]), dt_cam_imu=float(st["dt_cam_imu"]), v_IinG=np.asarray(st["v_IinG"]).tolist(), state_t=float(st["t"]))
        if not inited:
            est = PoseEstimate(int(t_ns), int(frame_index), None, TrackingState.INITIALIZING, confidence=0.0, num_features=n_feat, extra=extra)
            self._last = est; return est
        T_G_I = np.asarray(st["T_G_I"], np.float64); T_I_C = np.asarray(st["T_I_C"], np.float64); T = T_G_I @ T_I_C
        self._trace(int(t_ns), int(frame_index), T_I_C, float(st["dt_cam_imu"]), n_feat,
                    ba=np.asarray(st["ba"], np.float64), bg=np.asarray(st["bg"], np.float64), v=np.asarray(st["v_IinG"], np.float64))
        finite = bool(np.all(np.isfinite(T)))
        state = TrackingState.TRACKING
        if not finite or n_feat < self.min_lost: self._bad_streak += 1
        else: self._bad_streak = 0
        if not finite or self._bad_streak >= self.lost_after: state = TrackingState.LOST; self._n_lost += 1
        elif n_feat < self.min_deg or self._bad_streak > 0: state = TrackingState.DEGRADED
        conf = float(np.clip(n_feat / float(self.conf_feats), 0.0, 1.0)) if finite else 0.0
        est = PoseEstimate(int(t_ns), int(frame_index), T if (finite and state != TrackingState.LOST) else None, state, confidence=conf, num_features=n_feat, extra=extra)
        self._last = est; return est

    def _trace(self, t_ns: int, fi: int, T_I_C: np.ndarray, dt_s: float, n_feat: int, *, ba=None, bg=None, v=None) -> None:
        """Estimated camera<->IMU extrinsic (as OUR T_camera_imu = inv(T_I_C)) and time offset per frame, plus the delta to the loaded prior.
        Also the filter's IMU biases and velocity: a translation scale that is off with slow motion usually points at the accelerometer bias."""
        from handumi_collector.pose.se3 import T_to_pose7, rotation_angle_deg
        T_C_I = inv_T(T_I_C); row = dict(t_ns=t_ns, frame_index=fi, dt_cam_imu_ms=dt_s * 1e3, n_feat=n_feat)
        if ba is not None: row.update(ba_x=float(ba[0]), ba_y=float(ba[1]), ba_z=float(ba[2]), ba_norm=float(np.linalg.norm(ba)))
        if bg is not None: row.update(bg_x=float(bg[0]), bg_y=float(bg[1]), bg_z=float(bg[2]))
        if v is not None: row["speed_m_s"] = float(np.linalg.norm(v))
        for k, n in zip(T_to_pose7(T_C_I), ("x", "y", "z", "qx", "qy", "qz", "qw")): row[f"T_camera_imu_{n}"] = float(k)
        if self._T_C_I0 is not None:
            row["d_rot_deg"] = rotation_angle_deg(self._T_C_I0[:3, :3], T_C_I[:3, :3]); row["d_trans_mm"] = float(np.linalg.norm(T_C_I[:3, 3] - self._T_C_I0[:3, 3]) * 1e3)
        self.trace.append(row)

    def calibration_estimate(self) -> dict | None:
        """Final online estimate (last frame) as a dict ready for calib_converge / write_camera_imu; None before any frame."""
        if not self.trace: return None
        r = self.trace[-1]; p7 = [r[f"T_camera_imu_{n}"] for n in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        from handumi_collector.pose.se3 import pose7_to_T
        return dict(mode=self.mode, T_camera_imu=pose7_to_T(p7).round(9).tolist(), time_offset_ms=-r["dt_cam_imu_ms"],   # OpenVINS dt: t_imu = t_cam + dt  ->  our camera - imu = -dt
                    d_rot_deg=r.get("d_rot_deg"), d_trans_mm=r.get("d_trans_mm"), n_frames=len(self.trace))

    def get_quality(self) -> dict:
        q = dict(mode=self.mode, frames=self._n_frames, lost_frames=self._n_lost, config=str(self._config) if self._config else None, calibration_estimate=self.calibration_estimate())
        if self._bridge is not None:
            st = self._bridge.state(); q.update(initialized=bool(st["initialized"]), n_imu=int(st["n_imu"]), dt_cam_imu=float(st["dt_cam_imu"]), bg=np.asarray(st["bg"]).tolist(), ba=np.asarray(st["ba"]).tolist())
        return q

    def finish(self):
        return None
