"""Typed configuration loaded from YAML (configs/handumi/*.yaml). Everything device-specific lives here, not in code."""
from __future__ import annotations
import dataclasses as dc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "handumi"


@dataclass
class CameraCfg:
    name: str                       # head | left_wrist | right_wrist | head_depth
    role: str                       # policy_obs | pose_estimation | aux_depth
    backend: str = "uvc"            # uvc | orbbec | mock
    match_name: str | None = None   # AVFoundation/UVC product name substring (e.g. "C922")
    ordinal: int = 0                # n-th device with that name in the system listing
    index: int | None = None        # explicit OpenCV index override (wins over match)
    # Canonical physical identity (see docs). `serial` is the USB serial number; it is the identity that calibration and
    # dataset provenance are bound to, because a name or an ordinal is neither unique nor stable. `strict_identity`
    # makes the collector REFUSE to record rather than open a camera it cannot prove is the right one.
    serial: str | None = None
    strict_identity: bool = False
    strict_format: bool = False
    # Frozen UVC image controls, applied to the CAMERA before OpenCV opens it (devices/uvc_controls.py). Given as
    # literal uvc-util control names and values, in application order — an auto-mode flag must be cleared before the
    # manual value it governs. macOS AVFoundation refuses these controls, so this is the only path that works here; the
    # generic exposure/gain fields above are kept for platforms where cv2.VideoCapture.set() does work.
    # Every value is verified by read-back and the episode refuses to record on a mismatch.
    uvc_controls: dict | None = None
    uvc_util: str | None = None          # path to the uvc-util binary; None = PATH / $UVC_UTIL / known locations
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "MJPG"
    codec: str = "h264_videotoolbox"   # PyAV encoder; falls back to libx264
    bitrate_kbps: int = 6000
    required: bool = True
    record: bool = True
    # Manual image controls, frozen per side before fisheye calibration (production settings must not drift between the
    # calibration take and the data takes). None = leave the camera alone. What actually took effect is verified by
    # read-back and recorded in the session/episode device detail as `controls`; an unsupported control is never
    # reported as applied.
    exposure_auto: bool | None = None
    exposure: float | None = None
    gain: float | None = None
    white_balance_auto: bool | None = None
    white_balance: float | None = None


@dataclass
class ImuCfg:
    side: str                       # left | right
    backend: str = "teensy"         # teensy | mock
    port_glob: str = "/dev/cu.usbmodem*"
    serial_number: str | None = None   # Teensy USB serial (preferred identity)
    baud: int = 2_000_000
    rate_hz: int = 400
    required: bool = True


@dataclass
class GripperCfg:
    side: str
    backend: str = "feetech"        # feetech | mock
    port: str = "/dev/cu.usbserial-HANDUMI_L"
    port_serial: str | None = None     # USB serial number of the bus adapter (preferred identity; wins over `port`), e.g. "5B3D048082"
    port_glob: str = "/dev/cu.usb*"
    servo_id: int = 0
    baud: int = 1_000_000
    sample_hz: float = 100.0
    ticks_closed: int | None = None    # raw encoder at closed jaws (calibration)
    ticks_open: int | None = None      # raw encoder at open jaws
    norm_one_is: str = "closed"        # dataset convention: which end maps to 1.0 (robot grip01: 1 = closed)
    required: bool = True


@dataclass
class HardwareCfg:
    hardware_version: str = "handumi_v1"
    profile: str = "handumi_v1"
    calibration_version: str = "uncalibrated"
    gripper_calibration: str = "latest"     # "latest" | explicit version like gripper_v003
    cameras: list[CameraCfg] = field(default_factory=list)
    imus: list[ImuCfg] = field(default_factory=list)
    grippers: list[GripperCfg] = field(default_factory=list)


@dataclass
class TasksCfg:
    cube_set: str = "RBP"
    colors: dict[str, str] = field(default_factory=lambda: {"R": "red", "B": "blue", "P": "purple"})
    orders: list[str] = field(default_factory=lambda: ["RBP", "RPB", "BRP", "BPR", "PRB", "PBR"])
    target_per_order: int = 20
    instruction_template: str = ("Stack the {bottom} cube on the bottom, {middle} cube in the middle, "
                                 "and {top} cube on the top, on the plate.")
    datasets: dict = field(default_factory=lambda: {"Hpilot": {"target_per_order": 50}, "H120": {"target_per_order": 20}})

    def instruction(self, order: str) -> str:
        b, m, t = (self.colors[c] for c in order)
        return self.instruction_template.format(bottom=b, middle=m, top=t)


@dataclass
class CollectorCfg:
    dataset_root: str = "datasets/human_handumi_raw"
    session_prefix: str = "Hpilot"
    operator: str = "op01"
    slam_gate_recording: bool = False       # SLAM runs offline in V1; kept for later online estimators
    preview_hz: float = 15.0
    warn_camera_drop_frames: int = 3        # consecutive missing capture indices -> warning event
    warn_imu_gap_ms: float = 25.0           # > this between IMU samples -> imu_timeout event
    warn_gripper_gap_ms: float = 100.0
    warn_timestamp_jump_ms: float = 200.0
    min_episode_s: float = 2.0
    notes: str = ""
    imu_quality: dict = field(default_factory=lambda: {"green": {"min_hz": 380, "max_loss": 0.001, "max_age_ms": 50},
                                                       "yellow": {"min_hz": 300, "max_loss": 0.01, "max_age_ms": 250}})
    gripper_quality: dict = field(default_factory=lambda: {"green": {"min_hz": 80, "max_age_ms": 100},
                                                           "yellow": {"min_hz": 40, "max_age_ms": 500}})
    # M1.5 production UI
    disk: dict = field(default_factory=lambda: {"warn_gb": 50, "block_gb": 10, "gb_per_minute_estimate": 0.6})
    preflight: list = field(default_factory=list)
    preflight_required: bool = False
    # Gripper POLARITY TEST gate (2026-09-17, disabled 2026-09-19). It caught a real fault -- a servo whose zero
    # had moved, so `open` drove the jaw closed -- but it gates every REC on an 8 s manual sweep, and the
    # open-at-rest check below it catches the same fault without the operator doing anything. False: the button
    # is hidden and REC is not gated on it; the at-rest check still runs.
    polarity_test_required: bool = False
    home_countdown_s: float = 3.0           # legacy timer; superseded by `stillness` (kept for configs that still set it)
    # VIO init readiness gate (collector/stillness.py): REC only after every IMU has been still this long. Thresholds
    # measured on DIAG_T30 true-still windows (gyro p99 3.6 deg/s, accel std p99 0.16) -- do not tighten by feel.
    stillness: dict = field(default_factory=lambda: {"enabled": True, "min_still_s": 3.0, "window_s": 0.25, "gyro_max_deg_s": 6.0,
                                                     "accel_std_max": 0.35, "max_gap_ms": 25.0, "min_window_fill": 0.6, "countdown_s": 0.0,
                                                     "hold_after_rec_s": 3.0})
    # Hands-free loop with spoken cues (collector/autoloop.py): stack episode_s, reset reset_s, repeat until targets are met.
    auto_loop: dict = field(default_factory=lambda: {"enabled": True, "episode_s": 20.0, "reset_s": 10.0, "min_still_s": 1.0, "hold_after_rec_s": 3.0, "auto_keep_pass": True,
                                                     "pause_on_fail": True, "on_warn": "keep", "balance_orders": True, "voice": True, "voice_name": "Samantha",
                                                     "rate_wpm": None, "ready_cue_s": 2.0, "stop_when_targets_met": False})
    head_overlay: dict = field(default_factory=lambda: {"x0": 0.15, "y0": 0.2, "x1": 0.85, "y1": 0.95})
    head_motion_warn: float = 12.0
    # Depth-preview colour ramp (metres). The HandUMI working distance: below near_m / above far_m saturates.
    depth_preview: dict = field(default_factory=lambda: {"near_m": 0.30, "far_m": 1.20})
    replay_fps: int = 30


@dataclass
class PoseCfg:
    """configs/handumi/pose.yaml — see that file for the meaning of every block. Kept as plain dicts for the tunable blocks
    so thresholds are never hard-coded in code paths."""
    backend: str = "opencv_vo"
    tracking_mode: str = "episode_reset"
    sides: list[str] = field(default_factory=lambda: ["left", "right"])
    image_downscale: int = 2
    master_timeline: str = "head"
    fps: int = 30
    lead: int = 5
    horizon: int = 30
    max_pose_gap_ms: float = 100.0
    max_grip_gap_ms: float = 150.0
    offsets: dict = field(default_factory=lambda: {"left_camera_imu_offset_ms": 0.0, "right_camera_imu_offset_ms": 0.0, "measured": False})
    static_window: dict = field(default_factory=lambda: {"min_duration_s": 0.8, "max_duration_s": 3.0, "gyro_thresh_dps": 3.0,
                                                         "accel_std_thresh_m_s2": 0.35, "edge_trim_s": 0.15})
    home_return: dict = field(default_factory=lambda: {"pass": {"translation_mm": 10.0, "rotation_deg": 3.0},
                                                       "warn": {"translation_mm": 25.0, "rotation_deg": 7.0}})
    static_jitter: dict = field(default_factory=lambda: {"review": {"pos_std_mm": 3.0, "rot_std_deg": 1.0}})
    jumps: dict = field(default_factory=lambda: {"translation_m": 0.15, "rotation_deg": 20.0})
    imu_consistency: dict = field(default_factory=lambda: {"window_frames": 10, "warn_deg": 3.0, "reject_deg": 15.0, "max_frac_over_warn": 0.10})
    valid_ratio: dict = field(default_factory=lambda: {"warn": 0.98, "reject": 0.90})
    lost_events: dict = field(default_factory=lambda: {"reject_if_lost_longer_than_s": 1.0})
    imu_bias: dict = field(default_factory=lambda: {"estimate_from_static_start": True})
    backend_options: dict = field(default_factory=dict)
    # segmented relative VIO (pose/backends/segmented.py): static = IMU hold, moving = per-segment re-init of the backend
    segmented: dict = field(default_factory=lambda: {"enabled": False, "min_static_s": 0.5, "gyro_max_deg_s": 4.0, "accel_std_max": 0.25, "max_gap_ms": 25.0,
                                                     "window_s": 0.2, "prefix_s": 2.0, "tail_s": 0.5, "flow_width_px": 120, "flow_max_dn": 6.0, "flow_frames": 2,
                                                     "starve_frames": 0, "starve_min_features": 5})

    def camera_imu_offset_ns(self, side: str) -> int:
        return int(float(self.offsets.get(f"{side}_camera_imu_offset_ms", 0.0)) * 1e6)


def load_pose_cfg(path: str | Path | None = None) -> PoseCfg:
    path = Path(path) if path else DEFAULT_CONFIG_DIR / "pose.yaml"
    if not path.exists(): return PoseCfg()
    return _from_dict(PoseCfg, yaml.safe_load(path.read_text()))


@dataclass
class AppConfig:
    hardware: HardwareCfg
    tasks: TasksCfg
    collector: CollectorCfg
    config_dir: Path
    pose: "PoseCfg | None" = None

    def resolve(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (REPO_ROOT / p)


def _from_dict(cls, d: dict[str, Any] | None):
    d = dict(d or {})
    names = {f.name for f in dc.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**d)


def load_hardware(path: Path) -> HardwareCfg:
    raw = yaml.safe_load(path.read_text()) or {}
    return HardwareCfg(
        hardware_version=raw.get("hardware_version", "handumi_v1"),
        profile=raw.get("profile", path.stem.replace("hardware_", "")),
        calibration_version=raw.get("calibration_version", "uncalibrated"),
        gripper_calibration=str(raw.get("gripper_calibration", "latest")),
        cameras=[_from_dict(CameraCfg, c) for c in raw.get("cameras", [])],
        imus=[_from_dict(ImuCfg, c) for c in raw.get("imus", [])],
        grippers=[_from_dict(GripperCfg, c) for c in raw.get("grippers", [])],
    )


PROFILES = ("handumi_v1", "handumi_preimu", "handumi_rgbd", "rgbd_stagea_left", "temporal_sync_calib",
            "dev_current", "dev_3c922", "mock")
CALIB_PROFILES = ("temporal_sync_calib",)


def load_config(config_dir: str | Path = DEFAULT_CONFIG_DIR, *, mock: bool = False, hardware: str = "handumi_v1") -> AppConfig:
    """`hardware` = profile name (configs/handumi/hardware_<profile>.yaml) or a path to a yaml. --mock forces all backends to mock."""
    config_dir = Path(config_dir)
    hp = Path(hardware)
    if not hp.suffix:
        hp = config_dir / f"hardware_{hardware}.yaml"
    if not hp.exists():
        raise FileNotFoundError(f"hardware profile not found: {hp} (known: {PROFILES})")
    hw = load_hardware(hp)
    tasks = _from_dict(TasksCfg, yaml.safe_load((config_dir / "tasks.yaml").read_text()))
    coll = _from_dict(CollectorCfg, yaml.safe_load((config_dir / "collector.yaml").read_text()))
    if mock:
        for c in hw.cameras:
            c.backend = "mock"
        for i in hw.imus:
            i.backend = "mock"
        for g in hw.grippers:
            g.backend = "mock"
    # Synthetic runs must never be mistakable for real ones: a headless smoke used to land in the production dataset root
    # carrying `hardware_profile: handumi_v1`, distinguishable from a real episode only by a free-text note. Marking the
    # profile and moving the root happens here, so it covers the UI, the headless path and anything else that loads config.
    # A calibration take is a deliberately shaken, tapped, instruction-less half minute. It belongs in the same schema
    # as an episode -- that is what lets the same tools read it -- but not in the same pile, where a dataset build
    # scanning for episodes would find it.
    if hw.profile in CALIB_PROFILES and not coll.dataset_root.endswith("_calib"):
        coll.dataset_root = coll.dataset_root.rstrip("/").replace("_raw", "") + "_calib"
    if mock or any(d.backend == "mock" for d in (*hw.cameras, *hw.imus, *hw.grippers)):
        if not hw.profile.endswith("+mock") and hw.profile != "mock":
            hw.profile += "+mock"
        if not coll.dataset_root.endswith("_mock"):
            coll.dataset_root = coll.dataset_root.rstrip("/") + "_mock"
    pose = load_pose_cfg(config_dir / "pose.yaml")
    return AppConfig(hardware=hw, tasks=tasks, collector=coll, config_dir=config_dir, pose=pose)
