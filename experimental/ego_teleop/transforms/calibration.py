"""Calibrated rigid transforms the teleop stack needs, loaded from versioned files in configs/calibration/
(same versioning scheme as handumi_collector.pose.calibration: <prefix>_vNNN.yaml, latest wins, never overwritten):

    camera_imu_<side>_vNNN.yaml      T_camera_imu  (4x4)  = T_C_I     from Kalibr / allan + target calibration (offline tool)
    wrist_camera_<side>_vNNN.yaml    T_wrist_camera (4x4) = T_H_C     human wrist control frame <- camera mount
    rebot_aero_mount_vNNN.yaml       T_ee_aero (4x4)      = T_RE_AH   reBot EEF <- Aero Hand base
    human_robot_frame_vNNN.yaml      FrameMapperConfig fields          (falls back to configs/ego_teleop/frames.yaml)

Missing files load as None and are listed in `missing`; `require(...)` raises before any robot motion. Nothing here
is ever silently assumed identity — a deliberately-identity mount must be written to a file (e.g. wrist_camera_v001
with T = eye(4) and a note) so the episode's calibration/ folder records the decision."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from handumi_collector.pose.calibration import load_versioned, save_versioned, CAL_DIR
from .frames import FrameMapperConfig


def _T(d: dict, key: str) -> np.ndarray | None:
    if not d or key not in d: return None
    T = np.asarray(d[key], np.float64).reshape(4, 4)
    if not np.allclose(T[3], [0, 0, 0, 1]): raise ValueError(f"{key}: last row must be 0 0 0 1")
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or np.linalg.det(R) < 0: raise ValueError(f"{key}: rotation block is not a proper rotation")
    return T


@dataclass
class TeleopCalibration:
    side: str
    T_C_I: np.ndarray | None = None
    T_H_C: np.ndarray | None = None
    T_RE_AH: np.ndarray | None = None
    camera_imu_time_offset_ms: float | None = None    # camera - imu (ms), from camera_imu file; None = not measured
    frame_mapper: FrameMapperConfig = field(default_factory=FrameMapperConfig)
    versions: dict = field(default_factory=dict)      # prefix -> file stem or None

    @property
    def missing(self) -> list[str]:
        return [k for k, v in (("camera_imu", self.T_C_I), ("wrist_camera", self.T_H_C), ("rebot_aero_mount", self.T_RE_AH)) if v is None]

    def require(self, *keys: str) -> None:
        miss = [k for k in (keys or ("camera_imu", "wrist_camera")) if k in self.missing]
        if miss: raise RuntimeError(f"uncalibrated: {miss} (side={self.side}); write configs/calibration/<prefix>_vNNN.yaml first")

    def to_dict(self) -> dict:
        f = lambda T: None if T is None else np.asarray(T).round(9).tolist()
        return dict(side=self.side, T_C_I=f(self.T_C_I), T_H_C=f(self.T_H_C), T_RE_AH=f(self.T_RE_AH),
                    camera_imu_time_offset_ms=self.camera_imu_time_offset_ms, frame_mapper=self.frame_mapper.to_dict(), versions=dict(self.versions))


def load_teleop_calibration(side: str, *, cal_dir: Path | None = None, frames_yaml: Path | None = None) -> TeleopCalibration:
    cal_dir = cal_dir or CAL_DIR
    cal = TeleopCalibration(side)
    v, d = load_versioned(f"camera_imu_{side}", cal_dir=cal_dir); cal.versions["camera_imu"] = v
    cal.T_C_I = _T(d, "T_camera_imu"); cal.camera_imu_time_offset_ms = d.get("time_offset_ms") if d else None
    v, d = load_versioned(f"wrist_camera_{side}", cal_dir=cal_dir); cal.versions["wrist_camera"] = v
    cal.T_H_C = _T(d, "T_wrist_camera")
    v, d = load_versioned("rebot_aero_mount", cal_dir=cal_dir); cal.versions["rebot_aero_mount"] = v
    cal.T_RE_AH = _T(d, "T_ee_aero")
    v, d = load_versioned("human_robot_frame", cal_dir=cal_dir); cal.versions["human_robot_frame"] = v
    if d: cal.frame_mapper = FrameMapperConfig.from_dict(d)
    elif frames_yaml and Path(frames_yaml).exists(): cal.frame_mapper = FrameMapperConfig.from_yaml(frames_yaml)
    return cal


def save_transform(prefix: str, key: str, T: np.ndarray, *, notes: str = "", cal_dir: Path | None = None, **extra) -> Path:
    _T({key: T}, key)
    return save_versioned(prefix, {key: np.asarray(T, np.float64).round(9).tolist(), "notes": notes, **extra}, cal_dir=cal_dir)
