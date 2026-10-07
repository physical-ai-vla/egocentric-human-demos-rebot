"""The right/left wrist unit's complete calibration bundle — one versioned file that says everything OpenVINS (or any VIO)
needs, tied to the PHYSICAL wrist-unit configuration it was measured on.

    configs/calibration/wrist_bundle_<side>_vNNN.yaml
      hardware:   unit_id, side, mount_revision, camera (name/serial/resolution/fps/exposure), imu (serial/rate), notes
      camera:     model (kannala_brandt | pinhole), K, D, image_size          <- offline target calibration (checkerboard / circle grid)
      imu_noise:  gyroscope_noise_density, gyroscope_random_walk,
                  accelerometer_noise_density, accelerometer_random_walk      <- Allan variance (tools.imu_allan)
      camera_imu: T_camera_imu (4x4, IMU frame in camera frame), time_offset_ms (camera - imu)
                                                                              <- offline Kalibr (PRIMARY), OpenVINS online refine only
      provenance: per-field {source, tool, date, run(s), notes} — how each number was obtained

Policy (user, 2026-09-10): offline calibration is primary; OpenVINS `mode: calibration` only refines and validates it. A bundle
whose camera_imu provenance says `openvins_online` alone is INCOMPLETE for production and `require_production()` refuses it.

The bundle is a superset of the per-kind files the pose pipeline already reads (fisheye_<side>, camera_imu_<side>): `export_pose_files`
writes those from the bundle so `handumi_collector.pose.calibration.SideCalibration` keeps working unchanged."""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import yaml
from handumi_collector.pose.calibration import CAL_DIR, load_versioned, save_versioned, write_fisheye, write_camera_imu

BUNDLE_PREFIX = "wrist_bundle"
SCHEMA = "ego_teleop_wrist_calibration_bundle/v1"
# offline (target-based / Kalibr) sources are the only ones acceptable as the PRIMARY camera-IMU calibration
OFFLINE_CAMERA_IMU_SOURCES = ("kalibr", "basalt", "target_offline")
NOISE_FIELDS = ("gyroscope_noise_density", "gyroscope_random_walk", "accelerometer_noise_density", "accelerometer_random_walk")


@dataclass
class ImuNoise:
    gyroscope_noise_density: float | None = None        # rad/s/sqrt(Hz)
    gyroscope_random_walk: float | None = None          # rad/s^2/sqrt(Hz)
    accelerometer_noise_density: float | None = None    # m/s^2/sqrt(Hz)
    accelerometer_random_walk: float | None = None      # m/s^3/sqrt(Hz)
    update_rate: float = 400.0

    @property
    def complete(self) -> bool: return all(getattr(self, f) is not None for f in NOISE_FIELDS)

    def to_dict(self) -> dict: return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class WristCalibrationBundle:
    side: str
    hardware: dict = field(default_factory=dict)
    camera: dict | None = None                     # {model, K (3x3), D (n,), image_size (w,h)}
    imu_noise: ImuNoise = field(default_factory=ImuNoise)
    T_camera_imu: np.ndarray | None = None
    time_offset_ms: float | None = None            # camera - imu (positive = camera clock later)
    provenance: dict = field(default_factory=dict)  # field -> {source, tool, date, runs, notes}
    version: str | None = None                     # file stem once saved/loaded

    # ---------------------------------------------------------------- content
    @property
    def missing(self) -> list[str]:
        m = []
        if not self.camera: m.append("camera")
        if not self.imu_noise.complete: m.append("imu_noise")
        if self.T_camera_imu is None: m.append("T_camera_imu")
        if self.time_offset_ms is None: m.append("time_offset_ms")
        if not self.hardware.get("unit_id"): m.append("hardware.unit_id")
        return m

    @property
    def camera_imu_source(self) -> str | None:
        return (self.provenance.get("T_camera_imu") or {}).get("source")

    @property
    def offline_camera_imu(self) -> bool:
        return self.camera_imu_source in OFFLINE_CAMERA_IMU_SOURCES

    def require_production(self) -> None:
        """Raise unless the bundle is complete AND its camera-IMU calibration came from an offline procedure."""
        if self.missing: raise RuntimeError(f"wrist bundle {self.side} incomplete: {self.missing}")
        if not self.offline_camera_imu:
            raise RuntimeError(f"wrist bundle {self.side}: camera_imu source is {self.camera_imu_source!r}; production requires an offline "
                               f"calibration ({'/'.join(OFFLINE_CAMERA_IMU_SOURCES)}) — OpenVINS online refinement alone is not accepted")

    def hardware_hash(self) -> str:
        """Short hash of the physical configuration this calibration belongs to — changes when the mount/camera/IMU/mode changes."""
        keys = ("unit_id", "side", "mount_revision", "camera_serial", "camera_name", "imu_serial", "resolution", "fps", "exposure")
        payload = {k: self.hardware.get(k) for k in keys if self.hardware.get(k) is not None}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def matches_hardware(self, hardware: dict) -> tuple[bool, list[str]]:
        diff = [k for k, v in (hardware or {}).items() if k in self.hardware and self.hardware[k] != v]
        return (not diff), diff

    def set_provenance(self, field_name: str, *, source: str, tool: str = "", runs: list | None = None, notes: str = "") -> None:
        self.provenance[field_name] = dict(source=source, tool=tool, date=time.strftime("%Y-%m-%dT%H:%M:%S%z"), runs=list(runs or []), notes=notes)

    # ---------------------------------------------------------------- consumers
    def intrinsics(self) -> dict | None:
        if not self.camera: return None
        return dict(model=self.camera.get("model", "kannala_brandt"), K=np.asarray(self.camera["K"], np.float64).reshape(3, 3),
                    D=np.asarray(self.camera.get("D", np.zeros(4)), np.float64).reshape(-1), image_size=tuple(int(v) for v in self.camera["image_size"]))

    def export_pose_files(self, *, cal_dir: Path | None = None, notes: str = "") -> dict:
        """Write the per-kind files the pose pipeline reads (fisheye_<side>, camera_imu_<side>) from this bundle."""
        cal_dir = cal_dir or CAL_DIR; out = {}
        if self.camera:
            i = self.intrinsics()
            p = write_fisheye(self.side, i["K"], i["D"], i["image_size"], source=f"{BUNDLE_PREFIX}:{self.version or 'unsaved'}", notes=notes, cal_dir=cal_dir)
            if i["model"] != "kannala_brandt":                                  # write_fisheye hardcodes the KB model
                d = yaml.safe_load(p.read_text()); d["model"] = i["model"]; p.write_text(yaml.safe_dump(d, sort_keys=False))
            out["fisheye"] = p
        if self.T_camera_imu is not None:
            out["camera_imu"] = write_camera_imu(self.side, self.T_camera_imu, time_offset_ms=self.time_offset_ms, imu_noise=self.imu_noise.to_dict() or None,
                                                 source=f"{BUNDLE_PREFIX}:{self.version or 'unsaved'} ({self.camera_imu_source})", notes=notes, cal_dir=cal_dir)
        return out

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict:
        d = dict(schema=SCHEMA, side=self.side, hardware=dict(self.hardware), hardware_hash=self.hardware_hash(),
                 camera=None if not self.camera else dict(model=self.camera.get("model", "kannala_brandt"),
                                                          K=np.asarray(self.camera["K"], np.float64).reshape(3, 3).tolist(),
                                                          D=np.asarray(self.camera.get("D", np.zeros(4)), np.float64).reshape(-1).tolist(),
                                                          image_size=[int(v) for v in self.camera["image_size"]],
                                                          reprojection_rms_px=self.camera.get("reprojection_rms_px"), n_views=self.camera.get("n_views"),
                                                          target=self.camera.get("target")),
                 imu_noise=self.imu_noise.to_dict(),
                 camera_imu=dict(T_camera_imu=None if self.T_camera_imu is None else np.asarray(self.T_camera_imu, np.float64).round(9).tolist(),
                                 time_offset_ms=self.time_offset_ms, source=self.camera_imu_source),
                 provenance=self.provenance, missing=self.missing, production_ready=not self.missing and self.offline_camera_imu)
        return d

    @classmethod
    def from_dict(cls, d: dict, *, version: str | None = None) -> "WristCalibrationBundle":
        ci = d.get("camera_imu") or {}
        T = ci.get("T_camera_imu")
        b = cls(side=d["side"], hardware=dict(d.get("hardware") or {}), camera=d.get("camera") or None,
                imu_noise=ImuNoise(**{k: v for k, v in (d.get("imu_noise") or {}).items() if k in ImuNoise.__dataclass_fields__}),
                T_camera_imu=None if T is None else np.asarray(T, np.float64).reshape(4, 4), time_offset_ms=ci.get("time_offset_ms"),
                provenance=dict(d.get("provenance") or {}), version=version)
        return b


def load_bundle(side: str, which: str = "latest", cal_dir: Path | None = None) -> WristCalibrationBundle:
    v, d = load_versioned(f"{BUNDLE_PREFIX}_{side}", which, cal_dir or CAL_DIR)
    if not d: return WristCalibrationBundle(side)
    return WristCalibrationBundle.from_dict(d, version=v)


def save_bundle(b: WristCalibrationBundle, *, cal_dir: Path | None = None, notes: str = "") -> Path:
    p = save_versioned(f"{BUNDLE_PREFIX}_{b.side}", dict(b.to_dict(), notes=notes), cal_dir=cal_dir or CAL_DIR)
    b.version = p.stem
    return p
