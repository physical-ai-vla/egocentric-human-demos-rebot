"""Versioned pose calibrations in configs/calibration/ (latest wins, never overwritten, session/episode meta records the name):

    fisheye_<side>_vNNN.yaml     Arducam intrinsics: model kannala_brandt (cv2.fisheye) K (3x3), D (4), image_size [w,h]
    camera_imu_<side>_vNNN.yaml  T_camera_imu (4x4, IMU frame expressed in camera frame) + optional time_offset_ms (camera - imu)
    camera_tcp_<side>_vNNN.yaml  T_camera_tcp (4x4, pinch-centre TCP expressed in camera frame)

Nothing here is ever derived from a fiducial: intrinsics/extrinsics come from an offline calibration tool (e.g. Kalibr / a
target-based lens calibration run outside this pipeline) and are *imported* as files. Missing files load as None and every
consumer records `uncalibrated` explicitly instead of assuming identity silently."""
from __future__ import annotations
import re
import time
from pathlib import Path
import numpy as np
import yaml
from ..config import REPO_ROOT

CAL_DIR = REPO_ROOT / "configs" / "calibration"
KINDS = ("fisheye", "camera_imu", "camera_tcp")


def _re(prefix: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(prefix)}_v(\d{{3}})\.ya?ml$")


def versions(prefix: str, cal_dir: Path | None = None) -> list[Path]:
    cal_dir = cal_dir or CAL_DIR
    rx = _re(prefix)
    return sorted((p for p in cal_dir.glob(f"{prefix}_v*.y*ml") if rx.match(p.name)), key=lambda p: int(rx.match(p.name).group(1)))


def load_versioned(prefix: str, which: str = "latest", cal_dir: Path | None = None) -> tuple[str | None, dict]:
    vs = versions(prefix, cal_dir)
    if which != "latest": vs = [p for p in vs if p.stem == which]
    if not vs: return None, {}
    p = vs[-1]
    return p.stem, (yaml.safe_load(p.read_text()) or {})


def save_versioned(prefix: str, payload: dict, *, cal_dir: Path | None = None) -> Path:
    cal_dir = cal_dir or CAL_DIR; cal_dir.mkdir(parents=True, exist_ok=True)
    vs = versions(prefix, cal_dir); rx = _re(prefix)
    n = int(rx.match(vs[-1].name).group(1)) + 1 if vs else 1
    p = cal_dir / f"{prefix}_v{n:03d}.yaml"
    d = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z")); d.update(payload)
    p.write_text(yaml.safe_dump(d, sort_keys=False)); return p


def _mat(v, shape) -> np.ndarray:
    a = np.asarray(v, np.float64).reshape(shape)
    if shape == (4, 4) and not np.allclose(a[3], [0, 0, 0, 1]): raise ValueError("4x4 transform: last row must be 0 0 0 1")
    return a


class SideCalibration:
    """Everything the pose pipeline needs for one HandUMI side, with explicit `versions` (None = uncalibrated)."""

    def __init__(self, side: str, *, cal_dir: Path | None = None, which: dict | None = None) -> None:
        which = which or {}
        self.side = side
        self.versions: dict[str, str | None] = {}
        v, d = load_versioned(f"fisheye_{side}", which.get("fisheye", "latest"), cal_dir); self.versions["fisheye"] = v
        self.intrinsics: dict | None = None
        if d:
            self.intrinsics = dict(model=d.get("model", "kannala_brandt"), K=_mat(d["K"], (3, 3)), D=np.asarray(d["D"], np.float64).reshape(-1),
                                   image_size=tuple(int(x) for x in d["image_size"]))
            # Radius beyond which D is extrapolated rather than measured; None = never measured, so nothing is masked.
            if d.get("valid_radius_px") is not None:
                self.intrinsics["valid_radius_px"] = float(d["valid_radius_px"])
        # Which PHYSICAL wrist this camera sits on (scripts/verify_physical_side.py). Absent/false = never confirmed:
        # the pose would still be correct, but attributed to the wrong hand, and nothing downstream could reveal it.
        self.physical_side_verified: bool = bool(d.get("physical_side_verified", False)) if d else False
        v, d = load_versioned(f"camera_imu_{side}", which.get("camera_imu", "latest"), cal_dir); self.versions["camera_imu"] = v
        self.T_camera_imu: np.ndarray | None = _mat(d["T_camera_imu"], (4, 4)) if d else None
        self.imu_noise: dict | None = d.get("imu_noise") if d else None
        self.camera_imu_time_offset_ms: float | None = (float(d["time_offset_ms"]) if d and "time_offset_ms" in d else None)
        v, d = load_versioned(f"camera_tcp_{side}", which.get("camera_tcp", "latest"), cal_dir); self.versions["camera_tcp"] = v
        self.T_camera_tcp: np.ndarray | None = _mat(d["T_camera_tcp"], (4, 4)) if d else None

    @property
    def tcp_calibrated(self) -> bool: return self.T_camera_tcp is not None

    @property
    def imu_extrinsics_calibrated(self) -> bool: return self.T_camera_imu is not None

    def T_camera_tcp_or_identity(self) -> np.ndarray:
        """Identity fallback is allowed ONLY with the `uncalibrated` flag recorded next to the data (see run.py)."""
        return np.eye(4) if self.T_camera_tcp is None else self.T_camera_tcp

    def T_camera_imu_or_identity(self) -> np.ndarray:
        return np.eye(4) if self.T_camera_imu is None else self.T_camera_imu

    def summary(self) -> dict:
        return dict(side=self.side, versions=dict(self.versions), tcp_calibrated=self.tcp_calibrated,
                    imu_extrinsics_calibrated=self.imu_extrinsics_calibrated, intrinsics_present=self.intrinsics is not None,
                    physical_side_verified=self.physical_side_verified)


def write_fisheye(side: str, K, D, image_size, *, notes: str = "", source: str = "", extra: dict | None = None,
                  cal_dir: Path | None = None) -> Path:
    """`extra` carries the imaging state the intrinsics are only valid for — product name, reported serial, fps,
    exposure/gain/white-balance, calibration_version. Intrinsics are tied to the resolution AND to the frozen imaging
    preset they were measured under; a file that does not say which camera and which settings produced it cannot be
    checked against a live device later."""
    d = dict(schema="handumi_fisheye_intrinsics/v1", model="kannala_brandt", side=side,
             K=np.asarray(K, float).reshape(3, 3).tolist(), D=np.asarray(D, float).reshape(-1).tolist(),
             image_size=[int(image_size[0]), int(image_size[1])], source=source, notes=notes)
    d.update(extra or {})
    return save_versioned(f"fisheye_{side}", d, cal_dir=cal_dir)


def write_camera_imu(side: str, T_camera_imu, *, time_offset_ms: float | None = None, imu_noise: dict | None = None, notes: str = "",
                     source: str = "", cal_dir: Path | None = None) -> Path:
    d = dict(schema="handumi_camera_imu_extrinsics/v1", side=side, T_camera_imu=_mat(T_camera_imu, (4, 4)).tolist(), source=source, notes=notes)
    if time_offset_ms is not None: d["time_offset_ms"] = float(time_offset_ms)
    if imu_noise: d["imu_noise"] = imu_noise
    return save_versioned(f"camera_imu_{side}", d, cal_dir=cal_dir)


def write_head_mount(T_table_camera, *, geometry: dict, board: dict, rms_reproj_px: float, camera: str = "head_depth",
                     depth_check: dict | None = None, repeatability: dict | None = None, view_spread: dict | None = None,
                     notes: str = "", source: str = "", cal_dir: Path | None = None, stem: str = "head_mount",
                     intrinsics_source: str | None = None) -> Path:
    """The fixed head-camera mount: where it physically sits, and the table frame it sees.

    `T_table_camera` follows the same convention as everything else here — B expressed in A — so it takes a point in
    camera coordinates to table coordinates, which is the direction RGB-D points need to travel. The table frame has its
    origin at the calibration board's first corner with Z out of the table surface, right-handed.

    `geometry` is the hand-measured mount description (height, distance from the table edge, pitch/yaw/roll). It is not
    used by any computation: it is what lets someone physically restore the mount after it is removed, which is the
    thing a transform alone cannot give you. `repeatability` records the spread over remounts, so there is a number for
    how much of the transform survives taking the holder off and putting it back."""
    d = dict(schema="handumi_head_mount/v1", camera=camera, mount="fixed_table_holder",
             T_table_camera=_mat(T_table_camera, (4, 4)).tolist(), geometry=geometry, board=board,
             rms_reproj_px=round(float(rms_reproj_px), 4), source=source, notes=notes)
    if view_spread: d["view_spread"] = view_spread
    if depth_check: d["depth_check"] = depth_check
    if repeatability: d["repeatability"] = repeatability
    # Which intrinsics produced this transform. A table frame is only as good as the K it was solved with, and the
    # head camera changed from the Orbbec (which reported its own intrinsics per capture) to a C922 (which does not).
    if intrinsics_source: d["intrinsics_source"] = intrinsics_source
    return save_versioned(stem, d, cal_dir=cal_dir)


def load_head_mount(which: str = "latest", cal_dir: Path | None = None, stem: str = "head_mount") -> tuple[str | None, dict]:
    """(version_name, payload). Missing file returns (None, {}) — consumers must record `uncalibrated`, never assume identity.

    `stem` selects the namespace: head_mount_v001..v005 are the Orbbec-era solves and stay as legacy, the C922 table
    frame lives under head_mount_c922."""
    return load_versioned(stem, which, cal_dir=cal_dir)


def write_camera_tcp(side: str, T_camera_tcp, *, method: str = "", notes: str = "", cal_dir: Path | None = None) -> Path:
    return save_versioned(f"camera_tcp_{side}", dict(schema="handumi_camera_tcp/v1", side=side, T_camera_tcp=_mat(T_camera_tcp, (4, 4)).tolist(),
                                                     method=method, notes=notes), cal_dir=cal_dir)
