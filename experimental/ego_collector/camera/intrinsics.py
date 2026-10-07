"""Camera intrinsics file (configs/camera/<name>.yaml) + undistortion helpers.

The C922 has noticeable distortion; every 2D point used for pose estimation
must be undistorted (or PnP must be given the distortion coefficients).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass
class CameraIntrinsics:
    image_width: int
    image_height: int
    camera_matrix: np.ndarray  # (3, 3)
    distortion_coefficients: np.ndarray  # (k,) OpenCV pinhole order k1 k2 p1 p2 k3 [k4 k5 k6]
    model: str = "pinhole"
    rms_reprojection_error: float = float("nan")
    mean_reprojection_error: float = float("nan")
    num_views: int = 0
    calibration_date: str = ""
    camera_name: str = "c922"
    extra: dict[str, Any] = field(default_factory=dict)

    # -- io ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_name": self.camera_name,
            "model": self.model,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "camera_matrix": np.asarray(self.camera_matrix, dtype=float).tolist(),
            "distortion_coefficients": np.asarray(self.distortion_coefficients, dtype=float).reshape(-1).tolist(),
            "rms_reprojection_error": float(self.rms_reprojection_error),
            "mean_reprojection_error": float(self.mean_reprojection_error),
            "num_views": int(self.num_views),
            "calibration_date": self.calibration_date or datetime.now(timezone.utc).isoformat(),
            **self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraIntrinsics":
        known = {
            "image_width", "image_height", "camera_matrix", "distortion_coefficients", "model",
            "rms_reprojection_error", "mean_reprojection_error", "num_views", "calibration_date", "camera_name",
        }
        return cls(
            image_width=int(d["image_width"]),
            image_height=int(d["image_height"]),
            camera_matrix=np.asarray(d["camera_matrix"], dtype=np.float64).reshape(3, 3),
            distortion_coefficients=np.asarray(d["distortion_coefficients"], dtype=np.float64).reshape(-1),
            model=str(d.get("model", "pinhole")),
            rms_reprojection_error=float(d.get("rms_reprojection_error", float("nan"))),
            mean_reprojection_error=float(d.get("mean_reprojection_error", float("nan"))),
            num_views=int(d.get("num_views", 0)),
            calibration_date=str(d.get("calibration_date", "")),
            camera_name=str(d.get("camera_name", "c922")),
            extra={k: v for k, v in d.items() if k not in known},
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path

    @classmethod
    def load(cls, path: Path) -> "CameraIntrinsics":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    # -- geometry ------------------------------------------------------------

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """Distorted pixels (N,2) -> undistorted pixels (N,2) in the same K."""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if self.model == "fisheye":
            out = cv2.fisheye.undistortPoints(p, self.camera_matrix, self.distortion_coefficients.reshape(-1, 1), P=self.camera_matrix)
        else:
            out = cv2.undistortPoints(p, self.camera_matrix, self.distortion_coefficients, P=self.camera_matrix)
        return out.reshape(-1, 2)

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """3D camera-frame points (N,3) -> distorted pixels (N,2)."""
        pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 1, 3)
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        if self.model == "fisheye":
            proj, _ = cv2.fisheye.projectPoints(pts, rvec, tvec, self.camera_matrix, self.distortion_coefficients.reshape(-1, 1))
        else:
            proj, _ = cv2.projectPoints(pts, rvec, tvec, self.camera_matrix, self.distortion_coefficients)
        return proj.reshape(-1, 2)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        if self.model == "fisheye":
            return cv2.fisheye.undistortImage(image, self.camera_matrix, self.distortion_coefficients.reshape(-1, 1), Knew=self.camera_matrix)
        return cv2.undistort(image, self.camera_matrix, self.distortion_coefficients)

    @classmethod
    def ideal(cls, width: int, height: int, focal_px: float, name: str = "synthetic") -> "CameraIntrinsics":
        K = np.array([[focal_px, 0, width / 2.0], [0, focal_px, height / 2.0], [0, 0, 1.0]])
        return cls(image_width=width, image_height=height, camera_matrix=K, distortion_coefficients=np.zeros(5), camera_name=name, rms_reprojection_error=0.0, mean_reprojection_error=0.0)


__all__ = ["CameraIntrinsics"]
