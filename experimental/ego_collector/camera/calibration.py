"""ChArUco intrinsic calibration for the head camera.

Board spec lives in ``configs/charuco.yaml`` (default 5x7 squares, 30 mm squares,
15 mm markers, DICT_5X5_100 — identical to the HandUMI board so one print serves both).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from ego_collector.camera.intrinsics import CameraIntrinsics

log = logging.getLogger("ego_collector.calibration")


@dataclass(frozen=True)
class CharucoSpec:
    squares_x: int = 5
    squares_y: int = 7
    square_length_m: float = 0.030
    marker_length_m: float = 0.015
    dictionary: str = "DICT_5X5_100"
    legacy_pattern: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "CharucoSpec":
        d = yaml.safe_load(Path(path).read_text()) or {}
        d = d.get("charuco", d)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return {"squares_x": self.squares_x, "squares_y": self.squares_y, "square_length_m": self.square_length_m, "marker_length_m": self.marker_length_m, "dictionary": self.dictionary, "legacy_pattern": self.legacy_pattern}

    def board(self) -> cv2.aruco.CharucoBoard:
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        b = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_length_m, self.marker_length_m, d)
        b.setLegacyPattern(self.legacy_pattern)
        return b

    def image(self, px_per_m: float, margin_px: int = 40) -> np.ndarray:
        w = int(round(self.squares_x * self.square_length_m * px_per_m))
        h = int(round(self.squares_y * self.square_length_m * px_per_m))
        return self.board().generateImage((w + 2 * margin_px, h + 2 * margin_px), marginSize=margin_px)


@dataclass(frozen=True)
class CharucoDetection:
    object_points: np.ndarray  # (N,1,3)
    image_points: np.ndarray  # (N,1,2)
    ids: np.ndarray

    @property
    def count(self) -> int:
        return int(len(self.ids))


def detect_charuco(image_bgr: np.ndarray, spec: CharucoSpec, *, min_corners: int = 12) -> CharucoDetection | None:
    board = spec.board()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    detector = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_ids is None or ch_corners is None or len(ch_ids) < min_corners:
        return None
    obj, img = board.matchImagePoints(ch_corners, ch_ids)
    return CharucoDetection(np.asarray(obj, dtype=np.float32).reshape(-1, 1, 3), np.asarray(img, dtype=np.float32).reshape(-1, 1, 2), np.asarray(ch_ids).reshape(-1))


def calibrate_pinhole(detections: list[CharucoDetection], image_size: tuple[int, int], *, camera_name: str = "c922", rational: bool = False) -> CameraIntrinsics:
    if len(detections) < 10:
        raise ValueError(f"need >= 10 views, got {len(detections)}")
    obj = [d.object_points.reshape(-1, 3) for d in detections]
    img = [d.image_points.reshape(-1, 2) for d in detections]
    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, None, None, flags=flags)
    errors = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        errors.append(np.mean(np.linalg.norm(proj.reshape(-1, 2) - i, axis=1)))
    return CameraIntrinsics(
        image_width=image_size[0],
        image_height=image_size[1],
        camera_matrix=np.asarray(K, dtype=np.float64),
        distortion_coefficients=np.asarray(dist, dtype=np.float64).reshape(-1),
        model="pinhole",
        rms_reprojection_error=float(rms),
        mean_reprojection_error=float(np.mean(errors)),
        num_views=len(detections),
        calibration_date=datetime.now(timezone.utc).isoformat(),
        camera_name=camera_name,
    )


def pose_diversity_ok(new: CharucoDetection, accepted: list[CharucoDetection], *, min_shift_px: float = 60.0) -> bool:
    """Reject views whose board centroid is too close to an already accepted one."""
    c = new.image_points.reshape(-1, 2).mean(axis=0)
    for d in accepted:
        if np.linalg.norm(d.image_points.reshape(-1, 2).mean(axis=0) - c) < min_shift_px:
            return False
    return True


def draw_detection(image_bgr: np.ndarray, det: CharucoDetection | None) -> np.ndarray:
    out = image_bgr.copy()
    if det is not None:
        cv2.aruco.drawDetectedCornersCharuco(out, det.image_points, det.ids.reshape(-1, 1))
    return out


__all__ = ["CharucoDetection", "CharucoSpec", "calibrate_pinhole", "detect_charuco", "draw_detection", "pose_diversity_ok"]
