"""AprilTag detector interface with two backends.

* ``pupil`` — pupil-apriltags (AprilTag 3, any family incl. tagStandard41h12). Preferred.
* ``opencv`` — cv2.aruco (tag36h11 / 36h10 / 25h9 / 16h5 only). Fallback.

Both return corners normalized to **TL, TR, BR, BL** in *image pixel* coordinates
(distorted, as detected). Pose is never taken from the detector; PnP with the
calibrated intrinsics does that (:mod:`ego_collector.tracking.pnp`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger("ego_collector.detector")

OPENCV_FAMILIES = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag16h5": "DICT_APRILTAG_16h5",
}
PUPIL_FAMILIES = {"tag36h11", "tag36h10", "tag25h9", "tag16h5", "tagCircle21h7", "tagCircle49h12", "tagStandard41h12", "tagStandard52h13", "tagCustom48h12"}


@dataclass(frozen=True)
class TagDetection:
    tag_id: int
    corners: np.ndarray  # (4,2) TL,TR,BR,BL pixels
    center: np.ndarray  # (2,)
    decision_margin: float = float("nan")  # pupil only
    hamming: int = 0

    def as_row(self) -> dict:
        return {
            "tag_id": int(self.tag_id),
            "corners": self.corners.reshape(-1).astype(np.float64),
            "center": self.center.astype(np.float64),
            "decision_margin": float(self.decision_margin),
            "hamming": int(self.hamming),
        }


class TagDetector(Protocol):
    family: str
    backend: str

    def detect(self, gray: np.ndarray) -> list[TagDetection]: ...


def _to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class OpenCVTagDetector:
    backend = "opencv"

    def __init__(self, family: str = "tag36h11", *, subpixel: bool = True) -> None:
        if family not in OPENCV_FAMILIES:
            raise ValueError(f"OpenCV backend supports {sorted(OPENCV_FAMILIES)}, not {family!r} (install pupil-apriltags)")
        self.family = family
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, OPENCV_FAMILIES[family]))
        params = cv2.aruco.DetectorParameters()
        if subpixel:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
        params.minMarkerPerimeterRate = 0.015
        self._det = cv2.aruco.ArucoDetector(self.dictionary, params)

    def detect(self, image: np.ndarray) -> list[TagDetection]:
        corners, ids, _ = self._det.detectMarkers(_to_gray(image))
        if ids is None:
            return []
        out = []
        for c, i in zip(corners, ids.reshape(-1)):
            c = np.asarray(c, dtype=np.float64).reshape(4, 2)  # OpenCV: TL,TR,BR,BL
            out.append(TagDetection(tag_id=int(i), corners=c, center=c.mean(axis=0), hamming=0))
        return out

    def marker_image(self, tag_id: int, size_px: int) -> np.ndarray:
        return cv2.aruco.generateImageMarker(self.dictionary, int(tag_id), int(size_px))


class PupilTagDetector:
    backend = "pupil"

    def __init__(self, family: str = "tagStandard41h12", *, nthreads: int = 2, quad_decimate: float = 1.0, decode_sharpening: float = 0.25, refine_edges: bool = True) -> None:
        import pupil_apriltags  # type: ignore

        if family not in PUPIL_FAMILIES:
            raise ValueError(f"unknown AprilTag family {family!r}")
        self.family = family
        self._det = pupil_apriltags.Detector(families=family, nthreads=nthreads, quad_decimate=quad_decimate, quad_sigma=0.0, refine_edges=int(refine_edges), decode_sharpening=decode_sharpening)

    def detect(self, image: np.ndarray) -> list[TagDetection]:
        gray = np.ascontiguousarray(_to_gray(image))
        out = []
        for d in self._det.detect(gray):
            # apriltag3 reports corners wrapping counter-clockwise in the tag's own (y-up) frame;
            # relative to OpenCV's canonical TL,TR,BR,BL order that is [TR, TL, BL, BR]
            # (verified against OpenCV aruco on rendered tag36h11 markers, see tests/test_world_pose.py).
            p = np.asarray(d.corners, dtype=np.float64).reshape(4, 2)
            c = np.stack([p[1], p[0], p[3], p[2]])
            out.append(TagDetection(tag_id=int(d.tag_id), corners=c, center=np.asarray(d.center, dtype=np.float64), decision_margin=float(d.decision_margin), hamming=int(d.hamming)))
        return out


def make_detector(family: str, backend: str = "auto", **kwargs) -> TagDetector:
    if backend in ("auto", "pupil"):
        try:
            return PupilTagDetector(family, **kwargs)
        except ImportError:
            if backend == "pupil":
                raise
            log.warning("pupil-apriltags not installed; falling back to OpenCV aruco")
    return OpenCVTagDetector(family)


__all__ = ["OPENCV_FAMILIES", "PUPIL_FAMILIES", "OpenCVTagDetector", "PupilTagDetector", "TagDetection", "TagDetector", "make_detector"]
