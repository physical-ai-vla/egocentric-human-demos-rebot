"""PnP helpers shared by world-camera and wrist-tag pose estimation.

Tag frame: centre origin, +X right, +Y up, +Z out of the tag (towards a viewer
looking at it). Corner order TL, TR, BR, BL matches :mod:`detector`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ego_collector.camera.intrinsics import CameraIntrinsics
from ego_collector.tracking.transforms import T_from_rvec_tvec


def tag_corners_local(size_m: float) -> np.ndarray:
    h = float(size_m) / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)


@dataclass(frozen=True)
class PnPResult:
    T_camera_object: np.ndarray  # 4x4
    reprojection_error_px: float
    num_points: int
    inliers: int


def _reproj(obj: np.ndarray, img: np.ndarray, rvec, tvec, intr: CameraIntrinsics) -> float:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, intr.camera_matrix, intr.distortion_coefficients)
    return float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))


def solve_square_tag(corners_px: np.ndarray, size_m: float, intr: CameraIntrinsics) -> PnPResult | None:
    """Single planar tag: IPPE_SQUARE, keep the in-front solution with the lowest reprojection."""
    obj = tag_corners_local(size_m)
    img = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, intr.camera_matrix, intr.distortion_coefficients, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or not rvecs:
        return None
    best = None
    for rvec, tvec in zip(rvecs, tvecs):
        if float(np.asarray(tvec).reshape(3)[2]) <= 0:
            continue
        err = _reproj(obj, img, rvec, tvec, intr)
        if best is None or err < best[0]:
            best = (err, rvec, tvec)
    if best is None:
        return None
    err, rvec, tvec = best
    return PnPResult(T_from_rvec_tvec(rvec, tvec), err, 4, 4)


def solve_points(obj: np.ndarray, img: np.ndarray, intr: CameraIntrinsics, *, ransac: bool = True, ransac_reproj_px: float = 4.0) -> PnPResult | None:
    """General PnP over all corners of all visible tags (>= 2 tags).

    RANSAC first (when it succeeds), otherwise a robust fallback: SQPNP on all points,
    drop points whose reprojection exceeds ``ransac_reproj_px``, re-solve. Always
    finish with LM refinement on the inliers.
    """
    obj = np.ascontiguousarray(np.asarray(obj, dtype=np.float64).reshape(-1, 3))
    img = np.ascontiguousarray(np.asarray(img, dtype=np.float64).reshape(-1, 2))
    n = len(obj)
    if n < 4:
        return None
    K, dist = intr.camera_matrix, intr.distortion_coefficients
    rvec = tvec = None
    inlier_idx = np.arange(n)
    if ransac and n >= 6:
        try:
            ok, r, t, inl = cv2.solvePnPRansac(
                obj.astype(np.float32), img.astype(np.float32), K, dist,
                reprojectionError=ransac_reproj_px, iterationsCount=300, confidence=0.99, flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error:
            ok, inl = False, None
        if ok and inl is not None and len(inl) >= 4:
            rvec, tvec, inlier_idx = r, t, inl.reshape(-1)
    if rvec is None:
        flags = cv2.SOLVEPNP_SQPNP if n > 4 else cv2.SOLVEPNP_IPPE
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flags)
        if not ok:
            return None
        if n > 4:  # manual outlier rejection (coplanar world tags make OpenCV's RANSAC fail)
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            res = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
            keep = np.flatnonzero(res <= ransac_reproj_px)
            if 4 <= len(keep) < n:
                ok, rvec, tvec = cv2.solvePnP(obj[keep], img[keep], K, dist, flags=cv2.SOLVEPNP_SQPNP if len(keep) > 4 else cv2.SOLVEPNP_IPPE)
                if not ok:
                    return None
                inlier_idx = keep
    rvec, tvec = cv2.solvePnPRefineLM(obj[inlier_idx], img[inlier_idx], K, dist, rvec, tvec)
    if float(np.asarray(tvec).reshape(3)[2]) <= 0:
        return None
    err = _reproj(obj[inlier_idx], img[inlier_idx], rvec, tvec, intr)
    return PnPResult(T_from_rvec_tvec(rvec, tvec), err, n, int(len(inlier_idx)))


__all__ = ["PnPResult", "solve_points", "solve_square_tag", "tag_corners_local"]
