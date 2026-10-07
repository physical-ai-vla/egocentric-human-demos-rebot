"""Hide AprilTags in the head video (``head_rgb_raw`` -> ``head_rgb_clean``).

The recorded rows already carry every detected tag's pixel corners
(``observation.apriltag.{world,left,right}_corners_px``), so no re-detection is
needed: we dilate each quad by ``margin`` and either fill it with the local
border colour or blur it. Removing the markers prevents the policy from
learning an AprilTag -> action shortcut.
"""

from __future__ import annotations

from typing import Iterable, Literal

import cv2
import numpy as np

Method = Literal["fill", "blur", "noise"]


def corners_from_row(row: dict, *, sides: Iterable[str] = ("world", "left", "right")) -> list[np.ndarray]:
    """Collect (4,2) corner arrays from one dataset row (NaN-padded columns)."""
    quads: list[np.ndarray] = []
    for side in sides:
        values = row.get(f"observation.apriltag.{side}_corners_px")
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float64).reshape(-1, 8)
        for q in arr:
            if np.isfinite(q).all():
                quads.append(q.reshape(4, 2))
    return quads


def dilate_quad(quad: np.ndarray, margin: float) -> np.ndarray:
    """Scale a quad about its centroid by ``1 + margin`` (margin 0.15 = 15 % larger)."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    c = q.mean(axis=0)
    return c + (q - c) * (1.0 + float(margin))


def mask_tags(image: np.ndarray, quads: Iterable[np.ndarray], *, method: Method = "fill", margin: float = 0.2, blur_ksize: int = 31) -> np.ndarray:
    """Return a copy of ``image`` with every quad hidden."""
    out = np.ascontiguousarray(image.copy())
    h, w = out.shape[:2]
    for quad in quads:
        poly = np.round(dilate_quad(quad, margin)).astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        if not mask.any():
            continue
        if method == "fill":
            ring = cv2.dilate(mask, np.ones((9, 9), np.uint8)) & ~mask
            colour = out[ring > 0].reshape(-1, out.shape[2] if out.ndim == 3 else 1).mean(axis=0) if ring.any() else out[mask > 0].mean(axis=0)
            out[mask > 0] = np.round(colour).astype(out.dtype)
        elif method == "blur":
            k = blur_ksize | 1
            blurred = cv2.GaussianBlur(out, (k, k), 0)
            out[mask > 0] = blurred[mask > 0]
        elif method == "noise":
            rng = np.random.default_rng(int(mask.sum()))
            out[mask > 0] = rng.integers(0, 256, size=(int((mask > 0).sum()), out.shape[2]) if out.ndim == 3 else int((mask > 0).sum()), dtype=np.uint8)
        else:
            raise ValueError(f"unknown method {method!r}")
    return out


__all__ = ["Method", "corners_from_row", "dilate_quad", "mask_tags"]
