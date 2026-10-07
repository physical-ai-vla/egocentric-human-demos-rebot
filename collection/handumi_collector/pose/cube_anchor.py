"""Cube faces as an object-relative cue. REJECTED AS A PRIMARY wrist-translation estimator on 2026-09-16.

Measured on the five pilot episodes (PNP_FEASIBILITY_REPORT.md), right wrist, strict detections, while the gyro says
the hand is barely turning:

    translation jump   13.8-36.0 mm median,  64-1040 mm p95,  up to 1.8 m
    rotation jump      17.6-126 deg median,  ~178 deg p95   (the planar solution flipping between frames)
    inter-cube spacing 23-606 mm of spread, with medians of 105-136 cm between cubes that sit ~30 cm apart

and tightening a reprojection filter made the median WORSE while collapsing coverage from 62 % to 9 %, because a
small distant face fits beautifully and has the worst depth conditioning. So the failure is the conditioning of a
single planar 50 mm face, not a detector threshold, and no amount of detector work reaches a usable Δp from it.

What it is still for: object-relative geometry (where a cube is with respect to the hand), local sanity checks, and
diagnostics. Not for metric wrist translation -- that comes from VIO, with rotation from the gyro.

Detection and per-face PnP; no trajectory, no absolute frame.

Why this exists: the wrist pose has to come from somewhere, and `opencv_vo` does not supply it -- measured against the
wrist IMU on the 2026-09-15 pilot, its SHORT-WINDOW relative rotation disagrees by 3-10 deg median, so it fails as a
relative estimator and not merely as an integrator. The cubes are the other candidate: 50 mm edges, known size, in
view for most of every episode.

What this module does NOT do, deliberately:
  * integrate a trajectory. The representation is relative -- inv(T(t)) T(t+k) and object-relative geometry -- so a
    world-frame wrist path is never needed and would only accumulate error nobody consumes.
  * use HOME-return closure or a valid-ratio gate. Those judge an absolute trajectory, which is not what is built here.

A single square face is a weak depth constraint by construction: dZ ~= Z^2 * dpx / (f * s), so at Z = 0.3 m with
f = 780 px and s = 0.05 m, one pixel of corner error is 2.3 mm of range. That sets the floor a perfect detector could
reach, and it is why inter-cube geometry is worth measuring -- the baseline between two cubes is 4-8x the edge of one.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

CUBE_M = 0.05

# Hue bands for the task cubes on a white table, from the 2026-09-15 pilot frames. Red wraps zero, so it gets two.
COLOUR_BANDS = {
    "red":    (((0, 110, 70), (12, 255, 255)), ((168, 110, 70), (180, 255, 255))),
    "blue":   (((95, 90, 50), (128, 255, 255)),),
    "purple": (((128, 60, 50), (160, 255, 255)),),
}
MIN_AREA_PX = 400
MIN_SOLIDITY = 0.85
BORDER_PX = 6

# SOLVEPNP_IPPE_SQUARE's documented object-point order. Supplying a different one does not fail -- it returns a
# confident pose for a rotated or mirrored correspondence, which showed up as several hundred pixels of reprojection.
_OBJ = np.array([[-CUBE_M / 2,  CUBE_M / 2, 0], [CUBE_M / 2,  CUBE_M / 2, 0],
                 [ CUBE_M / 2, -CUBE_M / 2, 0], [-CUBE_M / 2, -CUBE_M / 2, 0]], np.float64)


@dataclass(frozen=True)
class Face:
    colour: str
    corners: np.ndarray          # (4, 2) image points, in detection order
    area_px: float
    solidity: float
    clipped: bool                # touches the frame edge: its corners are not the cube's


@dataclass(frozen=True)
class FacePose:
    colour: str
    t: np.ndarray                # (3,) cube centre in camera coordinates, metres
    R: np.ndarray                # (3, 3) -- ambiguous by 90 deg about the face normal; a square looks the same 4 ways
    rms_px: float


def detect_faces(bgr: np.ndarray, *, min_area_px: int = MIN_AREA_PX, min_solidity: float = MIN_SOLIDITY,
                 border_px: int = BORDER_PX, strict: bool = True) -> dict[str, Face]:
    """Largest quad-shaped blob per colour. `strict` rejects clipped or non-quadrilateral blobs.

    Crude on purpose: the corners come from `approxPolyDP` on a colour mask, with no sub-pixel refinement, so this
    UNDERSTATES what a real detector would achieve. A good number from it can be trusted; a bad one is not yet a
    verdict about the geometry."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    out: dict[str, Face] = {}
    for name, bands in COLOUR_BANDS.items():
        mask = None
        for lo, hi in bands:
            b = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = b if mask is None else cv2.bitwise_or(mask, b)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        best: Face | None = None
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = float(cv2.contourArea(c))
            if area < min_area_px:
                continue
            pts = c.reshape(-1, 2)
            clipped = bool(pts[:, 0].min() <= border_px or pts[:, 1].min() <= border_px or
                           pts[:, 0].max() >= w - border_px or pts[:, 1].max() >= h - border_px)
            hull = float(cv2.contourArea(cv2.convexHull(c)))
            quad = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
            sol = area / hull if hull > 0 else 0.0
            if len(quad) != 4:
                continue
            if strict and (clipped or sol <= min_solidity):
                continue
            if best is None or area > best.area_px:
                best = Face(name, quad.reshape(4, 2).astype(np.float64), area, sol, clipped)
        if best is not None:
            out[name] = best
    return out


def _cyclic(p: np.ndarray) -> np.ndarray:
    c = p.mean(axis=0)
    return p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]


def solve_face(face: Face, K: np.ndarray, D: np.ndarray, *, size_m: float = CUBE_M,
               fisheye: bool = True) -> FacePose | None:
    """Pose of one square face, taking the best of every correspondence a square admits.

    Which detected corner is 'first' is unknowable from a blob, and a square is symmetric under 90 degree turns, so
    all four rotations and both windings are tried and judged by reprojection rather than guessed. IPPE_SQUARE also
    returns BOTH planar solutions; the ambiguity it leaves is about the face normal and does not move the centre,
    which is the quantity this is for."""
    obj = _OBJ * (size_m / CUBE_M)
    pts = face.corners.reshape(-1, 1, 2)
    und = (cv2.fisheye.undistortPoints(pts, K, np.asarray(D, np.float64).reshape(-1, 1)[:4], P=K)
           if fisheye else cv2.undistortPoints(pts, K, np.asarray(D, np.float64).reshape(-1), P=K))
    base = _cyclic(und.reshape(-1, 2))
    best: FacePose | None = None
    for k in range(4):
        rolled = np.roll(base, k, axis=0)
        for img in (rolled, rolled[::-1]):
            ok, rvs, tvs, _ = cv2.solvePnPGeneric(obj, np.ascontiguousarray(img).reshape(-1, 1, 2), K, None,
                                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok or tvs is None:
                continue
            for j in range(len(tvs)):
                proj, _ = cv2.projectPoints(obj, rvs[j], tvs[j], K, None)
                rms = float(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(axis=1).mean()))
                if best is None or rms < best.rms_px:
                    R, _ = cv2.Rodrigues(rvs[j])
                    best = FacePose(face.colour, np.asarray(tvs[j], np.float64).reshape(3), R, rms)
    return best


def pair_distances(poses: dict[str, FacePose]) -> dict[str, float]:
    """Metre distance between every pair of solved cube centres.

    The cubes do not move relative to each other between grasps, so these distances are constants of the scene and
    their spread over time measures the solves directly -- without needing the constellation known in advance, and
    without a joint solver. If they are stable, a multi-cube solve has something to work with; if they are not, it
    has nothing, and the extra machinery would only average noise."""
    names = sorted(poses)
    return {f"{a}-{b}": float(np.linalg.norm(poses[a].t - poses[b].t))
            for i, a in enumerate(names) for b in names[i + 1:]}
