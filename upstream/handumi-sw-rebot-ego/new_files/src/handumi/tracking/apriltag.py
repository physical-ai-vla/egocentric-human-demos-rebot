"""AprilTag EEF tracking from the *head-mounted* camera (third HandUMI tracking backend).

One egocentric camera (Arducam B0202, 160° fisheye) sees, in every frame,

* fixed **world tags** around the table  -> ``T_camera_world`` (head localisation,
  no SLAM/VIO: the camera motion is removed per frame by the fixed markers),
* the **left / right UMI tag bundles**   -> ``T_camera_anchor`` per hand.

The usual controller->TCP (pivot) calibration then gives the TCP, exactly like
the Quest/PICO backends, with the bundle anchor tag playing the "controller"::

    T_world_TCP = inv(T_camera_world) · T_camera_anchor · T_anchor_TCP

Mapping onto the HandUMI sample schema (so record/QA/convert stay unchanged):

* ``*_controller_pose``          world-frame anchor pose (``observation.state``)
* ``*_device_controller_pose``   camera-frame anchor pose  (head-relative EEF)
* ``workspace_from_device_pose`` ``T_world_camera`` of *this frame*  (= head pose)
* ``hmd_pose`` / ``hmd_tracked`` head pose in world / world anchor visible
* ``observation.apriltag.*``     raw detector output for world/left/right
  (tag ids, pixel corners, reprojection error, camera-frame pose) so poses can
  be re-solved offline with a better calibration or PnP.

Fisheye handling: detection runs on a rectified (virtual pinhole) view built
with ``cv2.fisheye``; corners are stored in that rectified pixel space
(``observation.apriltag.corner_space == 1``) and can be mapped back to raw
pixels with :meth:`CameraGeometry.to_raw_pixels`.

Frame conventions: tag frame = centre origin, +X right, +Y up, +Z out of the tag
(OpenCV ``IPPE_SQUARE``); corners TL, TR, BR, BL. ``anchor_from_tag`` = T_anchor_tag,
``world_from_tag`` = T_world_tag (translation_m / quaternion_xyzw).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.calibration.spatial import (
    CameraIntrinsics,
    mean_transform,
    pose7_to_dict,
    transform_errors,
)
from handumi.cameras.base import CameraDevice, CameraSample
from handumi.robots.utils import IDENTITY_POSE7, mat_to_pose7, pose7_to_mat, pose_inv, pose_mul, quat_normalize
from handumi.tracking.base import ControllerPairSample, apply_tcp_calibration_pose7

log = logging.getLogger("handumi.tracking.apriltag")

SIDES = ("left", "right")
RAW_SIDES = ("world", "left", "right")
DEFAULT_BUNDLES_PATH = Path("configs/calibration/apriltag/bundles.yaml")
DEFAULT_WORLD_MAP_PATH = Path("outputs/calibration/world_map.yaml")
DEFAULT_DICTIONARY = "DICT_APRILTAG_36h11"
MAX_TAGS_PER_BUNDLE = 4
MAX_WORLD_TAGS = 8
DEFAULT_MAX_REPROJ_PX = 2.0
DEFAULT_STALE_TIMEOUT_S = 0.25
CORNER_SPACE_RAW = 0
CORNER_SPACE_RECTIFIED = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Tag / bundle / world-map configuration
# ---------------------------------------------------------------------------


def pose7_from_any(data: dict[str, Any]) -> np.ndarray:
    """Accept spatial-yaml keys (translation_m/quaternion_xyzw) or short ones (position/quaternion)."""
    pos = data.get("translation_m", data.get("position"))
    quat = data.get("quaternion_xyzw", data.get("quaternion"))
    if pos is None or quat is None:
        raise ValueError(f"pose dict needs translation_m/position and quaternion_xyzw/quaternion: {data}")
    out = np.concatenate([np.asarray(pos, dtype=np.float64).reshape(3), quat_normalize(np.asarray(quat, dtype=np.float64).reshape(4))])
    return out.astype(np.float64)


def tag_corners_local(size_m: float) -> np.ndarray:
    """(4, 3) TL, TR, BR, BL corners of a tag in its own frame (metres)."""
    h = float(size_m) / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)


@dataclass(frozen=True)
class TagSpec:
    id: int
    size_m: float
    anchor_from_tag: np.ndarray = field(default_factory=lambda: IDENTITY_POSE7.astype(np.float64).copy())
    measured: bool = True

    def corners_in_anchor(self) -> np.ndarray:
        T = pose7_to_mat(self.anchor_from_tag)
        pts = tag_corners_local(self.size_m)
        return (T[:3, :3] @ pts.T).T + T[:3, 3]


@dataclass(frozen=True)
class BundleSpec:
    """A rigid set of tags with one reference ("anchor") frame.

    For the hands the anchor is the anchor tag itself; for the world map the
    anchor frame is the world/table frame and ``anchor_from_tag`` = T_world_tag.
    """

    side: str
    anchor_tag: int
    tags: tuple[TagSpec, ...]

    def tag(self, tag_id: int) -> TagSpec | None:
        return next((t for t in self.tags if t.id == tag_id), None)

    @property
    def tag_ids(self) -> tuple[int, ...]:
        return tuple(t.id for t in self.tags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_tag": self.anchor_tag,
            "tags": [
                {"id": t.id, "size_m": t.size_m, "anchor_from_tag": pose7_to_dict(t.anchor_from_tag), "measured": t.measured}
                for t in self.tags
            ],
        }

    @classmethod
    def from_dict(cls, side: str, raw: dict[str, Any], *, pose_key: str = "anchor_from_tag") -> "BundleSpec":
        tags = []
        for t in raw.get("tags") or []:
            pose = t.get(pose_key, t.get("anchor_from_tag"))
            tags.append(
                TagSpec(
                    id=int(t["id"]),
                    size_m=float(t["size_m"]),
                    anchor_from_tag=pose7_from_any(pose) if isinstance(pose, dict) else IDENTITY_POSE7.astype(np.float64).copy(),
                    measured=bool(t.get("measured", True)),
                )
            )
        if not tags:
            raise ValueError(f"bundle {side!r} has no tags")
        anchor = int(raw.get("anchor_tag", tags[0].id))
        if anchor not in {t.id for t in tags}:
            raise ValueError(f"bundle {side!r}: anchor_tag {anchor} not among its tags")
        return cls(side=str(side), anchor_tag=anchor, tags=tuple(tags))


@dataclass(frozen=True)
class BundleConfig:
    dictionary: str
    bundles: dict[str, BundleSpec]
    max_reproj_px: float = DEFAULT_MAX_REPROJ_PX
    source: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: Path | None = None) -> "BundleConfig":
        bundles = {str(side): BundleSpec.from_dict(str(side), raw) for side, raw in (data.get("bundles") or {}).items()}
        ids = [t.id for b in bundles.values() for t in b.tags]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate tag ids across bundles: {ids}")
        return cls(
            dictionary=str(data.get("dictionary", DEFAULT_DICTIONARY)),
            bundles=bundles,
            max_reproj_px=float(data.get("max_reproj_px", DEFAULT_MAX_REPROJ_PX)),
            source=source,
        )

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_BUNDLES_PATH) -> "BundleConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {}, source=Path(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dictionary": self.dictionary,
            "max_reproj_px": self.max_reproj_px,
            "bundles": {side: b.to_dict() for side, b in self.bundles.items()},
        }

    def side_of(self, tag_id: int) -> str | None:
        for side, b in self.bundles.items():
            if tag_id in b.tag_ids:
                return side
        return None


@dataclass(frozen=True)
class WorldMap:
    """Fixed tags around the table; ``bundle.anchor_from_tag`` holds T_world_tag."""

    bundle: BundleSpec
    frame: str = "table"  # "table" (ChArUco-defined) or "tag<id>" (anchor tag frame)
    metrics: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    @property
    def tag_ids(self) -> tuple[int, ...]:
        return self.bundle.tag_ids

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: Path | None = None) -> "WorldMap":
        if data.get("kind") not in (None, "handumi_apriltag_world_map"):
            raise ValueError(f"not a world map: {data.get('kind')}")
        bundle = BundleSpec.from_dict("world", data, pose_key="world_from_tag")
        return cls(bundle=bundle, frame=str(data.get("frame", "table")), metrics=dict(data.get("metrics") or {}), source=source)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_WORLD_MAP_PATH) -> "WorldMap":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {}, source=Path(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "handumi_apriltag_world_map",
            "frame": self.frame,
            "created_at": _now_iso(),
            "anchor_tag": self.bundle.anchor_tag,
            "tags": [
                {"id": t.id, "size_m": t.size_m, "world_from_tag": pose7_to_dict(t.anchor_from_tag), "measured": t.measured}
                for t in self.bundle.tags
            ],
            "metrics": self.metrics,
        }

    def write_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


# ---------------------------------------------------------------------------
# Camera geometry (pinhole passthrough or fisheye rectification)
# ---------------------------------------------------------------------------


@dataclass
class CameraGeometry:
    """Where detection happens and which (K, dist) PnP must use."""

    intrinsics: CameraIntrinsics
    rectify: bool
    K: np.ndarray  # camera matrix valid for the detection image
    dist: np.ndarray  # distortion valid for the detection image
    map1: np.ndarray | None = None
    map2: np.ndarray | None = None

    @classmethod
    def for_intrinsics(cls, intrinsics: CameraIntrinsics, *, balance: float = 0.5, fov_scale: float = 1.0) -> "CameraGeometry":
        K = np.asarray(intrinsics.matrix, dtype=np.float64)
        D = np.asarray(intrinsics.distortion, dtype=np.float64).reshape(-1, 1)
        size = (int(intrinsics.width), int(intrinsics.height))
        if intrinsics.model != "fisheye":
            return cls(intrinsics=intrinsics, rectify=False, K=K, dist=D.reshape(-1))
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, np.eye(3), balance=balance, new_size=size, fov_scale=fov_scale)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, size, cv2.CV_16SC2)
        return cls(intrinsics=intrinsics, rectify=True, K=np.asarray(newK, dtype=np.float64), dist=np.zeros(5), map1=map1, map2=map2)

    @property
    def corner_space(self) -> int:
        return CORNER_SPACE_RECTIFIED if self.rectify else CORNER_SPACE_RAW

    def prepare(self, image: np.ndarray) -> np.ndarray:
        if not self.rectify:
            return image
        return cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def project(self, points3d: np.ndarray, rvec, tvec) -> np.ndarray:
        proj, _ = cv2.projectPoints(np.asarray(points3d, dtype=np.float64), rvec, tvec, self.K, self.dist)
        return proj.reshape(-1, 2)

    def detection_intrinsics(self) -> CameraIntrinsics:
        """Pinhole intrinsics describing the detection image (for ChArUco on the same view)."""
        if not self.rectify:
            return self.intrinsics
        return CameraIntrinsics(
            camera=self.intrinsics.camera + "_rectified",
            width=self.intrinsics.width,
            height=self.intrinsics.height,
            matrix=self.K,
            distortion=np.zeros((5, 1)),
            rms_px=self.intrinsics.rms_px,
            mean_error_px=self.intrinsics.mean_error_px,
            views=self.intrinsics.views,
            model="pinhole",
        )

    def to_raw_pixels(self, corners: np.ndarray) -> np.ndarray:
        """Map detection-image pixels back to raw (distorted) image pixels."""
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if not self.rectify:
            return pts
        Kinv = np.linalg.inv(self.K)
        norm = (Kinv @ np.concatenate([pts, np.ones((len(pts), 1))], axis=1).T).T[:, :2]
        K = np.asarray(self.intrinsics.matrix, dtype=np.float64)
        D = np.asarray(self.intrinsics.distortion, dtype=np.float64).reshape(-1, 1)
        distorted = cv2.fisheye.distortPoints(norm.reshape(-1, 1, 2), K, D)
        return distorted.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TagDetection:
    id: int
    corners: np.ndarray  # (4, 2) pixels in the detection image, TL TR BR BL


class AprilTagDetector:
    def __init__(self, dictionary: str = DEFAULT_DICTIONARY, *, subpixel: bool = True) -> None:
        self.dictionary_name = dictionary
        self._dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        params = cv2.aruco.DetectorParameters()
        if subpixel:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
        params.minMarkerPerimeterRate = 0.02
        self._detector = cv2.aruco.ArucoDetector(self._dict, params)

    def detect(self, image: np.ndarray) -> list[TagDetection]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        return [TagDetection(id=int(i), corners=np.asarray(c, dtype=np.float64).reshape(4, 2)) for c, i in zip(corners, ids.reshape(-1))]

    def marker_image(self, tag_id: int, size_px: int = 200) -> np.ndarray:
        return cv2.aruco.generateImageMarker(self._dict, int(tag_id), int(size_px))


# ---------------------------------------------------------------------------
# PnP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleSolve:
    side: str
    camera_from_anchor: np.ndarray  # pose7, T_camera_anchor (anchor = tag / world frame)
    tag_ids: tuple[int, ...]
    reproj_px: float
    num_points: int


def _rt_to_pose7(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return mat_to_pose7(T).astype(np.float64)


def _solve_points(obj: np.ndarray, px: np.ndarray, geometry: CameraGeometry, *, square: bool) -> tuple[np.ndarray, float] | None:
    flags = cv2.SOLVEPNP_IPPE_SQUARE if square else (cv2.SOLVEPNP_IPPE if len(px) == 4 else cv2.SOLVEPNP_SQPNP)
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, px, geometry.K, geometry.dist, flags=flags)
    if not ok or not rvecs:
        return None
    best = None
    for rvec, tvec in zip(rvecs, tvecs):
        if float(np.asarray(tvec).reshape(3)[2]) <= 0:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(obj, px, geometry.K, geometry.dist, rvec, tvec)
        err = float(np.mean(np.linalg.norm(geometry.project(obj, rvec, tvec) - px, axis=1)))
        if best is None or err < best[0]:
            best = (err, rvec, tvec)
    if best is None:
        return None
    err, rvec, tvec = best
    return _rt_to_pose7(rvec, tvec), err


def solve_single_tag(det: TagDetection, size_m: float, geometry: CameraGeometry) -> tuple[np.ndarray, float]:
    """``T_camera_tag`` (pose7) and reprojection error for one tag."""
    out = _solve_points(tag_corners_local(size_m), det.corners, geometry, square=True)
    if out is None:
        raise ValueError(f"PnP failed for tag {det.id}")
    return out


def solve_bundle(detections: Sequence[TagDetection], bundle: BundleSpec, geometry: CameraGeometry) -> BundleSolve | None:
    """Joint PnP over every visible tag of ``bundle`` -> ``T_camera_anchor``."""
    used = [(d, bundle.tag(d.id)) for d in detections if bundle.tag(d.id) is not None]
    if not used:
        return None
    obj = np.concatenate([spec.corners_in_anchor() for _, spec in used], axis=0)
    px = np.concatenate([d.corners for d, _ in used], axis=0)
    square = len(used) == 1 and bool(np.allclose(used[0][1].anchor_from_tag, IDENTITY_POSE7))
    out = _solve_points(obj, px, geometry, square=square)
    if out is None:
        return None
    pose, err = out
    return BundleSolve(side=bundle.side, camera_from_anchor=pose, tag_ids=tuple(d.id for d, _ in used), reproj_px=err, num_points=int(len(px)))


# ---------------------------------------------------------------------------
# Offline calibrations: bundle geometry, world map
# ---------------------------------------------------------------------------


def calibrate_bundle_geometry(
    frames: Sequence[Sequence[TagDetection]],
    bundle: BundleSpec,
    geometry: CameraGeometry,
    *,
    max_reproj_px: float = 1.0,
) -> tuple[BundleSpec, dict[str, dict[str, float]]]:
    """Estimate ``T_anchor_tag`` for every non-anchor tag from frames where both are visible."""
    anchor_spec = bundle.tag(bundle.anchor_tag)
    assert anchor_spec is not None
    rel: dict[int, list[np.ndarray]] = {t.id: [] for t in bundle.tags if t.id != bundle.anchor_tag}
    for dets in frames:
        by_id = {d.id: d for d in dets}
        if bundle.anchor_tag not in by_id:
            continue
        try:
            cam_anchor, err_a = solve_single_tag(by_id[bundle.anchor_tag], anchor_spec.size_m, geometry)
        except ValueError:
            continue
        if err_a > max_reproj_px:
            continue
        T_ca = pose7_to_mat(cam_anchor)
        for tag_id, acc in rel.items():
            if tag_id not in by_id:
                continue
            spec = bundle.tag(tag_id)
            assert spec is not None
            try:
                cam_tag, err_t = solve_single_tag(by_id[tag_id], spec.size_m, geometry)
            except ValueError:
                continue
            if err_t <= max_reproj_px:
                acc.append(np.linalg.inv(T_ca) @ pose7_to_mat(cam_tag))
    return _average_relative(bundle, rel)


def calibrate_world_map(
    frames: Sequence[tuple[Sequence[TagDetection], np.ndarray]],
    tag_sizes: dict[int, float],
    geometry: CameraGeometry,
    *,
    frame_name: str = "table",
    anchor_tag: int | None = None,
    max_reproj_px: float = 1.0,
) -> tuple[WorldMap, dict[str, dict[str, float]]]:
    """Fit ``T_world_tag`` for fixed tags.

    Each frame carries its detections and the reference ``T_camera_world`` (pose7)
    for that frame — from the ChArUco board on the table (``frame_name='table'``)
    or from the anchor tag (``frame_name='tag<id>'``, reference = that tag's pose).
    """
    ids = sorted(tag_sizes)
    anchor = int(anchor_tag if anchor_tag is not None else ids[0])
    bundle = BundleSpec(
        side="world",
        anchor_tag=anchor,
        tags=tuple(TagSpec(id=i, size_m=float(tag_sizes[i]), measured=False) for i in ids),
    )
    rel: dict[int, list[np.ndarray]] = {i: [] for i in ids}
    for dets, cam_world in frames:
        T_wc = np.linalg.inv(pose7_to_mat(np.asarray(cam_world, dtype=np.float64)))
        for d in dets:
            if d.id not in rel:
                continue
            try:
                cam_tag, err = solve_single_tag(d, tag_sizes[d.id], geometry)
            except ValueError:
                continue
            if err <= max_reproj_px:
                rel[d.id].append(T_wc @ pose7_to_mat(cam_tag))
    # every world tag (anchor included) is measured relative to the world frame
    new_tags: list[TagSpec] = []
    metrics: dict[str, dict[str, float]] = {}
    for spec in bundle.tags:
        samples = rel[spec.id]
        if len(samples) < 3:
            log.warning("world tag %d: only %d usable frames; left unmeasured", spec.id, len(samples))
            new_tags.append(spec)
            metrics[str(spec.id)] = {"views": float(len(samples))}
            continue
        rms_mm, max_mm, rms_deg, max_deg = transform_errors(samples)
        new_tags.append(replace(spec, anchor_from_tag=mat_to_pose7(mean_transform(samples)).astype(np.float64), measured=True))
        metrics[str(spec.id)] = {
            "views": float(len(samples)),
            "translation_rms_mm": rms_mm,
            "translation_max_mm": max_mm,
            "rotation_rms_deg": rms_deg,
            "rotation_max_deg": max_deg,
        }
    measured = tuple(t for t in new_tags if t.measured)
    if not measured:
        raise ValueError("no world tag could be measured")
    world = WorldMap(bundle=replace(bundle, tags=tuple(measured)), frame=frame_name, metrics=metrics)
    return world, metrics


def _average_relative(bundle: BundleSpec, rel: dict[int, list[np.ndarray]]) -> tuple[BundleSpec, dict[str, dict[str, float]]]:
    new_tags = []
    metrics: dict[str, dict[str, float]] = {}
    for spec in bundle.tags:
        if spec.id == bundle.anchor_tag:
            new_tags.append(replace(spec, anchor_from_tag=IDENTITY_POSE7.astype(np.float64).copy(), measured=True))
            continue
        samples = rel[spec.id]
        if len(samples) < 3:
            log.warning("tag %d: only %d co-visible frames; keeping previous geometry", spec.id, len(samples))
            new_tags.append(spec)
            metrics[str(spec.id)] = {"views": float(len(samples))}
            continue
        rms_mm, max_mm, rms_deg, max_deg = transform_errors(samples)
        new_tags.append(replace(spec, anchor_from_tag=mat_to_pose7(mean_transform(samples)).astype(np.float64), measured=True))
        metrics[str(spec.id)] = {
            "views": float(len(samples)),
            "translation_rms_mm": rms_mm,
            "translation_max_mm": max_mm,
            "rotation_rms_deg": rms_deg,
            "rotation_max_deg": max_deg,
        }
    return replace(bundle, tags=tuple(new_tags)), metrics


# ---------------------------------------------------------------------------
# Per-frame result + raw dataset schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SideTagResult:
    solve: BundleSolve | None
    detections: tuple[TagDetection, ...]

    @property
    def tracked(self) -> bool:
        return self.solve is not None


@dataclass(frozen=True)
class FrameResult:
    world: SideTagResult | None  # None when no world map is configured
    hands: dict[str, SideTagResult]
    detections: tuple[TagDetection, ...]
    corner_space: int = CORNER_SPACE_RAW

    @property
    def world_from_camera(self) -> np.ndarray | None:
        if self.world is None or self.world.solve is None:
            return None
        return pose_inv(np.asarray(self.world.solve.camera_from_anchor, dtype=np.float64))


def _max_tags(side: str) -> int:
    return MAX_WORLD_TAGS if side == "world" else MAX_TAGS_PER_BUNDLE


def apriltag_features() -> dict[str, Any]:
    """Raw detector output recorded per row (enables offline re-solving)."""
    features: dict[str, Any] = {}
    for side in RAW_SIDES:
        p = f"observation.apriltag.{side}"
        n = _max_tags(side)
        features[f"{p}_num_tags"] = {"dtype": "int64", "shape": (1,), "names": None}
        features[f"{p}_reproj_px"] = {"dtype": "float32", "shape": (1,), "names": None}
        features[f"{p}_tag_ids"] = {"dtype": "int64", "shape": (n,), "names": None}
        features[f"{p}_corners_px"] = {"dtype": "float32", "shape": (n * 8,), "names": None}  # tag-major TL.u TL.v TR.u ... NaN padded
        features[f"{p}_camera_pose"] = {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]}
    features["observation.apriltag.capture_time_ns"] = {"dtype": "int64", "shape": (1,), "names": None}
    features["observation.apriltag.sequence"] = {"dtype": "int64", "shape": (1,), "names": None}
    features["observation.apriltag.corner_space"] = {"dtype": "int64", "shape": (1,), "names": ["0=raw pixels, 1=rectified pinhole pixels"]}
    return features


def apriltag_frame(result: FrameResult, *, capture_time_ns: int, sequence: int) -> dict[str, np.ndarray]:
    frame: dict[str, np.ndarray] = {}
    per_side: dict[str, SideTagResult | None] = {"world": result.world, **result.hands}
    for side in RAW_SIDES:
        p = f"observation.apriltag.{side}"
        n = _max_tags(side)
        r = per_side.get(side)
        ids = np.full(n, -1, dtype=np.int64)
        corners = np.full(n * 8, np.nan, dtype=np.float32)
        pose = IDENTITY_POSE7.astype(np.float32).copy()
        count = 0
        err = float("nan")
        if r is not None:
            for k, det in enumerate(r.detections[:n]):
                ids[k] = det.id
                corners[k * 8 : (k + 1) * 8] = det.corners.reshape(-1).astype(np.float32)
            count = len(r.detections)
            if r.solve is not None:
                pose = np.asarray(r.solve.camera_from_anchor, dtype=np.float32)
                err = r.solve.reproj_px
        frame[f"{p}_num_tags"] = np.array([count], dtype=np.int64)
        frame[f"{p}_reproj_px"] = np.array([err], dtype=np.float32)
        frame[f"{p}_tag_ids"] = ids
        frame[f"{p}_corners_px"] = corners
        frame[f"{p}_camera_pose"] = pose
    frame["observation.apriltag.capture_time_ns"] = np.array([int(capture_time_ns)], dtype=np.int64)
    frame["observation.apriltag.sequence"] = np.array([int(sequence)], dtype=np.int64)
    frame["observation.apriltag.corner_space"] = np.array([int(result.corner_space)], dtype=np.int64)
    return frame


@dataclass(frozen=True)
class AprilTagPairSample(ControllerPairSample):
    """ControllerPairSample plus the raw per-frame tag results."""

    apriltag: dict[str, np.ndarray] | None = None

    def tracking_frame(self) -> dict[str, np.ndarray]:
        frame = super().tracking_frame()
        if self.apriltag:
            frame.update(self.apriltag)
        return frame


# ---------------------------------------------------------------------------
# Frame processing (pure) and sample composition
# ---------------------------------------------------------------------------


def process_frame(
    image: np.ndarray,
    *,
    detector: AprilTagDetector,
    config: BundleConfig,
    geometry: CameraGeometry,
    world_map: WorldMap | None = None,
) -> FrameResult:
    prepared = geometry.prepare(image)
    detections = tuple(detector.detect(prepared))
    world: SideTagResult | None = None
    if world_map is not None:
        dets = tuple(d for d in detections if d.id in world_map.tag_ids)
        solve = solve_bundle(dets, world_map.bundle, geometry) if dets else None
        if solve is not None and solve.reproj_px > config.max_reproj_px:
            solve = None
        world = SideTagResult(solve=solve, detections=dets)
    hands: dict[str, SideTagResult] = {}
    for side, bundle in config.bundles.items():
        dets = tuple(d for d in detections if d.id in bundle.tag_ids)
        solve = solve_bundle(dets, bundle, geometry) if dets else None
        if solve is not None and solve.reproj_px > config.max_reproj_px:
            solve = None
        hands[side] = SideTagResult(solve=solve, detections=dets)
    return FrameResult(world=world, hands=hands, detections=detections, corner_space=geometry.corner_space)


def build_pair_sample(
    result: FrameResult,
    *,
    calibration: ControllerTcpCalibration,
    capture_time_ns: int,
    sequence: int,
    fixed_world_from_camera: np.ndarray | None = None,
    require_world: bool = True,
    streaming: bool = True,
    include_raw: bool = True,
) -> AprilTagPairSample:
    """Compose the normalized sample.

    World frame comes from this frame's world-tag solve; if the world anchor is
    not visible, ``fixed_world_from_camera`` (a static camera session) is used
    when given, otherwise the hands are reported *untracked* (``require_world``)
    or left in the camera frame.
    """
    world_from_camera = result.world_from_camera
    world_visible = world_from_camera is not None
    if world_from_camera is None and fixed_world_from_camera is not None:
        world_from_camera = np.asarray(fixed_world_from_camera, dtype=np.float64)
    frame_valid = world_from_camera is not None or not require_world
    if world_from_camera is None:
        world_from_camera = IDENTITY_POSE7.astype(np.float64).copy()
    poses: dict[str, np.ndarray] = {}
    device_poses: dict[str, np.ndarray] = {}
    tracked: dict[str, bool] = {}
    for side in SIDES:
        r = result.hands.get(side)
        if r is not None and r.solve is not None:
            cam_anchor = np.asarray(r.solve.camera_from_anchor, dtype=np.float64)
            device_poses[side] = cam_anchor.astype(np.float32)
            poses[side] = pose_mul(world_from_camera, cam_anchor).astype(np.float32)
            tracked[side] = True
        else:
            device_poses[side] = IDENTITY_POSE7.astype(np.float32).copy()
            poses[side] = IDENTITY_POSE7.astype(np.float32).copy()
            tracked[side] = False
    left_tcp, right_tcp = apply_tcp_calibration_pose7(poses["left"], poses["right"], calibration)
    return AprilTagPairSample(
        device="apriltag",
        left_controller_pose=poses["left"],
        right_controller_pose=poses["right"],
        left_tcp_pose=left_tcp,
        right_tcp_pose=right_tcp,
        left_tracked=bool(streaming and frame_valid and tracked["left"]),
        right_tracked=bool(streaming and frame_valid and tracked["right"]),
        left_device_tracked=tracked["left"],
        right_device_tracked=tracked["right"],
        left_pose_valid=bool(frame_valid and tracked["left"]),
        right_pose_valid=bool(frame_valid and tracked["right"]),
        hmd_pose=world_from_camera.astype(np.float32),  # head camera pose in world
        left_device_controller_pose=device_poses["left"],  # head-relative anchor pose
        right_device_controller_pose=device_poses["right"],
        device_hmd_pose=IDENTITY_POSE7.astype(np.float32).copy(),
        hmd_tracked=bool(streaming and world_visible),
        workspace_from_device_pose=world_from_camera.astype(np.float32),
        device_time_ns=int(capture_time_ns),
        pc_monotonic_ns=int(capture_time_ns),
        aligned_time_ns=int(capture_time_ns),
        clock_offset_ns=0,
        clock_synced=True,  # camera timestamps already live on the PC monotonic clock
        connected=True,
        streaming=streaming,
        sequence=int(sequence),
        apriltag=apriltag_frame(result, capture_time_ns=capture_time_ns, sequence=sequence) if include_raw else None,
    )


# ---------------------------------------------------------------------------
# Live provider
# ---------------------------------------------------------------------------


class AprilTagTrackingProvider:
    """Head camera -> per-frame world localisation + bundle poses (TrackingProvider)."""

    device = "apriltag"

    def __init__(
        self,
        *,
        camera: CameraDevice,
        intrinsics: CameraIntrinsics,
        config: BundleConfig,
        calibration: ControllerTcpCalibration,
        world_map: WorldMap | None = None,
        fixed_world_from_camera: np.ndarray | None = None,
        require_world: bool = True,
        stale_timeout_s: float = DEFAULT_STALE_TIMEOUT_S,
        buffer_frames: int = 90,
        owns_camera: bool = True,
        rectify_balance: float = 0.5,
    ) -> None:
        self.camera = camera
        self.intrinsics = intrinsics
        self.geometry = CameraGeometry.for_intrinsics(intrinsics, balance=rectify_balance)
        self.config = config
        self.world_map = world_map
        self.calibration = calibration
        self.detector = AprilTagDetector(config.dictionary)
        self.require_world = require_world
        self.stale_timeout_ns = int(stale_timeout_s * 1e9)
        self.owns_camera = owns_camera
        self.workspace_locked = fixed_world_from_camera is not None
        self._fixed_world_from_camera = None if fixed_world_from_camera is None else np.asarray(fixed_world_from_camera, dtype=np.float64)
        self._samples: deque[AprilTagPairSample] = deque(maxlen=buffer_frames)
        self._latest_image: np.ndarray | None = None
        self._latest_result: FrameResult | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames_processed = 0
        self.last_error: str | None = None

    # -- TrackingProvider ----------------------------------------------------

    def start(self) -> None:
        if self.owns_camera:
            self.camera.connect()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="apriltag-tracker", daemon=True)
        self._thread.start()
        log.info(
            "AprilTag tracker started (dict=%s, hands=%s, world tags=%s, rectify=%s)",
            self.config.dictionary,
            list(self.config.bundles),
            list(self.world_map.tag_ids) if self.world_map else None,
            self.geometry.rectify,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.owns_camera:
            try:
                self.camera.disconnect()
            except Exception:  # noqa: BLE001
                pass

    def latest(self) -> ControllerPairSample:
        return self.sample_at(None)

    def sample_at(self, target_time_ns: int | None) -> ControllerPairSample:
        with self._lock:
            samples = tuple(self._samples)
        if not samples:
            return ControllerPairSample.empty(self.device)
        sample = samples[-1] if target_time_ns is None else min(samples, key=lambda s: abs(s.pc_monotonic_ns - int(target_time_ns)))
        if time.monotonic_ns() - samples[-1].pc_monotonic_ns > self.stale_timeout_ns:
            # Camera stalled: keep the cached pose but never advertise it as tracked.
            return replace(sample, streaming=False, left_tracked=False, right_tracked=False, hmd_tracked=False)
        return sample

    # -- optional static-camera fallback (upstream session contract) -------

    def set_workspace_from_device_pose(self, pose7: np.ndarray, *, locked: bool = True) -> None:
        self._fixed_world_from_camera = np.asarray(pose7, dtype=np.float64).reshape(7)
        self.workspace_locked = bool(locked)
        log.info("AprilTag: fixed world_from_camera fallback set (%s).", "locked" if locked else "unlocked")

    def reset_workspace(self) -> None:
        return None

    def extra_features(self) -> dict[str, Any]:
        return apriltag_features()

    def latest_frame(self) -> tuple[np.ndarray | None, FrameResult | None]:
        with self._lock:
            return self._latest_image, self._latest_result

    # -- worker --------------------------------------------------------------

    def _run(self) -> None:
        last_seq = None
        while not self._stop.is_set():
            try:
                cam_sample: CameraSample = self.camera.sample_at(None)
            except Exception as exc:  # noqa: BLE001 - camera hiccups are health, not crashes
                self.last_error = str(exc)
                time.sleep(0.01)
                continue
            if cam_sample.sequence == last_seq:
                time.sleep(0.002)
                continue
            last_seq = cam_sample.sequence
            try:
                result = process_frame(
                    cam_sample.image, detector=self.detector, config=self.config, geometry=self.geometry, world_map=self.world_map
                )
                sample = build_pair_sample(
                    result,
                    calibration=self.calibration,
                    capture_time_ns=int(cam_sample.capture_time_ns),
                    sequence=int(cam_sample.sequence),
                    fixed_world_from_camera=self._fixed_world_from_camera,
                    require_world=self.require_world,
                )
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                log.warning("AprilTag frame failed: %s", exc)
                continue
            with self._lock:
                self._samples.append(sample)
                self._latest_image = self.geometry.prepare(cam_sample.image) if self.geometry.rectify else cam_sample.image
                self._latest_result = result
                self.frames_processed += 1
                self.last_error = None


# ---------------------------------------------------------------------------
# Rig wiring (used by `handumi record --device apriltag` and tracker eval)
# ---------------------------------------------------------------------------


def build_provider_from_rig(
    rig_config: Path,
    calibration: ControllerTcpCalibration,
    *,
    spatial_path: Path | None = None,
    bundles_path: Path | None = None,
    world_map_path: Path | None = None,
) -> AprilTagTrackingProvider:
    """Head camera from rig.yaml + cameras.head intrinsics + bundles.yaml + world_map.yaml."""
    from handumi.cameras.usb import make_camera_device
    from handumi.config import load_rig_config
    from handumi.scripts.setup.calibrate_head_camera import DEFAULT_SPATIAL, head_camera_spec, load_head_intrinsics

    rig = load_rig_config(rig_config)
    cfg = rig.get("apriltag") or {}
    spatial = spatial_path or Path(str(cfg.get("spatial", DEFAULT_SPATIAL)))
    bundles = bundles_path or Path(str(cfg.get("bundles", DEFAULT_BUNDLES_PATH)))
    world_path = world_map_path or Path(str(cfg.get("world_map", DEFAULT_WORLD_MAP_PATH)))
    intrinsics = load_head_intrinsics(spatial)
    config = BundleConfig.from_yaml(bundles)
    world_map = WorldMap.from_yaml(world_path) if world_path.exists() else None
    if world_map is None:
        log.warning("No world map at %s: poses will be in the head-camera frame (require_world=False).", world_path)
    camera = make_camera_device(head_camera_spec(rig))
    return AprilTagTrackingProvider(
        camera=camera,
        intrinsics=intrinsics,
        config=config,
        calibration=calibration,
        world_map=world_map,
        require_world=bool(cfg.get("require_world", world_map is not None)),
        stale_timeout_s=float(cfg.get("frame_stale_timeout_s", DEFAULT_STALE_TIMEOUT_S)),
        rectify_balance=float(cfg.get("rectify_balance", 0.5)),
    )


# ---------------------------------------------------------------------------
# Drawing helper for previews
# ---------------------------------------------------------------------------


def draw_results(image: np.ndarray, result: FrameResult, geometry: CameraGeometry | None = None) -> np.ndarray:
    out = np.ascontiguousarray(image.copy())
    colors = {"world": (60, 220, 60), "left": (255, 120, 40), "right": (40, 160, 255)}
    per_side: dict[str, SideTagResult | None] = {"world": result.world, **result.hands}
    y = 25
    for side, r in per_side.items():
        if r is None:
            continue
        color = colors.get(side, (0, 255, 0))
        for det in r.detections:
            cv2.polylines(out, [det.corners.astype(np.int32).reshape(-1, 1, 2)], True, color, 2)
            cv2.putText(out, str(det.id), tuple(det.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if r.solve is not None and geometry is not None:
            T = pose7_to_mat(r.solve.camera_from_anchor)
            rvec, _ = cv2.Rodrigues(T[:3, :3])
            cv2.drawFrameAxes(out, geometry.K, geometry.dist, rvec, T[:3, 3], 0.05 if side == "world" else 0.03, 2)
            cv2.putText(out, f"{side}: {len(r.solve.tag_ids)} tags {r.solve.reproj_px:.2f}px", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(out, f"{side}: --", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25
    return out


__all__ = [
    "AprilTagDetector",
    "AprilTagPairSample",
    "AprilTagTrackingProvider",
    "BundleConfig",
    "BundleSolve",
    "BundleSpec",
    "CORNER_SPACE_RAW",
    "CORNER_SPACE_RECTIFIED",
    "CameraGeometry",
    "DEFAULT_BUNDLES_PATH",
    "DEFAULT_WORLD_MAP_PATH",
    "FrameResult",
    "MAX_TAGS_PER_BUNDLE",
    "MAX_WORLD_TAGS",
    "SideTagResult",
    "TagDetection",
    "TagSpec",
    "WorldMap",
    "apriltag_features",
    "apriltag_frame",
    "build_pair_sample",
    "build_provider_from_rig",
    "calibrate_bundle_geometry",
    "calibrate_world_map",
    "draw_results",
    "pose7_from_any",
    "process_frame",
    "solve_bundle",
    "solve_single_tag",
    "tag_corners_local",
]
