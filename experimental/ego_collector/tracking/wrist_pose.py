"""Wrist tags -> world-frame wrist poses.

    T_world_wrist = T_world_camera @ T_camera_tag @ T_tag_wrist

``T_tag_wrist`` comes from configs/wrist_extrinsics.yaml (identity in V0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from ego_collector.camera.intrinsics import CameraIntrinsics
from ego_collector.tracking.detector import TagDetection
from ego_collector.tracking.pnp import solve_square_tag
from ego_collector.tracking.transforms import T_from_xyz_rpy

SIDES = ("left", "right")
FINGERS = ("thumb", "index")
_DEFAULT_FINGER_IDS = {("left", "thumb"): 11, ("left", "index"): 12, ("right", "thumb"): 21, ("right", "index"): 22}


@dataclass(frozen=True)
class WristTagSpec:
    side: str
    tag_id: int
    tag_size_m: float
    T_tag_wrist: np.ndarray


@dataclass(frozen=True)
class FingerTagSpec:
    """Dorsal (fingernail-side) tag on the thumb or index finger.

    ``T_tag_tip`` maps the tag frame to the fingertip *contact point*: the tag sits on the
    nail, recessed from the tip, so the pad contact is a rigid offset measured once.
    """

    side: str
    finger: str  # "thumb" | "index"
    tag_id: int
    tag_size_m: float
    T_tag_tip: np.ndarray


@dataclass
class WristExtrinsics:
    tag_family: str
    sides: dict[str, WristTagSpec]
    fingers: dict[str, dict[str, FingerTagSpec]] = None  # side -> finger -> spec ({} when no finger tags)

    def __post_init__(self) -> None:
        if self.fingers is None:
            self.fingers = {}

    @classmethod
    def from_yaml(cls, path: Path) -> "WristExtrinsics":
        d = yaml.safe_load(Path(path).read_text()) or {}
        sides = {}
        fingers: dict[str, dict[str, FingerTagSpec]] = {}
        for side in SIDES:
            s = d.get(side) or {}
            ext = s.get("T_tag_wrist") or {}
            sides[side] = WristTagSpec(
                side=side,
                tag_id=int(s.get("tag_id", 10 if side == "left" else 20)),
                tag_size_m=float(s.get("tag_size_m", 0.05)),
                T_tag_wrist=T_from_xyz_rpy(ext.get("xyz", [0, 0, 0]), ext.get("rpy", [0, 0, 0])),
            )
            for finger in FINGERS:
                f = s.get(finger)
                if not f:
                    continue
                tip = f.get("T_tag_tip") or {}
                fingers.setdefault(side, {})[finger] = FingerTagSpec(
                    side=side,
                    finger=finger,
                    tag_id=int(f.get("tag_id", _DEFAULT_FINGER_IDS[(side, finger)])),
                    tag_size_m=float(f.get("tag_size_m", 0.016)),
                    T_tag_tip=T_from_xyz_rpy(tip.get("xyz", [0, 0, 0]), tip.get("rpy", [0, 0, 0])),
                )
        return cls(tag_family=str(d.get("tag_family", "tagStandard41h12")), sides=sides, fingers=fingers)

    @classmethod
    def default(cls, family: str = "tag36h11", *, with_fingers: bool = False, finger_size_m: float = 0.016) -> "WristExtrinsics":
        wr = cls(tag_family=family, sides={"left": WristTagSpec("left", 10, 0.05, np.eye(4)), "right": WristTagSpec("right", 20, 0.05, np.eye(4))})
        if with_fingers:
            wr.fingers = {side: {finger: FingerTagSpec(side, finger, _DEFAULT_FINGER_IDS[(side, finger)], finger_size_m, np.eye(4)) for finger in FINGERS} for side in SIDES}
        return wr

    @property
    def has_fingers(self) -> bool:
        return any(self.fingers.values()) if self.fingers else False

    def side_of(self, tag_id: int) -> str | None:
        return next((s for s, spec in self.sides.items() if spec.tag_id == tag_id), None)

    def finger_of(self, tag_id: int) -> tuple[str, str] | None:
        for side, d in (self.fingers or {}).items():
            for finger, spec in d.items():
                if spec.tag_id == tag_id:
                    return side, finger
        return None


@dataclass(frozen=True)
class WristSolve:
    side: str
    T_camera_tag: np.ndarray
    T_world_wrist: np.ndarray | None  # None when the camera is not localised
    T_camera_wrist: np.ndarray
    reprojection_error_px: float
    decision_margin: float


def solve_wrists(
    detections: Sequence[TagDetection],
    extrinsics: WristExtrinsics,
    intr: CameraIntrinsics,
    T_world_camera: np.ndarray | None,
    *,
    max_reproj_px: float = 3.0,
) -> dict[str, WristSolve]:
    out: dict[str, WristSolve] = {}
    for side, spec in extrinsics.sides.items():
        det = next((d for d in detections if d.tag_id == spec.tag_id), None)
        if det is None:
            continue
        pnp = solve_square_tag(det.corners, spec.tag_size_m, intr)
        if pnp is None or pnp.reprojection_error_px > max_reproj_px:
            continue
        T_cam_wrist = pnp.T_camera_object @ spec.T_tag_wrist
        T_world_wrist = None if T_world_camera is None else T_world_camera @ T_cam_wrist
        out[side] = WristSolve(side, pnp.T_camera_object, T_world_wrist, T_cam_wrist, pnp.reprojection_error_px, det.decision_margin)
    return out


@dataclass(frozen=True)
class FingerSolve:
    side: str
    finger: str
    p_camera: np.ndarray  # (3,) fingertip contact point in the camera frame
    p_world: np.ndarray | None  # None when the camera is not localised
    T_camera_tag: np.ndarray
    reprojection_error_px: float
    decision_margin: float


def solve_fingers(
    detections: Sequence[TagDetection],
    extrinsics: WristExtrinsics,
    intr: CameraIntrinsics,
    T_world_camera: np.ndarray | None,
    *,
    max_reproj_px: float = 3.0,
) -> dict[tuple[str, str], FingerSolve]:
    """Fingertip contact points from the thumb/index dorsal tags.

    Only the tip *position* is consumed downstream (aperture, pinch centre): IPPE's
    planar pose ambiguity flips the small tag's normal, not its translation, so tiny
    fingernail tags still give a stable point.
    """
    out: dict[tuple[str, str], FingerSolve] = {}
    for side, d in (extrinsics.fingers or {}).items():
        for finger, spec in d.items():
            det = next((x for x in detections if x.tag_id == spec.tag_id), None)
            if det is None:
                continue
            pnp = solve_square_tag(det.corners, spec.tag_size_m, intr)
            if pnp is None or pnp.reprojection_error_px > max_reproj_px:
                continue
            T_cam_tip = pnp.T_camera_object @ spec.T_tag_tip
            p_cam = T_cam_tip[:3, 3].copy()
            p_world = None if T_world_camera is None else (T_world_camera @ T_cam_tip)[:3, 3].copy()
            out[(side, finger)] = FingerSolve(side, finger, p_cam, p_world, pnp.T_camera_object, pnp.reprojection_error_px, det.decision_margin)
    return out


__all__ = ["FINGERS", "SIDES", "FingerSolve", "FingerTagSpec", "WristExtrinsics", "WristSolve", "WristTagSpec", "solve_fingers", "solve_wrists"]
