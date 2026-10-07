"""World tag map + head-camera localisation from all visible world-tag corners.

    T_camera_world  <- one PnP over every visible world corner (RANSAC + LM)
    T_world_camera   = inv(T_camera_world)

Never solve per tag and average translations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from ego_collector.camera.intrinsics import CameraIntrinsics
from ego_collector.tracking.detector import TagDetection
from ego_collector.tracking.pnp import PnPResult, solve_points, solve_square_tag, tag_corners_local
from ego_collector.tracking.transforms import T_from_xyz_rpy, invert_T, matrix_to_rpy


@dataclass
class WorldTagMap:
    tag_family: str
    tag_size_m: float
    tags: dict[int, np.ndarray]  # id -> T_world_tag (4x4)
    tag_sizes: dict[int, float] = field(default_factory=dict)  # per-tag override
    measured: bool = False
    source: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "WorldTagMap":
        d = yaml.safe_load(Path(path).read_text()) or {}
        tags: dict[int, np.ndarray] = {}
        sizes: dict[int, float] = {}
        for tid, spec in (d.get("tags") or {}).items():
            tags[int(tid)] = T_from_xyz_rpy(spec["xyz"], spec.get("rpy", [0, 0, 0]))
            if "size_m" in spec:
                sizes[int(tid)] = float(spec["size_m"])
        return cls(tag_family=str(d.get("tag_family", "tagStandard41h12")), tag_size_m=float(d.get("tag_size_m", 0.07)), tags=tags, tag_sizes=sizes, measured=bool(d.get("measured", False)), source=Path(path))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tag_family": self.tag_family, "tag_size_m": self.tag_size_m, "tags": {}, "measured": self.measured}
        for tid, T in sorted(self.tags.items()):
            entry = {"xyz": [float(v) for v in T[:3, 3]], "rpy": [float(v) for v in matrix_to_rpy(T[:3, :3])]}
            if tid in self.tag_sizes:
                entry["size_m"] = self.tag_sizes[tid]
            out["tags"][int(tid)] = entry
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def size_of(self, tag_id: int) -> float:
        return self.tag_sizes.get(int(tag_id), self.tag_size_m)

    def corners_world(self, tag_id: int) -> np.ndarray:
        T = self.tags[int(tag_id)]
        local = tag_corners_local(self.size_of(tag_id))
        return (T[:3, :3] @ local.T).T + T[:3, 3]

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.tags))


@dataclass(frozen=True)
class CameraPoseSolve:
    T_world_camera: np.ndarray
    T_camera_world: np.ndarray
    tag_ids: tuple[int, ...]
    corner_count: int
    inlier_count: int
    reprojection_error_px: float

    @property
    def tag_count(self) -> int:
        return len(self.tag_ids)


def solve_camera_pose(
    detections: Sequence[TagDetection],
    world_map: WorldTagMap,
    intr: CameraIntrinsics,
    *,
    max_reproj_px: float = 3.0,
    min_tags: int = 1,
    ransac_reproj_px: float = 4.0,
) -> CameraPoseSolve | None:
    world_dets = [d for d in detections if d.tag_id in world_map.tags]
    if len(world_dets) < min_tags or not world_dets:
        return None
    if len(world_dets) == 1:
        d = world_dets[0]
        single = solve_square_tag(d.corners, world_map.size_of(d.tag_id), intr)
        if single is None or single.reprojection_error_px > max_reproj_px:
            return None
        T_cam_tag = single.T_camera_object
        T_cam_world = T_cam_tag @ invert_T(world_map.tags[d.tag_id])
        result: PnPResult = PnPResult(T_cam_world, single.reprojection_error_px, 4, 4)
    else:
        obj = np.concatenate([world_map.corners_world(d.tag_id) for d in world_dets], axis=0)
        img = np.concatenate([d.corners for d in world_dets], axis=0)
        result_or_none = solve_points(obj, img, intr, ransac=True, ransac_reproj_px=ransac_reproj_px)
        if result_or_none is None or result_or_none.reprojection_error_px > max_reproj_px:
            return None
        result = result_or_none
    T_camera_world = result.T_camera_object
    return CameraPoseSolve(
        T_world_camera=invert_T(T_camera_world),
        T_camera_world=T_camera_world,
        tag_ids=tuple(sorted(d.tag_id for d in world_dets)),
        corner_count=result.num_points,
        inlier_count=result.inliers,
        reprojection_error_px=result.reprojection_error_px,
    )


__all__ = ["CameraPoseSolve", "WorldTagMap", "solve_camera_pose"]
