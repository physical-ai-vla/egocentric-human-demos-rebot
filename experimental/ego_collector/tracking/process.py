"""Offline STEP 2-4: video -> apriltags.parquet, camera_pose.parquet, wrist_pose.parquet.

Missing data stays missing (NaN / valid=False); no interpolation here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from tqdm import tqdm

from ego_collector.camera.intrinsics import CameraIntrinsics
from ego_collector.io.parquet import read_parquet, write_parquet
from ego_collector.recording.episode import EpisodePaths
from ego_collector.tracking.detector import TagDetection, TagDetector, make_detector
from ego_collector.tracking.transforms import IDENTITY_POSE7, T_to_pose7
from ego_collector.tracking.world_pose import WorldTagMap, solve_camera_pose
from ego_collector.tracking.wrist_pose import FINGERS, SIDES, WristExtrinsics, solve_fingers, solve_wrists

log = logging.getLogger("ego_collector.process_tags")
POSE_COLS = ("x", "y", "z", "qx", "qy", "qz", "qw")


@dataclass
class TagProcessingConfig:
    max_world_reproj_px: float = 3.0
    max_wrist_reproj_px: float = 3.0
    min_world_tags: int = 1
    backend: str = "auto"
    detector_kwargs: dict | None = None


def iter_video_frames(video: Path, timestamps: np.ndarray | None = None) -> Iterator[tuple[int, int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    i = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                return
            ts = int(timestamps[i]) if timestamps is not None and i < len(timestamps) else -1
            yield i, ts, img
            i += 1
    finally:
        cap.release()


def _pose_cols(prefix: str, T: np.ndarray | None) -> dict[str, float]:
    p = T_to_pose7(T) if T is not None else np.full(7, np.nan)
    return {f"{prefix}_{c}": float(v) for c, v in zip(POSE_COLS, p)}


def process_tags(
    paths: EpisodePaths,
    *,
    intr: CameraIntrinsics,
    world_map: WorldTagMap | None,
    wrists: WristExtrinsics,
    config: TagProcessingConfig = TagProcessingConfig(),
    detector: TagDetector | None = None,
    progress: bool = True,
) -> dict[str, float]:
    """Run detection + world/wrist PnP over every frame; write the three tracking parquets."""
    if world_map is not None and world_map.tag_family != wrists.tag_family:
        log.warning("world map family %s != wrist family %s", world_map.tag_family, wrists.tag_family)
    if world_map is None:
        log.info("no world map: head/camera frame only (wrist poses in T_camera_wrist, world_pose_valid=False)")
    det = detector or make_detector(wrists.tag_family, config.backend, **(config.detector_kwargs or {}))
    ts_table = read_parquet(paths.timestamps) if paths.timestamps.exists() else None
    timestamps = ts_table["timestamp_ns"].to_numpy() if ts_table is not None else None
    n_expected = len(timestamps) if timestamps is not None else None

    tag_rows: list[dict] = []
    cam_rows: list[dict] = []
    wrist_rows: list[dict] = []
    role_of = {tid: "world" for tid in (world_map.ids if world_map is not None else ())}
    role_of.update({spec.tag_id: side for side, spec in wrists.sides.items()})
    role_of.update({spec.tag_id: f"{side}_{finger}" for side, d in (wrists.fingers or {}).items() for finger, spec in d.items()})
    finger_rows: list[dict] = []

    frames = iter_video_frames(paths.video, timestamps)
    if progress:
        frames = tqdm(frames, total=n_expected, desc=f"tags {paths.root.name}", unit="f")
    for idx, ts, img in frames:
        dets = det.detect(img)
        for d in dets:
            tag_rows.append({"frame_index": idx, "timestamp_ns": ts, "role": role_of.get(d.tag_id, "unknown"), **d.as_row()})
        cam = solve_camera_pose(dets, world_map, intr, max_reproj_px=config.max_world_reproj_px, min_tags=config.min_world_tags) if world_map is not None else None
        T_wc = cam.T_world_camera if cam is not None else None
        cam_rows.append(
            {
                "frame_index": idx,
                "timestamp_ns": ts,
                "world_pose_valid": cam is not None,
                **_pose_cols("head", T_wc),
                "world_tag_count": cam.tag_count if cam else 0,
                "world_corner_count": cam.corner_count if cam else 0,
                "world_inlier_count": cam.inlier_count if cam else 0,
                "world_reprojection_error": cam.reprojection_error_px if cam else float("nan"),
                "world_tag_ids": list(cam.tag_ids) if cam else [],
                "visible_world_tags": sum(1 for d in dets if world_map is not None and d.tag_id in world_map.tags),
            }
        )
        solved = solve_wrists(dets, wrists, intr, T_wc, max_reproj_px=config.max_wrist_reproj_px)
        row: dict = {"frame_index": idx, "timestamp_ns": ts, "world_pose_valid": cam is not None}
        for side in SIDES:
            w = solved.get(side)
            row[f"{side}_tag_visible"] = w is not None
            row[f"{side}_wrist_valid"] = w is not None and w.T_world_wrist is not None
            row.update(_pose_cols(f"{side}_wrist", w.T_world_wrist if w else None))
            row.update(_pose_cols(f"{side}_wrist_cam", w.T_camera_wrist if w else None))
            row[f"{side}_tag_reprojection_error"] = w.reprojection_error_px if w else float("nan")
            row[f"{side}_tag_decision_margin"] = w.decision_margin if w else float("nan")
        wrist_rows.append(row)
        if wrists.has_fingers:
            fsolved = solve_fingers(dets, wrists, intr, T_wc, max_reproj_px=config.max_wrist_reproj_px)
            frow: dict = {"frame_index": idx, "timestamp_ns": ts, "world_pose_valid": cam is not None}
            for side in SIDES:
                tips = {}
                for finger in FINGERS:
                    f = fsolved.get((side, finger))
                    frow[f"{side}_{finger}_visible"] = f is not None
                    for j, ax in enumerate("xyz"):
                        frow[f"{side}_{finger}_cam_{ax}"] = float(f.p_camera[j]) if f else float("nan")
                        frow[f"{side}_{finger}_{ax}"] = float(f.p_world[j]) if f is not None and f.p_world is not None else float("nan")
                    frow[f"{side}_{finger}_reprojection_error"] = f.reprojection_error_px if f else float("nan")
                    if f is not None:
                        tips[finger] = f
                both = "thumb" in tips and "index" in tips
                frow[f"{side}_aperture_valid"] = both
                frow[f"{side}_aperture_m"] = float(np.linalg.norm(tips["thumb"].p_camera - tips["index"].p_camera)) if both else float("nan")
                mid_cam = (tips["thumb"].p_camera + tips["index"].p_camera) / 2 if both else None
                mid_world = (tips["thumb"].p_world + tips["index"].p_world) / 2 if both and tips["thumb"].p_world is not None else None
                for j, ax in enumerate("xyz"):
                    frow[f"{side}_pinch_cam_{ax}"] = float(mid_cam[j]) if mid_cam is not None else float("nan")
                    frow[f"{side}_pinch_{ax}"] = float(mid_world[j]) if mid_world is not None else float("nan")
            finger_rows.append(frow)

    n = len(cam_rows)
    if n_expected is not None and n != n_expected:
        log.warning("%s: %d video frames but %d timestamps", paths.root.name, n, n_expected)
    _write_rows(paths.apriltags, tag_rows, empty_schema={"frame_index": [], "timestamp_ns": [], "role": [], "tag_id": [], "corners": [], "center": [], "decision_margin": [], "hamming": []})
    _write_rows(paths.camera_pose, cam_rows)
    _write_rows(paths.wrist_pose, wrist_rows)
    if wrists.has_fingers:
        _write_rows(paths.finger_pose, finger_rows)
    valid = np.array([r["world_pose_valid"] for r in cam_rows], dtype=bool) if n else np.zeros(0, bool)
    summary = {
        "frames": n,
        "world_pose_valid_fraction": float(valid.mean()) if n else 0.0,
        "left_wrist_valid_fraction": float(np.mean([r["left_wrist_valid"] for r in wrist_rows])) if n else 0.0,
        "right_wrist_valid_fraction": float(np.mean([r["right_wrist_valid"] for r in wrist_rows])) if n else 0.0,
        "world_reprojection_error_median": float(np.nanmedian([r["world_reprojection_error"] for r in cam_rows])) if n and valid.any() else float("nan"),
        "detector_backend": det.backend,
        "tag_family": det.family,
        "world_tags": world_map is not None,
        "fingers": bool(wrists.has_fingers),
    }
    if wrists.has_fingers and n:
        for side in SIDES:
            summary[f"{side}_aperture_valid_fraction"] = float(np.mean([r[f"{side}_aperture_valid"] for r in finger_rows]))
            for finger in FINGERS:
                summary[f"{side}_{finger}_visible_fraction"] = float(np.mean([r[f"{side}_{finger}_visible"] for r in finger_rows]))
    paths.update_metadata(tracking=summary, tag_family=det.family)
    log.info("%s: %s", paths.root.name, summary)
    return summary


def _write_rows(path: Path, rows: list[dict], empty_schema: dict | None = None) -> None:
    if not rows:
        write_parquet(path, empty_schema or {"frame_index": np.zeros(0, dtype=np.int64)})
        return
    cols = {k: [r.get(k) for r in rows] for k in rows[0]}
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_parquet(path, index=False)


__all__ = ["POSE_COLS", "TagProcessingConfig", "iter_video_frames", "process_tags"]
