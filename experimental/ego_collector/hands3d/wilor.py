"""Offline 3D hand pose from WiLoR (wilor-mini) -> per-frame MANO track + downstream-compatible parquets.

Post-processing stage of the raw-ego-first pipeline (AprilTags shelved): the head video is
re-processed offline with a heavy model; recording only ever stores raw RGB.

Outputs per episode:
  hands3d/wilor_pose.parquet      raw track: per side visible/conf/bbox, wrist pose7 (CAMERA frame),
                                  21 keypoints 3D (cam) + 2D (px), thumb/index tips, aperture_m
  tracking/camera_pose.parquet    stub (world_pose_valid=False): monocular, no world frame
  tracking/wrist_pose.parquet     {side}_wrist_cam_* pose7 + {side}_tag_visible  -> generate-actions --frame head
  tracking/finger_pose.parquet    fingertip cam points + {side}_aperture_m       -> --grasp-source tags, QA, report

Conventions: camera frame +Z forward, +Y down (OpenCV). Wrist = MANO joint 0 translated by
pred_cam_t_full; wrist rotation = MANO global_orient. Monocular translation/scale is approximate:
prefer the relative deltas (dx..drz) downstream; aperture (a 3D distance) is scale-stable.
MANO keypoints: 0 wrist, 4 thumb tip, 8 index tip.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ego_collector.io.parquet import write_parquet
from ego_collector.recording.episode import EpisodePaths
from ego_collector.tracking.process import POSE_COLS, iter_video_frames
from ego_collector.tracking.transforms import T_to_pose7, make_T

log = logging.getLogger("ego_collector.hands3d")
SIDES = ("left", "right")
WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8


@dataclass
class HandFrame:
    visible: bool = False
    confidence: float = float("nan")
    bbox: np.ndarray | None = None  # (4,) xyxy
    pose7_cam: np.ndarray | None = None  # wrist pose in the camera frame
    kp3d: np.ndarray | None = None  # (21, 3) camera frame
    kp2d: np.ndarray | None = None  # (21, 2) pixels


def _first(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[0] if a.ndim > 2 or (a.ndim == 2 and a.shape[0] == 1 and a.shape[1] not in (2, 3)) else a


def parse_prediction(pred: dict) -> HandFrame:
    wp = pred["wilor_preds"]
    kp3d = np.asarray(wp["pred_keypoints_3d"], dtype=np.float64).reshape(-1, 21, 3)[0]
    cam_t = np.asarray(wp["pred_cam_t_full"], dtype=np.float64).reshape(-1, 3)[0]
    kp3d = kp3d + cam_t  # camera frame
    kp2d = np.asarray(wp["pred_keypoints_2d"], dtype=np.float64).reshape(-1, 21, 2)[0]
    go = np.asarray(wp["global_orient"], dtype=np.float64)
    if go.size == 3:  # axis-angle (wilor-mini)
        from scipy.spatial.transform import Rotation

        R = Rotation.from_rotvec(go.reshape(3)).as_matrix()
    else:  # rotation matrix; re-orthonormalize (the left-hand mirror flip breaks the det sign)
        R = go.reshape(-1, 3, 3)[0]
        u, _, vt = np.linalg.svd(R)
        R = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    conf = float(np.asarray(pred.get("detection_conf", np.nan)))
    return HandFrame(
        visible=True,
        confidence=conf,
        bbox=np.asarray(pred["hand_bbox"], dtype=np.float64).reshape(4),
        pose7_cam=T_to_pose7(make_T(R, kp3d[WRIST])),
        kp3d=kp3d,
        kp2d=kp2d,
    )


class WilorTracker:
    def __init__(self, *, device: str = "auto", hand_conf: float = 0.3, verbose: bool = False, focal_px: float | None = None, image_size: tuple[int, int] = (1920, 1080)) -> None:
        """focal_px: REAL camera focal length in pixels (e.g. ~1400 for a C922 at 1080p).
        wilor-mini defaults to an internal focal of 5000 (in 256-px crop units), which inflates
        the camera-frame depth ~27x at 1080p - always pass the real focal for metric wrist depth."""
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        log.info("WiLoR on %s (%s)", device, dtype)
        self.device = device
        self.hand_conf = hand_conf
        kwargs = {}
        if focal_px is not None:
            kwargs["focal_length"] = float(focal_px) * 256.0 / max(image_size)  # wilor-mini expects 256-crop units
        self.pipe = WiLorHandPose3dEstimationPipeline(device=torch.device(device), dtype=dtype, verbose=verbose, **kwargs)
        if device == "mps":
            # MPS conv2d rejects non-contiguous inputs (wilor's ViT patch_embed feeds one):
            # force-contiguous every conv input.
            import torch.nn as nn

            def _contig(_m, inputs):
                return tuple(x.contiguous() if isinstance(x, torch.Tensor) else x for x in inputs)

            for m in self.pipe.wilor_model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    m.register_forward_pre_hook(_contig)

    def __call__(self, bgr: np.ndarray) -> dict[str, HandFrame]:
        import cv2

        preds = self.pipe.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), hand_conf=self.hand_conf)
        out = {s: HandFrame() for s in SIDES}
        # keep the highest-confidence detection per side (duplicates happen)
        for p in preds:
            side = "right" if int(round(float(p["is_right"]))) == 1 else "left"
            hf = parse_prediction(p)
            cur = out[side]
            if not cur.visible or (np.nan_to_num(hf.confidence) >= np.nan_to_num(cur.confidence)):
                out[side] = hf
        return out


def _pose_cols(prefix: str, p7: np.ndarray | None) -> dict[str, float]:
    v = p7 if p7 is not None else np.full(7, np.nan)
    return {f"{prefix}_{c}": float(x) for c, x in zip(POSE_COLS, v)}


def _xyz(prefix: str, p: np.ndarray | None) -> dict[str, float]:
    v = p if p is not None else np.full(3, np.nan)
    return {f"{prefix}_{a}": float(x) for a, x in zip("xyz", v)}


def process_hands3d(paths: EpisodePaths, tracker, *, stride: int = 1, progress: bool = True, overlay: Path | None = None) -> dict:
    """Run the 3D hand tracker over the episode video and write all output parquets.

    stride > 1 subsamples frames (quality preview); skipped frames stay invisible/NaN
    so timestamps remain aligned 1:1 with the video.
    """
    import cv2
    from tqdm import tqdm

    from ego_collector.io.parquet import read_parquet

    ts_table = read_parquet(paths.timestamps) if paths.timestamps.exists() else None
    timestamps = ts_table["timestamp_ns"].to_numpy() if ts_table is not None else None
    n_expected = len(timestamps) if timestamps is not None else None
    raw_rows, cam_rows, wrist_rows, finger_rows = [], [], [], []
    writer = None
    frames = iter_video_frames(paths.video, timestamps)
    if progress:
        frames = tqdm(frames, total=n_expected, desc=f"wilor {paths.root.name}", unit="f")
    for idx, ts, img in frames:
        hands = tracker(img) if idx % stride == 0 else {s: HandFrame() for s in SIDES}
        raw = {"frame_index": idx, "timestamp_ns": ts, "processed": idx % stride == 0}
        wrow: dict = {"frame_index": idx, "timestamp_ns": ts, "world_pose_valid": False}
        frow: dict = {"frame_index": idx, "timestamp_ns": ts, "world_pose_valid": False}
        for side in SIDES:
            h = hands[side]
            raw[f"{side}_visible"] = h.visible
            raw[f"{side}_confidence"] = h.confidence
            raw[f"{side}_bbox"] = (h.bbox if h.bbox is not None else np.full(4, np.nan)).astype(np.float64)
            raw.update(_pose_cols(f"{side}_wrist_cam", h.pose7_cam))
            raw[f"{side}_keypoints_3d"] = (h.kp3d if h.kp3d is not None else np.full((21, 3), np.nan)).reshape(-1)
            raw[f"{side}_keypoints_2d"] = (h.kp2d if h.kp2d is not None else np.full((21, 2), np.nan)).reshape(-1)
            thumb = h.kp3d[THUMB_TIP] if h.kp3d is not None else None
            index = h.kp3d[INDEX_TIP] if h.kp3d is not None else None
            aperture = float(np.linalg.norm(thumb - index)) if h.kp3d is not None else float("nan")
            raw[f"{side}_aperture_m"] = aperture
            # tracking/wrist_pose.parquet schema (head-frame path: world columns stay NaN)
            wrow[f"{side}_tag_visible"] = h.visible
            wrow[f"{side}_wrist_valid"] = False
            wrow.update(_pose_cols(f"{side}_wrist", None))
            wrow.update(_pose_cols(f"{side}_wrist_cam", h.pose7_cam))
            wrow[f"{side}_tag_reprojection_error"] = float("nan")
            wrow[f"{side}_tag_decision_margin"] = h.confidence
            # tracking/finger_pose.parquet schema
            for name, p in (("thumb", thumb), ("index", index)):
                frow[f"{side}_{name}_visible"] = h.visible
                frow.update(_xyz(f"{side}_{name}_cam", p))
                frow.update(_xyz(f"{side}_{name}", None))
                frow[f"{side}_{name}_reprojection_error"] = float("nan")
            frow[f"{side}_aperture_valid"] = h.visible
            frow[f"{side}_aperture_m"] = aperture
            mid = (thumb + index) / 2 if h.kp3d is not None else None
            frow.update(_xyz(f"{side}_pinch_cam", mid))
            frow.update(_xyz(f"{side}_pinch", None))
        cam_rows.append(
            {"frame_index": idx, "timestamp_ns": ts, "world_pose_valid": False, **_pose_cols("head", None),
             "world_tag_count": 0, "world_corner_count": 0, "world_inlier_count": 0,
             "world_reprojection_error": float("nan"), "world_tag_ids": [], "visible_world_tags": 0}
        )
        raw_rows.append(raw)
        wrist_rows.append(wrow)
        finger_rows.append(frow)
        if overlay is not None:
            shown = draw_overlay(img, hands)
            if writer is None:
                overlay.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), 30, (shown.shape[1], shown.shape[0]))
            writer.write(shown)
    if writer is not None:
        writer.release()
    for path, rows in ((paths.hands3d_pose, raw_rows), (paths.camera_pose, cam_rows), (paths.wrist_pose, wrist_rows), (paths.finger_pose, finger_rows)):
        cols = {k: [r.get(k) for r in rows] for k in rows[0]} if rows else {}
        write_parquet(path, cols)
    n = len(raw_rows)
    summary = {
        "frames": n,
        "source": "wilor-mini",
        "stride": stride,
        "world_pose_valid_fraction": 0.0,
        "world_tags": False,
        "fingers": True,
        "left_wrist_valid_fraction": 0.0,
        "right_wrist_valid_fraction": 0.0,
    }
    for side in SIDES:
        vis = np.array([r[f"{side}_visible"] for r in raw_rows], dtype=bool) if n else np.zeros(0, bool)
        summary[f"{side}_hand_visible_fraction"] = float(vis.mean()) if n else 0.0
        summary[f"{side}_aperture_valid_fraction"] = summary[f"{side}_hand_visible_fraction"]
        for fg in ("thumb", "index"):
            summary[f"{side}_{fg}_visible_fraction"] = summary[f"{side}_hand_visible_fraction"]
    paths.update_metadata(tracking=summary, hands3d={"model": "wilor-mini", "stride": stride})
    log.info("%s: %s", paths.root.name, {k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items() if "visible" in k or k == "frames"})
    return summary


BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
COLORS = {"left": (255, 140, 40), "right": (40, 160, 255)}


def draw_overlay(img: np.ndarray, hands: dict[str, HandFrame]) -> np.ndarray:
    import cv2

    out = img.copy()
    y = 60
    for side in SIDES:
        h = hands[side]
        col = COLORS[side]
        if not h.visible:
            continue
        pts = h.kp2d.astype(int)
        for a, b in BONES:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), col, 2)
        for p in pts:
            cv2.circle(out, tuple(p), 3, col, -1)
        cv2.circle(out, tuple(pts[WRIST]), 8, (255, 255, 255), 2)
        cv2.line(out, tuple(pts[THUMB_TIP]), tuple(pts[INDEX_TIP]), (255, 255, 255), 2)
        ap = float(np.linalg.norm(h.kp3d[THUMB_TIP] - h.kp3d[INDEX_TIP])) * 1000
        w = h.pose7_cam
        cv2.putText(out, f"{side}: grip {ap:5.1f}mm  wrist [{w[0]:+.3f} {w[1]:+.3f} {w[2]:+.3f}]m conf {h.confidence:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        y += 30
    return out


__all__ = ["HandFrame", "WilorTracker", "draw_overlay", "parse_prediction", "process_hands3d"]
