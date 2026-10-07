"""Offline MediaPipe Hand Landmarker over a recorded video -> hands/hand_pose.parquet.

All 21 landmarks per hand are stored (2D pixels + MediaPipe 3D world landmarks), plus
handedness and confidence. Never run during recording.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ego_collector.io.parquet import read_parquet, write_parquet
from ego_collector.recording.episode import EpisodePaths
from ego_collector.tracking.process import iter_video_frames

log = logging.getLogger("ego_collector.hands")

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEFAULT_MODEL = Path("models/hand_landmarker.task")
N_LANDMARKS = 21


def ensure_model(path: Path = DEFAULT_MODEL) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading hand landmarker model -> %s", path)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


@dataclass
class HandResult:
    visible: bool
    confidence: float
    landmarks_2d: np.ndarray  # (21,2) pixels (NaN if not visible)
    landmarks_3d: np.ndarray  # (21,3) MediaPipe world landmarks, metres-ish (aux only)


@dataclass
class RawHandDetection:
    """One detection, before any left/right bookkeeping. `label` is MediaPipe's raw (selfie-mirrored) handedness."""
    label: str
    label_confidence: float
    landmarks_2d: np.ndarray  # (21,2) pixels
    landmarks_3d: np.ndarray  # (21,3) MediaPipe world landmarks (hand-centred, ~metres for an average hand)


def _empty() -> HandResult:
    return HandResult(False, 0.0, np.full((N_LANDMARKS, 2), np.nan), np.full((N_LANDMARKS, 3), np.nan))


class HandLandmarker:
    """Thin wrapper around mediapipe.tasks HandLandmarker in VIDEO mode."""

    def __init__(self, model_path: Path = DEFAULT_MODEL, *, num_hands: int = 2, min_detection: float = 0.5, min_presence: float = 0.5, min_tracking: float = 0.5, delegate: str = "auto", recycle_every: int = 0) -> None:
        """``recycle_every``: close and rebuild the underlying landmarker every N frames. 0 (default) = never.

        Why it exists (measured on this Mac, mediapipe 1.0.1, 2026-09-11): the macOS **GPU** graph leaks about
        3.4 MB per frame — two 848x480 RGBA buffers — in ``ImageCloneCalculator`` -> ``ConvertToGpu`` ->
        ``CreateCVPixelBufferWithoutPool``. RSS reaches 13.8 GB by frame 4000 and the process then aborts with
        CoreVideo ``kCVReturnAllocationFailed (-6662)`` around frame 7500. The CPU delegate is not an escape on
        macOS with 1.0.x (its graph aborts outright, see below), so a long-running caller has to bound the leak
        by recreating the landmarker. ``close()`` does release the buffers.

        Cost: a rebuild reloads the model, ~350-440 ms, and resets MediaPipe's internal frame-to-frame tracking, so
        the frame after a rebuild is a fresh detection. Use it for long-lived interactive tools; for a bounded batch
        (a 60 s take is 1800 frames, ~6 GB) leave it off so no periodic re-detection lands in the measurements."""
        import sys

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.recycle_every = int(recycle_every)
        self.recycles = 0
        self._frames = 0
        # mediapipe 1.0.x on macOS: the CPU graph aborts ("Service is unavailable") and the GPU graph
        # only accepts SRGBA frames; Linux CPU + SRGB is fine. (mediapipe 0.10.21 CPU also works on macOS.)
        if delegate == "auto":
            delegate = "gpu" if sys.platform == "darwin" else "cpu"
        self.delegate = delegate
        self._rgba = delegate == "gpu"
        self._opts = dict(model_path=str(ensure_model(model_path)), num_hands=num_hands, min_detection=min_detection,
                          min_presence=min_presence, min_tracking=min_tracking, delegate=delegate)
        self._lm = self._create()

    def _create(self):
        from mediapipe.tasks.python import BaseOptions, vision
        o = self._opts
        base = BaseOptions(model_asset_path=o["model_path"],
                           delegate=BaseOptions.Delegate.GPU if o["delegate"] == "gpu" else BaseOptions.Delegate.CPU)
        return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=o["num_hands"],
            min_hand_detection_confidence=o["min_detection"],
            min_hand_presence_confidence=o["min_presence"],
            min_tracking_confidence=o["min_tracking"],
        ))

    def due_for_recycle(self) -> bool:
        """True when the next frame should be preceded by a rebuild (frames counted since construction)."""
        return bool(self.recycle_every) and self._frames > 0 and self._frames % self.recycle_every == 0

    def recycle(self) -> None:
        self._lm.close()
        self._lm = self._create()          # a fresh instance restarts its own timestamp clock at the next frame
        self.recycles += 1

    def detect_all(self, image_bgr: np.ndarray, timestamp_ms: int) -> list["RawHandDetection"]:
        """Every hand MediaPipe found this frame, with its raw handedness label + score, in detection order."""
        import cv2

        if self.due_for_recycle(): self.recycle()
        self._frames += 1
        h, w = image_bgr.shape[:2]
        if self._rgba:
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGBA, data=np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)))
        else:
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
        res = self._lm.detect_for_video(mp_image, int(timestamp_ms))
        dets = []
        for handed, lms, wlms in zip(res.handedness, res.hand_landmarks, res.hand_world_landmarks):
            cat = handed[0]
            dets.append(RawHandDetection(
                cat.category_name.lower(),  # "Left"/"Right" are the *person's* hands, assuming a SELFIE-MIRRORED image
                float(cat.score),
                np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float64),
                np.array([[p.x, p.y, p.z] for p in wlms], dtype=np.float64),
            ))
        return dets

    def detect(self, image_bgr: np.ndarray, timestamp_ms: int) -> dict[str, HandResult]:
        """Best detection per side. Loses handedness ambiguity on purpose — for temporal identity use detect_all()."""
        out = {"left": _empty(), "right": _empty()}
        for d in self.detect_all(image_bgr, timestamp_ms):
            if d.label not in out or (out[d.label].visible and d.label_confidence <= out[d.label].confidence):
                continue
            out[d.label] = HandResult(True, d.label_confidence, d.landmarks_2d, d.landmarks_3d)
        return out

    def close(self) -> None:
        self._lm.close()


def process_hands(paths: EpisodePaths, *, model_path: Path = DEFAULT_MODEL, mirror_handedness: bool = False, progress: bool = True, landmarker: HandLandmarker | None = None, delegate: str = "auto") -> dict[str, float]:
    """Run the landmarker over every frame and write hands/hand_pose.parquet.

    ``mirror_handedness`` swaps left/right (use if the camera image is mirrored)."""
    ts_table = read_parquet(paths.timestamps) if paths.timestamps.exists() else None
    timestamps = ts_table["timestamp_ns"].to_numpy() if ts_table is not None else None
    lm = landmarker or HandLandmarker(model_path, delegate=delegate)
    rows: dict[str, list] = {"frame_index": [], "timestamp_ns": []}
    for side in ("left", "right"):
        for k in ("hand_visible", "confidence", "landmarks_2d", "landmarks_3d"):
            rows[f"{side}_{k}"] = []
    frames = iter_video_frames(paths.video, timestamps)
    if progress:
        frames = tqdm(frames, total=None if timestamps is None else len(timestamps), desc=f"hands {paths.root.name}", unit="f")
    t0 = None
    for idx, ts, img in frames:
        if t0 is None:
            t0 = ts if ts > 0 else 0
        ts_ms = int((ts - t0) / 1e6) if ts > 0 else int(idx * 1000 / 30)
        res = lm.detect(img, ts_ms)
        if mirror_handedness:
            res = {"left": res["right"], "right": res["left"]}
        rows["frame_index"].append(idx)
        rows["timestamp_ns"].append(ts)
        for side in ("left", "right"):
            r = res[side]
            rows[f"{side}_hand_visible"].append(r.visible)
            rows[f"{side}_confidence"].append(r.confidence)
            rows[f"{side}_landmarks_2d"].append(r.landmarks_2d.reshape(-1))
            rows[f"{side}_landmarks_3d"].append(r.landmarks_3d.reshape(-1))
    if landmarker is None:
        lm.close()
    write_parquet(paths.hand_pose, {k: (np.asarray(v) if k in ("frame_index", "timestamp_ns") else v) for k, v in rows.items()})
    n = len(rows["frame_index"])
    summary = {
        "frames": n,
        "delegate": lm.delegate,
        "left_hand_visible_fraction": float(np.mean(rows["left_hand_visible"])) if n else 0.0,
        "right_hand_visible_fraction": float(np.mean(rows["right_hand_visible"])) if n else 0.0,
    }
    paths.update_metadata(hands=summary)
    log.info("%s: %s", paths.root.name, summary)
    return summary


__all__ = ["DEFAULT_MODEL", "HandLandmarker", "HandResult", "N_LANDMARKS", "ensure_model", "process_hands"]
