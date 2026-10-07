"""Live detection monitors for the recording preview (never on the capture/encode path).

``LiveTagMonitor`` and ``LiveHandMonitor`` each run on their own thread, pull the *latest*
frame from the capture, and publish the newest result + per-episode visibility counters.
They are throttled (default ~10 Hz / ~6 Hz) so a slow detector only lowers overlay refresh,
never the recorded frame rate. Recording works with both monitors off.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ego_collector.camera.capture import CameraCapture
from ego_collector.tracking.detector import TagDetection, TagDetector

log = logging.getLogger("ego_collector.live")


@dataclass
class VisibilityCounter:
    frames: int = 0
    left: int = 0
    right: int = 0
    world: int = 0
    all_required: int = 0

    def reset(self) -> None:
        self.frames = self.left = self.right = self.world = self.all_required = 0

    def add(self, *, left: bool, right: bool, world: bool, need_world: bool) -> None:
        self.frames += 1
        self.left += left
        self.right += right
        self.world += world
        self.all_required += left and right and (world or not need_world)

    def ratios(self) -> dict[str, float]:
        n = max(self.frames, 1)
        return {"left": self.left / n, "right": self.right / n, "world": self.world / n, "all_required": self.all_required / n, "frames": self.frames}


@dataclass
class TagMonitorResult:
    frame_index: int
    detections: list[TagDetection]
    roles: dict[int, str]
    process_ms: float
    left_visible: bool
    right_visible: bool
    world_visible: int  # number of world tags seen
    hands: dict[str, dict] = field(default_factory=dict)  # side -> {frame, xyz, rpy_deg, aperture_mm, thumb, index}


class LiveTagMonitor(threading.Thread):
    def __init__(
        self,
        cam: CameraCapture,
        detector: TagDetector,
        *,
        left_id: int,
        right_id: int,
        world_ids: tuple[int, ...] = (),
        wrist_size_m: float = 0.028,
        focal_px: float | None = None,
        rate_hz: float = 10.0,
        extrinsics=None,  # WristExtrinsics: enables live wrist pose + finger aperture (needs intr)
        intr=None,  # CameraIntrinsics
        world_map=None,  # WorldTagMap: live poses become world-frame when its tags are visible
    ) -> None:
        super().__init__(name="live-tags", daemon=True)
        self.cam = cam
        self.detector = detector
        self.left_id, self.right_id, self.world_ids = left_id, right_id, tuple(world_ids)
        self.wrist_size_m = wrist_size_m
        self.focal_px = focal_px
        self.period = 1.0 / rate_hz
        self.enabled = True
        self.counter = VisibilityCounter()
        self.counting = False
        self.extrinsics, self.intr, self.world_map = extrinsics, intr, world_map
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: TagMonitorResult | None = None
        self.fps = 0.0

    @property
    def roles(self) -> dict[int, str]:
        r = {self.left_id: "left", self.right_id: "right"}
        r.update({i: "world" for i in self.world_ids})
        if self.extrinsics is not None:
            r.update({spec.tag_id: f"{side}_{finger}" for side, d in (self.extrinsics.fingers or {}).items() for finger, spec in d.items()})
        return r

    def _solve_hands(self, dets: list[TagDetection]) -> dict[str, dict]:
        """Live wrist pose + thumb-index aperture (same solvers as offline process-tags)."""
        if self.intr is None or self.extrinsics is None:
            return {}
        from ego_collector.tracking.transforms import matrix_to_rpy
        from ego_collector.tracking.world_pose import solve_camera_pose
        from ego_collector.tracking.wrist_pose import solve_fingers, solve_wrists

        cam_solve = solve_camera_pose(dets, self.world_map, self.intr) if self.world_map is not None else None
        T_wc = cam_solve.T_world_camera if cam_solve is not None else None
        wrists = solve_wrists(dets, self.extrinsics, self.intr, T_wc)
        fingers = solve_fingers(dets, self.extrinsics, self.intr, T_wc)
        out: dict[str, dict] = {}
        for side in ("left", "right"):
            w = wrists.get(side)
            th, ix = fingers.get((side, "thumb")), fingers.get((side, "index"))
            if w is None and th is None and ix is None:
                continue
            entry: dict = {"frame": "world" if T_wc is not None else "cam", "xyz": None, "rpy_deg": None, "aperture_mm": float("nan"), "thumb": th is not None, "index": ix is not None}
            if w is not None:
                T = w.T_world_wrist if w.T_world_wrist is not None else w.T_camera_wrist
                entry["xyz"] = T[:3, 3].copy()
                entry["rpy_deg"] = np.degrees(matrix_to_rpy(T[:3, :3]))
            if th is not None and ix is not None:
                entry["aperture_mm"] = float(np.linalg.norm(th.p_camera - ix.p_camera) * 1000)
            out[side] = entry
        return out

    def latest(self) -> TagMonitorResult | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        last_idx = -1
        t_fps, n_fps = time.monotonic(), 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            frame = self.cam.latest() if self.enabled else None
            if frame is None or frame.index == last_idx:
                time.sleep(0.005)
                continue
            last_idx = frame.index
            try:
                dets = self.detector.detect(frame.image)
            except Exception as exc:  # noqa: BLE001
                log.warning("live tag detection failed: %s", exc)
                dets = []
            ids = {d.tag_id for d in dets}
            try:
                hands = self._solve_hands(dets)
            except Exception as exc:  # noqa: BLE001
                log.warning("live hand solve failed: %s", exc)
                hands = {}
            res = TagMonitorResult(
                frame_index=frame.index,
                detections=dets,
                roles=self.roles,
                process_ms=(time.monotonic() - t0) * 1000,
                left_visible=self.left_id in ids,
                right_visible=self.right_id in ids,
                world_visible=sum(1 for i in self.world_ids if i in ids),
                hands=hands,
            )
            with self._lock:
                self._latest = res
                if self.counting:
                    self.counter.add(left=res.left_visible, right=res.right_visible, world=res.world_visible >= 1, need_world=bool(self.world_ids))
            n_fps += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                self.fps, t_fps, n_fps = n_fps / (now - t_fps), now, 0
            sleep = self.period - (now - t0)
            if sleep > 0:
                time.sleep(sleep)

    def start_counting(self) -> None:
        with self._lock:
            self.counter.reset()
            self.counting = True

    def stop_counting(self) -> dict[str, float]:
        with self._lock:
            self.counting = False
            return self.counter.ratios()

    def quad_px(self, det: TagDetection) -> float:
        c = det.corners
        return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))

    def distance_m(self, det: TagDetection, size_m: float) -> float | None:
        if self.focal_px is None:
            return None
        px = self.quad_px(det)
        return float(size_m * self.focal_px / px) if px > 1 else None


@dataclass
class HandMonitorResult:
    frame_index: int
    hands: dict[str, Any]  # side -> HandResult
    scale: float  # landmarks are in the *downscaled* image; multiply by 1/scale for full-res pixels
    process_ms: float


class LiveHandMonitor(threading.Thread):
    """MediaPipe on a downscaled frame at ~6 Hz. Optional; started only when enabled."""

    def __init__(self, cam: CameraCapture, *, model_path, scale: float = 0.5, rate_hz: float = 6.0, delegate: str = "auto") -> None:
        super().__init__(name="live-hands", daemon=True)
        self.cam = cam
        self.scale = scale
        self.period = 1.0 / rate_hz
        self.model_path = model_path
        self.delegate = delegate
        self.enabled = True
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: HandMonitorResult | None = None
        self.error: str | None = None
        self.fps = 0.0

    def latest(self) -> HandMonitorResult | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        import cv2

        try:
            from ego_collector.hands.mediapipe_tracker import HandLandmarker

            lm = HandLandmarker(self.model_path, delegate=self.delegate)
        except Exception as exc:  # noqa: BLE001
            self.error = f"hands unavailable: {exc}"
            log.warning(self.error)
            return
        t_start = time.monotonic()
        last_idx = -1
        t_fps, n_fps = time.monotonic(), 0
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                frame = self.cam.latest() if self.enabled else None
                if frame is None or frame.index == last_idx:
                    time.sleep(0.005)
                    continue
                last_idx = frame.index
                small = cv2.resize(frame.image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
                try:
                    hands = lm.detect(small, int((time.monotonic() - t_start) * 1000))
                except Exception as exc:  # noqa: BLE001
                    self.error = str(exc)
                    time.sleep(0.2)
                    continue
                with self._lock:
                    self._latest = HandMonitorResult(frame.index, hands, self.scale, (time.monotonic() - t0) * 1000)
                n_fps += 1
                now = time.monotonic()
                if now - t_fps >= 1.0:
                    self.fps, t_fps, n_fps = n_fps / (now - t_fps), now, 0
                sleep = self.period - (now - t0)
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            lm.close()


__all__ = ["HandMonitorResult", "LiveHandMonitor", "LiveTagMonitor", "TagMonitorResult", "VisibilityCounter"]
