"""Orbbec Gemini 336 RGB-D capture, drop-in for CameraCapture (main-thread poll mode).

Opens synchronized color + depth streams and hardware-aligns depth to color, so each
``Frame`` carries ``image`` (BGR uint8, from the MJPG color stream) and ``depth``
(uint16 aligned to color). Exposes the same surface the recorder uses from
``CameraCapture`` (open/close/poll/get/latest/drain, ``actual``, frame counters) plus
``intrinsics`` and ``depth_scale`` so the recorder can persist ``intrinsics.yaml``.

macOS needs the Orbbec device accessed with sudo (UVCAssistant claims the UVC
interface otherwise) and frames pumped from the main thread (poll), exactly like the
existing UVC path. Verified recipe: depth 848x480 UNKNOWN_FORMAT@30, color 848x480
MJPG@30, AlignFilter(align_to_stream=COLOR_STREAM).
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from ego_collector.camera.capture import CaptureConfig, Frame

log = logging.getLogger("ego_collector.orbbec")

_W, _H, _FPS = 848, 480, 30


class OrbbecCapture:
    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self._pipe = None
        self._align = None
        self._color_profile = None
        self.actual: dict[str, float] = {}
        self.intrinsics: dict[str, float] | None = None
        self.depth_scale: float | None = None
        self.frames_grabbed = 0
        self.frames_dropped = 0
        self.read_failures = 0
        self._latest: Frame | None = None
        self._latest_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        from pyorbbecsdk import (
            AlignFilter,
            Config,
            OBFormat,
            OBSensorType,
            OBStreamType,
            Pipeline,
        )

        pipe = Pipeline()
        cfg = Config()
        depth_profile = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_video_stream_profile(
            _W, _H, OBFormat.UNKNOWN_FORMAT, _FPS
        )
        cfg.enable_stream(depth_profile)
        color_profile = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(
            _W, _H, OBFormat.MJPG, _FPS
        )
        cfg.enable_stream(color_profile)
        pipe.start(cfg)
        self._pipe = pipe
        self._align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self._color_profile = color_profile
        self.actual = {"width": float(color_profile.get_width()), "height": float(color_profile.get_height()), "fps": float(_FPS)}
        intr = color_profile.get_intrinsic()
        self.intrinsics = {"fx": float(intr.fx), "fy": float(intr.fy), "cx": float(intr.cx), "cy": float(intr.cy)}
        # warm up: the first framesets after start can be incomplete
        for _ in range(30):
            if self._grab_one() is not None:
                break
            time.sleep(0.05)
        log.info("Orbbec RGB-D open: %s intrinsics=%s", self.actual, self.intrinsics)

    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:  # noqa: BLE001
                pass
            self._pipe = None

    def __enter__(self) -> "OrbbecCapture":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- access --------------------------------------------------------------

    def latest(self) -> Frame | None:
        with self._latest_lock:
            return self._latest

    def get(self, timeout: float = 0.5) -> Frame | None:
        # No background queue in poll mode; behave like a synchronous grab.
        return self._grab_one()

    def drain(self) -> None:
        # No queue to drain; poll() already returns fresh frames.
        return None

    # -- worker --------------------------------------------------------------

    def _grab_one(self) -> Frame | None:
        if self._pipe is None:
            raise RuntimeError("camera not open")
        frames = self._pipe.wait_for_frames(500)
        t_ns = time.monotonic_ns()  # stamp as close to the grab as possible
        if frames is None:
            self.read_failures += 1
            return None
        aligned = self._align.process(frames)
        if aligned is None:
            self.read_failures += 1
            return None
        aligned = aligned.as_frame_set() if hasattr(aligned, "as_frame_set") else aligned
        if aligned is None:
            self.read_failures += 1
            return None
        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        if cf is None or df is None:
            self.read_failures += 1
            return None
        image = cv2.imdecode(np.frombuffer(cf.get_data(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.read_failures += 1
            return None
        depth = np.frombuffer(df.get_data(), np.uint16).reshape(df.get_height(), df.get_width())
        self.depth_scale = float(df.get_depth_scale())
        frame = Frame(index=self.frames_grabbed, timestamp_ns=t_ns, image=image, depth=depth.copy())
        self.frames_grabbed += 1
        with self._latest_lock:
            self._latest = frame
        return frame

    def poll(self) -> Frame | None:
        """Synchronous grab on the caller's thread (matches CameraCapture poll mode)."""
        return self._grab_one()
