"""Threaded UVC capture with monotonic timestamps (Logitech C922, 1080p30 MJPEG).

The capture thread only grabs frames and stamps them with ``time.monotonic_ns()``;
nothing else (no detection, no ML) runs on this thread so frames are not lost.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue

import cv2
import numpy as np

log = logging.getLogger("ego_collector.capture")


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_ns: int
    image: np.ndarray  # BGR uint8 (OpenCV native)
    depth: np.ndarray | None = None  # uint16 depth aligned to color (Orbbec RGB-D); None for plain UVC cameras


@dataclass
class CaptureConfig:
    index: int | str = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    fourcc: str = "MJPG"
    queue_size: int = 300  # ~10 s at 30 fps of headroom for the writer thread
    backend: int | None = None  # e.g. cv2.CAP_AVFOUNDATION; None = default
    # False = no capture thread: the caller pumps frames with poll() from its own (main) thread.
    # macOS AVFoundation + an OpenCV preview window on the main thread can deadlock when grabbing
    # happens on a background thread, so the interactive recorder uses poll() there.
    threaded: bool = True
    # Manual image controls, for freezing production camera settings before calibration. None = leave the camera's
    # current setting alone (the default, so nothing changes for existing callers). Whether a UVC camera honours any of
    # these is camera- and backend-specific; `CameraCapture.controls` reports what actually took effect, verified by
    # read-back, and an unsupported control is never reported as applied.
    # Refuse the camera instead of accepting whatever it delivers. Off by default (several UVC cameras legitimately
    # substitute a nearby mode), on for production profiles: a camera that silently delivers a different format is
    # exactly how a take ends up unusable, or worse, attributed to the wrong intrinsics.
    strict_format: bool = False
    exposure_auto: bool | None = None
    exposure: float | None = None
    gain: float | None = None
    white_balance_auto: bool | None = None
    white_balance: float | None = None


# OpenCV property ids for the manual image controls, in the order they must be applied: the auto flags first, because a
# camera in auto mode overwrites a manual value the moment it is set.
_CONTROLS = (
    ("exposure_auto", cv2.CAP_PROP_AUTO_EXPOSURE),
    ("exposure", cv2.CAP_PROP_EXPOSURE),
    ("gain", cv2.CAP_PROP_GAIN),
    ("white_balance_auto", cv2.CAP_PROP_AUTO_WB),
    ("white_balance", cv2.CAP_PROP_WB_TEMPERATURE),
)


def _apply_controls(cap: "cv2.VideoCapture", c: CaptureConfig) -> list[dict]:
    """Apply the manual image controls that are set on the config and report, per control, what actually happened.

    `VideoCapture.set` returns True on several backends regardless of whether the camera honoured the request, so the
    verdict here is the READ-BACK: a control counts as applied only when the value read back matches what was asked
    for. Anything else is recorded as not applied, with the read-back value, and is never presented as a success —
    a camera settings freeze that only exists in a config file is worse than no freeze at all."""
    out: list[dict] = []
    for name, prop in _CONTROLS:
        want = getattr(c, name, None)
        if want is None:
            continue
        want = float(want) if not isinstance(want, bool) else (1.0 if want else 0.0)
        ok = bool(cap.set(prop, want))
        back = float(cap.get(prop))
        matched = abs(back - want) <= max(0.02 * abs(want), 1e-6)
        reason = ""
        if not ok:
            reason = "the backend rejected set()"
        elif not matched:
            reason = f"read back {back:g}, asked for {want:g}"
        out.append(dict(control=name, requested=want, readback=back, set_ok=ok, applied=bool(ok and matched),
                        reason=reason))
    return out


class CameraCapture:
    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.queue: Queue[Frame] = Queue(maxsize=config.queue_size)
        self.frames_grabbed = 0
        self.frames_dropped = 0  # queue full -> consumer too slow
        self.read_failures = 0
        self.actual: dict[str, float] = {}
        self.controls: list[dict] = []       # per-control outcome, written by open(); see _apply_controls
        self._latest: Frame | None = None
        self._latest_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        c = self.config
        cap = cv2.VideoCapture(c.index) if c.backend is None else cv2.VideoCapture(c.index, c.backend)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {c.index!r}")
        if c.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        cap.set(cv2.CAP_PROP_FPS, c.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.controls = _apply_controls(cap, c)
        # C922/UVC on macOS often return an empty first frame right after setting
        # properties; warm up with a short retry loop before declaring failure.
        ok, frame = False, None
        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"camera {c.index!r} opened but returned no frame")
        self.actual = {
            "width": float(frame.shape[1]),
            "height": float(frame.shape[0]),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        }
        for ctl in self.controls:
            if not ctl["applied"]:
                log.warning("camera %r: control %s NOT applied (%s)", c.index, ctl["control"], ctl["reason"])
        if (frame.shape[1], frame.shape[0]) != (c.width, c.height):
            msg = f"camera delivers {frame.shape[1]}x{frame.shape[0]} instead of the requested {c.width}x{c.height}"
            if c.strict_format:
                cap.release()
                raise RuntimeError(msg)
            log.warning("%s", msg)
        self._cap = cap
        self._stop.clear()
        if self.config.threaded:
            self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- access ----------------------------------------------------------------

    def latest(self) -> Frame | None:
        with self._latest_lock:
            return self._latest

    def get(self, timeout: float = 0.5) -> Frame | None:
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def drain(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                return

    # -- worker ------------------------------------------------------------------

    def _grab_one(self) -> Frame | None:
        assert self._cap is not None
        ok = self._cap.grab()
        t_ns = time.monotonic_ns()  # stamp as close to the grab as possible
        if not ok:
            self.read_failures += 1
            return None
        ok, image = self._cap.retrieve()
        if not ok or image is None:
            self.read_failures += 1
            return None
        frame = Frame(index=self.frames_grabbed, timestamp_ns=t_ns, image=image)
        self.frames_grabbed += 1
        with self._latest_lock:
            self._latest = frame
        return frame

    def poll(self) -> Frame | None:
        """Synchronous grab on the caller's thread (threaded=False mode)."""
        if self._cap is None:
            raise RuntimeError("camera not open")
        return self._grab_one()

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self._grab_one()
            if frame is None:
                time.sleep(0.002)
                continue
            try:
                self.queue.put_nowait(frame)
            except Full:
                self.frames_dropped += 1


__all__ = ["CameraCapture", "CaptureConfig", "Frame"]
