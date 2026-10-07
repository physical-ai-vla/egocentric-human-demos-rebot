"""LivePoseRunner (tracking_mode = continuous_session): runs an online-capable backend on one wrist camera + IMU during a
session, in its own thread, reading the same SampleBuffers the recorder drains? — NO: the recorder's buffers are drained by the
recorder; this runner subscribes via the devices' `latest()` frames and a private IMU tap, so it can never starve or slow raw
acquisition. Poses go to a side buffer that the recorder may optionally persist as `/<side>/pose_online` (derived, not raw).
Any exception inside the runner only flips its own status to error; raw recording is unaffected (tested)."""
from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
import numpy as np
from .estimator import PoseEstimate, PoseEstimator, TrackingState
from ..devices.base import SampleBuffer, now_ns

log = logging.getLogger("handumi.pose.live")


@dataclass
class LivePoseStatus:
    running: bool = False
    frames: int = 0
    state: str = "uninitialized"
    error: str | None = None
    fps: float = 0.0
    detail: dict = field(default_factory=dict)


class LivePoseRunner:
    def __init__(self, side: str, camera, imu, estimator: PoseEstimator, *, intrinsics=None, T_camera_imu=None, downscale: int = 2,
                 camera_imu_offset_ns: int = 0, buffer_s: float = 10.0, fps: float = 30.0) -> None:
        self.side, self.cam, self.imu, self.est = side, camera, imu, estimator
        self.intrinsics, self.T_camera_imu, self.downscale, self.offset_ns = intrinsics, T_camera_imu, downscale, camera_imu_offset_ns
        self.buffer: SampleBuffer[PoseEstimate] = SampleBuffer(int(buffer_s * fps))
        self.status = LivePoseStatus()
        self._stop = threading.Event(); self._thread: threading.Thread | None = None
        self._last_frame_index = None; self._imu_seq = None
        self._n_t: list[int] = []

    def start(self) -> None:
        try:
            self.est.initialize(intrinsics=self.intrinsics, T_camera_imu=self.T_camera_imu)
        except Exception as exc:
            self.status.error = f"initialize failed: {exc}"; log.warning("live pose %s: %s", self.side, self.status.error); return
        self._stop.clear(); self._thread = threading.Thread(target=self._run, name=f"live-pose-{self.side}", daemon=True); self._thread.start()
        self.status.running = True

    def stop(self) -> None:
        self._stop.set()
        if self._thread: self._thread.join(3.0); self._thread = None
        self.status.running = False

    def reset(self) -> None:
        """Operator-triggered tracker reset after severe drift / LOST (recovery step 3 in the design)."""
        try: self.est.reset()
        except Exception as exc: self.status.error = f"reset failed: {exc}"

    def _run(self) -> None:
        import cv2
        while not self._stop.is_set():
            try:
                f = self.cam.latest()
                if f is None or f.frame_index == self._last_frame_index:
                    self._stop.wait(0.005); continue
                self._last_frame_index = f.frame_index
                if self.imu is not None and self.est.info.uses_imu:
                    s = getattr(self.imu, "last_sample", None)
                    if s is not None and s.seq != self._imu_seq:            # latest-only tap (a full IMU tap needs a second buffer; V1 online is for QA/UI)
                        self._imu_seq = s.seq; self.est.push_imu(s.host_receive_ns + self.offset_ns, (s.gx, s.gy, s.gz), (s.ax, s.ay, s.az))
                img = f.image
                if self.downscale > 1: img = cv2.resize(img, (img.shape[1] // self.downscale, img.shape[0] // self.downscale), interpolation=cv2.INTER_AREA)
                e = self.est.push_image(f.capture_ns, f.frame_index, img)
                self.buffer.append(e); self.status.frames += 1; self.status.state = e.tracking_state.value
                self._n_t.append(now_ns()); self._n_t = self._n_t[-30:]
                if len(self._n_t) > 2: self.status.fps = (len(self._n_t) - 1) / max((self._n_t[-1] - self._n_t[0]) / 1e9, 1e-6)
            except Exception as exc:
                self.status.error = f"{exc!r}"; self.status.state = TrackingState.LOST.value
                log.exception("live pose %s failed; raw recording continues", self.side)
                self._stop.wait(0.5)
        self.status.running = False
