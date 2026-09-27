"""One H.264 mp4 per camera stream, encoded on its own thread with PyAV (VideoToolbox → libx264 fallback).
Frames are written in capture order; the writer never drops — if the encoder falls behind, the queue grows and the
recorder reports it. video frame n  <->  /<stream>/frame_meta message with video_frame == n."""
from __future__ import annotations
import logging
import queue
import threading
from fractions import Fraction
from pathlib import Path
import av
import numpy as np

log = logging.getLogger("handumi.video")


def _open_encoder(container, codec: str, w: int, h: int, fps: int, bitrate_kbps: int):
    for name in (codec, "libx264", "mpeg4"):
        try:
            s = container.add_stream(name, rate=fps)
            s.width, s.height, s.pix_fmt = w, h, "yuv420p"
            s.bit_rate = bitrate_kbps * 1000
            s.time_base = Fraction(1, fps)
            if name == "libx264":
                s.options = {"preset": "veryfast", "tune": "zerolatency", "crf": "18"}
            elif name == "h264_videotoolbox":
                s.options = {"realtime": "1", "allow_sw": "1"}
            # open the codec now: lazy init on the first encode() stalls the interpreter ~150-200 ms and would show up
            # as spurious sensor gaps right after START
            s.codec_context.open()
            return s, name
        except Exception as exc:  # codec not available -> next
            log.warning("encoder %s unavailable (%s)", name, exc)
    raise RuntimeError("no H.264/MPEG-4 encoder available")


class VideoStreamWriter:
    def __init__(self, path: Path, *, width: int, height: int, fps: int, codec: str = "h264_videotoolbox",
                 bitrate_kbps: int = 6000) -> None:
        self.path = Path(path)
        self.width, self.height, self.fps = width, height, fps
        self._container = av.open(str(self.path), mode="w")
        self._stream, self.codec_used = _open_encoder(self._container, codec, width, height, fps, bitrate_kbps)
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self.frames_written = 0
        self.frames_put = 0
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, name=f"enc-{self.path.stem}", daemon=True)
        self._thread.start()

    def put(self, bgr: np.ndarray) -> int:
        """Enqueue a frame; returns its video frame number."""
        n = self.frames_put
        self.frames_put += 1
        self._q.put(bgr)
        return n

    def backlog(self) -> int:
        return self.frames_put - self.frames_written

    def _run(self) -> None:
        pts = 0
        while True:
            img = self._q.get()
            if img is None: break
            try:
                if img.shape[1] != self.width or img.shape[0] != self.height:
                    import cv2
                    img = cv2.resize(img, (self.width, self.height))
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="bgr24")
                frame.pts = pts; pts += 1
                for pkt in self._stream.encode(frame):
                    self._container.mux(pkt)
                self.frames_written += 1
            except Exception as exc:
                self.error = str(exc); log.error("encode %s: %s", self.path.name, exc)
        try:
            for pkt in self._stream.encode(None):
                self._container.mux(pkt)
        except Exception as exc:
            self.error = self.error or str(exc)
        self._container.close()

    def close(self, timeout: float = 30.0) -> None:
        self._q.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            self.error = self.error or "encoder did not finish in time"
