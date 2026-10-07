"""Full-rate raw IMU fan-out for live VIO (M1-2 prerequisite).

    ICM42688P 400 Hz ──► device.buffer ──► raw recorder (drain)          unchanged
                     └─► tap buffer    ──► LiveVioFeeder ──► VIO        NEW (every sample, never the 30 Hz teleop clock)

`install_imu_tap(device)` swaps the device's SampleBuffer for a FanoutBuffer (same API) so every appended ImuSample also lands
in one or more tap buffers; the recorder's drain() is untouched. No handumi_collector file is edited.

Live device->host clock: the offline pipeline fits device_us -> host_ns over the whole episode (timing.fit_device_to_host);
live we keep a sliding lower-envelope fit (USB receive latency is one-sided) with the nominal 1000 ns/us slope until enough
samples exist for a slope fit. `LiveVioFeeder` pushes IMU (mapped to host ns + camera<->IMU offset) and camera frames into an
EstimatorWristPoseProvider on its own thread, so raw acquisition can never be slowed by the estimator."""
from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field
import numpy as np
from handumi_collector.devices.base import SampleBuffer, ImuSample, now_ns
from handumi_collector.pose.timing import fit_device_to_host

log = logging.getLogger("ego_teleop.imu_tap")


class FanoutBuffer(SampleBuffer):
    """Drop-in SampleBuffer whose append() also feeds every registered tap."""

    def __init__(self, primary: SampleBuffer) -> None:
        super().__init__(primary._dq.maxlen)
        with primary._lock:                                  # carry over anything not yet drained
            for s in primary._dq: self._dq.append(s)
            self.total, self.overflow = primary.total, primary.overflow
        self.taps: list[SampleBuffer] = []

    def add_tap(self, buffer_len: int) -> SampleBuffer:
        t = SampleBuffer(buffer_len); self.taps.append(t); return t

    def append(self, s) -> None:
        super().append(s)
        for t in self.taps: t.append(s)


def install_imu_tap(device, *, buffer_s: float = 2.0) -> SampleBuffer:
    """Returns a tap buffer receiving every ImuSample the device publishes (device.handle() appends via `device.buffer`)."""
    if not isinstance(device.buffer, FanoutBuffer): device.buffer = FanoutBuffer(device.buffer)
    rate = getattr(getattr(device, "cfg", None), "rate_hz", 400)
    return device.buffer.add_tap(int(rate * buffer_s))


@dataclass
class LiveImuClock:
    """Online device_us -> host_ns mapping: lower-envelope offset over a sliding window, nominal slope until `min_for_slope`."""
    window: int = 2000
    min_for_slope: int = 800
    slope_ns_per_us: float = 1000.0
    offset_ns: float | None = None
    _dev: list = field(default_factory=list, repr=False)
    _host: list = field(default_factory=list, repr=False)

    def update(self, device_us: int, host_receive_ns: int) -> None:
        self._dev.append(int(device_us)); self._host.append(int(host_receive_ns))
        if len(self._dev) > self.window: del self._dev[0]; del self._host[0]
        if len(self._dev) >= self.min_for_slope and len(self._dev) % 200 == 0:
            f = fit_device_to_host(np.asarray(self._dev, np.float64), np.asarray(self._host, np.float64))
            self.slope_ns_per_us, self.offset_ns = f.slope_ns_per_us, f.offset_ns
        else:
            cand = host_receive_ns - self.slope_ns_per_us * device_us      # one-sided latency: the smallest offset seen is the truest
            self.offset_ns = cand if self.offset_ns is None else min(self.offset_ns, cand) if len(self._dev) < self.min_for_slope else self.offset_ns

    def to_host_ns(self, device_us: int) -> int:
        if self.offset_ns is None: raise RuntimeError("LiveImuClock: no samples yet")
        return int(device_us * self.slope_ns_per_us + self.offset_ns)

    @property
    def n(self) -> int: return len(self._dev)


class LiveVioFeeder:
    """Thread: full-rate IMU tap + camera.latest() -> EstimatorWristPoseProvider (push_imu / push_image)."""

    def __init__(self, provider, *, imu_tap: SampleBuffer, camera, downscale: int = 1, clock: LiveImuClock | None = None, poll_s: float = 0.002) -> None:
        self.provider, self.tap, self.cam, self.downscale, self.poll_s = provider, imu_tap, camera, max(1, int(downscale)), poll_s
        self.clock = clock or LiveImuClock(); self._stop = threading.Event(); self._thread: threading.Thread | None = None
        self.stats = dict(imu=0, frames=0, imu_dropped_pre_clock=0, errors=0, last_error="", fps=0.0); self._last_frame_index = None; self._ft: list[int] = []

    def start(self) -> None:
        self._stop.clear(); self._thread = threading.Thread(target=self._run, name="live-vio-feeder", daemon=True); self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread: self._thread.join(3.0); self._thread = None

    def step(self) -> None:
        """One pump iteration (also used by tests): all pending IMU samples first (time order), then the newest frame."""
        import cv2
        for s in self.tap.drain():
            self.clock.update(s.device_timestamp_us, s.host_receive_ns)
            if self.clock.n < 50: self.stats["imu_dropped_pre_clock"] += 1; continue
            self.provider.push_imu(self.clock.to_host_ns(s.device_timestamp_us), (s.gx, s.gy, s.gz), (s.ax, s.ay, s.az)); self.stats["imu"] += 1
        f = self.cam.latest()
        if f is None or f.frame_index == self._last_frame_index: return
        self._last_frame_index = f.frame_index
        img = f.image
        if self.downscale > 1: img = cv2.resize(img, (img.shape[1] // self.downscale, img.shape[0] // self.downscale), interpolation=cv2.INTER_AREA)
        self.provider.push_image(f.capture_ns, f.frame_index, img); self.stats["frames"] += 1
        self._ft.append(now_ns()); self._ft = self._ft[-30:]
        if len(self._ft) > 2: self.stats["fps"] = (len(self._ft) - 1) / max((self._ft[-1] - self._ft[0]) / 1e9, 1e-6)

    def _run(self) -> None:
        while not self._stop.is_set():
            try: self.step()
            except Exception as exc:
                self.stats["errors"] += 1; self.stats["last_error"] = repr(exc); log.exception("live VIO feeder step failed; raw recording unaffected")
                self._stop.wait(0.2)
            self._stop.wait(self.poll_s)
