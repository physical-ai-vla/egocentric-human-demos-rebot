"""Mock devices with the real drivers' interfaces and timing (used without hardware and in tests)."""
from __future__ import annotations
import math
import threading
import time
import numpy as np
from .base import CameraFrame, DeviceStatus, GripperSample, ImuSample, RateMeter, SampleBuffer, now_ns
from ..config import CameraCfg, GripperCfg, ImuCfg
from .feetech_gripper import GripperQuality, normalize_ticks
from .teensy_imu import ImuQuality


class _Periodic:
    def __init__(self, name: str, hz: float, required: bool) -> None:
        self.name, self.hz, self.required = name, hz, required
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._status = DeviceStatus(name)
        self._rate = RateMeter()
        self.fail_after_s: float | None = None   # test hook: stop producing after N s (simulates a dropout)
        self.on_event = None
        self.last_sample = None

    def open(self) -> None:
        self._status.connected = True

    def start(self) -> None:
        if self._thread: return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True); self._thread.start()
        self._status.running = True; self._status.running_since_ns = now_ns()

    def stop(self) -> None:
        self._stop.set()
        if self._thread: self._thread.join(2.0); self._thread = None
        self._status.running = False

    def close(self) -> None:
        self.stop(); self._status.connected = False

    def status(self) -> DeviceStatus:
        self._status.rate_hz = self._rate.hz()
        return self._status

    def _tick(self, i: int, t_ns: int) -> None:  # override
        raise NotImplementedError

    def _run(self) -> None:
        t0 = time.perf_counter(); i = 0; period = 1.0 / self.hz
        while not self._stop.is_set():
            target = t0 + i * period
            d = target - time.perf_counter()
            if d > 0: self._stop.wait(d)
            if self._stop.is_set(): break
            if self.fail_after_s is not None and time.perf_counter() - t0 > self.fail_after_s:
                if self._status.error is None and self.on_event: self.on_event("device_error", dict(error="mock dropout"), now_ns())
                self._status.error = "mock dropout"; self._stop.wait(0.05); continue
            t_ns = now_ns(); self._tick(i, t_ns); self._rate.tick(t_ns); self._status.last_sample_ns = t_ns; i += 1


class MockCamera(_Periodic):
    def __init__(self, cfg: CameraCfg, *, buffer_s: float = 4.0) -> None:
        super().__init__(cfg.name, cfg.fps, cfg.required)
        self.cfg = cfg
        self.buffer: SampleBuffer[CameraFrame] = SampleBuffer(int(cfg.fps * buffer_s))
        self._latest: CameraFrame | None = None
        self._lock = threading.Lock()
        self.index = -1
        self._status.detail.update(index=-1, actual=dict(width=cfg.width, height=cfg.height, fps=cfg.fps), mock=True)

    def latest(self) -> CameraFrame | None:
        with self._lock: return self._latest

    def _tick(self, i: int, t_ns: int) -> None:
        h, w = self.cfg.height, self.cfg.width
        img = np.zeros((h, w, 3), np.uint8)
        x = int((0.5 + 0.4 * math.sin(i / 30.0)) * w); y = int((0.5 + 0.3 * math.cos(i / 20.0)) * h)
        img[:, :, 2] = 40; img[max(0, y - 20):y + 20, max(0, x - 20):x + 20] = (0, 255, 0)
        depth = (np.full((h, w), 800 + (i % 50), np.uint16) if self.cfg.role == "aux_depth" else None)
        f = CameraFrame(self.name, i, t_ns, img, depth)
        with self._lock: self._latest = f
        self.buffer.append(f)


class MockImu(_Periodic):
    # A mock unit lies on the table: tremor-level motion (gyro ~0.6 deg/s, accel ~0.01 m/s^2) so the stillness gate opens in
    # --mock the way it does on a real still unit. Tests raise these to make the mock "move".
    GYRO_AMP_RAD_S = 0.01
    ACCEL_AMP = 0.01

    def __init__(self, cfg: ImuCfg, *, buffer_s: float = 5.0) -> None:
        super().__init__(f"imu_{cfg.side}", cfg.rate_hz, cfg.required)
        self.cfg = cfg
        self.buffer: SampleBuffer[ImuSample] = SampleBuffer(int(cfg.rate_hz * buffer_s))
        self._t0_us = 1_000_000
        self.serial_number = f"MOCK-{cfg.side.upper()}"
        self.port = "mock"
        self._status.detail.update(mock=True, serial_number=self.serial_number, port=self.port)
        self.seq = None

    def quality(self) -> ImuQuality:
        return ImuQuality(hz=self._rate.hz(), age_ms=self._status.age_ms, connected=self._status.connected)

    def _tick(self, i: int, t_ns: int) -> None:
        t_us = self._t0_us + int(i * 1e6 / self.cfg.rate_hz)
        w = 2 * math.pi * 0.5 * i / self.cfg.rate_hz
        s = ImuSample(self.cfg.side, i, t_us, t_ns, self.ACCEL_AMP * math.sin(w), 0.0, 9.81, 0.0, self.GYRO_AMP_RAD_S * math.cos(w), 0.0, 35.0)
        self.buffer.append(s); self.last_sample = s


class MockGripper(_Periodic):
    def __init__(self, cfg: GripperCfg, *, buffer_s: float = 5.0) -> None:
        super().__init__(f"gripper_{cfg.side}", cfg.sample_hz, cfg.required)
        self.cfg = cfg
        self.buffer: SampleBuffer[GripperSample] = SampleBuffer(int(cfg.sample_hz * buffer_s))
        self._status.detail.update(mock=True)
        self.sim_closed, self.sim_open = 1000, 2000        # the simulated jaw sweeps between these ticks

    def set_calibration(self, ticks_closed, ticks_open) -> None:
        self.cfg.ticks_closed, self.cfg.ticks_open = ticks_closed, ticks_open

    def quality(self) -> GripperQuality:
        ls = self.last_sample
        return GripperQuality(hz=self._rate.hz(), age_ms=self._status.age_ms, connected=self._status.connected,
                              raw=ls.raw_position_mod if ls else None, unwrapped=ls.raw_position if ls else None, normalized=ls.normalized if ls else None)

    def _tick(self, i: int, t_ns: int) -> None:
        # Sweep between the CONFIGURED ticks, not a fixed 1000-2000: with a calibration outside that span the
        # normalised output clamps to a constant, and a stand-in whose jaw never appears to move cannot exercise
        # anything that reads grasp -- including the check that now refuses an episode whose jaw never moved.
        lo, hi = self.cfg.ticks_closed, self.cfg.ticks_open
        if lo is None or hi is None or lo == hi:
            lo, hi = self.sim_closed, self.sim_open
        raw = int(lo + (hi - lo) * 0.5 * (1 + math.sin(i / 50.0)))
        s = GripperSample(self.cfg.side, i, t_ns, raw, raw % 4096,
                          normalize_ticks(raw, self.cfg.ticks_closed, self.cfg.ticks_open, self.cfg.norm_one_is), None)
        self.buffer.append(s); self.last_sample = s
