"""Live wrist IMU -> AHRS orientation stream (2026-10-14). Wraps the production driver handumi_collector.devices.teensy_imu
(framing, CRC, reconnect, fail-closed LEFT/RIGHT identity by USB serial -- never port order) and runs one MahonyAhrs per side
on its own thread at the IMU rate. Device clock drives integration; a rolling device->host clock fit stamps the output on the
host clock (+ the measured camera<->IMU offset, imu/time_sync.py) so fusion can compare it with camera frames.

Health (spec 26): OK = AHRS ready, newest sample younger than max_age_ms, rate >= min_rate_hz."""
from __future__ import annotations
import threading
import time
from collections import deque
from dataclasses import dataclass
import numpy as np
from .ahrs import AhrsConfig, MahonyAhrs
from handumi_collector.pose.timing import fit_device_to_host


@dataclass
class ImuOrientation:
    side: str
    t_host_ns: int                 # sample time on the camera/host clock (clock fit + camera_imu offset)
    R_G_I: np.ndarray
    gyro_rad_s: np.ndarray         # bias-corrected, body frame
    healthy: bool
    reason: str = ""
    acc_m_s2: np.ndarray | None = None   # raw accelerometer of this sample (IMU frame) -- the INIT push-axis check


class ImuReader:
    def __init__(self, side: str, device, ahrs_cfg: dict | None = None, *, camera_imu_offset_ms: float | None = None,
                 max_age_ms: float = 50.0, min_rate_hz: float = 100.0, clock_window: int = 4000) -> None:
        """device: an opened handumi TeensyImu (or any object with .buffer.drain() of ImuSample)."""
        self.side, self.dev = side, device
        self.ahrs = MahonyAhrs(AhrsConfig.from_dict(ahrs_cfg))
        self.offset_ns = int(round((camera_imu_offset_ms or 0.0) * 1e6)); self.offset_measured = camera_imu_offset_ms is not None
        self.max_age_ms, self.min_rate_hz = float(max_age_ms), float(min_rate_hz)
        self._dev_us: deque = deque(maxlen=clock_window); self._host_ns: deque = deque(maxlen=clock_window)
        self._clock = None; self._lock = threading.Lock(); self._latest: ImuOrientation | None = None
        self._rate_t: deque = deque(maxlen=256); self.samples_for_recorder: deque = deque(maxlen=400000)   # 10-15: 20000 (100 s @ 200 Hz) truncated a 4 min recording
        self._stop = threading.Event(); self._thread: threading.Thread | None = None

    def set_offset_ms(self, ms: float) -> None: self.offset_ns = int(round(ms * 1e6)); self.offset_measured = True

    def ingest(self, samples) -> None:
        """Feed raw ImuSample objects (thread or tests)."""
        for s in samples:
            self._dev_us.append(s.device_timestamp_us); self._host_ns.append(s.host_receive_ns); self._rate_t.append(s.host_receive_ns)
            R = self.ahrs.update(s.device_timestamp_us, (s.gx, s.gy, s.gz), (s.ax, s.ay, s.az))
            self.samples_for_recorder.append(s)
            if R is None: continue
            if self._clock is None or len(self._dev_us) % 200 == 0:
                if len(self._dev_us) >= 50: self._clock = fit_device_to_host(np.asarray(self._dev_us), np.asarray(self._host_ns))
            t = int(self._clock.to_host_ns([s.device_timestamp_us])[0]) if self._clock else int(s.host_receive_ns)
            g = np.array([s.gx, s.gy, s.gz]) - self.ahrs.bias
            with self._lock: self._latest = ImuOrientation(self.side, t + self.offset_ns, R, g, True, "", np.array([s.ax, s.ay, s.az]))

    def latest(self, now_ns: int | None = None) -> ImuOrientation | None:
        now = int(now_ns if now_ns is not None else time.monotonic_ns())
        with self._lock: x = self._latest
        if x is None: return None
        reasons = []
        age_ms = (now - x.t_host_ns) / 1e6
        if age_ms > self.max_age_ms: reasons.append(f"imu_stale:{age_ms:.0f}ms")
        r = self.rate_hz()
        if r < self.min_rate_hz: reasons.append(f"imu_rate:{r:.0f}Hz")
        if not self.offset_measured: reasons.append("camera_imu_offset_unmeasured")
        return ImuOrientation(x.side, x.t_host_ns, x.R_G_I.copy(), x.gyro_rad_s.copy(), not reasons, ",".join(reasons), x.acc_m_s2)

    def rate_hz(self) -> float:
        t = list(self._rate_t)
        return 0.0 if len(t) < 2 or t[-1] == t[0] else (len(t) - 1) / ((t[-1] - t[0]) / 1e9)

    # ---- thread -------------------------------------------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear(); self._thread = threading.Thread(target=self._run, name=f"imu_reader_{self.side}", daemon=True); self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread: self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self.dev.buffer.drain()
            if batch: self.ingest(batch)
            else: time.sleep(0.002)


def open_teensy(side: str, cfg_imu: dict):
    """Open the production driver for one side (fails closed without a serial number)."""
    from handumi_collector.config import ImuCfg
    from handumi_collector.devices.teensy_imu import TeensyImu
    c = cfg_imu[side]
    dev = TeensyImu(ImuCfg(side=side, port_glob=c.get("port_glob", "/dev/cu.usbmodem*"), serial_number=c.get("serial_number"),
                           rate_hz=int(c.get("rate_hz", 200))))
    dev.open(); dev.start(); return dev
