"""HandUMI jaw encoder (Feetech STS3215, passive) sampled at a fixed rate on its own thread.
Timestamp = midpoint of the serial transaction (host monotonic). Encoder wrap (0/4095 seam) is unwrapped; the first
sample is trusted as-is, so start with the jaws roughly closed (away from the seam) — same rule as handumi-sw."""
from __future__ import annotations
import logging
import math
import threading
import time
from dataclasses import dataclass
from .base import DeviceStatus, GripperSample, RateMeter, SampleBuffer, now_ns, resolve_serial_port
from .feetech_bus import FeetechBus, describe_error_bits
from ..config import GripperCfg

log = logging.getLogger("handumi.gripper")
ENCODER_RESOLUTION = 4096
HALF_TURN = ENCODER_RESOLUTION // 2


class EncoderUnwrapper:
    def __init__(self) -> None:
        self._prev: int | None = None
        self._turns = 0

    def __call__(self, raw: int) -> int:
        if self._prev is not None:
            d = raw - self._prev
            if d > HALF_TURN: self._turns -= 1
            elif d < -HALF_TURN: self._turns += 1
        self._prev = raw
        return raw + self._turns * ENCODER_RESOLUTION


def normalize_ticks(ticks: float, closed: int | None, open_: int | None, one_is: str = "closed", resolution: int = ENCODER_RESOLUTION) -> float:
    """Circular normalisation: position along the jaw arc from `closed` towards `open_` (shortest way round the 4096-tick
    circle), clipped to the arc, so a jaw range that crosses the 4095->0 seam (e.g. closed 4069, open 5389) reads the same
    whether `ticks` is raw (mod 4096) or unwrapped and regardless of where the unwrapper started after a restart.
    Flipped so that `one_is` maps to 1.0. NaN until calibrated."""
    if closed is None or open_ is None or open_ == closed:
        return math.nan
    fwd = (open_ - closed) % resolution
    direction = 1 if fwd <= resolution // 2 else -1
    arc = ((open_ - closed) * direction) % resolution            # jaw arc length in ticks (assumed < half a turn)
    if arc == 0: return math.nan
    d = ((ticks - closed) * direction) % resolution               # distance from closed along the opening direction
    if d > arc: d = arc if (d - arc) < (resolution - d) else 0   # outside the arc: nearest end
    g = d / arc
    return 1.0 - g if one_is == "closed" else g


@dataclass
class GripperQuality:
    hz: float = 0.0
    age_ms: float | None = None
    connected: bool = False
    errors: int = 0
    reconnects: int = 0
    raw: int | None = None
    unwrapped: int | None = None
    normalized: float | None = None

    def grade(self, th: dict) -> str:
        if not self.connected or self.age_ms is None: return "RED"
        g, y = th.get("green", {}), th.get("yellow", {})
        if self.hz >= g.get("min_hz", 80) and self.age_ms <= g.get("max_age_ms", 100): return "GREEN"
        if self.hz >= y.get("min_hz", 40) and self.age_ms <= y.get("max_age_ms", 500): return "YELLOW"
        return "RED"


class FeetechGripper:
    def __init__(self, cfg: GripperCfg, *, buffer_s: float = 5.0, telemetry: bool = True) -> None:
        self.cfg = cfg
        self.total_errors = 0
        self.reconnects = 0
        self.last_sample: GripperSample | None = None
        self.on_event = None
        self.name = f"gripper_{cfg.side}"
        self.required = cfg.required
        self.buffer: SampleBuffer[GripperSample] = SampleBuffer(int(cfg.sample_hz * buffer_s))
        self.telemetry = telemetry
        self._bus = FeetechBus(cfg.port, cfg.baud)
        self._unwrap = EncoderUnwrapper()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._rate = RateMeter()
        self._status = DeviceStatus(self.name)
        self._seq = 0
        self._consecutive_errors = 0

    def open(self) -> None:
        if self.cfg.port_serial:
            port = resolve_serial_port(self.cfg.port_glob, serial_number=self.cfg.port_serial)
            if port is None: raise RuntimeError(f"{self.name}: no USB serial adapter with serial {self.cfg.port_serial!r}")
            self.cfg.port = port; self._bus = FeetechBus(port, self.cfg.baud)
        self._bus.open()
        if not self._bus.ping(self.cfg.servo_id):
            self._bus.close()
            raise RuntimeError(f"{self.name}: servo id {self.cfg.servo_id} does not answer on {self.cfg.port}")
        self._status.connected = True
        self._status.error = None
        self._status.detail["servo_error_bits"] = describe_error_bits(self._bus.last_error_bits)

    def start(self) -> None:
        if self._thread: return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()
        self._status.running = True; self._status.running_since_ns = now_ns()

    def stop(self) -> None:
        self._stop.set()
        if self._thread: self._thread.join(2.0); self._thread = None
        self._status.running = False

    def close(self) -> None:
        self.stop(); self._bus.close(); self._status.connected = False

    def set_calibration(self, ticks_closed: int | None, ticks_open: int | None) -> None:
        self.cfg.ticks_closed, self.cfg.ticks_open = ticks_closed, ticks_open

    def quality(self) -> GripperQuality:
        ls = self.last_sample
        return GripperQuality(hz=self._rate.hz(), age_ms=self._status.age_ms, connected=self._status.connected, errors=self.total_errors,
                              reconnects=self.reconnects, raw=ls.raw_position_mod if ls else None, unwrapped=ls.raw_position if ls else None,
                              normalized=ls.normalized if ls else None)

    def status(self) -> DeviceStatus:
        self._status.rate_hz = self._rate.hz()
        self._status.detail.update(port=self.cfg.port, servo_id=self.cfg.servo_id, errors=self.total_errors, reconnects=self.reconnects,
                                   ticks_closed=self.cfg.ticks_closed, ticks_open=self.cfg.ticks_open, port_serial=self.cfg.port_serial,
                                   servo_error_bits=describe_error_bits(getattr(self._bus, "last_error_bits", 0)))
        return self._status

    def _read_once(self) -> GripperSample:
        t0 = now_ns()
        tele = None
        if self.telemetry:
            try:
                tele = self._bus.read_status_block(self.cfg.servo_id, retries=0)
            except Exception:
                tele = None
        raw = int(tele["position"]) if tele else self._bus.read_position(self.cfg.servo_id, retries=0)
        t1 = now_ns()
        self._seq += 1
        ticks = self._unwrap(raw)
        return GripperSample(self.cfg.side, self._seq, (t0 + t1) // 2, ticks, raw,
                             normalize_ticks(ticks, self.cfg.ticks_closed, self.cfg.ticks_open, self.cfg.norm_one_is), tele)

    def _run(self) -> None:
        interval = 1.0 / self.cfg.sample_hz
        nxt = time.perf_counter()
        while not self._stop.is_set():
            try:
                s = self._read_once()
                self.buffer.append(s); self.last_sample = s; self._rate.tick(s.sample_ns)
                self._status.last_sample_ns = s.sample_ns; self._status.error = None
                self._consecutive_errors = 0
            except Exception as exc:
                self._consecutive_errors += 1; self.total_errors += 1
                self._status.error = str(exc)
                if self._consecutive_errors == 1:
                    log.warning("%s read failed: %s", self.name, exc)
                    if self.on_event: self.on_event("device_error", dict(error=str(exc)), now_ns())
                if self._consecutive_errors % 3 == 0:
                    try:
                        self._bus.reset_io(); self._bus.close(); time.sleep(0.02); self._bus.open(); self.reconnects += 1
                        if self.on_event: self.on_event("device_reconnect", dict(port=self.cfg.port), now_ns())
                    except Exception as rexc:
                        self._status.error = f"{exc}; reconnect failed: {rexc}"
            nxt += interval
            delay = nxt - time.perf_counter()
            if delay > 0: self._stop.wait(delay)
            else: nxt = time.perf_counter()
