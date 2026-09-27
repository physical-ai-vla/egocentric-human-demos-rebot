from __future__ import annotations
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Generic, Iterable, Protocol, TypeVar
import numpy as np


def now_ns() -> int:
    """Host monotonic clock used for every timestamp in the collector."""
    return time.monotonic_ns()


@dataclass(frozen=True)
class CameraFrame:
    stream: str                 # head | left_wrist | right_wrist | head_depth
    frame_index: int            # index in the capture sequence of this device (gaps = dropped by the camera/driver)
    capture_ns: int             # host monotonic right after grab()
    image: np.ndarray           # BGR uint8
    depth: np.ndarray | None = None   # uint16 (aux RGB-D only)


@dataclass(frozen=True)
class ImuSample:
    side: str
    seq: int                    # device sequence number (wraps at 2^32)
    device_timestamp_us: int    # Teensy micros() at INT1 data-ready
    host_receive_ns: int        # host monotonic when the packet was parsed
    ax: float; ay: float; az: float      # m/s^2
    gx: float; gy: float; gz: float      # rad/s
    temperature_c: float


@dataclass(frozen=True)
class GripperSample:
    side: str
    seq: int
    sample_ns: int              # midpoint of the serial transaction (host monotonic)
    raw_position: int           # unwrapped encoder ticks
    raw_position_mod: int       # raw 0..4095 as reported
    normalized: float           # 0..1 per config convention (NaN until calibrated)
    telemetry: dict | None = None


T = TypeVar("T")


class SampleBuffer(Generic[T]):
    """Thread-safe ring buffer. Producers append; the recorder drains everything since its last drain; the UI reads latest."""

    def __init__(self, maxlen: int) -> None:
        self._dq: deque[T] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.total = 0
        self.overflow = 0

    def append(self, s: T) -> None:
        with self._lock:
            if len(self._dq) == self._dq.maxlen:
                self.overflow += 1
            self._dq.append(s)
            self.total += 1

    def latest(self) -> T | None:
        with self._lock:
            return self._dq[-1] if self._dq else None

    def drain(self) -> list[T]:
        with self._lock:
            out = list(self._dq)
            self._dq.clear()
            return out

    def snapshot(self) -> list[T]:
        """Copy without consuming -- for readers that must not steal samples from the recorder (the stillness gate)."""
        with self._lock:
            return list(self._dq)

    def __len__(self) -> int:
        with self._lock:
            return len(self._dq)


@dataclass
class DeviceStatus:
    name: str
    connected: bool = False
    running: bool = False
    rate_hz: float = 0.0
    last_sample_ns: int | None = None
    running_since_ns: int | None = None      # set when the device starts; a rate cannot be judged without it
    error: str | None = None
    detail: dict = field(default_factory=dict)

    @property
    def age_ms(self) -> float | None:
        return None if self.last_sample_ns is None else (now_ns() - self.last_sample_ns) / 1e6

    @property
    def running_for_s(self) -> float:
        """How long there has been to measure. A camera reporting 0 Hz one millisecond after start() has told you
        nothing; the same reading two seconds later has told you everything."""
        return 0.0 if self.running_since_ns is None else (now_ns() - self.running_since_ns) / 1e9


class RateMeter:
    def __init__(self, window: int = 64) -> None:
        self._t: deque[int] = deque(maxlen=window)

    def tick(self, t_ns: int) -> None:
        self._t.append(t_ns)

    def hz(self) -> float:
        if len(self._t) < 2:
            return 0.0
        span = (self._t[-1] - self._t[0]) / 1e9
        return (len(self._t) - 1) / span if span > 0 else 0.0


class Device(Protocol):
    name: str
    required: bool

    def open(self) -> None: ...          # connect (may raise)
    def start(self) -> None: ...         # begin streaming into the buffer
    def stop(self) -> None: ...
    def close(self) -> None: ...
    def status(self) -> DeviceStatus: ...


def resolve_serial_port(glob_pattern: str, *, serial_number: str | None = None, exclude: Iterable[str] = ()) -> str | None:
    """Pick a serial device by USB serial number (preferred) or by glob; returns None when nothing matches."""
    import glob as _glob
    try:
        from serial.tools import list_ports
        ports = list(list_ports.comports())
    except Exception:
        ports = []
    if serial_number:
        for p in ports:
            if p.serial_number and serial_number in p.serial_number:
                return p.device
        return None
    cands = [c for c in sorted(_glob.glob(glob_pattern)) if c not in set(exclude)]
    return cands[0] if cands else None
