"""Session clock anchor + gap detection. All raw timestamps are host monotonic ns; the anchor lets offline code
convert to wall time. Device clocks (Teensy micros) are NOT corrected here — raw is raw; sync is offline."""
from __future__ import annotations
import time
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ClockAnchor:
    monotonic_ns: int
    wall_ns: int
    perf_counter_ns: int

    @classmethod
    def now(cls) -> "ClockAnchor":
        return cls(time.monotonic_ns(), time.time_ns(), time.perf_counter_ns())

    def to_wall_ns(self, monotonic_ns: int) -> int:
        return self.wall_ns + (monotonic_ns - self.monotonic_ns)

    def to_dict(self) -> dict:
        return asdict(self)


class GapDetector:
    """Flags inter-sample gaps above a threshold (per stream). Returns the gap in ms when it fires."""

    def __init__(self, max_gap_ms: float) -> None:
        self.max_gap_ns = int(max_gap_ms * 1e6)
        self._last: int | None = None
        self.count = 0
        self.worst_ms = 0.0

    def update(self, t_ns: int) -> float | None:
        gap = None
        if self._last is not None:
            d = t_ns - self._last
            if d > self.max_gap_ns:
                gap = d / 1e6; self.count += 1; self.worst_ms = max(self.worst_ms, gap)
            if d < 0:
                gap = d / 1e6; self.count += 1   # non-monotonic = also an event
        self._last = t_ns
        return gap

    def reset(self) -> None:
        self._last = None; self.count = 0; self.worst_ms = 0.0


def index_gaps(prev_index: int | None, index: int) -> int:
    """Number of capture indices skipped between two consecutive frames (0 = none)."""
    return 0 if prev_index is None else max(0, index - prev_index - 1)
