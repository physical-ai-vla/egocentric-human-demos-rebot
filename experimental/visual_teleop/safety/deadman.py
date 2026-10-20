"""Deadman + E-STOP latch (2026-10-14, spec 24, 36).

Deadman: the robot may follow ONLY while pressed. HeartbeatDeadman is fed by the operator UI while SPACE is held (keydown
repeats + explicit release); no heartbeat for timeout_ms = released (a crashed browser tab or lost network is a release).
Live mode refuses AlwaysPressed.

E-STOP latch: any source (UI button, hardware via the bridge /status estop flag, an internal fault) latches. It supersedes
deadman, clutch, policy and teleop. It never clears itself: reset() requires that every source is clear AND an explicit
operator call, after which the controller must re-initialise (re-anchor) before moving."""
from __future__ import annotations
import threading
import time
from typing import Protocol
from ego_teleop.robot.deadman import ScriptedDeadman  # noqa: F401  (re-export, tests/replay)


class Deadman(Protocol):
    def pressed(self, now_ns: int) -> bool: ...


class AlwaysPressed:
    live_allowed = False
    def pressed(self, now_ns: int) -> bool: return True


class HeartbeatDeadman:
    live_allowed = True

    def __init__(self, timeout_ms: float = 150.0) -> None:
        self.timeout_ns = int(timeout_ms * 1e6); self._last: int | None = None; self._released = True; self._lock = threading.Lock()

    def heartbeat(self, now_ns: int | None = None) -> None:
        with self._lock: self._last = int(now_ns if now_ns is not None else time.monotonic_ns()); self._released = False

    def release(self) -> None:
        with self._lock: self._released = True

    def pressed(self, now_ns: int) -> bool:
        with self._lock:
            return (not self._released) and self._last is not None and 0 <= now_ns - self._last <= self.timeout_ns


class EStopLatch:
    def __init__(self) -> None:
        self.latched = False; self.sources: set[str] = set(); self.history: list[tuple[int, str, str]] = []; self._lock = threading.Lock()

    def trip(self, source: str, now_ns: int | None = None) -> None:
        with self._lock:
            t = int(now_ns if now_ns is not None else time.monotonic_ns())
            if source not in self.sources: self.history.append((t, "trip", source))
            self.sources.add(source); self.latched = True

    def source_clear(self, source: str) -> None:
        """A source reports it is no longer active (e.g. hardware button released). Does NOT unlatch."""
        with self._lock: self.sources.discard(source)

    def reset(self, now_ns: int | None = None) -> tuple[bool, str]:
        with self._lock:
            if self.sources: return False, f"E-STOP sources still active: {sorted(self.sources)}"
            self.latched = False; self.history.append((int(now_ns if now_ns is not None else time.monotonic_ns()), "reset", "operator"))
            return True, ""
