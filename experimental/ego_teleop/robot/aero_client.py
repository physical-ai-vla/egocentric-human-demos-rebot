"""Aero Hand Open branch: canonical compact-7 targets (deg) -> official aero_open_sdk (invariant G: the SDK is the
hardware boundary; no servo protocol here). Telemetry is read through the SDK too (actuations, currents, temperatures)."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from ..tracking.interfaces import AERO_CHANNELS


@dataclass
class AeroState:
    timestamp_ns: int
    actuations_deg: np.ndarray | None          # (7) observed motor rotations (SDK get_actuations)
    compact_deg: np.ndarray | None = None      # (7) SDK actuations->joints (compact)
    currents: np.ndarray | None = None
    temperatures_c: np.ndarray | None = None
    speeds: np.ndarray | None = None
    ok: bool = True
    error: str = ""

    def to_row(self) -> dict:
        r = dict(t_ns=int(self.timestamp_ns), ok=self.ok, error=self.error)
        for i, n in enumerate(AERO_CHANNELS):
            r[f"obs_act_{n}_deg"] = float(self.actuations_deg[i]) if self.actuations_deg is not None else np.nan
            r[f"obs_{n}_deg"] = float(self.compact_deg[i]) if self.compact_deg is not None else np.nan
            r[f"cur_{n}"] = float(self.currents[i]) if self.currents is not None else np.nan
            r[f"temp_{n}_c"] = float(self.temperatures_c[i]) if self.temperatures_c is not None else np.nan
        return r


class AeroClient(Protocol):
    def home(self) -> None: ...
    def set_compact_deg(self, compact_deg: np.ndarray) -> tuple[bool, int]:
        """-> (ok, execution_timestamp_ns)"""
    def observe(self, *, telemetry: bool = False) -> AeroState: ...
    def close(self) -> None: ...


class SdkAeroClient:
    def __init__(self, port: str | None, *, baudrate: int = 921600, homing_timeout_s: float = 175.0, telemetry_every_n: int = 10) -> None:
        """`port` is required on macOS (the SDK's auto-detect only knows /dev/serial/by-id on Linux)."""
        from aero_open_sdk.aero_hand import AeroHand      # official SDK; pip install from ~/aero-hand/aero-hand-open/sdk
        self.hand = AeroHand(port=port, baudrate=baudrate); self.homing_timeout_s = homing_timeout_s
        self.telemetry_every_n = max(1, telemetry_every_n); self._n = 0; self.homed = False

    def home(self) -> None:
        self.hand.send_homing(timeout_s=self.homing_timeout_s); self.homed = True

    def set_compact_deg(self, compact_deg) -> tuple[bool, int]:
        c = [float(v) for v in np.asarray(compact_deg, np.float64).reshape(7)]
        self.hand.set_joint_positions(c)                   # 7 -> 16 -> actuations inside the SDK (thumb coupling handled there)
        return True, time.monotonic_ns()

    def observe(self, *, telemetry: bool = False) -> AeroState:
        self._n += 1; t = time.monotonic_ns()
        act = self.hand.get_actuations()
        if act is None: return AeroState(t, None, ok=False, error="get_actuations timeout")
        compact = self.hand.get_joint_positions_compact()
        cur = temp = None
        if telemetry or self._n % self.telemetry_every_n == 0:
            cur = self.hand.get_actuator_currents(); temp = self.hand.get_actuator_temperatures()
        f = lambda v: None if v is None else np.asarray(v, np.float64)
        return AeroState(t, f(act), f(compact), f(cur), f(temp))

    def close(self) -> None: self.hand.close()


class MockAeroClient:
    """First-order lag model of the hand; records every command with its execution timestamp."""

    def __init__(self, *, lag: float = 0.6) -> None:
        self.pos = np.zeros(7); self.lag = lag; self.commands: list[tuple[int, np.ndarray]] = []; self.homed = False

    def home(self) -> None: self.homed = True; self.pos[:] = 0

    def set_compact_deg(self, compact_deg) -> tuple[bool, int]:
        c = np.asarray(compact_deg, np.float64).reshape(7); self.pos += self.lag * (c - self.pos)
        t = time.monotonic_ns(); self.commands.append((t, c.copy())); return True, t

    def observe(self, *, telemetry: bool = False) -> AeroState:
        return AeroState(time.monotonic_ns(), self.pos.copy(), self.pos.copy(), np.zeros(7), np.full(7, 30.0))

    def close(self) -> None: pass
