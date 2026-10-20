"""Teleop clutch (2026-10-14, spec 25). ON: robot HOLD, human motion ignored, references follow the human continuously.
OFF: resume from a fresh relative reference (the controller re-anchors at the current human pose and the last EXECUTED robot
pose), so release can never jump. The controller owns the anchoring; this object only holds the operator intent + edges."""
from __future__ import annotations
import threading


class Clutch:
    def __init__(self) -> None: self._on = False; self._edge_release = False; self._lock = threading.Lock()

    def set(self, on: bool) -> None:
        with self._lock:
            if self._on and not on: self._edge_release = True
            self._on = bool(on)

    @property
    def on(self) -> bool: return self._on

    def take_release_edge(self) -> bool:
        with self._lock: e = self._edge_release; self._edge_release = False; return e
