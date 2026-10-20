"""Localhost UDP link between the three teleop processes (2026-10-14). Why three: the student fingertip model needs ultralytics 8.4
(~/finger_distill/.venv), the Orbbec SDK + MediaPipe + IMU serial live in ego_collector/.venv (Orbbec needs sudo), pyroki IK lives
in ~/xvla-mac. All three stamp with time.monotonic_ns() (system-wide mach clock on macOS), so timestamps compare directly.

Latest-value semantics: the control loop never waits for a message, it reads the newest per (kind, side) and checks its age
(spec 28). JSON datagrams, one message per datagram, < 8 kB."""
from __future__ import annotations
import json
import socket
import threading
import time
import numpy as np

HOST = "127.0.0.1"


def _enc(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.bool_): return bool(o)
    raise TypeError(type(o))


class Publisher:
    def __init__(self, port: int) -> None:
        self.addr = (HOST, int(port)); self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, kind: str, side: str | None = None, **payload) -> None:
        msg = dict(kind=kind, side=side, t_pub_ns=time.monotonic_ns(), **payload)
        try: self.s.sendto(json.dumps(msg, default=_enc).encode(), self.addr)
        except OSError: pass


class Latest:
    """Receives on `port` in a thread; keeps the newest message per (kind, side) and an optional queue of commands."""
    def __init__(self, port: int, *, keep_kinds: tuple = ()) -> None:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.s.bind((HOST, int(port))); self.s.settimeout(0.1)
        self._d: dict = {}; self._q: list = []; self.keep = set(keep_kinds); self._lock = threading.Lock(); self._stop = threading.Event()
        self.n = 0; self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try: data, _ = self.s.recvfrom(65535)
            except socket.timeout: continue
            except OSError: break
            try: m = json.loads(data)
            except ValueError: continue
            m["t_recv_ns"] = time.monotonic_ns()
            with self._lock:
                self._d[(m.get("kind"), m.get("side"))] = m; self.n += 1
                if m.get("kind") in self.keep: self._q.append(m)

    def get(self, kind: str, side: str | None = None, max_age_ms: float | None = None) -> dict | None:
        with self._lock: m = self._d.get((kind, side))
        if m is None: return None
        if max_age_ms is not None and (time.monotonic_ns() - m["t_recv_ns"]) / 1e6 > max_age_ms: return None
        return m

    def pop_all(self) -> list:
        with self._lock: q, self._q = self._q, []
        return q

    def close(self) -> None: self._stop.set(); self.s.close()
