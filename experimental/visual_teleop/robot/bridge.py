"""Robot bridge: ONE writer, ONE 14-vector per command cycle for both arms + both jaws (2026-10-14, spec 22).

Units/frames are converted ONLY by ego_teleop.robot.bimanual.build_bimanual_action (follower rad -> leader frame, g -> jaw cmd
0..45) and parsed back here (/observe follower DEGREES -> rad). Never POST /execute_step by hand (memory
rebot-arm-port-location-mapping: a follower-frame vector drove both arms to their limits).

    HttpBridge(live=False)  real /observe, NO POST: send() returns accepted=True, sent=False  (preview / Stage 0 against the robot)
    HttpBridge(live=True)   POST /execute_step; accepted only on HTTP 200 + {"ok": true} (MIT bridge contract)
    MockBridge              in-process robot for tests / offline replay; can be told to reject

A command is "accepted" only per the bridge reply; the controller commits state only then (spec 27)."""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
import numpy as np
from ego_teleop.robot.bimanual import build_bimanual_action
from ego_teleop.robot.rebot_client import GRIPPER_INDEX, N_ACTION, SIDE_SLICE, g_from_raw, GRIP_OBS_PER_CMD

_LIVE_URLS: set[str] = set()


@dataclass
class RobotObs:
    t_ns: int
    q: dict                      # side -> (6,) follower rad
    grip_raw: dict               # side -> raw jaw reading (0 closed .. -270 open, may wrap)
    joints_deg_all: list
    estop: bool = False
    latency_ms: float = 0.0

    def g(self, side: str) -> float: return g_from_raw(self.grip_raw[side])


@dataclass
class SendResult:
    accepted: bool
    sent: bool
    t_ns: int
    latency_ms: float
    error: str = ""
    action: list | None = None


def parse_observe(d: dict, t_ns: int, latency_ms: float = 0.0, estop: bool = False) -> RobotObs:
    j = [float(v) for v in d["joints_deg"]]
    if len(j) != N_ACTION: raise RuntimeError(f"/observe returned {len(j)} joints, expected {N_ACTION}")
    q = {s: np.radians(np.asarray(j[sl], np.float64)) for s, sl in SIDE_SLICE.items()}
    return RobotObs(t_ns, q, {s: j[i] for s, i in GRIPPER_INDEX.items()}, j, estop, latency_ms)


class HttpBridge:
    def __init__(self, base_url: str, *, live: bool = False, timeout_s: float = 0.2, session=None) -> None:
        import requests
        self.base_url = base_url.rstrip("/"); self.live = bool(live); self.timeout_s = timeout_s
        self.s = session or requests.Session()
        if self.live:
            if self.base_url in _LIVE_URLS: raise RuntimeError(f"a live HttpBridge for {self.base_url} already exists (single writer)")
            _LIVE_URLS.add(self.base_url)
        self.n_sent = 0; self.max_read_age_ms = 500.0; self.last_read_age_ms = None

    def close(self) -> None: _LIVE_URLS.discard(self.base_url)

    def status(self) -> dict: return self.s.get(self.base_url + "/status", timeout=self.timeout_s).json()

    def observe(self) -> RobotObs:
        t0 = time.monotonic_ns()
        d = self.s.get(self.base_url + "/observe", timeout=self.timeout_s * 3).json()
        if "joints_deg" not in d: raise RuntimeError(f"/observe: {d.get('error', d)}")
        st = self.status()
        t1 = time.monotonic_ns()
        # [10-15] the MIT bridge kept answering "connected" and accepting /execute_step while its serial port was gone (Broken
        # pipe, arm state 6293 s old): /observe then serves cached joints. A stale arm read is treated as E-STOP here.
        ages = [float(a.get("read_age_ms", 0) or 0) for a in (st.get("arms") or {}).values()]
        self.last_read_age_ms = max(ages) if ages else None
        stale = bool(ages) and max(ages) > self.max_read_age_ms
        return parse_observe(d, t1, (t1 - t0) / 1e6, bool(st.get("estop")) or not st.get("connected", False) or stale)

    def send(self, q: dict, g: dict) -> SendResult:
        t0 = time.monotonic_ns()
        try: act = build_bimanual_action(q["left"], q["right"], g["left"], g["right"])
        except Exception as e: return SendResult(False, False, t0, 0.0, f"build:{e}")
        if not self.live: return SendResult(True, False, time.monotonic_ns(), 0.0, "", act)
        try:
            r = self.s.post(self.base_url + "/execute_step", json={"action": act}, timeout=self.timeout_s)
            ok = r.status_code == 200 and bool(r.json().get("ok")); err = "" if ok else f"http{r.status_code}:{r.text[:160]}"
        except Exception as e: ok, err = False, f"{type(e).__name__}:{e}"
        t1 = time.monotonic_ns(); self.n_sent += 1
        return SendResult(ok, True, t1, (t1 - t0) / 1e6, err, act)

    def estop(self) -> None:
        try: self.s.post(self.base_url + "/estop", timeout=self.timeout_s)
        except Exception: pass


@dataclass
class MockBridge:
    """Instant robot: the executed joints are the last accepted command. `reject_next` / `reject_fn(q, g)` simulate refusals."""
    q: dict = field(default_factory=lambda: {"left": np.zeros(6), "right": np.zeros(6)})
    g: dict = field(default_factory=lambda: {"left": 1.0, "right": 1.0})
    estop_flag: bool = False
    reject_next: int = 0
    reject_fn: object = None
    sent: list = field(default_factory=list)
    n_estop_calls: int = 0
    live = True
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self) -> RobotObs:
        j = [0.0] * N_ACTION
        for s, sl in SIDE_SLICE.items(): j[sl] = list(np.degrees(self.q[s]))
        for s, i in GRIPPER_INDEX.items(): j[i] = self.g[s] * 45.0 * GRIP_OBS_PER_CMD
        return parse_observe({"joints_deg": j}, time.monotonic_ns(), 0.0, self.estop_flag)

    def send(self, q: dict, g: dict) -> SendResult:
        act = build_bimanual_action(q["left"], q["right"], g["left"], g["right"]); t = time.monotonic_ns()
        if self.estop_flag: return SendResult(False, True, t, 0.0, "estop", act)
        if self.reject_next > 0 or (self.reject_fn is not None and self.reject_fn(q, g)):
            self.reject_next = max(0, self.reject_next - 1); return SendResult(False, True, t, 0.0, "mock_reject", act)
        self.sent.append(act); self.q = {s: np.asarray(v, np.float64).copy() for s, v in q.items()}; self.g = dict(g)
        return SendResult(True, True, t, 0.0, "", act)

    def estop(self) -> None: self.n_estop_calls += 1
