"""Shared slow joint-space moves for the calibration / replay tools (2026-10-20), same pattern as ready_pose: HttpBridge.send only
(never a hand POST), smoothstep ramp at <= deg_s, E-STOP check every second, unpark via /torque before the first move."""
from __future__ import annotations
import subprocess
import sys
import time
import numpy as np

READY_Q_DEG = {"right": [1.8, -56.8, -32.5, -19.9, -1.3, 5.8]}       # = ready_pose.READY_Q_DEG (measured 10-15)


def say(t: str) -> None:
    print("[로봇]", t, flush=True)
    try: subprocess.Popen(["say", "-v", "Yuna", t])
    except Exception: pass


def unpark(br, s: str) -> None:
    st = br.status()
    if s in (st.get("parked") or []):
        r = br.s.post(br.base_url + "/torque", json={"arm": s, "on": True}, timeout=5.0).json(); print("토크 켬:", r.get("arms")); time.sleep(0.5)
    br.timeout_s = max(br.timeout_s, 0.5)


def ramp_to(br, s: str, qr, deg_s: float = 15.0, hz: float = 30.0, g=None) -> None:
    """Joint ramp from the observed q to qr (rad); the other arm and both jaws are held at their observed values."""
    ob = br.observe()
    if ob.estop: sys.exit("E-STOP/미연결/읽기 지연 -- 정지")
    q0 = ob.q[s].copy(); d = np.asarray(qr) - q0; other = "left" if s == "right" else "right"
    g = g or {x: ob.g(x) for x in ("left", "right")}
    n = max(2, int(np.ceil(np.degrees(np.abs(d)).max() / deg_s * hz))); t0 = time.monotonic()
    for k in range(1, n + 1):
        u = k / n; u = u * u * (3 - 2 * u)
        r = br.send({s: q0 + u * d, other: ob.q[other]}, g)
        if not r.accepted: br.estop(); sys.exit(f"브리지가 명령을 거부: {r.error} -- E-STOP 보냄")
        if k % int(hz) == 0 and br.observe().estop: br.estop(); sys.exit("이동 중 E-STOP/읽기 지연 -- 정지")
        time.sleep(max(0.0, t0 + k / hz - time.monotonic()))


def move_secs(q0, qr, deg_s: float) -> float:
    return float(np.degrees(np.abs(np.asarray(qr) - np.asarray(q0))).max() / deg_s)
