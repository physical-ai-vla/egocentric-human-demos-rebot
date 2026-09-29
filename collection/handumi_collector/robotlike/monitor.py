"""Live robot-likeness monitor for the robot_like_v1 collection protocol (2026-09-28).

While an ego demonstration is recorded, tell the operator -- live -- whether the wrist motion looks like the reBot's,
and summarise each episode. Only signals that are trustworthy LIVE are used: there is no reliable live metric pose on
this Mac (OpenVINS diverges on a waiting hand, mast3r_live has never run on a GPU, the Orbbec cannot share the host
with the wrist cameras), so mapped robot TCP / IK / joint margins are the OFFLINE check (robotlike/offline_check.py).

    wrist angular speed |w| and angular accel |dw|*15   gyro, 15 Hz grid       -> vs R150 p95 (probe_v2_motion.py)
    linear-accel proxy |a - g_lowpass|                   accelerometer          -> advisory only
    grasp / release events, per-arm stage                normalised jaw, hysteresis

Raw acquisition is never touched: IMU samples arrive through a fan-out tap on the device buffer (the recorder keeps
draining the original stream), the work runs on its own thread, and every failure is swallowed into `self.error`."""
from __future__ import annotations
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml
from ..devices.base import SampleBuffer, now_ns

log = logging.getLogger("handumi.robotlike")
PROTOCOL_DIR = Path(__file__).resolve().parents[2] / "configs" / "handumi"


def load_protocol(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.suffix: p = PROTOCOL_DIR / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(p.read_text()) or {}
    cfg.setdefault("protocol", p.stem); cfg["_path"] = str(p)
    return cfg


class _Fanout(SampleBuffer):
    """Same idea as ego_teleop.tracking.imu_tap.FanoutBuffer: append() also feeds every tap; drain() is untouched."""

    def __init__(self, primary: SampleBuffer) -> None:
        super().__init__(primary._dq.maxlen)
        with primary._lock:
            for s in primary._dq: self._dq.append(s)
            self.total, self.overflow = primary.total, primary.overflow
        self.taps: list[SampleBuffer] = []

    def add_tap(self, buffer_len: int) -> SampleBuffer:
        t = SampleBuffer(buffer_len); self.taps.append(t); return t

    def append(self, s) -> None:
        super().append(s)
        for t in self.taps: t.append(s)


def install_tap(device, buffer_s: float = 2.0) -> SampleBuffer:
    """Tap buffer receiving every sample the device publishes. Reuses an existing fan-out (e.g. ego_teleop's) if present."""
    if not hasattr(device.buffer, "add_tap"): device.buffer = _Fanout(device.buffer)
    rate = getattr(getattr(device, "cfg", None), "rate_hz", 400) or 400
    return device.buffer.add_tap(int(rate * buffer_s))


def lamp(value: float | None, p95: float, amber_x: float) -> str:
    if value is None or value != value: return "GREY"
    return "GREEN" if value <= p95 else "YELLOW" if value <= amber_x * p95 else "RED"


@dataclass
class _Side:
    side: str
    tap: SampleBuffer | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=4000))
    g_lp: np.ndarray | None = None
    w_prev: np.ndarray | None = None
    t_prev_ns: int | None = None
    w: float | None = None
    a: float | None = None
    lin: float | None = None
    grip: float | None = None
    grip_state: str | None = None           # "open" | "closed"
    w_hist: deque = field(default_factory=lambda: deque(maxlen=12))    # ~0.8 s of |w| at 15 Hz, for calm-transition test
    # per-episode accumulators
    ws: list = field(default_factory=list)
    as_: list = field(default_factory=list)
    lins: list = field(default_factory=list)
    grasps: int = 0
    releases: int = 0
    calm: int = 0

    def reset_episode(self) -> None:
        self.ws, self.as_, self.lins = [], [], []; self.grasps = self.releases = self.calm = 0


class RobotLikeMonitor:
    def __init__(self, cfg: dict, devices) -> None:
        self.cfg = cfg; self.devices = devices; self.fps = float(cfg.get("fps", 15.0))
        self.sides: dict[str, _Side] = {}
        for side in ("left", "right"):
            st = _Side(side)
            imu = devices.imus.get(side) if devices is not None else None
            if imu is not None:
                try: st.tap = install_tap(imu)
                except Exception as exc: log.warning("robot-like: IMU tap on %s failed: %s", side, exc)
            self.sides[side] = st
        self.recording = False; self.episode_dir: Path | None = None; self.t_begin_ns: int | None = None
        self.error: str | None = None; self.last_summary: dict | None = None
        self._lock = threading.Lock(); self._stop = threading.Event(); self._t: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._t is None:
            self._stop.clear(); self._t = threading.Thread(target=self._run, name="robotlike", daemon=True); self._t.start()

    def close(self) -> None:
        self._stop.set()
        if self._t: self._t.join(1.0); self._t = None

    def begin_episode(self, episode_dir: Path) -> None:
        with self._lock:
            for st in self.sides.values(): st.reset_episode()
            self.episode_dir = Path(episode_dir); self.t_begin_ns = now_ns(); self.recording = True

    def end_episode(self) -> dict:
        with self._lock:
            self.recording = False; summ = self._summary()
        self.last_summary = summ
        if self.episode_dir is not None:
            try:
                out = self.episode_dir / "derived" / "robot_like"; out.mkdir(parents=True, exist_ok=True)
                (out / "live_summary.json").write_text(json.dumps(summ, indent=1))
            except Exception as exc: self.error = f"write live_summary: {exc}"
        return summ

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        period = 1.0 / self.fps; nxt = time.monotonic()
        while not self._stop.is_set():
            try: self.tick(now_ns())
            except Exception as exc:                          # never let the monitor take anything down with it
                self.error = f"{type(exc).__name__}: {exc}"; log.debug("robot-like tick failed", exc_info=True)
            nxt += period; time.sleep(max(0.0, nxt - time.monotonic()))

    def tick(self, t_ns: int) -> None:
        """One 15 Hz step. Public so tests can drive it with synthetic samples and a synthetic clock."""
        c = self.cfg; win_ns = int(2e9 / (2 * self.fps))          # 2/30 s trailing window == probe's +-1/30 s, one tick late
        with self._lock:
            for side, st in self.sides.items():
                if st.tap is not None:
                    for s in st.tap.drain(): st.recent.append(s)
                while st.recent and st.recent[0].host_receive_ns < t_ns - 4 * win_ns: st.recent.popleft()
                win = [s for s in st.recent if t_ns - win_ns < s.host_receive_ns <= t_ns]
                if win:
                    w = np.array([[s.gx, s.gy, s.gz] for s in win]).mean(0)
                    acc = np.array([[s.ax, s.ay, s.az] for s in win]).mean(0)
                    dt = 1.0 / self.fps if st.t_prev_ns is None else max((t_ns - st.t_prev_ns) / 1e9, 1e-3)
                    k = dt / (float(c.get("gravity_tau_s", 0.5)) + dt)
                    st.g_lp = acc.copy() if st.g_lp is None else (1 - k) * st.g_lp + k * acc
                    st.w = float(np.linalg.norm(w)); st.lin = float(np.linalg.norm(acc - st.g_lp))
                    # |dw|*fps only across one nominal tick; a gap (no samples) must not read as a huge acceleration
                    st.a = float(np.linalg.norm(w - st.w_prev) * self.fps) if (st.w_prev is not None and dt < 2.5 / self.fps) else None
                    st.w_prev, st.t_prev_ns = w, t_ns; st.w_hist.append(st.w)
                    if self.recording:
                        st.ws.append(st.w); st.lins.append(st.lin)
                        if st.a is not None: st.as_.append(st.a)
                else:
                    st.w = st.a = st.lin = None; st.w_prev = None
                self._grip(side, st)

    def _grip(self, side: str, st: _Side) -> None:
        gr = self.devices.grippers.get(side) if self.devices is not None else None
        if gr is None: st.grip = None; return
        try: n = gr.quality().normalized
        except Exception: n = None
        if n is None or n != n: st.grip = None; return
        st.grip = float(n); lo, hi = float(self.cfg.get("grip_closed_below", 0.35)), float(self.cfg.get("grip_open_above", 0.65))
        if st.grip_state is None: st.grip_state = "closed" if n < 0.5 else "open"; return
        event = None
        if st.grip_state == "open" and n < lo: st.grip_state = "closed"; event = "grasp"
        elif st.grip_state == "closed" and n > hi: st.grip_state = "open"; event = "release"
        if event and self.recording:
            if event == "grasp": st.grasps += 1
            else: st.releases += 1
            # a staged transition = the wrist was calm just before it (robot pauses at grasp / release)
            if st.w_hist and float(np.median(list(st.w_hist)[-5:])) <= 0.3 * float(self.cfg["ang_w_p95"]): st.calm += 1

    # ------------------------------------------------------------------ outputs
    def snapshot(self) -> dict:
        c = self.cfg; ax = float(c.get("amber_x", 1.5))
        with self._lock:
            out = dict(recording=self.recording, error=self.error, sides={})
            for side, st in self.sides.items():
                stage = None if st.grip_state is None else ("HOLD / TRANSPORT" if st.grip_state == "closed" else "OPEN / APPROACH")
                out["sides"][side] = dict(
                    imu=st.tap is not None, w=st.w, a=st.a, lin=st.lin, grip=st.grip, stage=stage,
                    w_lamp=lamp(st.w, float(c["ang_w_p95"]), ax), a_lamp=lamp(st.a, float(c["ang_a_p95"]), ax),
                    lin_lamp=lamp(st.lin, float(c.get("lin_a_proxy_p95", 1.157)), ax),
                    grasps=st.grasps, releases=st.releases,
                    over_w=_frac_over(st.ws, float(c["ang_w_p95"])), over_a=_frac_over(st.as_, float(c["ang_a_p95"])))
        return out

    def _summary(self) -> dict:
        c = self.cfg; mx = float(c.get("max_over_frac", 0.10)); reasons = []; sides = {}
        for side, st in self.sides.items():
            d = dict(n_ticks=len(st.ws), ang_w=_q(st.ws), ang_a=_q(st.as_), lin_a_proxy=_q(st.lins),
                     over_frac_ang_w=_frac_over(st.ws, float(c["ang_w_p95"])), over_frac_ang_a=_frac_over(st.as_, float(c["ang_a_p95"])),
                     grasps=st.grasps, releases=st.releases, calm_transitions=st.calm)
            sides[side] = d
            if st.tap is None: reasons.append(f"{side}: no IMU (not judged)"); continue
            if d["n_ticks"] == 0: reasons.append(f"{side}: no IMU samples while recording"); continue
            if d["over_frac_ang_w"] > mx: reasons.append(f"{side} wrist rotation fast {d['over_frac_ang_w']:.0%} of the time (> R150 p95 {c['ang_w_p95']} rad/s)")
            if d["over_frac_ang_a"] > mx: reasons.append(f"{side} jerky rotation {d['over_frac_ang_a']:.0%} of the time (> R150 p95 {c['ang_a_p95']} rad/s^2)")
        grasps = sum(d["grasps"] for d in sides.values())
        if grasps < int(c.get("min_grasps_per_episode", 0)): reasons.append(f"only {grasps} grasp(s) (expected >= {c.get('min_grasps_per_episode')})")
        judged = [r for r in reasons if "not judged" not in r]
        return dict(schema="robot_like_live/v1", protocol=c.get("protocol"), reference=c.get("reference"),
                    thresholds=dict(ang_w_p95=c["ang_w_p95"], ang_a_p95=c["ang_a_p95"], lin_a_proxy_p95=c.get("lin_a_proxy_p95"),
                                    max_over_frac=mx, min_grasps_per_episode=c.get("min_grasps_per_episode")),
                    duration_s=None if self.t_begin_ns is None else round((now_ns() - self.t_begin_ns) / 1e9, 2),
                    sides=sides, grasps_total=grasps, verdict="PASS" if not judged else "FAIL", reasons=reasons,
                    note="live IMU/gripper proxy only; mapped robot TCP / IK / joint margins = offline check")


def _q(x) -> dict | None:
    if not x: return None
    a = np.asarray(x, float); return dict(p50=round(float(np.percentile(a, 50)), 4), p95=round(float(np.percentile(a, 95)), 4), max=round(float(a.max()), 4))


def _frac_over(x, th: float) -> float:
    return 0.0 if not x else round(float(np.mean(np.asarray(x, float) > th)), 4)
