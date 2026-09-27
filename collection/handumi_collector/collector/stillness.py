"""VIO init readiness gate: REC is allowed only after every IMU has been still for `min_still_s` (2026-09-16).

Why: OpenVINS's static initializer needs a still window BEFORE the first motion. The bimanual pilot episodes started
already moving (first second ~63 deg/s) and initialization took 4-12 s of a 25 s episode -> coverage 0.44. With a 3 s
still start the same pipeline initialized in 1.6-3.7 s and covered 0.99 of the rest. The old "HOME READY -> 3 s timer"
only waited; it never checked. This checks.

Stillness per IMU, on a rolling window of the newest `window_s` of samples:
    mean |gyro|      < gyro_max_deg_s
    std  |accel|     < accel_std_max          (norm, so orientation does not matter)
    max sample gap   < max_gap_ms             (a stalled stream is not "still", it is missing)
    enough samples for the window at the configured rate
Every window that passes extends the run; any window that fails resets it. Ready when every side's run >= min_still_s.

Thresholds come from the DIAG_T30 takes, not from taste: over 124 true-still 0.25 s windows the gyro mean was p99 3.6 /
max 5.5 deg/s and accel std p99 0.16 / max 0.18 m/s^2 (hand tremor while "holding still"); 6 deg/s + 0.35 passes 100 %
of them. Deliberate slow 4 cm/s moves pass 17 % of single windows, which the 3 s consecutive requirement makes irrelevant.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

DEFAULTS = dict(enabled=True, min_still_s=3.0, window_s=0.25, gyro_max_deg_s=6.0, accel_std_max=0.35, max_gap_ms=25.0,
                min_window_fill=0.6, countdown_s=0.0, hold_after_rec_s=3.0, hold_gyro_max_deg_s=12.0)
# hold_gyro_max_deg_s: the post-REC hold is JUDGED (logged as moved_before_go), not gated. On the first AUTO dry run every
#   'moved' flag was the left hand at 6.2-8.0 deg/s -- settling tremor just above the 6 deg/s gate bound, harmless to the
#   initializer (accelerometer variance and image disparity, not two degrees of drift). 12 keeps the flag for real motion.
# countdown_s: READY -> REC delay (0: record the moment the gate opens -- every second of stillness spent before REC is a
#   second the recording does not contain). hold_after_rec_s: how long the operator must KEEP still after REC starts, so the
#   still window OpenVINS initializes on is inside the data. Stacking episodes 3-4 on 2026-09-16 were started 4 s after a
#   verified still window and never initialized: the operator moved at REC and the recording held no stillness at all.


@dataclass
class SideState:
    run_start_ns: int | None = None       # when the current still run began (host ns); None = not still now
    still_s: float = 0.0                  # length of the current run
    last_eval_ns: int | None = None
    last_window: dict = field(default_factory=dict)   # the numbers behind the latest verdict, for the UI / metadata
    ready_at_ns: int | None = None        # first time still_s reached min_still_s in this arming


class StillnessGate:
    def __init__(self, cfg: dict | None = None, sides: tuple[str, ...] = ("left", "right"), rate_hz: float = 200.0) -> None:
        c = dict(DEFAULTS); c.update(cfg or {}); self.cfg = c
        self.sides = tuple(sides); self.rate_hz = float(rate_hz)
        self.state: dict[str, SideState] = {s: SideState() for s in self.sides}
        self.armed_ns: int | None = None

    # ------------------------------------------------------------------ lifecycle
    def arm(self, now_ns: int) -> None:
        self.armed_ns = now_ns; self.state = {s: SideState() for s in self.sides}

    def disarm(self) -> None:
        self.armed_ns = None

    # ------------------------------------------------------------------ evaluation
    def evaluate_window(self, samples) -> tuple[bool, dict]:
        """Verdict for one window of ImuSample (newest `window_s`). Pure; used by update() and by tests."""
        c = self.cfg; n = len(samples)
        need = int(self.rate_hz * float(c["window_s"]) * float(c["min_window_fill"]))
        if n < max(need, 2): return False, dict(n=n, reason=f"only {n} samples (need {max(need, 2)})")
        g = [math.sqrt(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz) for s in samples]
        a = [math.sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az) for s in samples]
        gyro_mean = math.degrees(sum(g) / n)
        am = sum(a) / n; accel_std = math.sqrt(sum((x - am) ** 2 for x in a) / n)
        ts = [s.host_receive_ns for s in samples]
        gap_ms = max((ts[i + 1] - ts[i]) for i in range(n - 1)) / 1e6
        w = dict(n=n, gyro_mean_deg_s=round(gyro_mean, 2), accel_std=round(accel_std, 3), max_gap_ms=round(gap_ms, 1))
        if gap_ms > float(c["max_gap_ms"]): w["reason"] = f"gap {gap_ms:.0f} ms"; return False, w
        if gyro_mean >= float(c["gyro_max_deg_s"]): w["reason"] = f"gyro {gyro_mean:.1f} deg/s"; return False, w
        if accel_std >= float(c["accel_std_max"]): w["reason"] = f"accel std {accel_std:.2f}"; return False, w
        return True, w

    def update(self, side: str, samples, now_ns: int) -> SideState:
        """Feed the newest samples of one side (any length; only the last window_s is judged)."""
        st = self.state[side]
        if self.armed_ns is None: return st
        win_ns = int(float(self.cfg["window_s"]) * 1e9)
        recent = [s for s in samples if s.host_receive_ns >= now_ns - win_ns]
        ok, w = self.evaluate_window(recent)
        st.last_eval_ns = now_ns; st.last_window = w
        if ok:
            if st.run_start_ns is None: st.run_start_ns = max(self.armed_ns, now_ns - win_ns)
            st.still_s = (now_ns - st.run_start_ns) / 1e9
            if st.ready_at_ns is None and st.still_s >= float(self.cfg["min_still_s"]): st.ready_at_ns = now_ns
        else:
            st.run_start_ns = None; st.still_s = 0.0; st.ready_at_ns = None
        return st

    # ------------------------------------------------------------------ verdict
    @property
    def min_still_s(self) -> float: return float(self.cfg["min_still_s"])

    def progress_s(self) -> float:
        """The side that is furthest behind decides."""
        return min((self.state[s].still_s for s in self.sides), default=0.0)

    @property
    def ready(self) -> bool:
        return self.armed_ns is not None and bool(self.sides) and all(self.state[s].still_s >= self.min_still_s for s in self.sides)

    def label(self) -> str:
        if not self.ready:
            p = self.progress_s()
            if p <= 0: return "WAIT STILL"
            return f"STILL {p:.1f} / {self.min_still_s:.1f} s"
        return "VIO INIT READY"

    def to_meta(self, *, started_ns: int | None = None) -> dict:
        """What the episode records about how it was started."""
        c = self.cfg
        out = dict(enabled=True, min_still_s=float(c["min_still_s"]), window_s=float(c["window_s"]), gyro_max_deg_s=float(c["gyro_max_deg_s"]),
                   accel_std_max=float(c["accel_std_max"]), max_gap_ms=float(c["max_gap_ms"]), countdown_s=float(c["countdown_s"]),
                   hold_after_rec_s=float(c.get("hold_after_rec_s", 0.0)), satisfied=self.ready, sides={})
        for s in self.sides:
            st = self.state[s]
            out["sides"][s] = dict(still_s=round(st.still_s, 2), ready=st.still_s >= self.min_still_s, last_window=st.last_window,
                                   ready_to_start_s=None if (st.ready_at_ns is None or started_ns is None) else round((started_ns - st.ready_at_ns) / 1e9, 2))
        return out
