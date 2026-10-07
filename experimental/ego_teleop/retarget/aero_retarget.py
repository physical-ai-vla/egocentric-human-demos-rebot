"""Canonical u7 (human, [0,1]) -> Aero Hand Open compact 7-joint targets (deg) -> 7 actuations (deg).

Per channel (spec section 9): human min/max (from calibration poses), robot min/max, gain, offset, sign, deadband,
low-pass filter, velocity limit, hard clamp. Order is ALWAYS AERO_CHANNELS. Raw MediaPipe angles never reach the
hand: u7 is the normalized semantic hand state (thumb abduction / CMC flexion / curl kept separate).

Calibration poses (section 10): OPEN -> human_min for every channel; CLOSED fist -> human_max for the four fingers and
the thumb curl/flexion; PINCH -> human_max for thumb abduction (and refines thumb flexion), because abduction at a
closed fist is not the abduction a pinch needs. The mapping is stored as aero_retarget.json in the episode."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from ego_collector.hands3d.aero import compact_to_full16_deg, full16_to_actuations_deg, COMPACT_LOWER_DEG, COMPACT_UPPER_DEG
from ..tracking.interfaces import AERO_CHANNELS

FINGER_IDX = (3, 4, 5, 6); THUMB_ABD, THUMB_FLEX, THUMB_CURL = 0, 1, 2


@dataclass
class ChannelMap:
    name: str
    human_min: float = 0.0            # u value at OPEN
    human_max: float = 1.0            # u value at CLOSED / PINCH
    robot_min_deg: float = 0.0        # Aero compact joint at human_min (before sign)
    robot_max_deg: float = 90.0
    gain: float = 1.0
    offset: float = 0.0
    sign: float = 1.0                 # -1 inverts (normalized 1-n)
    deadband: float = 0.02            # normalized units; smaller changes than this are ignored
    lpf_alpha: float = 1.0            # EMA coefficient on the normalized value (1 = no filtering)
    max_rate_deg_s: float = 400.0     # velocity limit on the robot joint
    clamp_min_deg: float | None = None  # hard clamp (defaults to Aero compact joint limits)
    clamp_max_deg: float | None = None
    min_span: float = 0.15            # calibration guard: human_max-human_min below this => refuse (dead channel)

    def normalized(self, u: float, prev_n: float | None) -> float:
        span = max(self.human_max - self.human_min, 1e-6)
        n = float(np.clip((u - self.human_min) / span, 0.0, 1.0))
        if self.sign < 0: n = 1.0 - n
        n = float(np.clip(n * self.gain + self.offset, 0.0, 1.0))
        if prev_n is not None:
            if abs(n - prev_n) < self.deadband: n = prev_n
            n = prev_n + self.lpf_alpha * (n - prev_n)
        return n

    def robot_deg(self, n: float) -> float:
        d = self.robot_min_deg + n * (self.robot_max_deg - self.robot_min_deg)
        lo = COMPACT_LOWER_DEG[AERO_CHANNELS.index(self.name)] if self.clamp_min_deg is None else self.clamp_min_deg
        hi = COMPACT_UPPER_DEG[AERO_CHANNELS.index(self.name)] if self.clamp_max_deg is None else self.clamp_max_deg
        return float(np.clip(d, lo, hi))


@dataclass
class AeroCommand:
    timestamp_ns: int
    compact_deg: np.ndarray           # (7) Aero compact joints, canonical order -> AeroHand.set_joint_positions(7)
    actuations_deg: np.ndarray        # (7) motor rotations via the SDK tendon model (logged; SDK computes its own)
    normalized: np.ndarray            # (7) post-filter normalized values
    u7: np.ndarray | None             # input (None for relaxed/hold)
    source: str = "retarget"          # retarget | hold | relaxed

    def to_row(self) -> dict:
        r = dict(t_ns=int(self.timestamp_ns), source=self.source)
        for i, n in enumerate(AERO_CHANNELS):
            r[f"cmd_{n}_deg"] = float(self.compact_deg[i]); r[f"act_{n}_deg"] = float(self.actuations_deg[i]); r[f"n_{n}"] = float(self.normalized[i])
            r[f"u_{n}"] = float(self.u7[i]) if self.u7 is not None else np.nan
        return r


class AeroRetargeter:
    def __init__(self, channels: list[ChannelMap] | None = None, *, relaxed_normalized: np.ndarray | None = None) -> None:
        self.ch = channels or [ChannelMap(n, robot_max_deg=COMPACT_UPPER_DEG[i]) for i, n in enumerate(AERO_CHANNELS)]
        assert tuple(c.name for c in self.ch) == AERO_CHANNELS, "channels must be in canonical AERO_CHANNELS order"
        self.relaxed_n = np.asarray(relaxed_normalized if relaxed_normalized is not None else [0.5, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1], np.float64)
        self._prev_n: np.ndarray | None = None; self._prev_deg: np.ndarray | None = None; self._prev_t: int | None = None
        self._cal: dict[str, np.ndarray] = {}

    # ---- calibration poses ------------------------------------------------------------------------------
    def calibrate_open(self, u7) -> None: self._cal["open"] = np.asarray(u7, np.float64).reshape(7)
    def calibrate_closed(self, u7) -> None: self._cal["closed"] = np.asarray(u7, np.float64).reshape(7)
    def calibrate_pinch(self, u7) -> None: self._cal["pinch"] = np.asarray(u7, np.float64).reshape(7)

    def finalize_calibration(self) -> dict:
        if not {"open", "closed"} <= set(self._cal): raise RuntimeError("need OPEN and CLOSED poses (PINCH recommended)")
        o, c = self._cal["open"], self._cal["closed"]; p = self._cal.get("pinch")
        report = {}
        for i, ch in enumerate(self.ch):
            hi = c[i]
            if i == THUMB_ABD and p is not None: hi = p[i]
            if i == THUMB_FLEX and p is not None: hi = max(c[i], p[i])
            lo = o[i]
            if hi - lo < ch.min_span: raise RuntimeError(f"channel {ch.name}: span {hi - lo:.3f} < min_span {ch.min_span} (finger not moving / not visible)")
            ch.human_min, ch.human_max = float(lo), float(hi); report[ch.name] = dict(human_min=ch.human_min, human_max=ch.human_max)
        self.reset_filters(); return report

    def reset_filters(self) -> None: self._prev_n = None; self._prev_deg = None; self._prev_t = None

    # ---- runtime -----------------------------------------------------------------------------------------
    def _finish(self, t_ns: int, n: np.ndarray, u7, source: str) -> AeroCommand:
        deg = np.array([ch.robot_deg(n[i]) for i, ch in enumerate(self.ch)])
        if self._prev_deg is not None and self._prev_t is not None:
            dt = max((int(t_ns) - self._prev_t) / 1e9, 1e-3)
            step = np.array([ch.max_rate_deg_s * dt for ch in self.ch])
            deg = self._prev_deg + np.clip(deg - self._prev_deg, -step, step)
        self._prev_n, self._prev_deg, self._prev_t = n, deg, int(t_ns)
        act = full16_to_actuations_deg(compact_to_full16_deg(deg))
        return AeroCommand(int(t_ns), deg, act, n.copy(), None if u7 is None else np.asarray(u7, np.float64), source)

    def retarget(self, t_ns: int, u7) -> AeroCommand:
        u = np.asarray(u7, np.float64).reshape(7)
        n = np.array([ch.normalized(u[i], None if self._prev_n is None else self._prev_n[i]) for i, ch in enumerate(self.ch)])
        return self._finish(t_ns, n, u, "retarget")

    def hold(self, t_ns: int) -> AeroCommand:
        n = self._prev_n if self._prev_n is not None else self.relaxed_n
        return self._finish(t_ns, n.copy(), None, "hold")

    def relaxed(self, t_ns: int) -> AeroCommand:
        return self._finish(t_ns, self.relaxed_n.copy(), None, "relaxed")

    # ---- persistence (episode calibration/aero_retarget.json) -------------------------------------------
    def to_dict(self) -> dict:
        return dict(channels=[asdict(c) for c in self.ch], relaxed_normalized=self.relaxed_n.tolist(), order=list(AERO_CHANNELS),
                    calibration_poses={k: v.tolist() for k, v in self._cal.items()})

    def save(self, path: str | Path) -> None: Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def from_dict(cls, d: dict) -> "AeroRetargeter":
        r = cls([ChannelMap(**c) for c in d["channels"]], relaxed_normalized=d.get("relaxed_normalized"))
        r._cal = {k: np.asarray(v) for k, v in d.get("calibration_poses", {}).items()}; return r

    @classmethod
    def load(cls, path: str | Path) -> "AeroRetargeter": return cls.from_dict(json.loads(Path(path).read_text()))
