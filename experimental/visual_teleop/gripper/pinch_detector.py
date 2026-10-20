"""Binary pinch with hysteresis + dropout policy (2026-10-14, spec 9-10).

    metric < close_threshold            -> pinch TRUE
    metric > open_threshold             -> pinch FALSE
    in between                          -> keep previous state (hysteresis, mandatory)
    no detection / low confidence:
        <= short_dropout_s              -> keep previous state (never toggle on a lost frame)
        >  long_dropout_s               -> status PAUSE (caller holds that hand)
        in between                      -> keep previous state, status WARN
Initial state is OPEN (pinch FALSE) only after the first confident measurement; before it the status is PAUSE."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PinchState:
    pinch: bool | None              # None = never measured
    status: str                     # OK | WARN | PAUSE
    metric: float | None
    dropout_s: float
    toggled: bool = False
    g_target: float | None = None    # continuous mode: gripper target 0 (closed) .. 1 (open); None = use the binary pinch


@dataclass
class PinchDetector:
    close_threshold: float
    open_threshold: float
    short_dropout_s: float = 0.5
    long_dropout_s: float = 2.0
    min_confidence: float = 0.4
    pinch: bool | None = None
    last_ok_ns: int | None = None
    n_toggles: int = 0

    def __post_init__(self) -> None:
        if not self.close_threshold < self.open_threshold: raise ValueError("hysteresis needs close_threshold < open_threshold")

    @classmethod
    def from_cfg(cls, g: dict, min_confidence: float) -> "PinchDetector":
        return cls(g["close_threshold"], g["open_threshold"], g["short_dropout_s"], g["long_dropout_s"], min_confidence)

    def update(self, metric: float | None, confidence: float, t_ns: int) -> PinchState:
        ok = metric is not None and confidence >= self.min_confidence and metric == metric      # NaN check
        if not ok:
            d = float("inf") if self.last_ok_ns is None else (t_ns - self.last_ok_ns) / 1e9
            st = "PAUSE" if (self.pinch is None or d > self.long_dropout_s) else ("WARN" if d > self.short_dropout_s else "OK")
            return PinchState(self.pinch, st, None, d)
        self.last_ok_ns = int(t_ns); prev = self.pinch
        if metric < self.close_threshold: self.pinch = True
        elif metric > self.open_threshold: self.pinch = False
        elif self.pinch is None: self.pinch = False          # first sample inside the band: start open
        toggled = prev is not None and prev != self.pinch
        if toggled: self.n_toggles += 1
        return PinchState(self.pinch, "OK", float(metric), 0.0, toggled)


@dataclass
class SigmoidPinch:
    """Continuous pinch (user 10-15: "sigmoid로 해줘"), ported from scripts/finger_grip_teleop.py --mode sigmoid (10-13):
        f = clip((metric - closed) / (open - closed), 0, 1)      per-mount range from calib_right pinch (p5 / p95)
        f_ema <- ema * f + (1 - ema) * f_ema
        g = (sig(f_ema) - sig(0)) / (sig(1) - sig(0)),  sig(x) = 1 / (1 + exp(-k (x - 0.5)))      0 = closed, 1 = open
    Continuous, so no on/off toggling to chatter; a lost hand never moves the jaw (same short/long dropout policy)."""
    closed: float
    open: float
    k: float = 10.0
    ema: float = 0.35
    short_dropout_s: float = 0.5
    long_dropout_s: float = 2.0
    min_confidence: float = 0.4
    dead: float = 0.2                    # bottom/top fraction of the range = fully closed/open (10-15: a normal squeeze read 5-9, not p5=2 -> never fully closed)
    f_ema: float | None = None
    last_ok_ns: int | None = None
    n_toggles: int = 0

    def __post_init__(self) -> None:
        if not self.closed < self.open: raise ValueError("sigmoid pinch needs closed < open (run calib_right pinch)")

    @classmethod
    def from_cfg(cls, g: dict, min_confidence: float) -> "SigmoidPinch":
        if g.get("pinch_closed") is None or g.get("pinch_open") is None: raise ValueError("gripper.pinch_closed/open missing -- run calib_right pinch")
        return cls(g["pinch_closed"], g["pinch_open"], g["sigmoid_k"], g["sigmoid_ema"], g["short_dropout_s"], g["long_dropout_s"], min_confidence,
                   g.get("sigmoid_dead", 0.2))

    def _g(self) -> float:
        import math
        sg = lambda x: 1.0 / (1.0 + math.exp(-self.k * (x - 0.5)))
        return (sg(self.f_ema) - sg(0.0)) / (sg(1.0) - sg(0.0))

    def update(self, metric: float | None, confidence: float, t_ns: int) -> PinchState:
        ok = metric is not None and confidence >= self.min_confidence and metric == metric
        if not ok:
            d = float("inf") if self.last_ok_ns is None else (t_ns - self.last_ok_ns) / 1e9
            st = "PAUSE" if (self.f_ema is None or d > self.long_dropout_s) else "HOLD"     # HOLD: keep the jaw where it is
            g = None if self.f_ema is None else self._g()
            return PinchState(None if g is None else g < 0.5, st, None, d, g_target=None)
        self.last_ok_ns = int(t_ns)
        f = (float(metric) - self.closed) / (self.open - self.closed)
        f = min(max((f - self.dead) / (1.0 - 2.0 * self.dead), 0.0), 1.0)
        self.f_ema = f if self.f_ema is None else self.ema * f + (1 - self.ema) * self.f_ema
        g = self._g()
        return PinchState(g < 0.5, "OK", float(metric), 0.0, g_target=g)

