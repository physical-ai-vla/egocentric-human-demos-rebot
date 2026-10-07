"""Hand-pose state machine (spec sections 16-17): INITIALIZING -> OK / DEGRADED / LOST, with a hold-on-loss policy.

Deliberately separate from `TrackingSupervisor` (wrist VIO): the two branches fail for different reasons and one must
never be read as the other. The arm can be tracking perfectly while the hand is occluded, and vice versa.

Failure policy for V1 is fixed by spec section 17: **HAND_LOST holds the current Aero target**. The hand is never
auto-opened on vision loss, because opening drops whatever is being carried. Resuming needs tracking back AND stable
for `recover_stable_ms`; until then the state is INITIALIZING, which the command layer also treats as hold."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .interfaces import HandPoseEstimate, HandPoseHealth


@dataclass
class HandPoseSupervisorConfig:
    min_valid_landmarks_ok: int = 18      # of 21; below this (but above the degraded floor) -> DEGRADED
    min_valid_landmarks_degraded: int = 12  # below this -> LOST (not enough geometry to retarget honestly)
    min_valid_tips_ok: int = 5            # of 5 fingertips
    min_valid_tips_degraded: int = 3
    min_depth_confidence_ok: float = 0.35  # mean over valid landmarks
    min_detector_confidence: float = 0.5
    max_wrist_jump_m: float = 0.20        # camera-frame wrist jump between consecutive frames -> LOST
    max_local_jump_m: float = 0.06        # palm-local articulation jump -> DEGRADED (noise) ...
    max_local_jump_lost_m: float = 0.15   # ... or LOST (a different hand / a detector glitch)
    stale_after_ms: float = 120.0         # no new estimate for this long -> LOST
    recover_stable_ms: float = 300.0      # continuous good frames required after a loss before commands resume
    degraded_streak_to_lost: int = 30     # ~1 s at 30 Hz of continuous DEGRADED -> give up and hold


@dataclass
class HandPoseStatus:
    health: HandPoseHealth
    estimate: HandPoseEstimate | None     # the accepted estimate (None while LOST/INITIALIZING with nothing fresh)
    reason: str = ""
    since_ok_ms: float = float("nan")
    recovering_ms: float = float("nan")

    @property
    def commandable(self) -> bool:
        """Only OK and DEGRADED produce new Aero targets; everything else holds the previous one."""
        return self.health in (HandPoseHealth.OK, HandPoseHealth.DEGRADED) and self.estimate is not None


class HandPoseSupervisor:
    def __init__(self, cfg: HandPoseSupervisorConfig | None = None) -> None:
        self.cfg = cfg or HandPoseSupervisorConfig()
        self._prev: HandPoseEstimate | None = None
        self._last_ok_ns: int | None = None
        self._recover_start_ns: int | None = None
        self._degraded_streak = 0
        self._state = HandPoseHealth.INITIALIZING
        self.counters = dict(ok=0, degraded=0, lost=0, initializing=0, discontinuity=0)

    # ---- classification of a single frame ---------------------------------------------------------------
    def _classify(self, est: HandPoseEstimate) -> tuple[HandPoseHealth, str]:
        c = self.cfg
        if est.detector_confidence < c.min_detector_confidence: return HandPoseHealth.LOST, "detector_confidence"
        if est.n_valid < c.min_valid_landmarks_degraded: return HandPoseHealth.LOST, "too_few_valid_landmarks"
        if est.n_valid_tips < c.min_valid_tips_degraded: return HandPoseHealth.LOST, "too_few_valid_tips"
        prev = self._prev
        if prev is not None and prev.side == est.side:
            w0, w1 = prev.landmarks_camera_3d[0], est.landmarks_camera_3d[0]
            if prev.metric and est.metric and np.all(np.isfinite(w0)) and np.all(np.isfinite(w1)):
                if float(np.linalg.norm(w1 - w0)) > c.max_wrist_jump_m:
                    self.counters["discontinuity"] += 1; return HandPoseHealth.LOST, "wrist_discontinuity"
            both = prev.valid & est.valid
            if both.any():
                d = float(np.max(np.linalg.norm(est.landmarks_local_3d[both] - prev.landmarks_local_3d[both], axis=1)))
                if d > c.max_local_jump_lost_m:
                    self.counters["discontinuity"] += 1; return HandPoseHealth.LOST, "articulation_discontinuity"
                if d > c.max_local_jump_m: return HandPoseHealth.DEGRADED, "articulation_jump"
        if est.n_valid < c.min_valid_landmarks_ok: return HandPoseHealth.DEGRADED, "depth_holes"
        if est.n_valid_tips < c.min_valid_tips_ok: return HandPoseHealth.DEGRADED, "tip_depth_holes"
        dc = est.depth_confidence[est.valid]
        if est.metric and dc.size and float(np.mean(dc)) < c.min_depth_confidence_ok: return HandPoseHealth.DEGRADED, "low_depth_confidence"
        return HandPoseHealth.OK, ""

    # ---- state machine ------------------------------------------------------------------------------------
    def update(self, est: HandPoseEstimate | None, now_ns: int) -> HandPoseStatus:
        c = self.cfg
        if est is None or est.health == HandPoseHealth.LOST:
            return self._to_lost(now_ns, "no_hand" if est is None else (est.extra.get("reason") or "provider_lost"))

        health, reason = self._classify(est)
        if health == HandPoseHealth.LOST:
            self._prev = None
            return self._to_lost(now_ns, reason)

        self._prev = est
        self._degraded_streak = self._degraded_streak + 1 if health == HandPoseHealth.DEGRADED else 0
        if self._degraded_streak >= c.degraded_streak_to_lost:
            return self._to_lost(now_ns, "degraded_too_long")

        # recovery gate: after any loss, require a continuous good stretch before commands resume
        if self._recover_start_ns is None and self._state in (HandPoseHealth.LOST, HandPoseHealth.INITIALIZING):
            self._recover_start_ns = int(est.timestamp_ns)
        if self._recover_start_ns is not None:
            waited = (int(est.timestamp_ns) - self._recover_start_ns) / 1e6
            if waited < c.recover_stable_ms:
                self._state = HandPoseHealth.INITIALIZING; self.counters["initializing"] += 1
                out = _with_health(est, HandPoseHealth.INITIALIZING, reason or "recovering")
                return HandPoseStatus(HandPoseHealth.INITIALIZING, out, reason or "recovering", recovering_ms=waited)
            self._recover_start_ns = None

        self._state = health; self._last_ok_ns = int(est.timestamp_ns)
        self.counters["ok" if health == HandPoseHealth.OK else "degraded"] += 1
        return HandPoseStatus(health, _with_health(est, health, reason), reason)

    def _to_lost(self, now_ns: int, reason: str) -> HandPoseStatus:
        self._state = HandPoseHealth.LOST; self._recover_start_ns = None; self._degraded_streak = 0
        self.counters["lost"] += 1
        since = float("nan") if self._last_ok_ns is None else (int(now_ns) - self._last_ok_ns) / 1e6
        return HandPoseStatus(HandPoseHealth.LOST, None, reason, since_ok_ms=since)

    def current(self, now_ns: int) -> HandPoseStatus:
        """Health as of `now_ns` without a new frame — a stale estimate ages into LOST (hold)."""
        if self._prev is None or (int(now_ns) - self._prev.timestamp_ns) / 1e6 > self.cfg.stale_after_ms:
            return self._to_lost(now_ns, "stale")
        return HandPoseStatus(self._state, self._prev, "")


def _with_health(est: HandPoseEstimate, health: HandPoseHealth, reason: str) -> HandPoseEstimate:
    import dataclasses
    extra = dict(est.extra); extra["health_reason"] = reason
    return dataclasses.replace(est, health=health, extra=extra)
