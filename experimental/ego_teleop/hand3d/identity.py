"""Temporal hand identity (spec section 12).

Per-frame handedness labels are not trustworthy: MediaPipe flips them under self-occlusion, on a rotated hand, or
when only part of the hand is in frame. Sending a left-hand pose to the right Aero is worse than sending nothing, so
this tracker resolves identity from *continuity* (where each hand's wrist was last frame) and only uses the
classifier label to bootstrap and to vote. A conflict does not pick a winner quietly — it reports the side as
ambiguous, which the supervisor turns into HAND_LOST (hold), per spec section 16.

Nothing here knows about Aero; it works on 2D wrist pixels plus, when available, the metric 3D wrist."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class HandCandidate:
    """One MediaPipe detection in a frame, with handedness ALREADY corrected for the mirror convention."""
    label: str                       # "left" | "right"
    label_confidence: float
    detector_confidence: float
    wrist_uv: np.ndarray             # (2,) px
    wrist_xyz: np.ndarray | None = None   # (3,) m, when the provider is metric
    payload: object = None           # whatever the provider wants back (landmarks, depth samples, ...)


@dataclass
class IdentityConfig:
    max_jump_px: float = 120.0        # wrist displacement between frames still considered the same hand
    max_jump_m: float = 0.25          # metric variant (used when both frames have a valid 3D wrist)
    track_timeout_ms: float = 400.0   # a track older than this is dropped; the next detection re-bootstraps by label
    min_detector_confidence: float = 0.5
    label_vote_len: int = 5           # rolling votes; a sustained disagreement with continuity is a real flip
    label_flip_votes: int = 4         # of label_vote_len, how many must disagree before we believe the label


@dataclass
class _Track:
    uv: np.ndarray
    xyz: np.ndarray | None
    t_ns: int
    votes: list[str] = field(default_factory=list)


@dataclass
class IdentityResult:
    tracks: dict[str, HandCandidate]   # side -> accepted candidate (missing key = nothing accepted this frame)
    ambiguous: set[str]                # sides that must be treated as LOST this frame
    events: list[str]                  # "identity_conflict:right", "handedness_flip:left", "reacquired:right", ...


class HandIdentityTracker:
    def __init__(self, cfg: IdentityConfig | None = None) -> None:
        self.cfg = cfg or IdentityConfig()
        self._tracks: dict[str, _Track] = {}
        self.swap_events = 0

    def _alive(self, side: str, t_ns: int) -> _Track | None:
        tr = self._tracks.get(side)
        if tr is None: return None
        if (t_ns - tr.t_ns) / 1e6 > self.cfg.track_timeout_ms:
            self._tracks.pop(side, None); return None
        return tr

    def _continuity_cost(self, tr: _Track, c: HandCandidate) -> float:
        """Normalised distance to a track (<=1 means "compatible"); inf when the jump is too large."""
        d_px = float(np.linalg.norm(np.asarray(c.wrist_uv, np.float64) - tr.uv)) / max(self.cfg.max_jump_px, 1e-9)
        if tr.xyz is not None and c.wrist_xyz is not None and np.all(np.isfinite(c.wrist_xyz)):
            d_m = float(np.linalg.norm(np.asarray(c.wrist_xyz, np.float64) - tr.xyz)) / max(self.cfg.max_jump_m, 1e-9)
            cost = max(d_px, d_m)
        else:
            cost = d_px
        return cost if cost <= 1.0 else float("inf")

    def update(self, t_ns: int, candidates: list[HandCandidate]) -> IdentityResult:
        cands = [c for c in candidates if c.detector_confidence >= self.cfg.min_detector_confidence]
        events: list[str] = []
        # 1. continuity assignment: every (side, candidate) pair that is geometrically compatible, cheapest first
        pairs = []
        for side in ("left", "right"):
            tr = self._alive(side, t_ns)
            if tr is None: continue
            for c in cands:
                cost = self._continuity_cost(tr, c)
                if np.isfinite(cost): pairs.append((cost, side, c))
        pairs.sort(key=lambda p: p[0])
        assigned: dict[str, HandCandidate] = {}
        taken: set[int] = set()
        for _, side, c in pairs:
            if side in assigned or id(c) in taken: continue
            assigned[side] = c; taken.add(id(c))
        # 2. bootstrap the rest from the classifier label (a fresh track, or one that timed out)
        for c in cands:
            if id(c) in taken: continue
            side = c.label
            if side in assigned:
                events.append(f"identity_conflict:{side}"); self.swap_events += 1
                continue
            assigned[side] = c; taken.add(id(c))
            if side not in self._tracks: events.append(f"reacquired:{side}")
        # 3. label vote: a sustained disagreement between continuity and the classifier is a genuine swap, not noise
        ambiguous: set[str] = set()
        for side, c in list(assigned.items()):
            tr = self._tracks.get(side)
            votes = (tr.votes if tr is not None else [])[-(self.cfg.label_vote_len - 1):] + [c.label]
            disagree = sum(1 for v in votes if v != side)
            if disagree >= self.cfg.label_flip_votes:
                # believe the classifier: drop this track entirely so the next frame re-bootstraps by label,
                # and report the side as ambiguous so the supervisor holds instead of driving the wrong hand.
                events.append(f"handedness_flip:{side}"); self.swap_events += 1
                ambiguous.add(side); self._tracks.pop(side, None)
                continue
            self._tracks[side] = _Track(np.asarray(c.wrist_uv, np.float64).copy(),
                                        None if c.wrist_xyz is None or not np.all(np.isfinite(c.wrist_xyz)) else np.asarray(c.wrist_xyz, np.float64).copy(),
                                        int(t_ns), votes)
        for side in ambiguous: assigned.pop(side, None)
        return IdentityResult(assigned, ambiguous, events)

    def reset(self) -> None:
        self._tracks.clear()
