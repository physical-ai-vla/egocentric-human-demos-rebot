"""Fixed RGB-D (Orbbec) -> both wrists in the world frame W (2026-10-14, spec 7). Thin adapter over the EXISTING metric front-end
ego_teleop.hand3d.providers.RgbdHandPoseProvider (MediaPipe 2D + aligned depth + hand identity tracker); nothing re-implemented.

    estimate(frame) -> {side: WristObservation(t_ns, wrist_W, conf, tips_mid_W | None, thumb_W, index_W)}

One MediaPipe pass per frame (a memoising landmarker is shared by the two per-side providers).
Elbow/shoulder are NOT produced (spec 7/21: optional, not needed for the MVP).
LEFT/RIGHT IDENTITY IS A CLAIM: MediaPipe labels assume a mirrored selfie image; for a fixed camera facing the operator set
`selfie_mirrored` in the YAML and VERIFY live (raise the left hand -> dashboard shows LEFT) before any robot motion (spec 44)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8


@dataclass
class WristObservation:
    side: str
    t_ns: int
    wrist_W: np.ndarray | None
    conf: float
    tips_mid_W: np.ndarray | None = None
    thumb_W: np.ndarray | None = None
    index_W: np.ndarray | None = None
    health: str = ""


class _MemoLandmarker:
    """detect_all() memoised on the frame time so two per-side providers cost one MediaPipe pass."""
    def __init__(self, lm) -> None: self.lm, self._key, self._val = lm, None, None
    def detect_all(self, img, t_ms):
        if self._key != t_ms: self._key, self._val = t_ms, self.lm.detect_all(img, t_ms)
        return self._val


class RgbdWristEstimator:
    def __init__(self, T_W_C: np.ndarray, *, selfie_mirrored: bool = False, landmarker=None, provider_kw: dict | None = None) -> None:
        from ego_teleop.hand3d.providers import RgbdHandPoseProvider, HandPoseProviderConfig
        if landmarker is None:
            from ego_collector.hands.mediapipe_tracker import HandLandmarker
            landmarker = HandLandmarker(num_hands=2)
        memo = _MemoLandmarker(landmarker)
        self.T_W_C = np.asarray(T_W_C, np.float64)
        self.providers = {s: RgbdHandPoseProvider(HandPoseProviderConfig(side=s, selfie_mirrored=selfie_mirrored, **(provider_kw or {})), landmarker=memo)
                          for s in ("left", "right")}

    def _to_W(self, p_C) -> np.ndarray | None:
        p = np.asarray(p_C, np.float64)
        return None if not np.all(np.isfinite(p)) else self.T_W_C[:3, :3] @ p + self.T_W_C[:3, 3]

    def estimate(self, frame) -> dict:
        out = {}
        for s, prov in self.providers.items():
            e = prov.get_hand_pose(frame)
            if e is None or not e.usable or not e.valid[WRIST]:
                out[s] = WristObservation(s, int(getattr(frame, "timestamp_ns", 0)), None, 0.0, health=getattr(e, "health", "none").value if e is not None else "none")
                continue
            L = e.landmarks_camera_3d
            th = self._to_W(L[THUMB_TIP]) if e.valid[THUMB_TIP] else None; ix = self._to_W(L[INDEX_TIP]) if e.valid[INDEX_TIP] else None
            out[s] = WristObservation(s, e.timestamp_ns, self._to_W(L[WRIST]), position_confidence(e), None if th is None or ix is None else 0.5 * (th + ix),
                                      th, ix, e.health.value)
        return out


def position_confidence(e) -> float:
    """Detector score x depth confidence at the wrist; 0 when the wrist depth is invalid. (vision/confidence.py re-exports.)"""
    if not e.valid[WRIST]: return 0.0
    dc = float(e.depth_confidence[WRIST]) if np.isfinite(e.depth_confidence[WRIST]) else 0.0
    return float(np.clip(e.detector_confidence, 0, 1) * np.clip(dc if dc > 0 else 1.0, 0, 1))
