"""Hand branch, human side: 21 landmarks -> wrist-relative normalized geometry -> canonical u7 in [0,1].

Landmark order = MediaPipe HandLandmarker = the layout hands3d.aero already uses (0 wrist; 1-4 thumb CMC,MCP,IP,tip;
5-8 index MCP,PIP,DIP,tip; 9-12 middle; 13-16 ring; 17-20 pinky). We reuse ego_collector.hands3d.aero.human_semantic7
(tendon-weighted flexion angles) + semantic7_normalized (fixed anatomical ranges) so the SAME canonical 7D describes a
human hand whether it came from HaWoR offline or MediaPipe live. Absolute landmark XYZ is never retargeted.

Finger-tracking CAMERA IS NOT DECIDED (2026-09-10): the wrist fisheye points at the scene for VIO and may not see the operator's
own fingers; the head C922 is a policy observation stream and must stay untouched by control logic; a dedicated small camera is
the third option. `FingerSource` makes this explicit — `LiveHandTracker` refuses to start while the source is UNVERIFIED, and
`HandRectifier` (Kannala-Brandt -> pinhole) is applied only when the chosen source is a fisheye."""
from __future__ import annotations
import enum
from dataclasses import dataclass
import numpy as np
from ego_collector.hands3d.aero import human_semantic7, semantic7_normalized, COMPACT_NAMES
from ..tracking.interfaces import HandFeatures, HandHealth, AERO_CHANNELS

assert len(COMPACT_NAMES) == len(AERO_CHANNELS) == 7
WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17


def normalize_landmarks(l3: np.ndarray) -> tuple[np.ndarray, float]:
    """(21,3) -> wrist-origin, palm-width-normalized landmarks, plus the palm scale (m or whatever unit came in)."""
    J = np.asarray(l3, np.float64).reshape(21, 3)
    rel = J - J[WRIST]
    palm = 0.5 * (np.linalg.norm(J[INDEX_MCP] - J[PINKY_MCP]) + np.linalg.norm(J[MIDDLE_MCP] - J[WRIST]))
    if not np.isfinite(palm) or palm < 1e-6: raise ValueError("degenerate hand landmarks (palm scale ~ 0)")
    return rel / palm, float(palm)


def features_from_landmarks(t_ns: int, l3: np.ndarray, confidence: float, *, side: str = "") -> HandFeatures:
    """Build HandFeatures (health OK) from one visible hand. u7 uses the canonical AERO_CHANNELS order."""
    rel, palm = normalize_landmarks(l3)
    sem7 = human_semantic7(np.asarray(l3, np.float64).reshape(1, 21, 3))[0]     # angles: scale-invariant
    u7 = semantic7_normalized(sem7)
    return HandFeatures(int(t_ns), rel, u7, float(confidence), HandHealth.OK, side, extra=dict(palm_scale=palm, sem7_rad=sem7.tolist()))


@dataclass
class HandSupervisorConfig:
    hold_ms: float = 300.0        # landmarks missing up to this long: HAND_HOLD (keep previous Aero target)
    min_confidence: float = 0.5   # below: treated as not visible
    fresh_ms: float = 60.0        # a feature older than this is no longer returned as OK by get_features(now)


class HandTrackingSupervisor:
    """Turns a stream of (possibly missing) HandFeatures into OK / HOLD / LOST: OK while landmarks are fresh, HOLD for up
    to hold_ms after they vanish (previous target kept), LOST afterwards (caller moves the hand to the relaxed pose)."""

    def __init__(self, cfg: HandSupervisorConfig | None = None) -> None:
        self.cfg = cfg or HandSupervisorConfig(); self._last_ok: HandFeatures | None = None

    def update(self, feat: HandFeatures | None, now_ns: int) -> HandFeatures:
        if feat is not None and feat.visible and feat.confidence >= self.cfg.min_confidence:
            self._last_ok = feat; return feat
        if self._last_ok is None:
            return HandFeatures(int(now_ns), None, None, 0.0, HandHealth.LOST)
        age_ms = (int(now_ns) - self._last_ok.timestamp_ns) / 1e6
        hold = age_ms <= self.cfg.hold_ms
        return HandFeatures(int(now_ns), None, self._last_ok.u7.copy() if hold else None, 0.0,
                            HandHealth.HOLD if hold else HandHealth.LOST, self._last_ok.side, dict(age_ms=age_ms))

    def current(self, now_ns: int) -> HandFeatures:
        """Health as of `now_ns` without a new observation (fresh last OK -> returned as OK)."""
        if self._last_ok is not None and (int(now_ns) - self._last_ok.timestamp_ns) / 1e6 <= self.cfg.fresh_ms: return self._last_ok
        return self.update(None, now_ns)


class FingerSource(str, enum.Enum):
    UNVERIFIED = "unverified"          # default until the physical FOV check is done
    WRIST_FISHEYE = "wrist_fisheye"    # only if the wrist camera demonstrably sees the fingers
    DEDICATED_CAMERA = "dedicated_camera"
    HEAD_C922_READONLY = "head_c922"   # allowed ONLY as a passive image consumer; never gives the head a control role
    HEAD_RGBD = "head_rgbd"            # head RGB-D metric hand pose (ego_teleop.hand3d) — the Aero branch of the
                                       # ActiveUMI+RGB-D spec. Passive for the ARM too: it only drives the HAND.
                                       # Activated ONLY after the A1 acceptance gate passes, never because the
                                       # camera happens to be plugged in (spec section 26).


class HandRectifier:
    """Raw fisheye frame -> pinhole view of `fov_deg` for the landmark model (VIO keeps the raw frame)."""

    def __init__(self, intrinsics: dict, *, fov_deg: float = 100.0, out_size: tuple[int, int] = (640, 480)) -> None:
        from handumi_collector.pose.backends.opencv_vo import FisheyeRectifier
        self._r = FisheyeRectifier(intrinsics, out_size=out_size, fov_deg=fov_deg)

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray: return self._r.rectify(image_bgr)


class LiveHandTracker:
    """MediaPipe HandLandmarker (ego_collector.hands.mediapipe_tracker) -> HandFeatures for the configured side.
    Lazy import: mediapipe is heavy and absent on the robot host is a valid configuration (hand branch off)."""

    def __init__(self, side: str, *, source: FingerSource | str = FingerSource.UNVERIFIED, rectifier: HandRectifier | None = None,
                 mirror_handedness: bool = False, **landmarker_kw) -> None:
        source = FingerSource(source)
        if source == FingerSource.UNVERIFIED:
            raise RuntimeError("finger-tracking camera not decided: verify the FOV and set teleop.yaml finger_source (wrist_fisheye | dedicated_camera | head_c922)")
        if source != FingerSource.WRIST_FISHEYE and rectifier is not None:
            raise ValueError("HandRectifier is for a fisheye source only")
        from ego_collector.hands.mediapipe_tracker import HandLandmarker
        self.source = source; self.side = side; self.rect = rectifier; self.mirror = mirror_handedness
        self._lm = HandLandmarker(num_hands=1, **landmarker_kw)
        self.supervisor = HandTrackingSupervisor()
        self._latest: HandFeatures | None = None

    def push_image(self, t_ns: int, image_bgr: np.ndarray) -> HandFeatures:
        img = self.rect(image_bgr) if self.rect else image_bgr
        res = self._lm.detect(img, int(t_ns // 1_000_000))
        side = self.side
        if self.mirror: side = "left" if side == "right" else "right"
        r = res.get(side)
        feat = None
        if r is not None and r.visible:
            try: feat = features_from_landmarks(t_ns, r.landmarks_3d, r.confidence, side=self.side)
            except ValueError: feat = None
        self._latest = self.supervisor.update(feat, t_ns); return self._latest

    def get_features(self, now_ns: int | None = None) -> HandFeatures | None:
        if self._latest is None: return None
        return self._latest if now_ns is None else self.supervisor.current(int(now_ns))
