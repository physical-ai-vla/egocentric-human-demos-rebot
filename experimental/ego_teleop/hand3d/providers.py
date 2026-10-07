"""The two `HandPoseProvider` implementations of spec sections 5-6.

    MediaPipeHandPoseProvider   B0 — the official monocular baseline: MediaPipe world landmarks, palm-normalised.
    RgbdHandPoseProvider        B1 — ours: MediaPipe supplies detection / handedness / landmark identity / 2D position,
                                the depth sensor supplies the metric geometry.

Both emit the same `HandPoseEstimate`, and both accept the same frame object, so a single recorded RGB-D episode can
be pushed through both and attributed cleanly (spec section 30). `world_landmarks` is never the final 3D source in
RGB-D mode; it is used only as a *shape prior* to fill landmarks whose depth the sensor could not measure, and every
filled landmark stays `valid == False` so the acceptance gates still measure real depth coverage.

Mirroring: production runs unmirrored (spec section 19). The image is not flipped, geometry is not negated, and the
handedness label is swapped instead (`selfie_mirrored=False`) — see docs/ego_teleop/AERO_A0_AUDIT.md."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np
from . import aero_mocap as M
from .depth import DepthSampler, DepthSamplerConfig, reject_depth_outliers
from .identity import HandCandidate, HandIdentityTracker, IdentityConfig
from .interfaces import HandPoseEstimate, HandPoseHealth


@dataclass
class HandPoseProviderConfig:
    side: str = "right"                 # which human hand drives Aero in V1 (interfaces stay side-aware for the second)
    selfie_mirrored: bool = False       # head camera looks outward -> MediaPipe handedness labels must be swapped
    num_hands: int = 2                  # detect both, so an identity conflict is visible instead of silently accepted
    min_detection: float = 0.5
    min_presence: float = 0.5
    min_tracking: float = 0.5
    fill_missing: bool = True           # reconstruct depth-less landmarks from the monocular shape prior
    max_fill: int = 6                   # more missing landmarks than this -> refuse to fill, report the frame as LOST
    depth: DepthSamplerConfig = field(default_factory=DepthSamplerConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity (R, t, s) with dst ~= s*R@src + t. Used to fit the monocular hand shape to the
    metric landmarks we DID measure, so the missing ones can be predicted in the same metric frame."""
    S, D = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    if S.shape != D.shape or S.shape[0] < 3: raise ValueError(f"need >=3 matched points, got {S.shape}")
    mu_s, mu_d = S.mean(0), D.mean(0)
    Sc, Dc = S - mu_s, D - mu_d
    U, sv, Vt = np.linalg.svd(Dc.T @ Sc / S.shape[0])
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ d @ Vt
    var = float((Sc ** 2).sum() / S.shape[0])
    s = float((sv * np.diag(d)).sum() / var) if var > 1e-12 else 1.0
    return R, mu_d - s * R @ mu_s, s


class MediaPipeHandPoseProvider:
    """B0: exactly what `webcam_mocap.py` does, minus ROS. `landmarks_camera_3d` holds MediaPipe *world* landmarks —
    hand-centred and only approximately metric for an average hand — which is why `metric` is False."""

    source = "mediapipe_mono"

    def __init__(self, cfg: HandPoseProviderConfig | None = None, *, landmarker=None, **landmarker_kw) -> None:
        self.cfg = cfg or HandPoseProviderConfig()
        self._lm = landmarker
        self._lm_kw = landmarker_kw
        self.identity = HandIdentityTracker(self.cfg.identity)
        self.last_events: list[str] = []

    def _landmarker(self):
        if self._lm is None:
            from ego_collector.hands.mediapipe_tracker import HandLandmarker
            self._lm = HandLandmarker(num_hands=self.cfg.num_hands, min_detection=self.cfg.min_detection,
                                      min_presence=self.cfg.min_presence, min_tracking=self.cfg.min_tracking, **self._lm_kw)
        return self._lm

    def _detect(self, frame) -> tuple[np.ndarray, int, list]:
        color = getattr(frame, "color_bgr", frame)
        t_ns = int(getattr(frame, "timestamp_ns", time.monotonic_ns()))
        dets = self._landmarker().detect_all(color, int(t_ns // 1_000_000))
        return color, t_ns, dets

    def _pick(self, t_ns: int, dets) -> object | None:
        cands = [HandCandidate(M.handedness_for_image(d.label, selfie_mirrored=self.cfg.selfie_mirrored),
                               d.label_confidence, d.label_confidence, d.landmarks_2d[M.WRIST], None, d) for d in dets]
        res = self.identity.update(t_ns, cands)
        self.last_events = res.events
        c = res.tracks.get(self.cfg.side)
        return c

    def get_hand_pose(self, frame) -> HandPoseEstimate | None:
        color, t_ns, dets = self._detect(frame)
        c = self._pick(t_ns, dets)
        if c is None:
            return HandPoseEstimate.empty(t_ns, self.cfg.side, self.source, extra=dict(reason="no_hand", events=list(self.last_events)))
        d = c.payload
        world = np.asarray(d.landmarks_3d, np.float64).reshape(21, 3)
        if self.cfg.selfie_mirrored: world = world * np.array([-1.0, 1.0, 1.0])   # undo the mirror, as upstream does
        try:
            local = M.to_palm_local(world, self.cfg.side)
        except ValueError as e:
            return HandPoseEstimate.empty(t_ns, self.cfg.side, self.source, extra=dict(reason=f"palm_frame:{e}"))
        return HandPoseEstimate(t_ns, self.cfg.side, np.asarray(d.landmarks_2d, np.float64), world, local,
                                np.ones(21, bool), np.full(21, np.nan), np.zeros(21), float(d.label_confidence),
                                HandPoseHealth.OK, self.source, metric=False, palm_scale_m=M.palm_scale_m(world),
                                handedness_confidence=float(c.label_confidence),
                                source_timestamp_ns=getattr(frame, "source_timestamp_ns", None),
                                pose_done_ns=time.monotonic_ns(),
                                extra=dict(events=list(self.last_events), camera_frame=False))


class RgbdHandPoseProvider(MediaPipeHandPoseProvider):
    """B1: 2D landmarks from MediaPipe, metric XYZ from the aligned depth frame (spec section 6).

    Input is an `RgbdFrame`. The calibration's `aligned` flag is checked once per frame before any depth[u,v] read."""

    source = "head_rgbd"

    def get_hand_pose(self, frame) -> HandPoseEstimate | None:
        if not hasattr(frame, "depth_m"):
            raise TypeError("RgbdHandPoseProvider needs an RgbdFrame (colour + aligned depth), got a plain image")
        frame.calib.require_aligned()
        sampler = DepthSampler(self.cfg.depth)
        color, t_ns, dets = self._detect(frame)

        cands = []
        for d in dets:
            side = M.handedness_for_image(d.label, selfie_mirrored=self.cfg.selfie_mirrored)
            s = sampler.sample(frame.depth_m, d.landmarks_2d[M.WRIST][None, :])
            xyz = frame.calib.color.deproject(d.landmarks_2d[M.WRIST][None, :], s.depth_m)[0] if s.valid[0] else None
            cands.append(HandCandidate(side, d.label_confidence, d.label_confidence, d.landmarks_2d[M.WRIST], xyz, d))
        res = self.identity.update(t_ns, cands)
        self.last_events = res.events
        c = res.tracks.get(self.cfg.side)
        if c is None:
            reason = "identity_ambiguous" if self.cfg.side in res.ambiguous else "no_hand"
            return HandPoseEstimate.empty(t_ns, self.cfg.side, self.source, extra=dict(reason=reason, events=list(res.events)))

        d = c.payload
        uv = np.asarray(d.landmarks_2d, np.float64).reshape(21, 2)
        smp = reject_depth_outliers(sampler.sample(frame.depth_m, uv), self.cfg.depth.max_hand_depth_span_m)
        cam = np.full((21, 3), np.nan)
        if smp.valid.any():
            cam[smp.valid] = frame.calib.color.deproject(uv[smp.valid], smp.depth_m[smp.valid])
        valid = smp.valid.copy()
        filled = np.zeros(21, bool)

        n_missing = int(21 - valid.sum())
        if n_missing:
            if not self.cfg.fill_missing or n_missing > self.cfg.max_fill or valid.sum() < 4:
                return HandPoseEstimate(t_ns, self.cfg.side, uv, cam, np.full((21, 3), np.nan), valid, smp.depth_m,
                                        smp.confidence, float(d.label_confidence), HandPoseHealth.LOST, self.source, True,
                                        extra=dict(reason="insufficient_metric_landmarks", n_missing=n_missing, events=list(res.events)))
            # monocular shape prior, fitted (similarity) to the landmarks depth DID measure: metric frame preserved
            world = np.asarray(d.landmarks_3d, np.float64).reshape(21, 3)
            if self.cfg.selfie_mirrored: world = world * np.array([-1.0, 1.0, 1.0])
            R, t, s = umeyama_similarity(world[valid], cam[valid])
            cam[~valid] = (s * (R @ world[~valid].T).T) + t
            filled = ~valid

        try:
            local = M.to_palm_local(cam, self.cfg.side)
        except ValueError as e:
            return HandPoseEstimate.empty(t_ns, self.cfg.side, self.source, extra=dict(reason=f"palm_frame:{e}", events=list(res.events)))
        return HandPoseEstimate(t_ns, self.cfg.side, uv, cam, local, valid, smp.depth_m, smp.confidence,
                                float(d.label_confidence), HandPoseHealth.OK, self.source, metric=True,
                                palm_scale_m=M.palm_scale_m(cam), filled=filled,
                                handedness_confidence=float(c.label_confidence),
                                source_timestamp_ns=frame.source_timestamp_ns, pose_done_ns=time.monotonic_ns(),
                                extra=dict(events=list(res.events), camera_frame=True, n_filled=int(filled.sum()),
                                           depth_spread_m=float(np.nanmedian(smp.spread_m[valid])) if valid.any() else float("nan")))
