"""UMI AprilTag on the fixed RGB-D (2026-10-15, user: "MVP에서는 UMI에 작은 AprilTag 하나"). Gives the UMI 6DoF in W:

    T_W_tag = T_W_C . T_C_tag(PnP, pupil_apriltags, metric from the tag size; optional depth refinement of the centre)
    TCP     = T_W_tag . T_tag_tcp          (rigid, config fusion.tag.T_tag_tcp; unmeasured -> the tag centre is the TCP)

Roles (user): tag/RGB-D = POSITION primary; IMU = ORIENTATION primary; tag orientation = only a weak yaw-drift anchor
(OrientationFusion.observe_vision with its rejection gate). The tag->H rotation is not assumed: HandFusion learns it once from
the IMU right after INIT (chordal mean of R_W_tag^T R_W_H over the first samples) and freezes it.
Detection confidence = decision_margin scaled to [0, 1]; a tag below min_margin, the wrong id, or a large reprojection error = no observation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class TagObservation:
    t_ns: int
    T_W_tag: np.ndarray
    p_W_tcp: np.ndarray
    conf: float
    margin: float
    depth_refined: bool


class UmiTagTracker:
    def __init__(self, T_W_C: np.ndarray, *, family: str = "tag36h11", tag_id: int = 0, size_m: float = 0.03,
                 T_tag_tcp: np.ndarray | None = None, min_margin: float = 30.0, use_depth: bool = True, detector=None) -> None:
        if detector is None:
            from pupil_apriltags import Detector
            detector = Detector(families=family, nthreads=2, quad_decimate=1.0, refine_edges=True)
        self.det, self.T_W_C = detector, np.asarray(T_W_C, np.float64)
        self.tag_id, self.size_m, self.min_margin, self.use_depth = int(tag_id), float(size_m), float(min_margin), bool(use_depth)
        self.T_tag_tcp = np.eye(4) if T_tag_tcp is None else np.asarray(T_tag_tcp, np.float64)

    roi_half_px: int = 160          # tracking window half-size (full-res px) around the last detection
    roi_upscale: float = 3.0        # 10-15: the UMI tag is ~17 px in the Orbbec view; found only in an upscaled crop
    full_upscale: float = 2.0       # re-acquisition over the whole frame
    reacquire_every: int = 5        # while lost, the slow whole-frame search runs only every Nth frame (keeps the loop rate)
    _last_uv: tuple | None = None

    def _detect_in(self, gray, K, x0: int, y0: int, s: float):
        import cv2
        img = gray if s == 1.0 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        cam = (K.fx * s, K.fy * s, (K.cx - x0) * s, (K.cy - y0) * s)       # intrinsics of the cropped + scaled image
        out = []
        for d in self.det.detect(img, estimate_tag_pose=True, camera_params=cam, tag_size=self.size_m):
            if d.tag_id == self.tag_id and d.decision_margin >= self.min_margin:
                d.center_full = (d.center[0] / s + x0, d.center[1] / s + y0); out.append(d)
        return out

    def detect(self, frame) -> TagObservation | None:
        """frame: RgbdFrame (color_bgr, depth_m, calib.color intrinsics, timestamp_ns)."""
        import cv2
        K = frame.calib.color
        gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY); H, W = gray.shape[:2]
        dets = []
        if self._last_uv is not None:                                        # fast path: upscaled window around the last tag
            u, v = self._last_uv; r = self.roi_half_px
            x0, y0 = int(max(0, u - r)), int(max(0, v - r)); x1, y1 = int(min(W, u + r)), int(min(H, v + r))
            dets = self._detect_in(gray[y0:y1, x0:x1], K, x0, y0, self.roi_upscale)
        self._n_lost = getattr(self, "_n_lost", 0)
        if not dets and (self._last_uv is not None or self._n_lost % self.reacquire_every == 0):
            dets = self._detect_in(gray, K, 0, 0, self.full_upscale)              # re-acquire over the whole frame (~84 ms)
        self._n_lost = 0 if dets else self._n_lost + 1
        if not dets: self._last_uv = None; return None
        d = max(dets, key=lambda x: x.decision_margin); self._last_uv = d.center_full
        d.center = d.center_full
        T_C_tag = np.eye(4); T_C_tag[:3, :3] = np.asarray(d.pose_R); T_C_tag[:3, 3] = np.asarray(d.pose_t).reshape(3)
        refined = False
        if self.use_depth and getattr(frame, "depth_m", None) is not None:
            u, v = int(round(d.center[0])), int(round(d.center[1])); h, w = frame.depth_m.shape[:2]
            win = frame.depth_m[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
            z = float(np.nanmedian(win)) if np.isfinite(win).any() else float("nan")
            if np.isfinite(z) and z > 0.1 and abs(z - T_C_tag[2, 3]) < 0.15:       # depth agrees with PnP: use the metric depth
                T_C_tag[:3, 3] *= z / T_C_tag[2, 3]; refined = True
        T_W_tag = self.T_W_C @ T_C_tag
        p = (T_W_tag @ self.T_tag_tcp)[:3, 3]
        conf = float(np.clip(0.6 + (d.decision_margin - self.min_margin) / 50.0, 0.0, 1.0))   # at min_margin 0.6 (usable, WARN band)
        return TagObservation(int(frame.timestamp_ns), T_W_tag, p, conf, float(d.decision_margin), refined)
