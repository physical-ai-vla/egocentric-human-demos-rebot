"""Markerless UMI jaw-tip tracking on the fixed RGB-D (2026-10-15, user: no tag, no tape at runtime).
A YOLOv8n-pose model with ONE keypoint (jaw tip) -- trained on recordings auto-labelled by the tag-guided tape tracker
(scripts/umi_kpt_dataset.py; tag/tape are the teacher only) -- exported to ONNX and run here with cv2.dnn, because the hub's venv
has an older ultralytics that cannot load the 8.4 checkpoint. Position = depth median around the keypoint -> camera 3-D -> W.
Continuity: a detection farther than max_jump_m (3-D) from the last accepted tip is refused unless the tip was lost > lost_max_s."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class KptObservation:
    t_ns: int
    p_W: np.ndarray
    uv: tuple
    conf: float
    depth_m: float


class UmiKptTracker:
    def __init__(self, onnx_path: str, T_W_C, *, imgsz: int = 640, conf: float = 0.5, max_jump_m: float = 0.15, lost_max_s: float = 1.0,
                 depth_band_m=(0.2, 3.0), v_max_m_s: float = 1.5, margin_m: float = 0.03) -> None:
        self.v_max, self.margin = float(v_max_m_s), float(margin_m)
        import cv2
        from pathlib import Path
        self.net = cv2.dnn.readNetFromONNX(str(Path(onnx_path).expanduser())); self.imgsz, self.conf = int(imgsz), float(conf)
        self.T_W_C = np.asarray(T_W_C, np.float64); self.max_jump, self.lost_max, self.band = float(max_jump_m), float(lost_max_s), tuple(depth_band_m)
        self.prev_p: np.ndarray | None = None; self._last_t = 0.0

    def infer(self, img) -> list:
        """-> [(conf, u, v)] in image px, best first (letterbox, cv2.dnn, YOLOv8-pose head: 4 box + 1 cls + 3 kpt)."""
        import cv2
        h, w = img.shape[:2]; r = self.imgsz / max(h, w); nw, nh = int(round(w * r)), int(round(h * r))
        pad = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8); px, py = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        pad[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
        self.net.setInput(cv2.dnn.blobFromImage(pad, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True))
        out = self.net.forward()[0]                      # (8, N)
        sc = out[4]; idx = np.argsort(-sc)[:5]
        return [(float(sc[i]), float((out[5, i] - px) / r), float((out[6, i] - py) / r)) for i in idx if sc[i] >= self.conf]

    def detect(self, frame) -> KptObservation | None:
        """Also leaves self.last_diag = dict(n_det, best_conf, rejected_jump, rejected_depth, accepted) for telemetry."""
        now = time.monotonic(); K = frame.calib.color; dets = self.infer(frame.color_bgr)
        self.last_diag = dict(n_det=len(dets), best_conf=dets[0][0] if dets else 0.0, rejected_jump=0, rejected_depth=0, accepted=False)
        for c, u, v in dets:
            ui, vi = int(round(u)), int(round(v)); d = frame.depth_m[max(0, vi - 4):vi + 5, max(0, ui - 4):ui + 5]
            d = d[np.isfinite(d) & (d > self.band[0]) & (d < self.band[1])]
            if len(d) < 5: self.last_diag["rejected_depth"] += 1; continue
            z = float(np.median(d)); p = self.T_W_C[:3, :3] @ np.array([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z]) + self.T_W_C[:3, 3]
            # velocity-aware gate (10-15 review): allowed = v_max * dt + margin (30 Hz -> ~8 cm), capped at max_jump_m
            if self.prev_p is not None and now - self._last_t <= self.lost_max:
                allow = min(self.max_jump, self.v_max * (now - self._last_t) + self.margin)
                if np.linalg.norm(p - self.prev_p) > allow: self.last_diag["rejected_jump"] += 1; continue
            self.prev_p = p; self._last_t = now; self.last_diag["accepted"] = True
            return KptObservation(int(frame.timestamp_ns), p, (u, v), c, z)
        return None
