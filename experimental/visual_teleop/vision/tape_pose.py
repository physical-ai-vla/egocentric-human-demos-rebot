"""UMI position from the orange jaw-tip tape seen by the fixed RGB-D (2026-10-15, user: "rgbd 쓰는데 태그 없이 안되?").
No tag: the orange tape on the UMI jaw tips is the only orange thing in the teleop view (Orbbec frame 14:31: one blob, 89 px at
1280x720, at the UMI tip). Position = depth median over the blob -> camera 3D -> W. Orientation stays with the IMU; there is no
visual yaw anchor in this mode (INIT re-aligns the IMU yaw every session).
Selection: the blob nearest the previous pick (<= max_jump_px), else the largest; too small / no valid depth = no observation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class TapeObservation:
    t_ns: int
    p_W: np.ndarray
    uv: tuple
    area_px: int
    depth_m: float
    conf: float


class UmiTapeTracker:
    def __init__(self, T_W_C, *, hsv_lo=(0, 120, 80), hsv_hi=(22, 255, 255), min_area_px: int = 12, max_jump_px: float = 120.0,
                 depth_band_m=(0.2, 3.0)) -> None:
        self.T_W_C = np.asarray(T_W_C, np.float64); self.lo, self.hi = np.array(hsv_lo), np.array(hsv_hi)
        self.min_area, self.max_jump, self.band = int(min_area_px), float(max_jump_px), tuple(depth_band_m)
        self.prev_uv: tuple | None = None

    hint_radius_px: float = 200.0     # tag-guided: the tape tip must lie within this of the tag centre (same UMI)
    max_jump_m: float = 0.15          # tape-only tracking: a blob must be within this (3-D) of the last accepted UMI tip
    lost_max_s: float = 1.0           # after this long without any accepted tip, re-acquire only next to the tag
    prev_p: np.ndarray | None = None
    _last_t: float = 0.0
    _confirmed_t: float = 0.0

    def detect(self, frame, hint_uv=None) -> TapeObservation | None:
        """hint_uv: the UMI tag centre in this frame (px) when the tag is visible -> pick the blob near it (cannot lock onto a
        distractor); without a hint, continuity is trusted only for unconfirmed_max_s after the last tag confirmation."""
        import cv2, time as _t
        hsv = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2HSV)
        m = cv2.morphologyEx(cv2.inRange(hsv, self.lo, self.hi), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        cands = [k for k in range(1, n) if st[k, 4] >= self.min_area]
        if not cands: return None
        now = _t.monotonic(); K = frame.calib.color
        def p_of(q):                                   # 3-D (W) of a blob: depth median over its pixels
            d = frame.depth_m[lab == q]; d = d[np.isfinite(d) & (d > self.band[0]) & (d < self.band[1])]
            if len(d) < max(3, 0.2 * st[q, 4]): return None
            z = float(np.median(d)); u, v = cen[q]; pc = np.array([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z])
            return self.T_W_C[:3, :3] @ pc + self.T_W_C[:3, 3]
        if hint_uv is not None:                        # tag visible: the blob next to the tag IS the UMI tip (acquire / confirm)
            near = [k for k in cands if np.hypot(*(cen[k] - np.asarray(hint_uv))) <= self.hint_radius_px]
            if not near: return None
            k = min(near, key=lambda q: np.hypot(*(cen[q] - np.asarray(hint_uv)))); self._confirmed_t = now
        elif self.prev_p is not None and now - self._last_t <= self.lost_max_s:
            # 10-15: the small side tag is unreadable at an angle (lifted UMI: squashed 2:1, ~15 px) -> follow the tape alone,
            # gated in 3-D: only a blob within max_jump_m of the last accepted UMI tip (a distractor elsewhere is physically far)
            best = None
            for q in sorted(cands, key=lambda q: np.hypot(*(cen[q] - self.prev_uv))):
                pq = p_of(q)
                if pq is not None and np.linalg.norm(pq - self.prev_p) <= self.max_jump_m: best = (q, pq); break
            if best is None: return None
            k = best[0]
        else: self.prev_p = None; return None                                         # long loss: re-acquire only next to the tag
        d = frame.depth_m[lab == k]; d = d[np.isfinite(d) & (d > self.band[0]) & (d < self.band[1])]
        if len(d) < max(3, 0.2 * st[k, 4]): return None
        z = float(np.median(d)); u, v = float(cen[k][0]), float(cen[k][1]); K = frame.calib.color
        p_C = np.array([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z])
        self.prev_uv = (u, v); p_W = self.T_W_C[:3, :3] @ p_C + self.T_W_C[:3, 3]
        if hint_uv is None and self.prev_p is not None and np.linalg.norm(p_W - self.prev_p) > self.max_jump_m: return None
        self.prev_p = p_W; self._last_t = now
        return TapeObservation(int(frame.timestamp_ns), p_W, (u, v), int(st[k, 4]), z, 1.0)
