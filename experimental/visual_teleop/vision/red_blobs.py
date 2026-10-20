"""Red-tape body markers on the UMI, seen by the fixed RGB-D (2026-10-15, user added 4 red tapes as rigid body keypoints).

detect_blobs(): HSV red (both hue ends) -> connected components -> centroid + robust depth (median of the eroded blob) -> 3D in W.
Only blobs inside a 3D radius around a prior (the last UMI position) are returned, which removes the robot's orange parts, the fire
extinguisher, boxes... Correspondence to the calibrated layout is solved by solve_tip() (layout matching under the IMU rotation)."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
import numpy as np


@dataclass
class Blob:
    uv: tuple
    area: int
    p_W: np.ndarray
    depth_m: float


def red_mask(bgr, s_min: int = 110, v_min: int = 70, hue_lo: int = 12, hue_hi: int = 168):
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (((h <= hue_lo) | (h >= hue_hi)) & (s >= s_min) & (v >= v_min)).astype(np.uint8)


def detect_blobs(bgr, depth_m, K, T_W_C, *, prior_W=None, radius_m: float = 0.30, min_area: int = 12, max_area: int = 4000,
                 roi_px: int = 260, depth_band=(0.3, 3.0)) -> list[Blob]:
    """K = (fx, fy, cx, cy) of the colour image; depth_m aligned to colour (same size). prior_W: 3D centre for the radius gate and,
    projected, for a pixel ROI (speed)."""
    import cv2
    fx, fy, cx, cy = K; H, W = bgr.shape[:2]; x0, y0 = 0, 0; img = bgr
    if prior_W is not None:
        pc = np.linalg.inv(T_W_C) @ np.r_[prior_W, 1.0]
        if pc[2] > 0.2:
            u, v = fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy
            x0, y0 = int(max(0, u - roi_px)), int(max(0, v - roi_px)); x1, y1 = int(min(W, u + roi_px)), int(min(H, v + roi_px))
            if x1 - x0 > 20 and y1 - y0 > 20: img = bgr[y0:y1, x0:x1]
            else: x0, y0 = 0, 0
    m = red_mask(img); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < min_area or a > max_area: continue
        u, v = cen[i][0] + x0, cen[i][1] + y0
        ys, xs = np.nonzero(lab == i); ys, xs = ys + y0, xs + x0
        d = depth_m[ys, xs]; d = d[np.isfinite(d) & (d > depth_band[0]) & (d < depth_band[1])]
        if d.size < max(4, a // 4): continue
        z = float(np.median(d)); p = T_W_C[:3, :3] @ np.array([(u - cx) * z / fx, (v - cy) * z / fy, z]) + T_W_C[:3, 3]
        if prior_W is not None and np.linalg.norm(p - prior_W) > radius_m: continue
        out.append(Blob((float(u), float(v)), a, p, z))
    return merge_split(out)


def merge_split(blobs: list[Blob], px: float = 14.0, m3: float = 0.02) -> list[Blob]:
    """One tape often comes out as 2 components (specular stripe / shadow): merge blobs closer than px in the image or m3 in 3D,
    area-weighted (10-15 layout recording: 8 clusters for 4 tapes)."""
    out: list[Blob] = []
    for b in sorted(blobs, key=lambda x: -x.area):
        for i, o in enumerate(out):
            if np.hypot(b.uv[0] - o.uv[0], b.uv[1] - o.uv[1]) < px or np.linalg.norm(b.p_W - o.p_W) < m3:
                w = o.area + b.area
                out[i] = Blob(((o.uv[0] * o.area + b.uv[0] * b.area) / w, (o.uv[1] * o.area + b.uv[1] * b.area) / w), w,
                              (o.p_W * o.area + b.p_W * b.area) / w, (o.depth_m * o.area + b.depth_m * b.area) / w); break
        else: out.append(b)
    return out


def solve_tip(blobs: list[Blob], R_WH: np.ndarray, layout_H: np.ndarray, *, tol_m: float = 0.025, min_pts: int = 2):
    """layout_H (N,3): marker positions in the UMI/TCP frame (row 0 may be the jaw tip itself). Find the assignment blob->marker
    that makes all R_WH @ P_i + t agree (t = tip in W), robustly. Returns (t_W or None, matched [(blob_idx, marker_idx)], rms_m)."""
    Q = np.array([b.p_W for b in blobs]) if blobs else np.zeros((0, 3)); RP = layout_H @ R_WH.T; best = (None, [], np.inf)
    if len(Q) < min_pts: return best
    # hypothesis: blob j is marker i -> t = q_j - R P_i; count inliers (each other blob explained by some unused marker)
    for j in range(len(Q)):
        for i in range(len(layout_H)):
            t = Q[j] - RP[i]; used, pairs = {i}, [(j, i)]
            for jj in range(len(Q)):
                if jj == j: continue
                d = np.linalg.norm(Q[jj] - (RP + t), axis=1); d[list(used)] = np.inf; k = int(np.argmin(d))
                if d[k] < tol_m: used.add(k); pairs.append((jj, k))
            if len(pairs) < min_pts: continue
            tt = np.mean([Q[a] - RP[b] for a, b in pairs], axis=0); rms = float(np.sqrt(np.mean([np.sum((Q[a] - RP[b] - tt) ** 2) for a, b in pairs])))
            if len(pairs) > len(best[1]) or (len(pairs) == len(best[1]) and rms < best[2]): best = (tt, pairs, rms)
    return best
