"""Metric geometry for the RGB-D hand branch: pinhole intrinsics, deprojection, and robust per-landmark depth.

Spec sections 7-9. Two rules the rest of the stack depends on:
  * a landmark whose depth could not be measured is marked INVALID — never silently filled with 0, and never
    interpolated here (that is a policy decision, made one level up in the provider);
  * depth[u, v] is only read when the depth frame is known to be aligned to the colour frame. `HeadRgbdCalibration`
    (head_camera.py) carries `aligned`, and the sampler refuses to run on an unaligned pair without an explicit
    RGB->depth mapping.

Internal units are metres, everywhere, always."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_dict(cls, d: dict, size: tuple[int, int] | None = None) -> "CameraIntrinsics":
        w, h = size if size is not None else (d["width"], d["height"])
        return cls(float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]), int(w), int(h))

    def to_dict(self) -> dict:
        return dict(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, width=self.width, height=self.height)

    def deproject(self, uv: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """(N,2) pixels + (N,) metric depth -> (N,3) camera-frame XYZ in metres (OpenCV frame: x right, y down, z fwd).

        The vendor's own deprojection is preferred when the device exposes one (Orbbec/RealSense both do, for lens
        models this pinhole formula does not cover); `HeadRgbdCamera.deproject` routes to it when available."""
        uv = np.asarray(uv, np.float64).reshape(-1, 2); d = np.asarray(depth_m, np.float64).reshape(-1)
        if uv.shape[0] != d.shape[0]: raise ValueError(f"uv {uv.shape} and depth {d.shape} disagree")
        x = (uv[:, 0] - self.cx) * d / self.fx
        y = (uv[:, 1] - self.cy) * d / self.fy
        return np.stack([x, y, d], axis=1)

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """(N,3) metres -> (N,2) pixels (visualisation / round-trip tests)."""
        p = np.asarray(xyz, np.float64).reshape(-1, 3); z = np.where(np.abs(p[:, 2]) < 1e-9, np.nan, p[:, 2])
        return np.stack([p[:, 0] / z * self.fx + self.cx, p[:, 1] / z * self.fy + self.cy], axis=1)

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)


@dataclass
class DepthSamplerConfig:
    window: int = 5                    # odd neighbourhood side; finger edges need >1 px but a big window eats the finger
    min_range_m: float = 0.15          # below: sensor blind zone / flying pixel
    max_range_m: float = 1.20          # above: not the operator's own hand
    min_valid_px: int = 4              # fewer usable pixels in the window -> landmark invalid
    max_spread_m: float = 0.06         # robust spread (MAD-based) above this -> foreground/background mix -> invalid
    edge_reject_m: float = 0.04        # pixels this far behind the window median are dropped (background bleed)
    max_hand_depth_span_m: float = 0.15  # hand-level check: a landmark this far from the hand's median depth is
                                         # background showing through a gap between fingers, not the finger
    min_landmarks_ok: int = 16         # provider-level: fewer valid landmarks than this -> not OK
    min_tip_landmarks_ok: int = 4      # of the 5 fingertips


@dataclass
class DepthSample:
    """Per-landmark depth result. `valid` is the only thing downstream code may trust."""
    depth_m: np.ndarray                # (N,) metres, NaN where invalid
    confidence: np.ndarray             # (N,) in [0,1]: fraction of usable window pixels, damped by spread
    valid: np.ndarray                  # (N,) bool
    n_used: np.ndarray                 # (N,) int, window pixels that survived filtering
    spread_m: np.ndarray               # (N,) robust spread of the window


class DepthSampler:
    """Robust per-landmark depth from an aligned depth image (spec section 8).

    For each 2D landmark: take a window, drop zeros/NaNs/out-of-range, drop pixels far behind the window median
    (background bleed / flying pixels at finger edges), then take the median of what is left. A landmark with too
    few surviving pixels or too large a spread is returned as INVALID with NaN depth."""

    def __init__(self, cfg: DepthSamplerConfig | None = None) -> None:
        self.cfg = cfg or DepthSamplerConfig()
        if self.cfg.window % 2 == 0 or self.cfg.window < 1: raise ValueError("window must be odd and >= 1")

    def sample(self, depth_m: np.ndarray, uv: np.ndarray) -> DepthSample:
        """`depth_m`: (H,W) float metres (0/NaN = no measurement). `uv`: (N,2) pixel coordinates in the SAME frame."""
        D = np.asarray(depth_m, np.float64)
        if D.ndim != 2: raise ValueError(f"depth image must be (H,W), got {D.shape}")
        P = np.asarray(uv, np.float64).reshape(-1, 2)
        H, W = D.shape; r = self.cfg.window // 2
        n = P.shape[0]
        out = DepthSample(np.full(n, np.nan), np.zeros(n), np.zeros(n, bool), np.zeros(n, int), np.full(n, np.nan))

        for i, (u, v) in enumerate(P):
            if not (np.isfinite(u) and np.isfinite(v)): continue
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < W and 0 <= vi < H): continue
            win = D[max(vi - r, 0): vi + r + 1, max(ui - r, 0): ui + r + 1].ravel()
            total = max(win.size, 1)
            ok = np.isfinite(win) & (win >= self.cfg.min_range_m) & (win <= self.cfg.max_range_m)
            if ok.sum() < self.cfg.min_valid_px: continue
            vals = win[ok]
            med = float(np.median(vals))
            near = vals[vals <= med + self.cfg.edge_reject_m]        # keep the foreground side of the window
            if near.size < self.cfg.min_valid_px: continue
            d = float(np.median(near))
            spread = float(1.4826 * np.median(np.abs(near - d)))
            out.n_used[i] = near.size; out.spread_m[i] = spread
            if spread > self.cfg.max_spread_m: continue
            out.depth_m[i] = d; out.valid[i] = True
            out.confidence[i] = float(np.clip(near.size / total, 0.0, 1.0) * np.clip(1.0 - spread / max(self.cfg.max_spread_m, 1e-9), 0.0, 1.0))
        return out


def reject_depth_outliers(sample: DepthSample, max_span_m: float) -> DepthSample:
    """Invalidate landmarks that sit far from the hand's own depth (spec section 8, "obvious discontinuities").

    A single robust window median is not enough: between two fingers the window can be entirely background, which is
    internally consistent and would pass as a confident measurement metres behind the hand. A hand is ~20 cm deep, so
    anything further from the hand's median depth than `max_span_m` is not part of it."""
    v = sample.valid
    if v.sum() < 3: return sample
    med = float(np.median(sample.depth_m[v]))
    bad = v & (np.abs(sample.depth_m - med) > max_span_m)
    if not bad.any(): return sample
    sample.valid = v & ~bad
    sample.depth_m = np.where(bad, np.nan, sample.depth_m)
    sample.confidence = np.where(bad, 0.0, sample.confidence)
    return sample


def depth_image_to_m(raw: np.ndarray, depth_scale_m: float) -> np.ndarray:
    """Raw sensor depth (uint16 counts) -> metres, with 0 (no measurement) turned into NaN, not 0 m."""
    d = np.asarray(raw).astype(np.float64) * float(depth_scale_m)
    return np.where(np.asarray(raw) == 0, np.nan, d)
