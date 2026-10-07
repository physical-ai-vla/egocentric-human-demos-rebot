"""Thumb-index aperture -> continuous grasp signal (0 = closed pinch, 1 = open).

MediaPipe landmark indices: WRIST 0, THUMB_TIP 4, INDEX_MCP 5, INDEX_TIP 8, PINKY_MCP 17.
Aperture is normalized by a hand-scale (index_mcp <-> pinky_mcp) so it does not depend on
distance to the camera; MediaPipe 3D is kept only as an auxiliary value, never as metric GT.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

THUMB_TIP, INDEX_MCP, INDEX_TIP, PINKY_MCP, WRIST = 4, 5, 8, 17, 0


def aperture_features(landmarks_2d: np.ndarray, landmarks_3d: np.ndarray | None = None) -> dict[str, float]:
    """landmarks_2d (21,2) pixels; landmarks_3d (21,3) MediaPipe world/normalized (aux)."""
    lm = np.asarray(landmarks_2d, dtype=np.float64).reshape(21, 2)
    raw = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]))
    scale = float(np.linalg.norm(lm[INDEX_MCP] - lm[PINKY_MCP]))
    norm = raw / scale if scale > 1e-6 else float("nan")
    out = {"aperture_raw_px": raw, "hand_scale_px": scale, "aperture_norm": norm, "aperture_model_3d": float("nan")}
    if landmarks_3d is not None:
        l3 = np.asarray(landmarks_3d, dtype=np.float64).reshape(21, 3)
        out["aperture_model_3d"] = float(np.linalg.norm(l3[THUMB_TIP] - l3[INDEX_TIP]))
    return out


@dataclass
class GraspCalibration:
    """Per-hand open/closed normalized apertures."""

    left_open: float = 1.6
    left_closed: float = 0.25
    right_open: float = 1.6
    right_closed: float = 0.25
    source: str = "default"

    def bounds(self, side: str) -> tuple[float, float]:
        return (getattr(self, f"{side}_open"), getattr(self, f"{side}_closed"))

    def grasp(self, side: str, aperture_norm: np.ndarray | float) -> np.ndarray | float:
        open_, closed = self.bounds(side)
        span = open_ - closed
        a = np.asarray(aperture_norm, dtype=np.float64)
        g = np.clip((a - closed) / span, 0.0, 1.0) if abs(span) > 1e-9 else np.full_like(a, np.nan)
        return np.where(np.isfinite(a), g, np.nan)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: Path) -> "GraspCalibration":
        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_samples(cls, open_norm: dict[str, np.ndarray], closed_norm: dict[str, np.ndarray], *, percentile: float = 10.0) -> "GraspCalibration":
        """Robust estimate: open = 90th pct of open samples, closed = 10th pct of closed samples."""
        kw: dict[str, Any] = {"source": "calibrate_grasp"}
        for side in ("left", "right"):
            o = np.asarray(open_norm.get(side, []), dtype=np.float64)
            c = np.asarray(closed_norm.get(side, []), dtype=np.float64)
            o, c = o[np.isfinite(o)], c[np.isfinite(c)]
            if len(o) >= 5 and len(c) >= 5:
                kw[f"{side}_open"] = float(np.percentile(o, 100 - percentile))
                kw[f"{side}_closed"] = float(np.percentile(c, percentile))
        return cls(**kw)


@dataclass
class ApertureCalibration:
    """Per-hand open/closed *metric* thumb-index apertures (from the finger AprilTags).

    Unlike :class:`GraspCalibration` (MediaPipe, normalized by hand scale) this is in meters,
    so it maps directly to a gripper width: grasp = clip((aperture - closed) / (open - closed)).
    """

    left_open_m: float = 0.085
    left_closed_m: float = 0.010
    right_open_m: float = 0.085
    right_closed_m: float = 0.010
    source: str = "default"

    def bounds(self, side: str) -> tuple[float, float]:
        return (getattr(self, f"{side}_open_m"), getattr(self, f"{side}_closed_m"))

    def grasp(self, side: str, aperture_m: np.ndarray | float) -> np.ndarray | float:
        open_, closed = self.bounds(side)
        span = open_ - closed
        a = np.asarray(aperture_m, dtype=np.float64)
        g = np.clip((a - closed) / span, 0.0, 1.0) if abs(span) > 1e-9 else np.full_like(a, np.nan)
        return np.where(np.isfinite(a), g, np.nan)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: Path) -> "ApertureCalibration":
        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_samples(cls, open_m: dict[str, np.ndarray], closed_m: dict[str, np.ndarray], *, percentile: float = 10.0) -> "ApertureCalibration":
        """Robust estimate: open = 90th pct of open samples, closed = 10th pct of closed samples."""
        kw: dict[str, Any] = {"source": "calibrate_aperture"}
        for side in ("left", "right"):
            o = np.asarray(open_m.get(side, []), dtype=np.float64)
            c = np.asarray(closed_m.get(side, []), dtype=np.float64)
            o, c = o[np.isfinite(o)], c[np.isfinite(c)]
            if len(o) >= 5 and len(c) >= 5:
                kw[f"{side}_open_m"] = float(np.percentile(o, 100 - percentile))
                kw[f"{side}_closed_m"] = float(np.percentile(c, percentile))
        return cls(**kw)


__all__ = ["INDEX_MCP", "INDEX_TIP", "PINKY_MCP", "THUMB_TIP", "WRIST", "ApertureCalibration", "GraspCalibration", "aperture_features"]
