"""Confidence helpers (2026-10-14, spec 26). One place that turns raw detector numbers into the per-hand confidences the
dashboard shows and HandFusion gates on (thresholds: teleop_rebot.yaml tracking.*)."""
from __future__ import annotations
from .rgbd_pose import position_confidence  # noqa: F401


def fingertip_confidence(result) -> float:
    """FingertipResult -> [0, 1]; 0 when no pinch metric could be measured."""
    return float(result.confidence) if result is not None and result.detected else 0.0
