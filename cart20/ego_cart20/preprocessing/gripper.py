"""Gripper aperture normalization.  Contract (D2): g = aperture_mm / w_open_mm, 0 = closed, 1 = open.

Calibration happens BEFORE clipping: raw ticks -> mm through the monotone caliper knots, then mm / w_open, then the
[0, 1] clip.  ``validate_gripper`` is a hard check on the stored values.
"""
import numpy as np


def ticks_to_mm(ticks, side_cal):
    d = side_cal.get("direction", 1) * (np.asarray(ticks, np.float64) - side_cal["touch_tick"])
    return np.interp(d, side_cal["knots_delta_ticks"], side_cal["knots_mm"], left=0.0, right=side_cal["w_open_mm"])


def mm_to_g(mm, w_open_mm):
    return np.clip(np.asarray(mm, np.float64) / w_open_mm, 0.0, 1.0)


def validate_gripper(g, name="gripper"):
    g = np.asarray(g); f = g[np.isfinite(g)]
    if f.size and (f.min() < 0.0 or f.max() > 1.0):
        raise ValueError(f"{name} outside [0, 1]: min {f.min()} max {f.max()}")
    return dict(min=float(f.min()) if f.size else None, max=float(f.max()) if f.size else None)
