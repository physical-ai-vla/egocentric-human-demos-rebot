"""Derived (never raw-modifying) signals for a HandUMI episode.

* relative EEF motion          -> :func:`handumi.dataset.eef_actions.local_delta`
* bimanual relation            -> :func:`bimanual_relation`  (T_left_right, distance, rel. velocity)
* gripper events               -> :func:`gripper_events`     (opening/closing/grasp_start/release)
* motor-current contact proxy  -> :func:`contact_proxy`      (baseline, delta, contact_estimate, slip)

All inputs are plain arrays sampled at the dataset rate; outputs are arrays of the
same length so they can be written as extra columns next to the raw episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from handumi.dataset.eef_actions import local_delta

# ---------------------------------------------------------------------------
# Bimanual relation
# ---------------------------------------------------------------------------


def bimanual_relation(left_tcp: np.ndarray, right_tcp: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Right TCP expressed in the left TCP frame, plus distance and its rate."""
    L = np.asarray(left_tcp, dtype=np.float64)
    R = np.asarray(right_tcp, dtype=np.float64)
    rl = Rotation.from_quat(L[:, 3:7])
    rr = Rotation.from_quat(R[:, 3:7])
    rel_pos = rl.inv().apply(R[:, :3] - L[:, :3])
    rel_rot = rl.inv() * rr
    dist = np.linalg.norm(R[:, :3] - L[:, :3], axis=1)
    vel = np.gradient(dist) * float(fps) if len(dist) > 1 else np.zeros_like(dist)
    return {
        "left_from_right": np.concatenate([rel_pos, rel_rot.as_quat()], axis=1).astype(np.float32),
        "relative_xyz": rel_pos.astype(np.float32),
        "relative_rotvec": rel_rot.as_rotvec().astype(np.float32),
        "hand_distance": dist.astype(np.float32),
        "hand_distance_rate": vel.astype(np.float32),
    }


def relative_motion(tcp: np.ndarray, horizon: int = 1) -> np.ndarray:
    """(T, 6) local increments, zero-padded at the end to keep the row count."""
    d = local_delta(tcp, horizon)
    pad = np.zeros((min(horizon, len(tcp)), 6), dtype=np.float32)
    return np.concatenate([d, pad], axis=0)


def head_relative_tcp(device_controller_pose7: np.ndarray, controller_to_tcp7: np.ndarray) -> np.ndarray:
    """TCP in the head-camera frame: ``T_cam_anchor · T_anchor_TCP`` per row (AprilTag backend)."""
    from handumi.robots.utils import pose_mul

    poses = np.asarray(device_controller_pose7, dtype=np.float64).reshape(-1, 7)
    off = np.asarray(controller_to_tcp7, dtype=np.float64).reshape(7)
    return np.stack([pose_mul(p, off) for p in poses], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Gripper events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GripperEventConfig:
    speed_threshold_mm_s: float = 15.0  # |d width/dt| above this = moving
    smooth_window: int = 5  # moving-average window (samples) on width before differentiating
    grasp_max_width_mm: float = 60.0  # a stop below this width after closing counts as a grasp
    min_hold_samples: int = 3  # stationary samples required to confirm grasp/release


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return np.asarray(x, dtype=np.float64)
    kernel = np.ones(window) / window
    padded = np.pad(np.asarray(x, dtype=np.float64), (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def gripper_events(width_mm: np.ndarray, fps: float, config: GripperEventConfig = GripperEventConfig()) -> dict[str, np.ndarray]:
    """Classify jaw motion and mark grasp/release onsets.

    ``grasp_start[t]`` is 1 on the first stationary sample after a closing
    phase that ended below ``grasp_max_width_mm``; ``release[t]`` is 1 on the
    first opening sample after a grasp. ``grasped`` is 1 between the two.
    """
    w = np.asarray(width_mm, dtype=np.float64)
    n = len(w)
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return {"velocity_mm_s": np.zeros(0, np.float32), "closing": z, "opening": z, "grasp_start": z, "release": z, "grasped": z}
    ws = _smooth(w, config.smooth_window)
    v = np.gradient(ws) * float(fps) if n > 1 else np.zeros(1)
    closing = v < -config.speed_threshold_mm_s
    opening = v > config.speed_threshold_mm_s
    stationary = ~closing & ~opening
    grasp_start = np.zeros(n, dtype=bool)
    release = np.zeros(n, dtype=bool)
    grasped = np.zeros(n, dtype=bool)
    was_closing = False
    holding = False
    for t in range(n):
        if closing[t]:
            was_closing = True
            continue
        if was_closing and stationary[t] and ws[t] < config.grasp_max_width_mm and not holding:
            end = min(n, t + config.min_hold_samples)
            if stationary[t:end].all():
                grasp_start[t] = True
                holding = True
            was_closing = False
        elif opening[t]:
            was_closing = False
            if holding:
                release[t] = True
                holding = False
        if holding:
            grasped[t] = True
    return {
        "velocity_mm_s": v.astype(np.float32),
        "closing": closing.astype(np.int64),
        "opening": opening.astype(np.int64),
        "grasp_start": grasp_start.astype(np.int64),
        "release": release.astype(np.int64),
        "grasped": grasped.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Motor-current contact proxy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContactProxyConfig:
    baseline_percentile: float = 20.0  # free-motion current level
    contact_delta_ma: float = 80.0  # current rise (mA) above baseline that reads as contact
    slope_ma: float = 30.0  # logistic softness
    slip_drop_ma: float = 60.0  # sudden drop while closed/holding -> possible slip/release
    slip_window: int = 3


def contact_proxy(
    current_ma: np.ndarray,
    width_mm: np.ndarray,
    fps: float,
    *,
    events: dict[str, np.ndarray] | None = None,
    config: ContactProxyConfig = ContactProxyConfig(),
) -> dict[str, np.ndarray]:
    """Turn jaw motor current into a soft contact estimate and a slip flag.

    closing/closed + current rise  -> contact likely
    closed + sudden current drop   -> possible slip / release
    """
    c = np.asarray(current_ma, dtype=np.float64)
    n = len(c)
    if n == 0:
        z = np.zeros(0, np.float32)
        return {"current_baseline_ma": z, "current_delta_ma": z, "contact_estimate": z, "possible_slip": np.zeros(0, np.int64)}
    if events is None:
        events = gripper_events(width_mm, fps)
    baseline = float(np.percentile(c, config.baseline_percentile))
    delta = c - baseline
    logistic = 1.0 / (1.0 + np.exp(-(delta - config.contact_delta_ma) / max(config.slope_ma, 1e-6)))
    engaged = (events["closing"] | events["grasped"]).astype(bool)
    contact = np.where(engaged, logistic, 0.0)
    slip = np.zeros(n, dtype=bool)
    k = max(1, config.slip_window)
    for t in range(k, n):
        if events["grasped"][t - k] and (c[t - k] - c[t]) > config.slip_drop_ma and contact[t - k] > 0.5:
            slip[t] = True
    return {
        "current_baseline_ma": np.full(n, baseline, dtype=np.float32),
        "current_delta_ma": delta.astype(np.float32),
        "contact_estimate": contact.astype(np.float32),
        "possible_slip": slip.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Episode-level assembly
# ---------------------------------------------------------------------------


def derive_episode(
    *,
    left_tcp: np.ndarray,
    right_tcp: np.ndarray,
    left_width_mm: np.ndarray,
    right_width_mm: np.ndarray,
    fps: float,
    left_current_ma: np.ndarray | None = None,
    right_current_ma: np.ndarray | None = None,
    timestamps_s: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Flat column dict (all length T) ready for a DataFrame/parquet."""
    T = len(left_tcp)
    cols: dict[str, np.ndarray] = {}
    cols["timestamp"] = np.asarray(timestamps_s, dtype=np.float32) if timestamps_s is not None else (np.arange(T) / float(fps)).astype(np.float32)
    for side, tcp in (("left", left_tcp), ("right", right_tcp)):
        d = relative_motion(tcp)
        for i, name in enumerate(("dx", "dy", "dz", "drx", "dry", "drz")):
            cols[f"{side}.delta.{name}"] = d[:, i]
    rel = bimanual_relation(left_tcp, right_tcp, fps)
    for i, name in enumerate(("x", "y", "z", "qx", "qy", "qz", "qw")):
        cols[f"bimanual.left_from_right.{name}"] = rel["left_from_right"][:, i]
    cols["bimanual.hand_distance"] = rel["hand_distance"]
    cols["bimanual.hand_distance_rate"] = rel["hand_distance_rate"]
    for side, width, current in (("left", left_width_mm, left_current_ma), ("right", right_width_mm, right_current_ma)):
        ev = gripper_events(width, fps)
        for k, v in ev.items():
            cols[f"{side}.gripper.{k}"] = v
        if current is not None:
            cp = contact_proxy(current, width, fps, events=ev)
            for k, v in cp.items():
                cols[f"{side}.motor.{k}"] = v
    return cols


__all__ = [
    "ContactProxyConfig",
    "GripperEventConfig",
    "bimanual_relation",
    "contact_proxy",
    "derive_episode",
    "gripper_events",
    "head_relative_tcp",
    "relative_motion",
]
