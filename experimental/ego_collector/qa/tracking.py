"""Tracking QA metrics over processed tracking parquets (used by `qa` and `tracking-test`)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ego_collector.tracking.transforms import pose7_to_T, rotation_angle_deg

POSE = ("x", "y", "z", "qx", "qy", "qz", "qw")


def pose_array(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"{prefix}_{k}" for k in POSE]].to_numpy(dtype=np.float64)


@dataclass
class JumpStats:
    max_translation_speed_m_s: float
    max_rotation_step_deg: float
    catastrophic_jumps: int


def jump_stats(poses7: np.ndarray, valid: np.ndarray, timestamps_ns: np.ndarray, *, max_speed_m_s: float = 4.0, max_rot_step_deg: float = 60.0) -> JumpStats:
    """Speed / rotation step between consecutive *valid* frames; counts frames exceeding the limits."""
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return JumpStats(0.0, 0.0, 0)
    speeds, steps = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        dt = (timestamps_ns[b] - timestamps_ns[a]) / 1e9
        if dt <= 0:
            continue
        dp = np.linalg.norm(poses7[b, :3] - poses7[a, :3])
        speeds.append(dp / dt)
        steps.append(rotation_angle_deg(pose7_to_T(poses7[a]), pose7_to_T(poses7[b])))
    speeds, steps = np.asarray(speeds), np.asarray(steps)
    jumps = int(np.sum((speeds > max_speed_m_s) | (steps > max_rot_step_deg)))
    return JumpStats(float(speeds.max()) if speeds.size else 0.0, float(steps.max()) if steps.size else 0.0, jumps)


@dataclass
class StaticJitter:
    samples: int
    sigma_xyz_mm: list[float]
    translation_rms_mm: float
    rotation_rms_deg: float


def static_jitter(poses7: np.ndarray, valid: np.ndarray) -> StaticJitter:
    p = poses7[valid]
    if len(p) < 5:
        return StaticJitter(int(len(p)), [float("nan")] * 3, float("nan"), float("nan"))
    centred = p[:, :3] - np.median(p[:, :3], axis=0)
    from scipy.spatial.transform import Rotation

    q = p[:, 3:7].copy()
    q[(q @ q[0]) < 0] *= -1
    rots = Rotation.from_quat(q)
    rel = rots.mean().inv() * rots
    rotvec = np.degrees(rel.as_rotvec())
    return StaticJitter(
        samples=int(len(p)),
        sigma_xyz_mm=[float(v) for v in centred.std(axis=0) * 1000],
        translation_rms_mm=float(np.sqrt(np.mean(np.sum(centred**2, axis=1))) * 1000),
        rotation_rms_deg=float(np.sqrt(np.mean(np.sum(rotvec**2, axis=1)))),
    )


@dataclass
class Repeatability:
    dwells: int
    dwell_positions_m: list[list[float]]
    max_spread_mm: float
    mean_spread_mm: float


def dwell_repeatability(poses7: np.ndarray, valid: np.ndarray, timestamps_ns: np.ndarray, *, speed_thresh_m_s: float = 0.03, min_dwell_s: float = 0.5, cluster_radius_m: float = 0.04) -> Repeatability:
    """Find low-speed dwell segments, take their median positions, and measure the spread of dwells
    that revisit the same physical point (clusters within ``cluster_radius_m``)."""
    idx = np.flatnonzero(valid)
    if len(idx) < 10:
        return Repeatability(0, [], float("nan"), float("nan"))
    p = poses7[idx, :3]
    t = timestamps_ns[idx] / 1e9
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
    slow = np.concatenate([[False], v < speed_thresh_m_s])
    dwells = []
    start = None
    for i, s in enumerate(slow):
        if s and start is None:
            start = i
        if (not s or i == len(slow) - 1) and start is not None:
            end = i if not s else i + 1
            if t[end - 1] - t[start] >= min_dwell_s:
                dwells.append(np.median(p[start:end], axis=0))
            start = None
    if len(dwells) < 2:
        return Repeatability(len(dwells), [d.tolist() for d in dwells], float("nan"), float("nan"))
    centres = np.asarray(dwells)
    spreads = []
    used = np.zeros(len(centres), bool)
    for i in range(len(centres)):
        if used[i]:
            continue
        members = np.flatnonzero(np.linalg.norm(centres - centres[i], axis=1) < cluster_radius_m)
        used[members] = True
        if len(members) >= 2:
            c = centres[members].mean(axis=0)
            spreads.append(np.max(np.linalg.norm(centres[members] - c, axis=1)) * 1000)
    return Repeatability(len(dwells), centres.tolist(), float(max(spreads)) if spreads else float("nan"), float(np.mean(spreads)) if spreads else float("nan"))


def aperture_noise_mm(aperture_m: np.ndarray, valid: np.ndarray) -> float:
    """Grip-width noise: median |frame-to-frame aperture change| between consecutive valid frames, mm.

    During normal manipulation the true aperture moves slowly per frame (30 Hz), so the median
    step is dominated by measurement noise, not motion.
    """
    idx = np.flatnonzero(np.asarray(valid, bool) & np.isfinite(aperture_m))
    pairs = idx[1:][np.diff(idx) == 1]
    if len(pairs) < 5:
        return float("nan")
    return float(np.median(np.abs(aperture_m[pairs] - aperture_m[pairs - 1])) * 1000)


def finger_summary(finger: pd.DataFrame) -> dict:
    out: dict = {}
    n = len(finger)
    for side in ("left", "right"):
        for f in ("thumb", "index"):
            out[f"{side}_{f}_visible_fraction"] = float(finger[f"{side}_{f}_visible"].mean()) if n else 0.0
        valid = finger[f"{side}_aperture_valid"].to_numpy(bool) if n else np.zeros(0, bool)
        a = finger[f"{side}_aperture_m"].to_numpy(dtype=np.float64) if n else np.zeros(0)
        out[f"{side}_aperture_valid_fraction"] = float(valid.mean()) if n else 0.0
        out[f"{side}_aperture_noise_mm"] = aperture_noise_mm(a, valid)
        fin = a[np.isfinite(a)]
        out[f"{side}_aperture_min_mm"] = float(fin.min() * 1000) if fin.size else float("nan")
        out[f"{side}_aperture_max_mm"] = float(fin.max() * 1000) if fin.size else float("nan")
    return out


def tracking_summary(cam: pd.DataFrame, wrist: pd.DataFrame, finger: pd.DataFrame | None = None) -> dict:
    ts = cam["timestamp_ns"].to_numpy()
    out: dict = {
        "frames": int(len(cam)),
        "world_pose_valid_fraction": float(cam["world_pose_valid"].mean()) if len(cam) else 0.0,
        "world_reprojection_error_median_px": float(np.nanmedian(cam["world_reprojection_error"])) if len(cam) and cam["world_pose_valid"].any() else float("nan"),
        "world_tag_count_mean": float(cam["world_tag_count"].mean()) if len(cam) else 0.0,
    }
    head = jump_stats(pose_array(cam, "head"), cam["world_pose_valid"].to_numpy(bool), ts)
    out["head"] = asdict(head)
    # Visibility ratios (the FOV bottleneck metric for a single 78-deg head camera):
    world_vis = (cam["visible_world_tags"].to_numpy() >= 1) if "visible_world_tags" in cam else cam["world_pose_valid"].to_numpy(bool)
    left_vis = wrist["left_tag_visible"].to_numpy(bool)
    right_vis = wrist["right_tag_visible"].to_numpy(bool)
    n = min(len(world_vis), len(left_vis), len(right_vis))
    out["world_visible_ratio"] = float(world_vis[:n].mean()) if n else 0.0
    out["left_wrist_visible_ratio"] = float(left_vis[:n].mean()) if n else 0.0
    out["right_wrist_visible_ratio"] = float(right_vis[:n].mean()) if n else 0.0
    out["all_required_visible_ratio"] = float((world_vis[:n] & left_vis[:n] & right_vis[:n]).mean()) if n else 0.0
    for side in ("left", "right"):
        valid = wrist[f"{side}_wrist_valid"].to_numpy(bool)
        out[f"{side}_wrist_valid_fraction"] = float(valid.mean()) if len(valid) else 0.0
        out[f"{side}_tag_visible_fraction"] = float(wrist[f"{side}_tag_visible"].mean()) if len(valid) else 0.0
        out[f"{side}_tag_reprojection_error_median_px"] = float(np.nanmedian(wrist[f"{side}_tag_reprojection_error"])) if valid.any() else float("nan")
        out[f"{side}_wrist"] = asdict(jump_stats(pose_array(wrist, f"{side}_wrist"), valid, ts))
    if finger is not None:
        out["fingers"] = finger_summary(finger)
    return out


__all__ = ["JumpStats", "Repeatability", "StaticJitter", "aperture_noise_mm", "dwell_repeatability", "finger_summary", "jump_stats", "pose_array", "static_jitter", "tracking_summary"]
