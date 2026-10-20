"""Camera <-> IMU time offset per side, per session (2026-10-14, spec section 12).

The wrist UVC cameras carry HOST receive stamps; the IMU carries its own device clock (mapped to host by
handumi_collector.pose.timing.fit_device_to_host). Nothing triggers the camera from the IMU, so the offset is MEASURED every
session by cross-correlating per-frame rotation magnitude from vision against the integrated gyro
(timing.estimate_camera_imu_offset -- reused, not re-implemented) and graded here:

    OK     peak_corr >= min_peak_corr_ok  and sharpness >= min_sharpness
    WARN   peak_corr >= min_peak_corr_warn
    PAUSE  otherwise -> teleop initialisation must not proceed on this side

Every estimate is recorded (offset, correlation, sharpness, confidence, side) into the episode metadata."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from handumi_collector.pose.timing import estimate_camera_imu_offset, fit_device_to_host, ClockFit  # noqa: F401  (re-export)


@dataclass
class OffsetEstimate:
    side: str
    offset_ms: float | None          # add to IMU host-clock times to land on the camera clock
    peak_corr: float
    sharpness: float
    confidence: float
    verdict: str                     # OK | WARN | PAUSE
    reason: str = ""
    n_visual: int = 0
    n_imu: int = 0

    def to_dict(self) -> dict: return asdict(self)


def grade(side: str, est: dict, cfg: dict) -> OffsetEstimate:
    off = est.get("offset_ms"); pc = float(est.get("peak_corr", 0.0) or 0.0); sh = float(est.get("sharpness", 0.0) or 0.0)
    if off is None:
        return OffsetEstimate(side, None, 0.0, 0.0, 0.0, "PAUSE", est.get("reason", "no estimate"))
    if abs(off) >= float(cfg["max_offset_ms"]) - 1e-6:
        return OffsetEstimate(side, off, pc, sh, 0.0, "PAUSE", "peak at the search boundary", est.get("n_visual", 0), est.get("n_imu", 0))
    if pc >= cfg["min_peak_corr_ok"] and sh >= cfg["min_sharpness"]: v, why = "OK", ""
    elif pc >= cfg["min_peak_corr_warn"]: v, why = "WARN", f"peak_corr {pc:.2f} / sharpness {sh:.3f} below OK"
    else: v, why = "PAUSE", f"peak_corr {pc:.2f} < {cfg['min_peak_corr_warn']}"
    return OffsetEstimate(side, float(off), pc, sh, float(np.clip(pc, 0, 1)), v, why, est.get("n_visual", 0), est.get("n_imu", 0))


def estimate_side(side: str, t_cam_ns, R_cam, t_imu_host_ns, gyro, cfg: dict, gyro_bias=None) -> OffsetEstimate:
    """t_cam_ns/R_cam: vision-derived wrist (or camera) rotations on the camera clock; t_imu_host_ns: IMU samples already
    mapped device->host (fit_device_to_host). Requires real rotation in the window (wave the wrist ~2 s)."""
    est = estimate_camera_imu_offset(t_cam_ns, R_cam, t_imu_host_ns, gyro, max_offset_ms=float(cfg["max_offset_ms"]), gyro_bias=gyro_bias)
    return grade(side, est, cfg)
