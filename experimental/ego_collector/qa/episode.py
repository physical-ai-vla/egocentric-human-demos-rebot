"""Episode QA -> qa/report.json and dataset tier (action_labelled | video_only).

Bad tracking never deletes video: the episode simply becomes ``video_only``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ego_collector.io.parquet import read_parquet
from ego_collector.qa.tracking import tracking_summary
from ego_collector.recording.episode import EpisodePaths


@dataclass
class QAThresholds:
    min_world_valid_fraction: float = 0.95
    min_wrist_valid_fraction: float = 0.95
    min_hand_valid_fraction: float = 0.80
    max_catastrophic_jumps: int = 0
    max_world_reprojection_px: float = 2.0
    max_wrist_reprojection_px: float = 2.0
    max_video_drop_fraction: float = 0.01
    min_duration_s: float = 1.0
    require_both_hands: bool = True
    require_world: bool = True  # False = head-frame-only dataset (no world tags): world checks become soft
    min_fps_warn: float = 28.0  # soft: C922 lowers its frame rate in dim light
    sides: str = "both"  # "both" | "left" | "right" - which hands the dataset requires (single-hand collection: "right")
    min_aperture_valid_fraction: float = 0.80  # hard when finger tags were processed (grip-width availability)
    max_aperture_noise_mm: float = 5.0  # soft: median frame-to-frame grip-width step

    @property
    def required_sides(self) -> tuple[str, ...]:
        return ("left", "right") if self.sides == "both" else (self.sides,)

    @classmethod
    def from_yaml(cls, path: Path | None) -> "QAThresholds":
        if path is None or not Path(path).exists():
            return cls()
        d = yaml.safe_load(Path(path).read_text()) or {}
        d = d.get("qa", d)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class QAReport:
    episode: str
    tier: str
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check(checks: dict, name: str, ok: bool, value: Any, threshold: Any, *, hard: bool = True) -> bool:
    checks[name] = {"pass": bool(ok), "value": value, "threshold": threshold, "hard": hard}
    return bool(ok)


def evaluate_episode(paths: EpisodePaths, thresholds: QAThresholds = QAThresholds()) -> QAReport:
    meta = paths.read_metadata()
    report = QAReport(episode=paths.root.name, tier="video_only")
    checks = report.checks
    hard_ok = True

    # -- video / recording -------------------------------------------------------
    frames = int(meta.get("frame_count", 0))
    drop = int(meta.get("dropped_frames", 0))
    drop_frac = drop / max(frames + drop, 1)
    hard_ok &= _check(checks, "video_drop_fraction", drop_frac <= thresholds.max_video_drop_fraction, drop_frac, thresholds.max_video_drop_fraction)
    hard_ok &= _check(checks, "duration_s", float(meta.get("duration_s", 0.0)) >= thresholds.min_duration_s, meta.get("duration_s", 0.0), thresholds.min_duration_s)
    fps_measured = float(meta.get("fps_measured", 30.0) or 0.0)
    if frames > 10 and fps_measured < thresholds.min_fps_warn:
        report.warnings.append(f"camera delivered {fps_measured:.1f} fps (< {thresholds.min_fps_warn}): auto-exposure lengthened -> add light / fix exposure")
    _check(checks, "fps_measured", frames <= 10 or fps_measured >= thresholds.min_fps_warn, fps_measured, thresholds.min_fps_warn, hard=False)

    # -- tracking ---------------------------------------------------------------------
    if not (paths.camera_pose.exists() and paths.wrist_pose.exists()):
        report.warnings.append("tracking not processed (run process-tags)")
        _check(checks, "tracking_processed", False, False, True)
        report.metrics["meta"] = {k: meta.get(k) for k in ("task", "instruction", "frame_count", "duration_s")}
        return report
    cam = read_parquet(paths.camera_pose)
    wrist = read_parquet(paths.wrist_pose)
    finger = read_parquet(paths.finger_pose) if paths.finger_pose.exists() else None
    ts = tracking_summary(cam, wrist, finger)
    report.metrics["tracking"] = ts
    sides = thresholds.required_sides
    report.metrics["visibility"] = {k: ts[k] for k in ("world_visible_ratio", "left_wrist_visible_ratio", "right_wrist_visible_ratio", "all_required_visible_ratio")}
    world_hard = thresholds.require_world and bool(meta.get("tracking", {}).get("world_tags", True))
    ok = ts["world_pose_valid_fraction"] >= thresholds.min_world_valid_fraction
    if world_hard:
        hard_ok &= _check(checks, "world_pose_valid_fraction", ok, ts["world_pose_valid_fraction"], thresholds.min_world_valid_fraction)
    else:
        _check(checks, "world_pose_valid_fraction", ok, ts["world_pose_valid_fraction"], thresholds.min_world_valid_fraction, hard=False)
        report.warnings.append("head-frame-only episode (no world tags): actions are egocentric, not table-referenced")
    for side in sides:
        # without world tags the wrist is "valid" whenever its tag is visible (head frame)
        frac = ts[f"{side}_wrist_valid_fraction"] if world_hard else ts[f"{side}_tag_visible_fraction"]
        hard_ok &= _check(checks, f"{side}_wrist_valid_fraction", frac >= thresholds.min_wrist_valid_fraction, frac, thresholds.min_wrist_valid_fraction)
        hard_ok &= _check(checks, f"{side}_wrist_jumps", ts[f"{side}_wrist"]["catastrophic_jumps"] <= thresholds.max_catastrophic_jumps, ts[f"{side}_wrist"]["catastrophic_jumps"], thresholds.max_catastrophic_jumps)
        rep = ts[f"{side}_tag_reprojection_error_median_px"]
        _check(checks, f"{side}_tag_reprojection_px", (not np.isfinite(rep)) or rep <= thresholds.max_wrist_reprojection_px, rep, thresholds.max_wrist_reprojection_px, hard=False)

    # -- finger tags / metric grasp -----------------------------------------------------
    tag_grasp = False
    if finger is not None:
        fs = ts["fingers"]
        tag_grasp = True
        for side in sides:
            frac = fs[f"{side}_aperture_valid_fraction"]
            ok = frac >= thresholds.min_aperture_valid_fraction
            hard_ok &= _check(checks, f"{side}_aperture_valid_fraction", ok, frac, thresholds.min_aperture_valid_fraction)
            tag_grasp &= ok
            noise = fs[f"{side}_aperture_noise_mm"]
            _check(checks, f"{side}_aperture_noise_mm", (not np.isfinite(noise)) or noise <= thresholds.max_aperture_noise_mm, noise, thresholds.max_aperture_noise_mm, hard=False)
    if world_hard:
        hard_ok &= _check(checks, "head_jumps", ts["head"]["catastrophic_jumps"] <= thresholds.max_catastrophic_jumps, ts["head"]["catastrophic_jumps"], thresholds.max_catastrophic_jumps)
    rep = ts["world_reprojection_error_median_px"]
    _check(checks, "world_reprojection_px", (not np.isfinite(rep)) or rep <= thresholds.max_world_reprojection_px, rep, thresholds.max_world_reprojection_px, hard=False)

    # -- hands / grasp ----------------------------------------------------------------
    hands_hard = thresholds.require_both_hands and not tag_grasp  # tag aperture passing makes MediaPipe optional
    if paths.hand_pose.exists():
        hp = read_parquet(paths.hand_pose)
        for side in sides:
            frac = float(hp[f"{side}_hand_visible"].mean()) if len(hp) else 0.0
            report.metrics[f"{side}_hand_visible_fraction"] = frac
            ok = frac >= thresholds.min_hand_valid_fraction
            if hands_hard:
                hard_ok &= _check(checks, f"{side}_hand_valid_fraction", ok, frac, thresholds.min_hand_valid_fraction)
            else:
                _check(checks, f"{side}_hand_valid_fraction", ok, frac, thresholds.min_hand_valid_fraction, hard=False)
    else:
        if not tag_grasp:
            report.warnings.append("hands not processed (run process-hands); grasp will be NaN")
        if hands_hard:
            hard_ok &= _check(checks, "hands_processed", False, False, True)
    action_file = paths.pseudo_actions if world_hard or not paths.pseudo_actions_head.exists() else paths.pseudo_actions_head
    if action_file.exists():
        pa = read_parquet(action_file)
        av = np.ones(len(pa), bool)
        for side in sides:
            av &= pa[f"{side}_delta_valid"].to_numpy(bool) & np.isfinite(pa[f"{side}_grasp"].to_numpy(dtype=np.float64))
        frac = float(av.mean()) if len(pa) else 0.0
        report.metrics["action_valid_fraction"] = frac
        _check(checks, "action_valid_fraction", frac >= min(thresholds.min_wrist_valid_fraction, thresholds.min_hand_valid_fraction) * 0.9, frac, None, hard=False)

    soft_fails = [k for k, c in checks.items() if not c["pass"] and not c["hard"]]
    if soft_fails:
        report.warnings.append(f"soft checks failed: {soft_fails}")
    report.tier = "action_labelled" if hard_ok else "video_only"
    report.metrics["meta"] = {k: meta.get(k) for k in ("task", "instruction", "frame_count", "duration_s", "outcome", "recovery")}
    return report


def run_qa(paths: EpisodePaths, thresholds: QAThresholds = QAThresholds()) -> QAReport:
    report = evaluate_episode(paths, thresholds)
    paths.qa_report.parent.mkdir(parents=True, exist_ok=True)
    paths.qa_report.write_text(json.dumps(report.to_dict(), indent=2, default=float))
    paths.update_metadata(dataset_tier=report.tier, qa_failed_checks=[k for k, c in report.checks.items() if not c["pass"] and c["hard"]])
    return report


__all__ = ["QAReport", "QAThresholds", "evaluate_episode", "run_qa"]
