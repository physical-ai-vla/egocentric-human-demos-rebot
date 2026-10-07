"""STEP 6-7: wrist_pose.parquet (+ hand_pose.parquet) -> pseudo_actions.parquet, bimanual_features.parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ego_collector.actions.bimanual import bimanual_features
from ego_collector.actions.pseudo_action import POSE, SIDES, pseudo_action_table, states_from_arrays
from ego_collector.filtering.pose_filter import OneEuroConfig, one_euro_pose7
from ego_collector.hands.grasp import ApertureCalibration, GraspCalibration, aperture_features
from ego_collector.io.parquet import read_parquet, stack_column, write_parquet
from ego_collector.recording.episode import EpisodePaths

log = logging.getLogger("ego_collector.actions")


def grasp_signals(paths: EpisodePaths, cal: GraspCalibration, n_frames: int) -> dict[str, dict[str, np.ndarray]]:
    """Per side: aperture_norm, aperture_raw_px, aperture_model_3d, grasp (NaN when the hand is not visible)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    hp = read_parquet(paths.hand_pose) if paths.hand_pose.exists() else None
    for side in SIDES:
        cols = {k: np.full(n_frames, np.nan) for k in ("aperture_norm", "aperture_raw_px", "aperture_model_3d", "hand_confidence")}
        if hp is not None:
            vis = hp[f"{side}_hand_visible"].to_numpy(bool)
            l2 = stack_column(hp, f"{side}_landmarks_2d", 42).reshape(len(hp), 21, 2)
            l3 = stack_column(hp, f"{side}_landmarks_3d", 63).reshape(len(hp), 21, 3)
            conf = hp[f"{side}_confidence"].to_numpy(dtype=np.float64)
            for i in range(min(n_frames, len(hp))):
                if vis[i]:
                    f = aperture_features(l2[i], l3[i])
                    cols["aperture_norm"][i] = f["aperture_norm"]
                    cols["aperture_raw_px"][i] = f["aperture_raw_px"]
                    cols["aperture_model_3d"][i] = f["aperture_model_3d"]
                    cols["hand_confidence"][i] = conf[i]
        cols["grasp"] = np.asarray(cal.grasp(side, cols["aperture_norm"]), dtype=np.float64)
        out[side] = cols
    return out


def tag_apertures(paths: EpisodePaths, n_frames: int) -> dict[str, np.ndarray] | None:
    """Per-side metric thumb-index aperture from tracking/finger_pose.parquet (None if absent)."""
    if not paths.finger_pose.exists():
        return None
    fp = read_parquet(paths.finger_pose)
    out = {}
    for side in SIDES:
        a = np.full(n_frames, np.nan)
        col = fp[f"{side}_aperture_m"].to_numpy(dtype=np.float64)
        a[: min(n_frames, len(col))] = col[:n_frames]
        out[side] = a
    return out


FRAMES = ("world", "head")


def _action_columns(wp, ts, n, g, *, frame: str, horizon: int, filter_config: OneEuroConfig | None) -> dict[str, np.ndarray]:
    """Pseudo-action table for one reference frame.

    world: T_world_wrist (needs world tags; valid = wrist_valid)
    head : T_camera_wrist, i.e. egocentric / head-relative (valid = tag_visible only, no world tags needed).
    """
    if frame == "world":
        poses = {s: wp[[f"{s}_wrist_{k}" for k in POSE]].to_numpy(dtype=np.float64) for s in SIDES}
        valid = {s: wp[f"{s}_wrist_valid"].to_numpy(bool) for s in SIDES}
    elif frame == "head":
        poses = {s: wp[[f"{s}_wrist_cam_{k}" for k in POSE]].to_numpy(dtype=np.float64) for s in SIDES}
        valid = {s: wp[f"{s}_tag_visible"].to_numpy(bool) for s in SIDES}
    else:
        raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
    states = states_from_arrays(ts, poses, valid, {s: g[s]["grasp"] for s in SIDES})
    cols = pseudo_action_table(states, horizon=horizon)
    cols["frame_index"] = wp["frame_index"].to_numpy(dtype=np.int64)
    cols["reference_frame"] = np.full(n, frame, dtype=object)
    for s in SIDES:
        for k in g[s]:
            if k != "grasp":
                cols[f"{s}_{k}"] = g[s][k]
        if filter_config is not None:
            filt = one_euro_pose7(poses[s], valid[s], ts, filter_config)
            for j, k in enumerate(POSE):
                cols[f"{s}_{k}_filtered"] = filt[:, j]
    return cols, poses, valid


def generate_actions(
    paths: EpisodePaths,
    *,
    grasp_cal: GraspCalibration,
    aperture_cal: ApertureCalibration | None = None,
    grasp_source: str = "auto",
    horizon: int = 1,
    filter_config: OneEuroConfig | None = OneEuroConfig(),
    frames: tuple[str, ...] = ("world", "head"),
) -> dict[str, float]:
    """Write actions/pseudo_actions.parquet (world) and/or actions/pseudo_actions_head.parquet (head-relative).

    The head-relative variant needs no world tags: it is the egocentric ablation (hand motion
    *and* head motion mixed), useful for video-conditioned pretraining, not for robot retargeting.

    grasp_source: "tags" = metric thumb-index aperture from the finger AprilTags (finger_pose.parquet),
    "mediapipe" = normalized MediaPipe aperture, "auto" = tags when available, else MediaPipe.
    """
    wp = read_parquet(paths.wrist_pose)
    ts = wp["timestamp_ns"].to_numpy(dtype=np.int64)
    n = len(wp)
    g = grasp_signals(paths, grasp_cal, n)
    tag_ap = tag_apertures(paths, n) if grasp_source in ("auto", "tags") else None
    use_tags = tag_ap is not None and (grasp_source == "tags" or any(np.isfinite(a).any() for a in tag_ap.values()))
    if grasp_source == "tags" and tag_ap is None:
        raise FileNotFoundError(f"{paths.finger_pose} missing: run process-tags with finger tags configured, or use --grasp-source mediapipe")
    if use_tags:
        cal = aperture_cal or ApertureCalibration()
        for side in SIDES:
            g[side]["aperture_m"] = tag_ap[side]
            g[side]["grasp"] = np.asarray(cal.grasp(side, tag_ap[side]), dtype=np.float64)
    summary: dict = {"frames": int(n), "horizon": int(horizon), "grasp_calibration": grasp_cal.to_dict(), "reference_frames": list(frames), "grasp_source": "tags" if use_tags else "mediapipe"}
    if use_tags:
        summary["aperture_calibration"] = (aperture_cal or ApertureCalibration()).to_dict()
    summary["left_grasp_valid_fraction"] = float(np.mean(np.isfinite(g["left"]["grasp"]))) if n else 0.0
    summary["right_grasp_valid_fraction"] = float(np.mean(np.isfinite(g["right"]["grasp"]))) if n else 0.0
    for frame in frames:
        cols, poses, valid = _action_columns(wp, ts, n, g, frame=frame, horizon=horizon, filter_config=filter_config)
        out = paths.pseudo_actions if frame == "world" else paths.pseudo_actions_head
        write_parquet(out, cols)
        summary[f"{frame}_action_valid_fraction"] = float(np.mean(cols["action_valid"])) if n else 0.0
        summary[f"{frame}_pose_valid_fraction"] = float(np.mean(valid["left"] & valid["right"])) if n else 0.0
        if frame == "world":
            bf = bimanual_features(ts, poses["left"], poses["right"], valid["left"], valid["right"])
            bf["frame_index"] = cols["frame_index"]
            write_parquet(paths.bimanual_features, bf)
    summary["action_valid_fraction"] = summary.get("world_action_valid_fraction", summary.get("head_action_valid_fraction", 0.0))
    paths.update_metadata(actions=summary)
    log.info("%s: action valid %s", paths.root.name, {f: round(summary[f"{f}_action_valid_fraction"], 3) for f in frames})
    return summary


__all__ = ["generate_actions", "grasp_signals", "tag_apertures"]
