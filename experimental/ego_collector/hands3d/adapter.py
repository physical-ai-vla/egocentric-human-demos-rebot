"""HaWoR world-frame track -> the standard tracking parquets -> pseudo-actions (G3).

The SLAM world frame plays the role the AprilTag world board played: we write
tracking/{camera_pose,wrist_pose,finger_pose}.parquet with REAL world columns, so the
existing generate-actions produces absolute + relative-delta (ΔT = T_t^-1 T_t+1) actions
in BOTH world and head frames, and qa/report/visualize work unchanged.

Action v1 (what training consumes): wrist SE(3) + thumb-index aperture per hand.
Everything else in hawor_export.npz (21 MANO joints, root orient, camera traj) stays as
derived metadata - stored, not fed to the WAM.

Cleaning gates (from the 6-episode pilot findings):
  - trim_s: drop the first seconds (DROID-SLAM init + C922 auto-exposure warmup diverges)
  - max_speed_m_s: a wrist step faster than this marks both endpoint frames invalid (teleports)
  - max_cam_dist_m: a wrist further than this from the head camera is anatomically impossible
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ego_collector.io.parquet import read_parquet, write_parquet
from ego_collector.recording.episode import EpisodePaths
from ego_collector.tracking.transforms import T_to_pose7, make_T

log = logging.getLogger("ego_collector.hawor")
SIDES = ("left", "right")
POSE_COLS = ("x", "y", "z", "qx", "qy", "qz", "qw")
WRIST, THUMB, INDEX = 0, 4, 8


@dataclass
class HaworGates:
    trim_s: float = 2.0
    max_speed_m_s: float = 1.5
    max_cam_dist_m: float = 1.2
    dilate: int = 1  # also invalidate this many neighbours around a teleport


def _pose_cols(prefix: str, p7: np.ndarray | None) -> dict[str, float]:
    v = p7 if p7 is not None else np.full(7, np.nan)
    return {f"{prefix}_{c}": float(x) for c, x in zip(POSE_COLS, v)}


def _xyz(prefix: str, p) -> dict[str, float]:
    v = p if p is not None else np.full(3, np.nan)
    return {f"{prefix}_{a}": float(x) for a, x in zip("xyz", v)}


def gate_mask(w: np.ndarray, valid: np.ndarray, cam_t: np.ndarray, ts_s: np.ndarray, g: HaworGates) -> np.ndarray:
    """Combined validity: HaWoR valid AND after trim AND no teleport step AND within arm's reach."""
    ok = valid.copy()
    ok &= ts_s >= g.trim_s
    ok &= np.linalg.norm(w - cam_t, axis=1) <= g.max_cam_dist_m
    dt = np.maximum(np.diff(ts_s), 1e-6)
    speed = np.linalg.norm(np.diff(w, axis=0), axis=1) / dt
    bad_step = np.flatnonzero(speed > g.max_speed_m_s)
    bad = set()
    for i in bad_step:
        bad.update(range(max(0, i - g.dilate), min(len(w), i + 1 + g.dilate + 1)))
    if bad:
        ok[sorted(bad)] = False
    # second pass on the VALID SUBSEQUENCE: a jump ACROSS a gated gap (consecutive valid
    # frames that are not adjacent in time) is a teleport too - drop the re-entry frame(s)
    changed = True
    while changed:
        changed = False
        vi = np.flatnonzero(ok)
        if len(vi) < 2:
            break
        v_dt = np.maximum(ts_s[vi[1:]] - ts_s[vi[:-1]], 1e-6)
        v_sp = np.linalg.norm(w[vi[1:]] - w[vi[:-1]], axis=1) / np.maximum(v_dt, 1.0 / 30)
        for k in np.flatnonzero(v_sp > g.max_speed_m_s):
            ok[vi[k + 1]] = False
            changed = True
    return ok


def hawor_to_tracking(paths: EpisodePaths, *, gates: HaworGates = HaworGates()) -> dict:
    """hands3d/hawor_export.npz -> tracking/{camera_pose,wrist_pose,finger_pose}.parquet (world + cam frames)."""
    d = dict(np.load(paths.root / "hands3d" / "hawor_export.npz", allow_pickle=True))
    ts_table = read_parquet(paths.timestamps)
    timestamps = ts_table["timestamp_ns"].to_numpy()
    R_c2w, t_c2w = d["R_c2w"], d["t_c2w"]
    T = min(len(R_c2w), len(timestamps), *(len(d[f"{s}_wrist_pos"]) for s in SIDES))
    ts_s = (timestamps[:T] - timestamps[0]) / 1e9

    side_data = {}
    for s in SIDES:
        w = d[f"{s}_wrist_pos"][:T]
        R_w = Rotation.from_rotvec(d[f"{s}_root_orient_aa"][:T]).as_matrix()
        ok = gate_mask(w, d[f"{s}_valid"][:T].astype(bool), t_c2w[:T], ts_s, gates)
        side_data[s] = dict(w=w, R=R_w, ok=ok,
                            thumb=d[f"{s}_thumb_tip"][:T], index=d[f"{s}_index_tip"][:T],
                            ap=d[f"{s}_aperture_m"][:T], raw_valid=d[f"{s}_valid"][:T].astype(bool))

    cam_rows, wrist_rows, finger_rows = [], [], []
    for i in range(T):
        Rc, tc = R_c2w[i], t_c2w[i]
        T_wc = make_T(Rc, tc)
        cam_rows.append({"frame_index": i, "timestamp_ns": int(timestamps[i]), "world_pose_valid": True,
                         **_pose_cols("head", T_to_pose7(T_wc)),
                         "world_tag_count": 0, "world_corner_count": 0, "world_inlier_count": 0,
                         "world_reprojection_error": float("nan"), "world_tag_ids": [], "visible_world_tags": 0})
        wrow: dict = {"frame_index": i, "timestamp_ns": int(timestamps[i]), "world_pose_valid": True}
        frow: dict = {"frame_index": i, "timestamp_ns": int(timestamps[i]), "world_pose_valid": True}
        for s in SIDES:
            sd = side_data[s]
            ok = bool(sd["ok"][i])
            T_ww = make_T(sd["R"][i], sd["w"][i])
            T_cw = make_T(Rc.T @ sd["R"][i], Rc.T @ (sd["w"][i] - tc))
            wrow[f"{s}_tag_visible"] = ok
            wrow[f"{s}_wrist_valid"] = ok
            wrow.update(_pose_cols(f"{s}_wrist", T_to_pose7(T_ww) if ok else None))
            wrow.update(_pose_cols(f"{s}_wrist_cam", T_to_pose7(T_cw) if ok else None))
            wrow[f"{s}_tag_reprojection_error"] = float("nan")
            wrow[f"{s}_tag_decision_margin"] = 1.0 if sd["raw_valid"][i] else 0.0  # 0 = infilled by HaWoR
            for name, p in (("thumb", sd["thumb"][i]), ("index", sd["index"][i])):
                frow[f"{s}_{name}_visible"] = ok
                frow.update(_xyz(f"{s}_{name}", p if ok else None))
                frow.update(_xyz(f"{s}_{name}_cam", (Rc.T @ (p - tc)) if ok else None))
                frow[f"{s}_{name}_reprojection_error"] = float("nan")
            frow[f"{s}_aperture_valid"] = ok
            frow[f"{s}_aperture_m"] = float(sd["ap"][i]) if ok else float("nan")
            mid = (sd["thumb"][i] + sd["index"][i]) / 2
            frow.update(_xyz(f"{s}_pinch", mid if ok else None))
            frow.update(_xyz(f"{s}_pinch_cam", (Rc.T @ (mid - tc)) if ok else None))
        wrist_rows.append(wrow)
        finger_rows.append(frow)

    for path, rows in ((paths.camera_pose, cam_rows), (paths.wrist_pose, wrist_rows), (paths.finger_pose, finger_rows)):
        write_parquet(path, {k: [r.get(k) for r in rows] for k in rows[0]})
    summary = {"frames": T, "source": "hawor", "world_tags": True, "world_pose_valid_fraction": 1.0, "fingers": True,
               "gates": {"trim_s": gates.trim_s, "max_speed_m_s": gates.max_speed_m_s, "max_cam_dist_m": gates.max_cam_dist_m}}
    for s in SIDES:
        sd = side_data[s]
        summary[f"{s}_wrist_valid_fraction"] = float(sd["ok"].mean())
        summary[f"{s}_aperture_valid_fraction"] = float(sd["ok"].mean())
        summary[f"{s}_infilled_fraction"] = float((~sd["raw_valid"]).mean())
        summary[f"{s}_gated_out_fraction"] = float((sd["raw_valid"] & ~sd["ok"]).mean())
        for f in ("thumb", "index"):
            summary[f"{s}_{f}_visible_fraction"] = summary[f"{s}_wrist_valid_fraction"]
    paths.update_metadata(tracking=summary, action_source="hawor")
    log.info("%s: %s", paths.root.name, {k: round(v, 3) for k, v in summary.items() if isinstance(v, float)})
    return summary


__all__ = ["HaworGates", "gate_mask", "hawor_to_tracking"]
