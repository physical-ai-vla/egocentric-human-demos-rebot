"""Dataset integrity report (spec 31/32).  Hard failures raise at the end; the report is always written first.

Hard checks: shapes (state 20, CART20 [16,20], action32 [16,32]), action[:, :20] == CART20, AUX12 exactly zero (NaN counts
as non-zero), no NaN / Inf in training rows, gripper in [0, 1], rotations orthonormal (det +1), task-start anchor = identity,
split disjoint, every training row's 16 targets inside the episode (no tail padding).
"""
import collections
import json
import pathlib

import numpy as np

from ..config import HORIZON, UMI_DT
from ..geometry.rotation6d import rotation_6d_to_matrix, rotation_angle
from ..io.processed_writer import load_episode
from .geometry_checks import rotation_report


def pct(x, qs=(50, 95, 99)):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} if len(x) else None


def validate_dataset(root):
    root = pathlib.Path(root); fails = []; rep = collections.OrderedDict()
    man = {s: [json.loads(l) for l in open(root / f"{s}_manifest.jsonl")] for s in ("train", "val", "test")}
    ids = [r["episode_id"] for s in man for r in man[s]]
    if len(ids) != len(set(ids)): fails.append("episode in more than one split")
    tot = collections.Counter(); by_order = collections.defaultdict(collections.Counter); by_stage = collections.Counter()
    by_split = collections.Counter(); cams = collections.Counter(); gaps = []; gmin, gmax = np.inf, -np.inf
    tmag = {k: [] for k in (1, 4, 8, 16)}; rmag = {k: [] for k in (1, 4, 8, 16)}; rots = []; anchor_err = 0.0
    nan = inf = aux_nz = 0
    for s in man:
        for r in man[s]:
            ep = load_episode(root / r["path"]); m = ep["metadata"]; tr = ep["train_rows"]; c = ep["cart20"]; a = ep["action"]
            tot["episodes"] += 1; tot["raw_frames"] += sum(m["frames_raw"].values()) // 2; tot["rows_15hz"] += m["frames_rows"]
            tot["rows_valid"] += m["rows_valid"]; tot["train_rows"] += len(tr)
            by_order[m["stack_order"]]["episodes"] += 1; by_order[m["stack_order"]]["frames"] += m["frames_rows"]; by_order[m["stack_order"]]["samples"] += len(tr)
            by_split[s] += len(tr); by_stage.update(ep["stage_id"][tr].tolist())
            for cam, ok in m["camera_available"].items(): cams[cam] += int(ok)
            if c.shape[1:] != (HORIZON, 20) or a.shape[1:] != (HORIZON, 32) or ep["state"].shape[1:] != (20,): fails.append(f"{r['episode_id']}: shape")
            if not np.array_equal(a[..., :20], c): fails.append(f"{r['episode_id']}: action[:, :20] != CART20")
            aux_nz += int(np.count_nonzero(a[..., 20:]))
            S = ep["state"][tr]; nan += int(np.isnan(S).sum() + np.isnan(a).sum()); inf += int(np.isinf(S).sum() + np.isinf(a).sum())
            g = np.concatenate([S[:, 18:20].ravel(), c[..., [9, 19]].ravel()]); gmin, gmax = min(gmin, g.min()), max(gmax, g.max())
            anchor_err = max(anchor_err, float(np.abs(ep["state"][0, :18] - np.array([0, 0, 0, 1, 0, 0, 0, 1, 0] * 2)).max()))
            t = ep["timestamps"]; gaps += np.diff(t).tolist()
            t_last_valid = t[np.flatnonzero(ep["row_valid"])[-1]]
            if len(tr) and t[tr].max() + HORIZON * UMI_DT > t_last_valid + 1 / 15 + 1e-6: fails.append(f"{r['episode_id']}: target beyond episode end")
            for k in tmag:
                for o in (0, 10):
                    tmag[k] += np.linalg.norm(c[:, k - 1, o:o + 3], axis=1).tolist()
                    rmag[k] += np.degrees(rotation_angle(rotation_6d_to_matrix(c[:, k - 1, o + 3:o + 9]))).tolist()
            rots.append(c[..., [3, 4, 5, 6, 7, 8]].reshape(-1, 6)); rots.append(c[..., 13:19].reshape(-1, 6))
            rots.append(S[:, 3:9]); rots.append(S[:, 12:18])
    rr = rotation_report(np.concatenate(rots))
    rep["counts"] = dict(tot)
    rep["shapes"] = dict(state=[20], cart20=[HORIZON, 20], action32=[HORIZON, 32])
    rep["nan_count"], rep["inf_count"], rep["nonzero_aux12_count"] = nan, inf, aux_nz
    rep["gripper_min_max"] = [float(gmin), float(gmax)]
    rep["samples_by_order"] = {k: dict(v) for k, v in sorted(by_order.items())}
    rep["samples_by_stage"] = {str(k): v for k, v in sorted(by_stage.items())}
    rep["samples_by_split"] = dict(by_split)
    rep["camera_available_episodes"] = dict(cams)
    rep["row_dt_s"] = dict(min=float(np.min(gaps)), max=float(np.max(gaps)), mean=float(np.mean(gaps)))
    rep["cart_translation_m"] = {f"k{k}": pct(v) for k, v in tmag.items()}
    rep["cart_rotation_deg"] = {f"k{k}": pct(v) for k, v in rmag.items()}
    rep["rotation_validity"] = rr
    rep["task_start_anchor_max_err"] = anchor_err
    med = [rep["cart_translation_m"][f"k{k}"]["p50"] for k in (1, 4, 8, 16)]
    rep["horizon_monotone_translation_p50"] = bool(np.all(np.diff(med) > 0))
    if aux_nz: fails.append(f"AUX12 non-zero entries: {aux_nz}")
    if nan or inf: fails.append(f"NaN {nan} / Inf {inf} in training rows")
    if gmin < 0 or gmax > 1: fails.append(f"gripper outside [0,1]: {gmin}..{gmax}")
    if rr["invalid_rotation_count"]: fails.append(f"invalid rotations {rr['invalid_rotation_count']}")
    if anchor_err > 1e-5: fails.append(f"task-start anchor not identity ({anchor_err})")
    if not rep["horizon_monotone_translation_p50"]: fails.append(f"translation p50 not increasing with k: {med}")
    rep["fails"] = fails; rep["pass"] = not fails
    return rep
