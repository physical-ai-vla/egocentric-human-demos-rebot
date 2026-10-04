"""State continuity on the 15 Hz row grid (2026-10-01).  Adjacency is decided by the ROW INDEX only (two rows are neighbours
iff row_j = row_i + 1 and both are valid) -- never by the order of stored / LeRobot frames, which skip invalid rows
(ego LeRobot frames carry aux.row / aux.time_s for exactly this).  Also re-scans the RAW tracks for steps the filter left in."""
import json
import pathlib

import numpy as np

from ..io.processed_writer import load_episode
from ..io.raw_episode_loader import load_raw_episode
from ..preprocessing.pose_filter import discontinuity_filter


def continuity_report(proc_root, raw_root=None, cfg=None):
    proc_root = pathlib.Path(proc_root); man = [json.loads(l) for s in ("train", "val") for l in open(proc_root / f"{s}_manifest.jsonl")]
    J, Jt, raw_fast = [], [], 0
    for r in man:
        ep = load_episode(proc_root / r["path"]); S = ep["state"]; rv = ep["row_valid"]; tr = set(ep["train_rows"].tolist())
        ok = rv[1:] & rv[:-1]; i = np.flatnonzero(ok)
        j = np.maximum(np.linalg.norm(S[i + 1, 0:3] - S[i, 0:3], axis=1), np.linalg.norm(S[i + 1, 9:12] - S[i, 9:12], axis=1)) * 1e3
        J.append(j); Jt.append(np.array([j[k] for k, ii in enumerate(i) if ii in tr and ii + 1 in tr]))
        if raw_root is not None:
            raw = load_raw_episode(pathlib.Path(raw_root) / r["episode_id"])
            for a, arm in raw.arms.items():
                v = arm.valid
                if cfg is not None and cfg.discontinuity_filter:
                    v, _ = discontinuity_filter(arm.t_s, arm.T, arm.valid, cfg.jump_max_trans_per_row_m, cfg.jump_max_rot_per_row_deg,
                                                window=cfg.jump_window, isolated_step_m=cfg.jump_isolated_step_m)
                b = v[1:] & v[:-1]; dp = np.linalg.norm(np.diff(arm.T[:, :3, 3], axis=0), axis=1); dt = np.maximum(np.diff(arm.t_s), 1 / 30)
                raw_fast += int((b & (dp / dt > 3.0)).sum())
    J, Jt = np.concatenate(J), np.concatenate([x for x in Jt if len(x)])
    f = lambda x: dict(n=int(len(x)), gt30mm=int((x > 30).sum()), gt100mm=int((x > 100).sum()), max_mm=float(x.max()), p999_mm=float(np.percentile(x, 99.9)))
    return dict(all_valid_adjacent_rows=f(J), training_adjacent_rows=f(Jt), raw_steps_gt_3mps_left_in_valid=raw_fast)
