"""Raw ego episode -> processed CART20 episode (the whole label pipeline for one episode).

    1  one PoseTrack per arm on its own raw clock (no pairing to another arm's frames)
    2  task start t_s = first raw instant at which BOTH arms are valid; row grid t_s + n / 15 Hz up to the last instant
       both arms have a valid raw sample
    3  rows: both arms sampled at the row time (position lerp, rotation SLERP, gripper lerp)
    4  canonical / state pose: per arm inv(T(t_s)) @ T(t)                    (canonical_frame.per_arm_task_start_v1)
    5  state.npy (RELCART20 [L9|R9|gL gR]); state_prevrel.npy with T(t - UMI_DT) from the raw track (diagnostic)
    6  CART20 chunk per row: A_k = inv(T(t)) T(t + k UMI_DT), k = 1..16, raw-track interpolation; packed to [16,32]
    7  a TRAINING row needs: both arms valid at t, all 16 targets of both arms valid, every required camera frame within
       image_tol_s.  Nothing is padded.  Invalid rows keep NaN state and are never sampled.
"""
import numpy as np

from .config import ARMS, CONTRACT, HORIZON, Cart20Config
from .geometry import canonical_frame
from .io.raw_episode_loader import CAMERAS
from .labels.build_cart20 import build_cart20_chunk
from .labels.build_state20 import build_state20_prevrel, build_state20_relcart20
from .labels.pack_action32 import check_action32, pack_cart20_to_action32
from .labels.stage import stage_ids
from .preprocessing.gripper import validate_gripper
from .preprocessing.pose_filter import discontinuity_filter
from .preprocessing.resample import PoseTrack, canonical_times
from .preprocessing.synchronize import common_end_time, nearest_index, task_start_time


def convert_episode(raw, cfg=Cart20Config()):
    valid = {a: raw.arms[a].valid for a in ARMS}; jump_events = {}
    if cfg.discontinuity_filter:                                                   # v2b: raw MASt3R silent-jump filter (before anything else)
        for a in ARMS:
            valid[a], jump_events[a] = discontinuity_filter(raw.arms[a].t_s, raw.arms[a].T, raw.arms[a].valid,
                                                            cfg.jump_max_trans_per_row_m, cfg.jump_max_rot_per_row_deg,
                                                            window=cfg.jump_window, isolated_step_m=cfg.jump_isolated_step_m)
    tr = {a: PoseTrack(raw.arms[a].t_s, raw.arms[a].T, valid[a], raw.arms[a].gripper, cfg.max_raw_gap_s) for a in ARMS}
    for a in ARMS: validate_gripper(tr[a].g[tr[a].valid], f"{a} raw gripper")
    t_s = task_start_time([tr[a] for a in ARMS]); t_e = common_end_time([tr[a] for a in ARMS])
    times = canonical_times(t_s, t_e, cfg.row_fps); N = len(times)
    S = {a: tr[a].sample(times) for a in ARMS}                                 # (T, g, ok)
    row_valid = S["left"][2] & S["right"][2]; assert row_valid[0], "row 0 must be the task start"
    anchor = {a: S[a][0][0].copy() for a in ARMS}

    # canonical poses (NaN on invalid rows)
    pose = {}
    for a in ARMS:
        P = canonical_frame.canonicalize(S[a][0], anchor[a]); P[~row_valid] = np.nan; pose[a] = P
    g = {a: np.where(row_valid, S[a][1], np.nan) for a in ARMS}

    # state (main) -- RELCART20 task anchor
    state = build_state20_relcart20(anchor["left"], S["left"][0], S["left"][1], anchor["right"], S["right"][0], S["right"][1])
    state[~row_valid] = np.nan

    # state_prevrel (diagnostic) -- inv(T(t)) T(t - UMI_DT) on the raw track; identity + mask 0 where history is invalid
    Pv = {a: tr[a].sample(times - cfg.target_dt_s) for a in ARMS}
    prev_ok = Pv["left"][2] & Pv["right"][2] & row_valid
    Tp = {a: np.where(prev_ok[:, None, None], Pv[a][0], S[a][0]) for a in ARMS}
    state_prevrel = build_state20_prevrel(Tp["left"], S["left"][0], S["left"][1], Tp["right"], S["right"][0], S["right"][1])
    state_prevrel[~row_valid] = np.nan
    state_prev_valid = prev_ok.astype(np.float32)

    # cameras: nearest frame within tolerance, per row
    cam_valid = np.zeros((N, len(CAMERAS)), bool); cameras = {}
    for ci, c in enumerate(CAMERAS):
        if c not in raw.cams: continue
        ct, cf = raw.cams[c]; j, dt = nearest_index(ct, times, cfg.image_tol_s)
        cam_valid[:, ci] = j >= 0
        cameras[c] = dict(frame_index=np.where(j >= 0, cf[np.maximum(j, 0)], -1).astype(np.int64),
                          video=raw.meta.get("videos", {}).get(c), tol_s=cfg.image_tol_s,
                          dt_ms_p99=float(np.percentile(dt[j >= 0], 99) * 1e3) if (j >= 0).any() else None)
    need = np.ones(N, bool)
    for c in cfg.required_cameras: need &= cam_valid[:, CAMERAS.index(c)]

    # CART20 chunks
    rows, chunks = [], []
    for r in np.flatnonzero(row_valid & need):
        c, ok = build_cart20_chunk(tr["left"], tr["right"], times[r], cfg.horizon, cfg.target_dt_s,
                                   T_left_t=S["left"][0][r], T_right_t=S["right"][0][r])
        if ok: rows.append(r); chunks.append(c)
    cart20 = np.stack(chunks).astype(np.float32) if chunks else np.zeros((0, HORIZON, 20), np.float32)
    action = pack_cart20_to_action32(cart20); check_action32(action, cart20)
    train_rows = np.asarray(rows, np.int64)

    st, st_info = stage_ids(times, g["left"], g["right"], row_valid)
    meta = dict(
        episode_id=raw.episode_id, source=raw.meta.get("source", "human_egocentric"), task=raw.meta["task"],
        instruction=raw.meta["instruction"], stack_order=raw.meta["stack_order"],
        fps=cfg.row_fps, **CONTRACT, canonical_frame=canonical_frame.NAME,
        task_start_time_s=float(t_s), clock_offset_ns=int(raw.t0_ns), duration_s=float(times[-1] - times[0]),
        frames_raw={a: int(len(raw.arms[a].t_s)) for a in ARMS}, frames_rows=int(N), rows_valid=int(row_valid.sum()),
        train_rows=int(len(train_rows)), cameras=list(cameras), camera_order=list(CAMERAS),
        camera_available={c: bool(c in cameras) for c in CAMERAS}, stage=st_info, config=cfg.to_dict(),
        tool_frame=raw.meta.get("tool_frame"), gripper_calibration=raw.meta.get("gripper_calibration"),
        raw_provenance=raw.meta.get("provenance"), jump_events=jump_events if cfg.discontinuity_filter else None,
    )
    return dict(timestamps=times.astype(np.float64), row_valid=row_valid, train_rows=train_rows, state=state,
                state_prevrel=state_prevrel, state_prev_valid=state_prev_valid, cart20=cart20, action=action,
                left_pose=pose["left"], right_pose=pose["right"], left_gripper=g["left"].astype(np.float32),
                right_gripper=g["right"].astype(np.float32), stage_id=st.astype(np.int8), camera_valid=cam_valid,
                cameras=cameras, metadata=meta, _anchor=anchor, _tracks=tr)
