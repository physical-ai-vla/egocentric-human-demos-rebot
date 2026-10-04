"""Pose-jump diagnostics.  Diagnostic ONLY: nothing here changes data or validity unless the caller applies the mask
explicitly (the default pipeline does not)."""
import numpy as np
from ..geometry.rotation6d import rotation_angle


def jump_mask(t_s, T, valid, max_lin_mps=3.0, max_ang_rps=30.0):
    """True at raw sample i when the step (i-1 -> i) between two valid samples exceeds the speed limits"""
    t_s = np.asarray(t_s, np.float64); T = np.asarray(T, np.float64); m = np.zeros(len(t_s), bool)
    dt = np.diff(t_s); both = valid[1:] & valid[:-1]
    v = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) / dt
    w = rotation_angle(np.swapaxes(T[:-1, :3, :3], -1, -2) @ T[1:, :3, :3]) / dt
    m[1:] = both & ((v > max_lin_mps) | (w > max_ang_rps))
    return m


# [2026-10-01] v2b discontinuity filter (raw MASt3R-SLAM silent jumps: translation jumps with no tracking-lost flag).
ROW_S = 1.0 / 15.0


def discontinuity_filter(t_s, T, valid, max_trans_per_row_m=0.100, max_rot_per_row_deg=45.0, group_gap=5, return_frac=0.3, return_abs_m=0.030,
                         window="frame", isolated_step_m=None, isolated_neighbor_mps=0.3):
    """Hard jump detector on ONE arm's raw track.  A raw step i -> i+1 (both valid) is a hard jump if
        |dp| > max_trans_per_row * max(dt, 1/30 s) / (1/15 s)     (100 mm per 66.7 ms row = 1.5 m/s; floor dt at one raw frame)
     or rot > max_rot_per_row * max(dt, 1/30 s) / (1/15 s)       (45 deg per row)
    Consecutive flagged steps (gaps <= group_gap raw samples) form one EVENT spanning samples [a, b+1].
    The event is a SPIKE if the pose after it returns to the pose before it (|p(b+1) - p(a)| < max(return_abs, return_frac * largest step)
    and the rotation also returns within the rotation limit); then only samples a+1..b are invalidated, so no interpolation,
    state, or action window can cross it.  Otherwise it is ONE-WAY: every sample from a+1 to the end is invalidated (the task-anchor
    state would stay offset for the rest of the episode).  Returns (new_valid, events)."""
    t_s = np.asarray(t_s, np.float64); T = np.asarray(T, np.float64); v = np.asarray(valid, bool).copy(); n = len(t_s)
    idx = np.flatnonzero(v); events = []
    if len(idx) < 2: return v, events
    a_, b_ = idx[:-1], idx[1:]; cons = (b_ - a_) == 1
    dt = np.maximum(t_s[b_] - t_s[a_], 1.0 / 30.0); scale = dt / ROW_S
    dp = np.linalg.norm(T[b_, :3, 3] - T[a_, :3, 3], axis=1)
    dr = np.degrees(rotation_angle(np.swapaxes(T[a_, :3, :3], -1, -2) @ T[b_, :3, :3]))
    if window == "frame":                             # rate limit checked on every raw step (strict)
        r_t = cons & (dp > max_trans_per_row_m * scale); r_r = cons & (dr > max_rot_per_row_deg * scale); bad = r_t | r_r
    else:                                             # "row": displacement over one canonical row (66.7 ms) of raw samples, literal
        j = np.clip(np.searchsorted(t_s, t_s[a_] + ROW_S - 1e-3), 0, n - 1)
        okw = v[j] & ((j - a_) <= 3)
        dpw = np.linalg.norm(T[j, :3, 3] - T[a_, :3, 3], axis=1)
        drw = np.degrees(rotation_angle(np.swapaxes(T[a_, :3, :3], -1, -2) @ T[j, :3, :3]))
        loc = (dp > 0.5 * max_trans_per_row_m) | (dr > 0.5 * max_rot_per_row_deg)   # the row jump must sit in THIS raw step
        r_t = cons & okw & (dpw > max_trans_per_row_m) & loc; r_r = cons & okw & (drw > max_rot_per_row_deg) & loc
        bad = r_t | r_r
    if isolated_step_m is not None:                   # glitch signature: a big single step whose neighbouring steps are slow
        spd = dp / dt
        prev = np.r_[np.inf, spd[:-1]]; nxt = np.r_[spd[1:], np.inf]
        r_i = cons & (dp > isolated_step_m) & (np.minimum(prev, nxt) < isolated_neighbor_mps) & (np.maximum(np.where(np.isfinite(prev), prev, 0), np.where(np.isfinite(nxt), nxt, 0)) < isolated_neighbor_mps)
        bad = bad | r_i
    else:
        r_i = np.zeros_like(bad)
    tags = {"hard_jump_100mm": r_t, "hard_rot_45deg": r_r, "isolated_glitch_50mm": r_i}
    step_reason = {int(a_[k]): sorted(nm for nm, mk in tags.items() if mk[k]) for k in np.flatnonzero(bad)}
    steps = a_[bad]                                   # step index = sample before the jump
    groups = []
    for s in steps:
        if groups and s - groups[-1][-1] <= group_gap: groups[-1].append(s)
        else: groups.append([s])
    cut = None
    for g in groups:
        a, b = g[0], g[-1] + 1                        # pose before the event = a, after = b
        big = float(max(dp[np.searchsorted(a_, x)] for x in g))
        ret_p = np.linalg.norm(T[b, :3, 3] - T[a, :3, 3]); ret_r = float(np.degrees(rotation_angle(T[a, :3, :3].T @ T[b, :3, :3])))
        spike = ret_p < max(return_abs_m, return_frac * big) and ret_r < max_rot_per_row_deg
        events.append(dict(kind="spike" if spike else "one_way", reason=sorted({x for st in g for x in step_reason[int(st)]}),
                           first_sample=int(a), last_sample=int(b), t_s=float(t_s[a + 1]), max_step_mm=round(big * 1e3, 1),
                           return_mm=round(float(ret_p) * 1e3, 1), return_deg=round(ret_r, 1), n_steps=len(g)))
        if spike: v[a + 1:b] = False
        else:
            cut = a + 1; break                        # first one-way jump ends the usable track
    if cut is not None: v[cut:] = False
    return v, events
