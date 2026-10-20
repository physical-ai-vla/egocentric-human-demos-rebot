"""[2026-10-14] Visual binary gripper for servo-less HandUMI takes (hardware_profile handumi_rgbd_noservo, UMIalign 14:42 on).

Input: <episode>/derived/jaw/jaw_{side}.npz from ego_collector/scripts/umi_fusion/jaw_track.py v2 (red tape / red-X distance
between the jaw tips, px at 1920 width, NaN = not found). Pooled UMIalign histograms (144936/153819/160900/112920, 188k
frames) are bimodal on both sides: closed 110-150 px, open 280-450 px, valley 200-260 px. Checked on 16 random
labelled frames (16/16) plus the failure cases listed in jaw_track.py.

Rule: > OUTLIER_PX -> NaN (stray red objects); hysteresis CLOSED_PX / OPEN_PX (between = previous state); NaN runs
<= FILL_S take the state of their neighbours (both sides agree) or the nearer one; longer runs keep the nearest state
but are marked invalid. binary 1 = open, 0 = closed (same convention as hfold_export.gripper_binary).
"""
import pathlib

import numpy as np

CLOSED_PX, OPEN_PX, OUTLIER_PX, FILL_S = 200.0, 260.0, 650.0, 0.5


def binary_from_px(t_ns, d):
    d = np.where(d > OUTLIER_PX, np.nan, d); n = len(d); b = np.full(n, np.nan); s = np.nan
    for i in range(n):
        if np.isfinite(d[i]):
            if d[i] >= OPEN_PX: s = 1.0
            elif d[i] <= CLOSED_PX: s = 0.0
            elif not np.isfinite(s): s = float(d[i] >= (CLOSED_PX + OPEN_PX) / 2)
            b[i] = s
    valid = np.isfinite(b); out = b.copy(); idx = np.flatnonzero(valid)
    if not len(idx): return np.zeros(n), np.zeros(n, bool)
    i = 0
    while i < n:                                                       # fill NaN runs
        if valid[i]: i += 1; continue
        j = i
        while j < n and not valid[j]: j += 1
        a, c = (b[i - 1] if i > 0 else np.nan), (b[j] if j < n else np.nan)
        dur = ((t_ns[min(j, n - 1)] - t_ns[max(i - 1, 0)]) / 1e9)
        if np.isfinite(a) and np.isfinite(c) and a == c: fill = a
        elif np.isfinite(a) and np.isfinite(c):
            mid = (i + j) // 2; out[i:mid] = a; out[mid:j] = c; fill = None
        else: fill = a if np.isfinite(a) else c
        if fill is not None: out[i:j] = fill
        if dur <= FILL_S: valid[i:j] = True
        i = j
    return out, valid


def visual_gripper(ep, side, t_ns):
    """-> (binary at t_ns, valid at t_ns, info). Nearest jaw frame per query time (same wrist capture clock)."""
    z = np.load(pathlib.Path(ep) / "derived/jaw" / f"jaw_{side}.npz", allow_pickle=True)
    tj, d = z["t_ns"], z["d_px"]; b, v = binary_from_px(tj, d)
    j = np.clip(np.searchsorted(tj, t_ns), 1, len(tj) - 1); j = np.where(np.abs(tj[j - 1] - t_ns) < np.abs(tj[j] - t_ns), j - 1, j)
    near = np.abs(tj[j] - t_ns) <= 40e6
    info = dict(source="visual_jaw", tool=str(z["tool"]) if "tool" in z.files else None, closed_px=CLOSED_PX, open_px=OPEN_PX,
                outlier_px=OUTLIER_PX, fill_s=FILL_S, detect_frac=round(float(np.isfinite(d).mean()), 4),
                valid_frac=round(float((v[j] & near).mean()), 4), open_fraction_of_samples=round(float(b[j].mean()), 4))
    return b[j], v[j] & near, info
