"""G0-0 Mount Qualification: can ONE wrist Arducam be both the gripper-aperture sensor and the VI camera?

The wrist unit is remounted lateral-oblique (30-45 deg) toward thumb + index. The same image stream then has to serve
two consumers that want opposite things:

    hand / gripper tracker   wants the thumb, the index and part of the palm large and always in view
    VI (MASt3R-Fusion)       wants environment. The hand is camera-FIXED foreground (it moves with the camera), so
                             every feature on it says "the world is not moving" — it must be masked out

So this decides the mount BEFORE any calibration or retargeter code, from one ~100 s take:

    record    one take through the collector's own device path + spoken protocol cues; segment boundaries are
              written as `g0_segment` events, so analysis reads recorded boundaries, never assumed ones
    analyze   hand  : MediaPipe detection / dropout on the raw fisheye AND a rectified view, key-landmark
                      visibility, and three aperture signals per frame
                        A  |thumb_tip - index_tip| in pixels
                        B  A / |index_MCP - index_PIP|           (a segment the lateral view always sees)
                        C  |thumb_tip - index_tip| in MediaPipe world landmarks (hand-centred, ~metres)
                      scored by OPEN/HALF/PINCH separation, within-state jitter, monotonic HALF, hysteresis chatter
              mask  : camera-fixed region = pixels that do not change while the camera moves (motion block)
                      UNION the landmark-hull occupancy, dilated
              VI    : usable environment ratio, GFTT corners + grid coverage outside the mask, KLT + essential-matrix
                      inliers masked vs unmasked, and the visual rotation angle per frame against the gyro (the
                      ANGLE is extrinsic-free, so this needs no camera-IMU calibration — which the new mount does
                      not have yet)
              verdict: the 2x2 matrix (hand OK? x masked VI OK?) -> share one camera / split VI camera /
                      custom gap detector / re-aim the mount
    export    masked, rectified frames (+ raw IMU) in EuRoC layout for an RGB-only MASt3R-SLAM replay on the 5090 —
              the "real trajectory" check, run once the cheap checks above pass

    .venv/bin/python -m ego_teleop.tools.g0_mount_qual record --hardware g0_mount_right
    .venv/bin/python -m ego_teleop.tools.g0_mount_qual analyze EPISODE [--out DIR]
    .venv/bin/python -m ego_teleop.tools.g0_mount_qual export EPISODE --out SEQ_ROOT [--mask DIR/mask_vi.png]

Gates are PRE-MEASUREMENT starting points (same rule as every gate block in this repo): re-fit them from the first
real takes before a PASS is read as evidence. Nothing here touches the robot, the retargeters or the coordinator."""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------ protocol
# (name, seconds, spoken cue). Grip windows are held with the forearm still; motion windows keep the hand relaxed
# OPEN so the hand silhouette is as fixed as it will ever be (that is what the static-region mask measures).
PROTOCOL: list[tuple[str, float, str]] = [
    ("still_open", 5.0, "손 펴고 정지"),
    *[s for i in range(3) for s in (("open", 4.0, "펴기"), ("half", 4.0, "반쯤"), ("pinch", 4.0, "집기"))],
    ("pinch_fast", 8.0, "빠르게 집었다 펴기 반복"),
    ("tx", 6.0, "손 편 채로 앞뒤로"),
    ("ty", 6.0, "좌우로"),
    ("tz", 6.0, "위아래로"),
    ("roll", 6.0, "손목 돌리기"),
    ("pitch", 6.0, "손목 위아래로 꺾기"),
    ("yaw", 6.0, "손목 좌우로 꺾기"),
    ("natural", 12.0, "자연스럽게 집어서 옮기기"),
    ("return_still", 5.0, "처음 자세로 정지"),
]
GRIP_STATES = ("open", "half", "pinch")
MOTION_WINDOWS = ("tx", "ty", "tz", "roll", "pitch", "yaw", "natural")
MAX_FRAME_ROT_DEG = 20.0      # ~600 deg/s at 30 Hz: anything above is a decomposition artefact
SETTLE_S = 0.8                  # skip this much after each grip cue: the operator is still changing posture

TIP_THUMB, TIP_INDEX, INDEX_MCP, INDEX_PIP, MIDDLE_MCP, MIDDLE_TIP, WRIST = 4, 8, 5, 6, 9, 12, 0
KEY_LANDMARKS = (TIP_THUMB, TIP_INDEX, INDEX_MCP, INDEX_PIP, MIDDLE_MCP)   # Aero later also needs palm + middle
SIGNALS = ("A_px", "B_ratio", "C_world")

GATES = dict(
    hand=dict(min_detect_grip=0.95, min_detect_pinch=0.90, max_miss_run_grip_ms=300.0, min_tips_in_view=0.95,
              min_margin=0.20, max_chatter=0, max_jitter_frac=0.10),
    vi=dict(min_env_ratio=0.50, min_inliers_p50=80, min_inliers_p10=30, min_grid_cells_p50=6,
            min_gyro_r=0.80, gyro_slope=(0.85, 1.15)),
)


@dataclass
class Window:
    name: str
    t0_ns: int
    t1_ns: int


# ------------------------------------------------------------------------------------------------ windows
def windows_from_events(events: list[dict], t_stop_ns: int) -> list[Window]:
    """Recorded `g0_segment` events -> windows; each runs to the next segment start (the last to the stop)."""
    seg = sorted((int(e["t_ns"]), e["detail"]["name"]) for e in events if e.get("kind") == "g0_segment")
    return [Window(n, t, seg[i + 1][0] if i + 1 < len(seg) else int(t_stop_ns)) for i, (t, n) in enumerate(seg)]


def windows_from_args(specs: list[str], t_start_ns: int) -> list[Window]:
    out = []
    for s in specs:
        name, a, b = s.split(":"); out.append(Window(name, t_start_ns + int(float(a) * 1e9), t_start_ns + int(float(b) * 1e9)))
    return out


def label_frames(t_ns: np.ndarray, windows: list[Window], *, settle_s: float = SETTLE_S) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: the window name ('' outside) and whether it is HELD (past the settle time of a grip window)."""
    name = np.full(len(t_ns), "", dtype=object); held = np.zeros(len(t_ns), bool); idx = np.full(len(t_ns), -1)
    for k, w in enumerate(windows):
        m = (t_ns >= w.t0_ns) & (t_ns < w.t1_ns); name[m] = w.name; idx[m] = k
        if w.name in GRIP_STATES: held |= m & (t_ns >= w.t0_ns + int(settle_s * 1e9))
    return name, held, idx


def miss_runs_ms(ok: np.ndarray, t_ns: np.ndarray) -> list[float]:
    """Each outage measured to the NEXT good frame (a 10-frame miss at 30 Hz is 333 ms of hold, not 300)."""
    runs, i, n = [], 0, len(ok)
    while i < n:
        if ok[i]: i += 1; continue
        j = i
        while j < n and not ok[j]: j += 1
        t_prev = t_ns[i - 1] if i > 0 else t_ns[i]
        t_next = t_ns[j] if j < n else t_ns[-1]
        runs.append((t_next - t_prev) / 1e6); i = j
    return runs


# ------------------------------------------------------------------------------------------------ hand signals
def aperture_signals(lm2d: np.ndarray, lm3d: np.ndarray) -> dict:
    a = float(np.linalg.norm(lm2d[TIP_THUMB] - lm2d[TIP_INDEX]))
    seg = float(np.linalg.norm(lm2d[INDEX_MCP] - lm2d[INDEX_PIP]))
    c = float(np.linalg.norm(lm3d[TIP_THUMB] - lm3d[TIP_INDEX]))
    return dict(A_px=a, B_ratio=a / seg if seg > 1e-6 else np.nan, C_world=c)


def hysteresis(x: np.ndarray, close_th: float, open_th: float, init: str = "open") -> np.ndarray:
    """Binary gripper state with a dead band: below close_th -> CLOSE, above open_th -> OPEN, else keep. NaN keeps."""
    s = np.empty(len(x), dtype=object); cur = init
    for i, v in enumerate(x):
        if np.isfinite(v):
            if v < close_th: cur = "close"
            elif v > open_th: cur = "open"
        s[i] = cur
    return s


def signal_metrics(df, sig: str) -> dict:
    """Separation of one aperture signal across the HELD grip windows. `df` needs columns sig, state, held, win, t_ns."""
    import pandas as pd
    v = df[df.held & np.isfinite(df[sig])]
    st = {}
    for s in GRIP_STATES:
        x = v.loc[v.state == s, sig].to_numpy(float)
        st[s] = dict(n=int(len(x)), median=float(np.median(x)) if len(x) else np.nan,
                     p05=float(np.percentile(x, 5)) if len(x) else np.nan, p95=float(np.percentile(x, 95)) if len(x) else np.nan,
                     std=float(np.std(x)) if len(x) else np.nan)
    o, p, h = st["open"], st["pinch"], st["half"]
    span = o["median"] - p["median"]
    out = dict(states=st, span=float(span))
    if not (np.isfinite(span) and span > 0):
        out.update(ok=False, why="OPEN median not above PINCH median" if np.isfinite(span) else "OPEN or PINCH not measured"); return out
    dprime = span / np.sqrt(max((o["std"] ** 2 + p["std"] ** 2) / 2, 1e-12))
    margin = (o["p05"] - p["p95"]) / span
    mono = bool(np.isfinite(h["median"]) and p["median"] < h["median"] < o["median"])
    # jitter: frame-to-frame noise inside each held window, as a fraction of the OPEN-PINCH span
    jit = []
    for _, g in v.groupby("win"):
        x = g[sig].to_numpy(float)
        if len(x) >= 5: jit.append(np.std(np.diff(x)) / np.sqrt(2))
    jitter_frac = float(np.median(jit) / span) if jit else np.nan
    gap_lo, gap_hi = p["p95"], o["p05"]
    if gap_hi > gap_lo: close_th, open_th = gap_lo + 0.25 * (gap_hi - gap_lo), gap_hi - 0.25 * (gap_hi - gap_lo)
    else: close_th, open_th = p["median"] + 0.33 * span, p["median"] + 0.66 * span   # overlapping: still report
    # chatter: state flips INSIDE held OPEN/PINCH windows; wrong: a held window whose final state is not its own
    grip = df[df.state.isin(["open", "pinch"])].copy()
    chatter, wrong = 0, 0
    if len(grip):
        hs = hysteresis(df[sig].to_numpy(float), close_th, open_th)
        df2 = df.assign(_hs=hs)
        for _, g in df2[df2.held & df2.state.isin(["open", "pinch"])].groupby("win"):
            s = g._hs.to_numpy(); chatter += int(np.sum(s[1:] != s[:-1]))
            want = "open" if g.state.iloc[0] == "open" else "close"
            wrong += int(s[-1] != want)
    out.update(ok=True, dprime=float(dprime), margin=float(margin), monotonic_half=mono, jitter_frac=jitter_frac,
               close_th=float(close_th), open_th=float(open_th), chatter=int(chatter), wrong_windows=int(wrong))
    return out


def pick_signal(per_sig: dict) -> str | None:
    """Best = largest OPEN/PINCH margin among monotonic signals; jitter breaks near-ties (<0.05 margin apart)."""
    cand = [(k, m) for k, m in per_sig.items() if m.get("ok")]
    if not cand: return None
    mono = [c for c in cand if c[1]["monotonic_half"]] or cand
    mono.sort(key=lambda c: (-round(c[1]["margin"] / 0.05), c[1]["jitter_frac"]))
    return mono[0][0]


def hand_gate(view: dict, best: dict | None) -> tuple[bool, list[str]]:
    g = GATES["hand"]; fails = []
    if view["detect_grip"] < g["min_detect_grip"]: fails.append(f"detect_grip {view['detect_grip']:.2f} < {g['min_detect_grip']}")
    if view["detect_pinch"] < g["min_detect_pinch"]: fails.append(f"detect_pinch {view['detect_pinch']:.2f} < {g['min_detect_pinch']}")
    if view["miss_run_grip_ms_max"] > g["max_miss_run_grip_ms"]: fails.append(f"longest grip miss {view['miss_run_grip_ms_max']:.0f} ms")
    if view["tips_in_view"] < g["min_tips_in_view"]: fails.append(f"thumb/index tips in view {view['tips_in_view']:.2f}")
    if best is None: fails.append("no aperture signal separates OPEN from PINCH"); return False, fails
    if best["margin"] < g["min_margin"]: fails.append(f"margin {best['margin']:.2f} < {g['min_margin']}")
    if best["chatter"] > g["max_chatter"]: fails.append(f"chatter {best['chatter']}")
    if not (best["jitter_frac"] <= g["max_jitter_frac"]): fails.append(f"jitter {best['jitter_frac']:.3f} of span")
    if not best["monotonic_half"]: fails.append("HALF not between OPEN and PINCH")
    return not fails, fails


# ------------------------------------------------------------------------------------------------ views
def make_views(cal: dict, *, det_downscale: int, vi_downscale: int, hand_balance: float):
    """raw  = fisheye downscaled (what MediaPipe sees without help); rect = wide pinhole (balance `hand_balance`);
    vi   = the exact pinhole view MASt3R would get (balance 0, `vi_downscale`) — masks and VI metrics live here."""
    from ..transforms.fisheye import make_rectifier
    K, D, size = cal["K"], cal["D"], cal["image_size"]
    rect = make_rectifier(K, D, size, downscale=det_downscale, balance=hand_balance)
    vi = make_rectifier(K, D, size, downscale=vi_downscale, balance=0.0)
    return rect, vi


def raw_px_to_vi(pts_raw_view: np.ndarray, cal: dict, det_downscale: int, vi) -> np.ndarray:
    import cv2
    p = np.asarray(pts_raw_view, np.float64).reshape(-1, 1, 2) * det_downscale
    out = cv2.fisheye.undistortPoints(p, np.asarray(cal["K"], float), np.asarray(cal["D"], float).reshape(4, 1), P=vi.K_new)
    return out.reshape(-1, 2)


def rect_px_to_vi(pts: np.ndarray, rect, vi) -> np.ndarray:
    p = np.c_[np.asarray(pts, np.float64), np.ones(len(pts))]
    q = (vi.K_new @ np.linalg.inv(rect.K_new) @ p.T).T
    return q[:, :2] / q[:, 2:3]


# ------------------------------------------------------------------------------------------------ hand pass
def run_hand_pass(frames_iter, view: str, prep, *, stride: int = 1):
    """MediaPipe over one view. Picks, per frame, the detection nearest the previous wrist (two hands may be seen)."""
    import pandas as pd
    from ego_collector.hands.mediapipe_tracker import HandLandmarker
    lm = HandLandmarker(num_hands=2)
    rows, prev_wrist, last_ms = [], None, -1
    try:
        for k, (vf, fi, t_ns, img) in enumerate(frames_iter):
            if k % stride: continue
            im = prep(img); h, w = im.shape[:2]
            t0 = time.perf_counter()
            ts_ms = max(int(t_ns // 1_000_000), last_ms + 1); last_ms = ts_ms   # MediaPipe VIDEO mode rejects a repeated ms
            dets = lm.detect_all(im, ts_ms); ms = (time.perf_counter() - t0) * 1e3
            r = dict(t_ns=t_ns, frame_index=fi, view=view, detected=bool(dets), n_hands=len(dets), ms=ms, w=w, h=h)
            if dets:
                d = min(dets, key=lambda d: np.linalg.norm(d.landmarks_2d[WRIST] - prev_wrist)) if prev_wrist is not None else max(dets, key=lambda d: d.label_confidence)
                prev_wrist = d.landmarks_2d[WRIST]
                inb = (d.landmarks_2d[:, 0] >= 0) & (d.landmarks_2d[:, 0] < w) & (d.landmarks_2d[:, 1] >= 0) & (d.landmarks_2d[:, 1] < h)
                r.update(label=d.label, score=d.label_confidence, tips_in_view=bool(inb[TIP_THUMB] and inb[TIP_INDEX]),
                         keys_in_view=int(inb[list(KEY_LANDMARKS)].sum()), lm2d=d.landmarks_2d.ravel().tolist(),
                         lm3d=d.landmarks_3d.ravel().tolist(), **aperture_signals(d.landmarks_2d, d.landmarks_3d))
            else:
                r.update(label="", score=0.0, tips_in_view=False, keys_in_view=0, lm2d=None, lm3d=None, A_px=np.nan, B_ratio=np.nan, C_world=np.nan)
            rows.append(r)
    finally:
        lm.close()
    return pd.DataFrame(rows)


def summarize_view(df, windows) -> dict:
    t = df.t_ns.to_numpy(np.int64)
    state, held, win = label_frames(t, windows)
    df = df.assign(state=state, held=held, win=win)
    ok = df.detected.to_numpy(bool)
    grip = df.state.isin(GRIP_STATES).to_numpy()
    pinch = (df.state == "pinch").to_numpy()
    runs_grip = []
    for _, g in df[grip].groupby("win"):
        runs_grip += miss_runs_ms(g.detected.to_numpy(bool), g.t_ns.to_numpy(np.int64))
    runs_all = miss_runs_ms(ok, t)
    det = df[df.detected]
    per_sig = {s: signal_metrics(df, s) for s in SIGNALS} if grip.any() else {}
    best = pick_signal(per_sig) if per_sig else None
    by_win = {}
    for name in dict.fromkeys(df.state):
        if not name: continue
        m = (df.state == name).to_numpy(); by_win[name] = float(ok[m].mean())
    view = dict(n_frames=int(len(df)), detect_all=float(ok.mean()) if len(ok) else 0.0,
                detect_grip=float(ok[grip].mean()) if grip.any() else float("nan"),
                detect_pinch=float(ok[pinch].mean()) if pinch.any() else float("nan"),
                detect_by_window=by_win,
                miss_run_all_ms=dict(n=len(runs_all), p50=float(np.median(runs_all)) if runs_all else 0.0,
                                     p95=float(np.percentile(runs_all, 95)) if runs_all else 0.0, max=float(max(runs_all, default=0.0))),
                miss_run_grip_ms_max=float(max(runs_grip, default=0.0)),
                tips_in_view=float(det.tips_in_view.mean()) if len(det) else 0.0,
                keys_in_view_mean=float(det.keys_in_view.mean()) if len(det) else 0.0,
                ms_per_frame_p50=float(df.ms.median()) if len(df) else 0.0,
                signals=per_sig, best_signal=best)
    if grip.any():
        ok_gate, fails = hand_gate(view, per_sig.get(best) if best else None)
        view.update(gate_pass=ok_gate, gate_fails=fails)
    else:
        view.update(gate_pass=None, gate_fails=["NOT MEASURED: no OPEN/HALF/PINCH windows in this take"])
    return view, df


# ------------------------------------------------------------------------------------------------ mask
def static_region_mask(diff_stack: np.ndarray, grad_stack: np.ndarray | None = None, *, rel_thresh: float = 0.25,
                       abs_min: float = 3.0, tex_rel: float = 0.6, moving_pct: float = 75.0) -> np.ndarray:
    """Camera-fixed pixels = those whose frame-to-frame change stays small WHILE THE CAMERA MOVES. `diff_stack` is
    (N, h, w) |I_t - I_{t-1}| over motion frames; `grad_stack` the matching gradient magnitude. Returns a bool mask
    (True = camera-fixed / to be masked).

    Measured on a real take (HRL80, jaw mount), each rule below exists because its absence over-masked:
      * a still stretch makes every pixel unchanging -> only the most-moving quarter of frames is used
      * a textureless region (white table) does not change under motion either -> low change counts only where the
        pixel is textured (relative to this image's median texture)
      * the smooth inside of a hand / jaw then has no evidence of its own -> each static blob is filled by its
        convex hull (a hole fill does not work: a jaw or forearm runs off the image edge and encloses nothing)"""
    import cv2
    g = diff_stack.reshape(len(diff_stack), -1).mean(axis=1)
    sel = g >= np.percentile(g, moving_pct) if len(diff_stack) >= 40 else np.ones(len(diff_stack), bool)
    med = np.median(diff_stack[sel], axis=0).astype(np.float32)
    thr = max(abs_min, rel_thresh * float(np.percentile(med, 75)))
    m = med < thr
    if grad_stack is not None:
        G = np.mean(grad_stack[sel].astype(np.float32), axis=0); m &= G >= tex_rel * float(np.median(G))
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.005 * m.size: continue
        ys, xs = np.nonzero(lab == i)
        cv2.fillConvexPoly(keep, cv2.convexHull(np.c_[xs, ys].astype(np.int32)), 1)
    return keep.astype(bool)


def texture_map(grad_stack: np.ndarray, *, tex_rel: float = 0.6) -> np.ndarray:
    G = np.mean(grad_stack, axis=0); return G >= tex_rel * float(np.median(G))


def hull_occupancy(points_vi: list[np.ndarray], size_wh: tuple[int, int], scale: float) -> np.ndarray:
    import cv2
    W, H = size_wh; w, h = int(W * scale), int(H * scale)
    acc = np.zeros((h, w), np.float32)
    for p in points_vi:
        q = np.asarray(p, np.float32) * scale
        if not np.all(np.isfinite(q)): continue
        hull = cv2.convexHull(q.astype(np.int32)); tmp = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(tmp, hull, 1); acc += tmp
    return acc / max(len(points_vi), 1)


def build_mask(static_small: np.ndarray | None, occ_small: np.ndarray | None, size_wh, *, occ_min: float = 0.05,
               dilate_frac: float = 0.03, poly: list | None = None) -> np.ndarray:
    import cv2
    W, H = size_wh; m = np.zeros((H, W), np.uint8)
    for small in (static_small, None if occ_small is None else occ_small >= occ_min):
        if small is not None: m |= cv2.resize(small.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    if poly: cv2.fillPoly(m, [np.asarray(poly, np.int32)], 1)
    r = max(1, int(dilate_frac * W)); m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return m.astype(bool)


# ------------------------------------------------------------------------------------------------ VI pass
def pair_rotation(prev: np.ndarray, cur: np.ndarray, K: np.ndarray, region: np.ndarray | None, *, max_corners: int = 600,
                  grid: int = 4) -> dict:
    """GFTT in `region` of prev -> KLT -> forward/backward check -> essential RANSAC. The rotation ANGLE does not
    depend on the camera-IMU extrinsic, which is why it can be compared with the gyro before any calibration."""
    import cv2
    msk = None if region is None else region.astype(np.uint8) * 255
    pts = cv2.goodFeaturesToTrack(prev, max_corners, 0.01, 8, mask=msk)
    out = dict(corners=0, grid_cells=0, tracked=0, inliers=0, angle_deg=np.nan, inliers_in_mask=0)
    if pts is None or len(pts) < 8: return out
    out["corners"] = int(len(pts))
    H, W = prev.shape; cells = set((int(x * grid / W), int(y * grid / H)) for x, y in pts.reshape(-1, 2)); out["grid_cells"] = len(cells)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, winSize=(21, 21), maxLevel=3)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - pts).reshape(-1, 2), axis=1) < 1.0)
    p0, p1 = pts.reshape(-1, 2)[good], nxt.reshape(-1, 2)[good]; out["tracked"] = int(len(p0))
    if len(p0) < 8: return out
    # Two models. Essential matrix needs parallax and is DEGENERATE under pure rotation (the roll/pitch/yaw windows are
    # nearly that); the rotation-only model is exact there and biased under real translation. Take E unless the
    # rotation model explains (almost) as many tracks — then there is no parallax for E to use.
    Rr, inl_r = rotation_ransac(p0, p1, K)
    best = ("rot", Rr, inl_r)
    E, inl = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is not None and E.shape == (3, 3):
        _n, Re, _t, _ = cv2.recoverPose(E, p0, p1, K, mask=inl.copy())
        inl_e = inl.ravel().astype(bool)
        ang_e = np.degrees(np.linalg.norm(cv2.Rodrigues(Re)[0]))
        # > MAX_FRAME_ROT_DEG between two 30 Hz frames is the E decomposition's twisted-pair ambiguity, not a wrist
        if inl_e.sum() > 1.25 * inl_r.sum() and ang_e <= MAX_FRAME_ROT_DEG: best = ("ess", Re, inl_e)
    model, R, inl_b = best
    out["inliers"] = int(inl_b.sum()); out["model"] = model
    if out["inliers"] >= 8: out["angle_deg"] = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
    out["_inlier_pts"] = p0[inl_b]
    return out


def rotation_ransac(p0: np.ndarray, p1: np.ndarray, K: np.ndarray, *, iters: int = 200, thresh_px: float = 1.5,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pure-rotation model on bearing vectors: 2-point Kabsch hypotheses, inliers by reprojection angle, refit."""
    Ki = np.linalg.inv(K)
    f0 = (Ki @ np.c_[p0, np.ones(len(p0))].T).T; f0 /= np.linalg.norm(f0, axis=1, keepdims=True)
    f1 = (Ki @ np.c_[p1, np.ones(len(p1))].T).T; f1 /= np.linalg.norm(f1, axis=1, keepdims=True)
    thr = thresh_px / K[0, 0]; rng = np.random.default_rng(seed)

    def kabsch(a, b):
        U, _, Vt = np.linalg.svd(a.T @ b); d = np.sign(np.linalg.det(Vt.T @ U.T))
        return Vt.T @ np.diag([1, 1, d]) @ U.T                      # R with b ≈ R a

    best_R, best_in = np.eye(3), np.zeros(len(p0), bool)
    for _ in range(iters):
        i = rng.choice(len(p0), 2, replace=False); R = kabsch(f0[i], f1[i])
        err = np.arccos(np.clip(np.sum((f0 @ R.T) * f1, axis=1), -1, 1)); inl = err < thr
        if inl.sum() > best_in.sum(): best_R, best_in = R, inl
    if best_in.sum() >= 3: best_R = kabsch(f0[best_in], f1[best_in])
    return best_R, best_in


def gyro_angles(t_frames: np.ndarray, t_imu: np.ndarray, gyro: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """|∫ω dt| between consecutive frame times, degrees (small-angle sum; 33 ms windows)."""
    out = np.full(len(t_frames), np.nan)
    w = gyro - bias
    for i in range(1, len(t_frames)):
        m = (t_imu >= t_frames[i - 1]) & (t_imu < t_frames[i])
        if m.sum() < 2: continue
        tt = t_imu[m]; dt = np.diff(np.r_[t_frames[i - 1], tt]) / 1e9
        out[i] = float(np.degrees(np.linalg.norm((w[m] * dt[:, None]).sum(axis=0))))
    return out


def gyro_agreement(vis: np.ndarray, gyr: np.ndarray, *, min_deg: float = 0.3) -> dict:
    m = np.isfinite(vis) & np.isfinite(gyr) & (gyr > min_deg)
    if m.sum() < 20: return dict(n=int(m.sum()), r=np.nan, slope=np.nan, note="too little rotation to compare")
    r = float(np.corrcoef(vis[m], gyr[m])[0, 1]); slope = float(np.median(vis[m] / gyr[m]))
    return dict(n=int(m.sum()), r=r, slope=slope)


def best_lag(vis: np.ndarray, t_frames: np.ndarray, t_imu: np.ndarray, gyro: np.ndarray, bias: np.ndarray,
             motion: np.ndarray, *, max_ms: float = 100.0, step_ms: float = 5.0) -> float:
    """Shift of the IMU clock (ms) that best aligns gyro angle with visual angle on the motion frames."""
    best, best_r = 0.0, -2.0
    for s in np.arange(-max_ms, max_ms + 1e-9, step_ms):
        g = gyro_angles(t_frames, t_imu + int(s * 1e6), gyro, bias)
        m = motion & np.isfinite(vis) & np.isfinite(g)
        if m.sum() < 20: continue
        r = np.corrcoef(vis[m], g[m])[0, 1]
        if r > best_r: best, best_r = float(s), float(r)
    return best


def vi_gate(vi: dict) -> tuple[bool, list[str]]:
    g = GATES["vi"]; f = []
    if vi["env_ratio"] < g["min_env_ratio"]: f.append(f"env_ratio {vi['env_ratio']:.2f} < {g['min_env_ratio']}")
    mm = vi["masked_motion"]
    if mm["inliers_p50"] < g["min_inliers_p50"]: f.append(f"masked inliers p50 {mm['inliers_p50']:.0f} < {g['min_inliers_p50']}")
    if mm["inliers_p10"] < g["min_inliers_p10"]: f.append(f"masked inliers p10 {mm['inliers_p10']:.0f} < {g['min_inliers_p10']}")
    if mm["grid_cells_p50"] < g["min_grid_cells_p50"]: f.append(f"grid coverage p50 {mm['grid_cells_p50']:.0f}/16 < {g['min_grid_cells_p50']}")
    ga = vi.get("gyro_masked")
    if ga is None: f.append("NOT MEASURED: no IMU in this take")
    else:
        if not (ga["r"] >= g["min_gyro_r"]): f.append(f"gyro r {ga['r']:.2f} < {g['min_gyro_r']}")
        lo, hi = g["gyro_slope"]
        if not (lo <= ga["slope"] <= hi): f.append(f"visual/gyro angle slope {ga['slope']:.2f} outside {lo}-{hi}")
    return not f, f


def verdict(hand_ok: bool | None, vi_ok: bool | None) -> str:
    if hand_ok is None or vi_ok is None: return "INCOMPLETE — a half of the matrix is NOT MEASURED"
    return {(True, True): "SHARE: one wrist camera for VI + gripper (mount PASS)",
            (True, False): "SPLIT: keep this camera for hand/gripper; VI needs its own camera (or re-aim and retake)",
            (False, True): "GAP DETECTOR: VI keeps this camera; gripper needs a custom thumb-index gap detector",
            (False, False): "RE-AIM: change the mount angle and retake"}[(bool(hand_ok), bool(vi_ok))]


# ------------------------------------------------------------------------------------------------ analyze
def analyze(ep_dir: Path, *, side: str, stream: str | None, out: Path, det_downscale: int = 2, vi_downscale: int = 2,
            hand_balance: float = 0.6, views: tuple[str, ...] = ("raw", "rect"), segments: list[str] | None = None,
            mask_poly: list | None = None, hand_stride: int = 1) -> dict:
    import cv2
    import pandas as pd
    from handumi_collector.pose.episode_io import RawEpisode
    from handumi_collector.pose.timing import fit_device_to_host, imu_host_times_ns
    from ..transforms.fisheye import load_wrist_fisheye
    ep = RawEpisode.load(ep_dir); stream = stream or f"{side}_wrist"; out.mkdir(parents=True, exist_ok=True)
    windows = windows_from_args(segments, ep.t_start_ns) if segments else windows_from_events(ep.events, ep.t_stop_ns)
    cal = load_wrist_fisheye(side); rect, vi = make_views(cal, det_downscale=det_downscale, vi_downscale=vi_downscale, hand_balance=hand_balance)
    rep: dict = dict(episode=str(ep_dir), side=side, stream=stream, fisheye_version=cal["version"], windows=[w.__dict__ for w in windows],
                     gates=GATES, views={}, vi={})

    # ---- hand: one MediaPipe pass per view (sequential, each landmarker closed: the macOS GPU graph leaks per frame)
    preps = dict(raw=lambda im: cv2.resize(im, (im.shape[1] // det_downscale, im.shape[0] // det_downscale), interpolation=cv2.INTER_AREA),
                 rect=lambda im: rect(cv2.resize(im, (im.shape[1] // det_downscale, im.shape[0] // det_downscale), interpolation=cv2.INTER_AREA)))
    hand_dfs = {}
    for v in views:
        df = run_hand_pass(ep.iter_frames(stream), v, preps[v], stride=hand_stride)
        summ, df = summarize_view(df, windows); rep["views"][v] = summ; hand_dfs[v] = df
        df.drop(columns=["lm2d", "lm3d"]).to_parquet(out / f"hand_{v}.parquet")
        print(f"[hand:{v}] detect {summ['detect_all']:.2f} grip {summ['detect_grip']:.2f} pinch {summ['detect_pinch']:.2f} "
              f"best={summ['best_signal']} gate={summ['gate_pass']}")
    scored = [(v, s) for v, s in rep["views"].items() if s.get("gate_pass") is not None]
    best_view = max(scored, key=lambda vs: (vs[1]["gate_pass"], vs[1]["detect_grip"]))[0] if scored else (views[0] if views else None)
    rep["hand_best_view"] = best_view
    hand_ok = rep["views"][best_view]["gate_pass"] if best_view else None

    # ---- mask: landmark hull occupancy (from the best view) + camera-fixed static region over the motion frames
    pts_vi = []
    if best_view:
        for r in hand_dfs[best_view].itertuples():
            if r.lm2d is None: continue
            p = np.asarray(r.lm2d, float).reshape(21, 2)
            pts_vi.append(raw_px_to_vi(p, cal, det_downscale, vi) if best_view == "raw" else rect_px_to_vi(p, rect, vi))
    scale = 0.25; W, H = vi.size
    fm = ep.frames[stream]; t_all = fm.capture_ns.astype(np.int64)
    name_all, _, _ = label_frames(t_all, windows)
    motion_by_t = {int(t): (n in MOTION_WINDOWS) for t, n in zip(t_all, name_all)} if windows else {int(t): True for t in t_all}
    diffs, grads, prev_small = [], [], None
    for vf, fi, t_ns, img in ep.iter_frames(stream, gray=True):
        g = vi(cv2.resize(img, (img.shape[1] // vi_downscale, img.shape[0] // vi_downscale), interpolation=cv2.INTER_AREA))
        sm = cv2.resize(g, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA).astype(np.int16)
        if prev_small is not None and motion_by_t.get(int(t_ns), False):
            diffs.append(np.abs(sm - prev_small).astype(np.uint8))
            s8 = sm.astype(np.float32); grads.append(np.hypot(cv2.Sobel(s8, cv2.CV_32F, 1, 0), cv2.Sobel(s8, cv2.CV_32F, 0, 1)).astype(np.float16) / 4)
        prev_small = sm
    static_small = static_region_mask(np.stack(diffs), np.stack(grads)) if len(diffs) >= 30 else None
    tex_small = texture_map(np.stack(grads).astype(np.float32)) if len(grads) >= 30 else None
    occ_small = hull_occupancy(pts_vi, (W, H), scale) if pts_vi else None
    mask = build_mask(static_small, occ_small, (W, H), poly=mask_poly)
    cv2.imwrite(str(out / "mask_vi.png"), mask.astype(np.uint8) * 255)
    env = ~mask; rep["vi"]["env_ratio"] = float(env.mean())
    if tex_small is not None:     # how much of the UNMASKED image actually carries texture (what VI can use)
        tex = cv2.resize(tex_small.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        rep["vi"]["env_textured_ratio"] = float((env & tex).mean()); cv2.imwrite(str(out / "texture_vi.png"), tex.astype(np.uint8) * 255)
    rep["vi"]["mask_sources"] = dict(static_region=static_small is not None, landmark_hull=occ_small is not None, manual_poly=bool(mask_poly),
                                     static_frac=float(static_small.mean()) if static_small is not None else None,
                                     hull_frac=float((occ_small >= 0.05).mean()) if occ_small is not None else None)

    # ---- VI pass: masked vs unmasked pair tracking
    rows, prev, prev_t = [], None, None
    for vf, fi, t_ns, img in ep.iter_frames(stream, gray=True):
        g = vi(cv2.resize(img, (img.shape[1] // vi_downscale, img.shape[0] // vi_downscale), interpolation=cv2.INTER_AREA))
        r = dict(t_ns=int(t_ns), motion=motion_by_t.get(int(t_ns), False))
        if prev is not None:
            a = pair_rotation(prev, g, vi.K_new, env); b = pair_rotation(prev, g, vi.K_new, None)
            ip = b.pop("_inlier_pts", None); a.pop("_inlier_pts", None)
            if ip is not None and len(ip):
                xi, yi = np.clip(ip[:, 0].astype(int), 0, W - 1), np.clip(ip[:, 1].astype(int), 0, H - 1)
                b["inliers_in_mask"] = int(mask[yi, xi].sum())
            r.update({f"m_{k}": v for k, v in a.items()}); r.update({f"u_{k}": v for k, v in b.items()})
        rows.append(r); prev, prev_t = g, t_ns
    vdf = pd.DataFrame(rows)
    mot = vdf.motion.to_numpy(bool) & vdf.get("m_inliers", pd.Series(np.nan, index=vdf.index)).notna().to_numpy()

    def blk(prefix, sel):
        x = vdf[sel]
        return dict(n=int(len(x)), inliers_p50=float(x[f"{prefix}inliers"].median()), inliers_p10=float(x[f"{prefix}inliers"].quantile(0.1)),
                    corners_p50=float(x[f"{prefix}corners"].median()), grid_cells_p50=float(x[f"{prefix}grid_cells"].median()),
                    fail_frac=float((x[f"{prefix}inliers"] < 15).mean()))
    if mot.any():
        rep["vi"]["masked_motion"] = blk("m_", mot); rep["vi"]["unmasked_motion"] = blk("u_", mot)
        u = vdf[mot]; rep["vi"]["unmasked_inliers_on_hand_frac"] = float((u.u_inliers_in_mask / u.u_inliers.clip(lower=1)).median())
    # ---- gyro agreement (rotation angle is extrinsic-free)
    if side in ep.imu and mot.any():
        imu = ep.imu[side]; clock = fit_device_to_host(imu.device_us, imu.host_ns); t_imu = imu_host_times_ns(imu, clock, 0)
        order = np.argsort(t_imu, kind="stable"); t_imu, gy = t_imu[order], np.asarray(imu.gyro, float)[order]
        still = np.zeros(len(t_imu), bool)
        for w in windows:
            if w.name in ("still_open", "return_still"): still |= (t_imu >= w.t0_ns) & (t_imu < w.t1_ns)
        bias = np.median(gy[still], axis=0) if still.sum() > 50 else np.zeros(3)
        tf = vdf.t_ns.to_numpy(np.int64)
        lag = best_lag(vdf.m_angle_deg.to_numpy(float), tf, t_imu, gy, bias, mot)
        g = gyro_angles(tf, t_imu + int(lag * 1e6), gy, bias); vdf["gyro_angle_deg"] = g
        rep["vi"]["imu_lag_ms"] = lag
        rep["vi"]["gyro_masked"] = gyro_agreement(np.where(mot, vdf.m_angle_deg, np.nan), g)
        rep["vi"]["gyro_unmasked"] = gyro_agreement(np.where(mot, vdf.u_angle_deg, np.nan), g)
    vdf.to_parquet(out / "vi_pairs.parquet")
    if "masked_motion" in rep["vi"]:
        vi_ok, vi_fails = vi_gate(rep["vi"])
        if any(s.startswith("NOT MEASURED") for s in vi_fails) and len(vi_fails) == 1: vi_ok = None
    else:
        vi_ok, vi_fails = None, ["NOT MEASURED: no motion frames"]
    rep["vi"].update(gate_pass=vi_ok, gate_fails=vi_fails)
    rep["hand_gate_pass"] = hand_ok; rep["verdict"] = verdict(hand_ok, vi_ok)
    write_overlays(ep, stream, out, mask, vi, vi_downscale, hand_dfs.get(best_view), best_view, rect, cal, det_downscale)
    write_signal_plot(hand_dfs.get(best_view), windows, rep["views"].get(best_view, {}), out)
    (out / "g0_report.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return rep


# ------------------------------------------------------------------------------------------------ pictures
def write_overlays(ep, stream, out, mask, vi, vi_downscale, hdf, view, rect, cal, det_downscale, n: int = 6) -> None:
    """A strip of VI-view frames with the mask tinted and (when detected) the landmarks drawn."""
    import cv2
    fm = ep.frames[stream]; picks = set(np.linspace(0, len(fm) - 1, n).astype(int).tolist())
    lm_by_t = {} if hdf is None else {int(r.t_ns): r.lm2d for r in hdf.itertuples() if r.lm2d is not None}
    tiles = []
    for k, (vf, fi, t_ns, img) in enumerate(ep.iter_frames(stream)):
        if k not in picks: continue
        g = vi(cv2.resize(img, (img.shape[1] // vi_downscale, img.shape[0] // vi_downscale), interpolation=cv2.INTER_AREA))
        tint = g.copy(); tint[mask] = (0.45 * tint[mask] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        p = lm_by_t.get(int(t_ns))
        if p is not None:
            q = np.asarray(p, float).reshape(21, 2)
            q = raw_px_to_vi(q, cal, det_downscale, vi) if view == "raw" else rect_px_to_vi(q, rect, vi)
            for j, (x, y) in enumerate(q):
                if np.isfinite(x): cv2.circle(tint, (int(x), int(y)), 5 if j in (TIP_THUMB, TIP_INDEX) else 3, (0, 255, 0) if j in (TIP_THUMB, TIP_INDEX) else (255, 255, 0), -1)
        cv2.putText(tint, f"t={(t_ns - ep.t_start_ns) / 1e9:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(cv2.resize(tint, (480, int(480 * tint.shape[0] / tint.shape[1]))))
    if tiles:
        rows = [np.hstack(tiles[i:i + 3] + [np.zeros_like(tiles[0])] * (3 - len(tiles[i:i + 3]))) for i in range(0, len(tiles), 3)]
        cv2.imwrite(str(out / "overlay.jpg"), np.vstack(rows))


def write_signal_plot(hdf, windows, view_summary, out) -> None:
    if hdf is None or not len(hdf): return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t0 = int(hdf.t_ns.min()); t = (hdf.t_ns - t0) / 1e9
    fig, axs = plt.subplots(len(SIGNALS) + 1, 1, figsize=(12, 9), sharex=True)
    colors = dict(open="#cfe8cf", half="#f3e6c4", pinch="#f2caca")
    for ax, s in zip(axs, SIGNALS):
        for w in windows:
            if w.name in colors: ax.axvspan((w.t0_ns - t0) / 1e9, (w.t1_ns - t0) / 1e9, color=colors[w.name], lw=0)
        ax.plot(t, hdf[s], ".", ms=2, color="#333")
        m = view_summary.get("signals", {}).get(s, {})
        if m.get("ok"):
            ax.axhline(m["close_th"], color="#c33", lw=0.8); ax.axhline(m["open_th"], color="#3a3", lw=0.8)
            ax.set_title(f"{s}  margin {m['margin']:.2f}  d' {m['dprime']:.1f}  jitter {m['jitter_frac']:.3f}  chatter {m['chatter']}", fontsize=9, loc="left")
        else:
            ax.set_title(f"{s}  {m.get('why', 'not scored')}", fontsize=9, loc="left")
        ax.set_ylabel(s)
    axs[-1].plot(t, hdf.detected.astype(int), drawstyle="steps-post", color="#36c"); axs[-1].set_ylabel("detected"); axs[-1].set_xlabel("s")
    fig.tight_layout(); fig.savefig(out / "signals.png", dpi=110); plt.close(fig)


# ------------------------------------------------------------------------------------------------ record
def record(args) -> int:
    from handumi_collector.collector.autoloop import Speaker
    from handumi_collector.collector.session import CollectorSession
    from handumi_collector.config import DEFAULT_CONFIG_DIR, load_config
    cfg = load_config(DEFAULT_CONFIG_DIR, hardware=args.hardware, mock=args.mock)
    root = Path(args.root); sd = root / f"G0MOUNT_{time.strftime('%Y%m%d_%H%M%S')}"
    s = CollectorSession.create(cfg, session_dir=sd)
    sp = Speaker(enabled=not args.mock, voice=args.voice)
    try:
        time.sleep(2.5)
        ok, why = s.can_record(); print("devices:", {n: (st.connected, st.error) for n, st in s.devices.statuses().items()})
        if not ok: print("cannot record:", why); return 2
        sp.say("녹화 시작합니다"); time.sleep(2.0)
        s.start()
        for name, dur, cue in PROTOCOL:
            s.recorder.event("g0_segment", "g0", dict(name=name, dur_s=dur, cue=cue)); sp.say(cue)
            print(f"  {name:13s} {dur:4.1f}s  {cue}"); time.sleep(dur * args.time_scale)
        s.stop(); m = s.keep(notes=f"G0-0 mount qualification ({args.notes})" if args.notes else "G0-0 mount qualification")
        sp.say("끝"); ep = Path(sd) / m["episode_dir"]
        print("episode:", ep, "streams:", {k: v["frames"] for k, v in m["streams"].items()}, "sensors:", m["sensor_messages"])
        print(f"next: .venv/bin/python -m ego_teleop.tools.g0_mount_qual analyze {ep}")
        return 0
    finally:
        s.close()


# ------------------------------------------------------------------------------------------------ preview
def preview(args) -> int:
    """Aiming aid while the mount is being adjusted: live wrist frame + landmarks + A/B/C + IMU health, written as a
    JPEG + status JSON (the UI polls them; no cv2 window, same reason as every HUD here). Never run during a take —
    it holds the camera and runs MediaPipe on every frame."""
    import cv2, signal
    from handumi_collector.config import DEFAULT_CONFIG_DIR, load_config
    from handumi_collector.devices.manager import DeviceManager
    from ego_collector.hands.mediapipe_tracker import HandLandmarker
    cfg = load_config(DEFAULT_CONFIG_DIR, hardware=args.hardware, mock=args.mock)
    dm = DeviceManager(cfg.hardware); dm.build(); dm.connect_all()
    cam = dm.cameras.get(f"{args.side}_wrist") or next(iter(dm.cameras.values()))
    imu = dm.imus.get(args.side)
    jpg, stp = Path(args.jpeg), Path(args.status); jpg.parent.mkdir(parents=True, exist_ok=True)
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True)); signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    lm = HandLandmarker(num_hands=2, recycle_every=900)        # non-root UVC: recycling works and bounds the GPU leak
    last_fi, last_ms, n, t_rate, hist, fps = None, -1, 0, time.monotonic(), [], float('nan')
    try:
        while not stop["v"]:
            f = cam.latest()
            if f is None or f.frame_index == last_fi: time.sleep(0.005); continue
            last_fi = f.frame_index; img = f.image
            sm = cv2.resize(img, (img.shape[1] // args.downscale, img.shape[0] // args.downscale), interpolation=cv2.INTER_AREA)
            ts = max(int(f.capture_ns // 1_000_000), last_ms + 1); last_ms = ts
            dets = lm.detect_all(sm, ts); st = dict(t=time.time(), detected=bool(dets), n_hands=len(dets))
            vis = sm.copy()
            if dets:
                d = max(dets, key=lambda d: d.label_confidence); p = d.landmarks_2d
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 17), (0, 17)):
                    cv2.line(vis, tuple(map(int, p[a])), tuple(map(int, p[b])), (255, 255, 0), 2)
                cv2.line(vis, tuple(map(int, p[TIP_THUMB])), tuple(map(int, p[TIP_INDEX])), (0, 255, 0), 3)
                for j in (TIP_THUMB, TIP_INDEX): cv2.circle(vis, tuple(map(int, p[j])), 7, (0, 255, 0), -1)
                h, w = sm.shape[:2]; inb = lambda j: 0 <= p[j, 0] < w and 0 <= p[j, 1] < h
                st.update(score=d.label_confidence, tips_in_view=bool(inb(TIP_THUMB) and inb(TIP_INDEX)),
                          keys_in_view=int(sum(inb(j) for j in KEY_LANDMARKS)), **aperture_signals(p, d.landmarks_3d))
                hist.append(st.get("B_ratio", np.nan))
            else:
                hist.append(np.nan)
            hist = hist[-150:]
            # B trace along the bottom: the operator sees open/pinch separation while aiming
            H, W = vis.shape[:2]; y0 = H - 10; arr = np.array(hist, float)
            if np.isfinite(arr).any():
                top = max(np.nanmax(arr), 1e-6)
                pts = [(int(W * i / 150), int(y0 - 80 * v / top)) for i, v in enumerate(arr) if np.isfinite(v)]
                for a, b in zip(pts[:-1], pts[1:]): cv2.line(vis, a, b, (0, 200, 255), 2)
            n += 1
            if time.monotonic() - t_rate > 1.0: fps = n / (time.monotonic() - t_rate); n, t_rate = 0, time.monotonic()
            st["fps"] = fps
            if imu is not None:
                s = imu.status(); st.update(imu_rate_hz=s.rate_hz, imu_age_ms=s.age_ms, imu_connected=s.connected)
                buf = getattr(imu, "buffer", None); ls = buf.latest() if buf is not None else None
                if ls is not None: st["gyro_dps"] = float(np.degrees(np.linalg.norm([ls.gx, ls.gy, ls.gz])))
            cs = cam.status(); st.update(cam_rate_hz=cs.rate_hz, cam_connected=cs.connected)
            tmp = jpg.with_suffix(".tmp.jpg"); cv2.imwrite(str(tmp), vis, [cv2.IMWRITE_JPEG_QUALITY, 75]); tmp.replace(jpg)
            ts_ = stp.with_suffix(".tmp"); ts_.write_text(json.dumps(st, default=float)); ts_.replace(stp)
    finally:
        lm.close(); dm.close_all()
    return 0


# ------------------------------------------------------------------------------------------------ export
def export(args) -> int:
    """Masked rectified gray frames + raw-clock IMU in EuRoC layout, for an RGB-only MASt3R-SLAM replay. No calib.yaml
    Tic is written: the new mount has no camera-IMU extrinsic yet, and a stale one would be worse than none."""
    import cv2
    from handumi_collector.pose.episode_io import RawEpisode
    from handumi_collector.pose.timing import fit_device_to_host, imu_host_times_ns
    from ..transforms.fisheye import load_wrist_fisheye, make_rectifier
    ep_dir = Path(args.episode); ep = RawEpisode.load(ep_dir); side = args.side; stream = f"{side}_wrist"
    cal = load_wrist_fisheye(side); vi = make_rectifier(cal["K"], cal["D"], cal["image_size"], downscale=args.downscale, balance=0.0)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE) > 127 if args.mask else None
    if mask is not None and mask.shape != (vi.size[1], vi.size[0]): raise SystemExit(f"mask {mask.shape} != VI view {vi.size[::-1]} (same --downscale as analyze?)")
    name = f"g0_{ep_dir.parent.name[-15:]}_{ep_dir.name[-3:]}_{side}{'_masked' if mask is not None else ''}"
    out = Path(args.out) / name; (out / "mav0/cam0/data").mkdir(parents=True, exist_ok=True); (out / "mav0/imu0").mkdir(parents=True, exist_ok=True)
    rows = []
    for vf, fi, t_ns, img in ep.iter_frames(stream, gray=True):
        g = vi(cv2.resize(img, (img.shape[1] // args.downscale, img.shape[0] // args.downscale), interpolation=cv2.INTER_AREA))
        if mask is not None: g = g.copy(); g[mask] = 0
        fn = f"{t_ns}.png"; cv2.imwrite(str(out / "mav0/cam0/data" / fn), g); rows.append(f"{t_ns},{fn}")
    (out / "mav0/cam0/data.csv").write_text("#timestamp [ns],filename\n" + "\n".join(rows) + "\n")
    K = vi.K_new
    (out / "mav0/cam0/sensor.yaml").write_text(json.dumps(dict(camera_model="pinhole", resolution=list(vi.size),
                                                              intrinsics=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]], distortion_coefficients=[0, 0, 0, 0])))
    if side in ep.imu:
        imu = ep.imu[side]; clock = fit_device_to_host(imu.device_us, imu.host_ns); t = imu_host_times_ns(imu, clock, 0)
        o = np.argsort(t, kind="stable"); lines = [f"{int(t[i])},{imu.gyro[i][0]:.7f},{imu.gyro[i][1]:.7f},{imu.gyro[i][2]:.7f},{imu.accel[i][0]:.6f},{imu.accel[i][1]:.6f},{imu.accel[i][2]:.6f}" for i in o]
        (out / "mav0/imu0/data.csv").write_text("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n" + "\n".join(lines) + "\n")
    (out / "export_meta.json").write_text(json.dumps(dict(episode=str(ep_dir), side=side, n_frames=len(rows), masked=mask is not None,
                                                          mask=str(args.mask) if args.mask else None, downscale=args.downscale, balance=0.0,
                                                          fisheye_version=cal["version"], camera_imu="NONE (mount not calibrated)"), indent=1))
    print(f"{name}: {len(rows)} frames {vi.size} -> {out}"); return 0


# ------------------------------------------------------------------------------------------------ cli
def print_summary(rep: dict) -> None:
    print("\n=== G0-0 mount qualification ===")
    for v, s in rep["views"].items():
        b = s["signals"].get(s["best_signal"], {}) if s["best_signal"] else {}
        print(f"hand[{v}]  detect grip {s['detect_grip']:.2f} pinch {s['detect_pinch']:.2f} longest grip miss {s['miss_run_grip_ms_max']:.0f} ms "
              f"tips-in-view {s['tips_in_view']:.2f}  best {s['best_signal']} margin {b.get('margin', float('nan')):.2f} "
              f"jitter {b.get('jitter_frac', float('nan')):.3f} chatter {b.get('chatter', '-')}  -> {s['gate_pass']}")
        for f in s["gate_fails"]: print("      ·", f)
    v = rep["vi"]; mm, um = v.get("masked_motion", {}), v.get("unmasked_motion", {})
    print(f"VI  env {v['env_ratio']:.2f}  inliers masked p50 {mm.get('inliers_p50', float('nan')):.0f} p10 {mm.get('inliers_p10', float('nan')):.0f} "
          f"(unmasked p50 {um.get('inliers_p50', float('nan')):.0f}, on-hand {v.get('unmasked_inliers_on_hand_frac', float('nan')):.2f})  "
          f"grid {mm.get('grid_cells_p50', float('nan')):.0f}/16")
    for k in ("gyro_masked", "gyro_unmasked"):
        if k in v: print(f"    {k}: r {v[k]['r']:.2f} slope {v[k]['slope']:.2f} (n {v[k]['n']}, lag {v.get('imu_lag_ms', 0):.0f} ms)")
    print(f"    -> {v['gate_pass']}"); [print("      ·", f) for f in v["gate_fails"]]
    print("VERDICT:", rep["verdict"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record"); r.add_argument("--hardware", default="g0_mount_right")
    r.add_argument("--root", default="datasets/human_handumi_raw/G0MOUNT"); r.add_argument("--voice", default="Yuna"); r.add_argument("--notes", default="")
    r.add_argument("--mock", action="store_true", help="mock devices, silent: dry-run the flow without hardware")
    r.add_argument("--time-scale", type=float, default=1.0, help="dry runs only: shrink every window by this factor")
    a = sub.add_parser("analyze"); a.add_argument("episode", type=Path); a.add_argument("--side", default="right", choices=("left", "right"))
    a.add_argument("--stream", default=None); a.add_argument("--out", type=Path, default=None)
    a.add_argument("--views", default="raw,rect"); a.add_argument("--det-downscale", type=int, default=2); a.add_argument("--vi-downscale", type=int, default=2)
    a.add_argument("--hand-balance", type=float, default=0.6); a.add_argument("--hand-stride", type=int, default=1)
    a.add_argument("--segment", action="append", default=[], metavar="name:t0:t1", help="override recorded windows (seconds from episode start)")
    a.add_argument("--mask-poly", default=None, help="json [[x,y],...] in VI-view pixels, unioned into the mask")
    e = sub.add_parser("export"); e.add_argument("episode"); e.add_argument("--side", default="right", choices=("left", "right"))
    e.add_argument("--out", required=True); e.add_argument("--mask", default=None); e.add_argument("--downscale", type=int, default=2)
    pv = sub.add_parser("preview"); pv.add_argument("--hardware", default="g0_mount_right"); pv.add_argument("--side", default="right")
    pv.add_argument("--mock", action="store_true"); pv.add_argument("--downscale", type=int, default=2)
    pv.add_argument("--jpeg", default="runs/g0_ui/preview.jpg"); pv.add_argument("--status", default="runs/g0_ui/preview.json")
    args = ap.parse_args(argv)
    if args.cmd == "record": return record(args)
    if args.cmd == "preview": return preview(args)
    if args.cmd == "export": return export(args)
    out = args.out or (args.episode / "derived" / "g0_mount")
    rep = analyze(args.episode, side=args.side, stream=args.stream, out=out, det_downscale=args.det_downscale, vi_downscale=args.vi_downscale,
                  hand_balance=args.hand_balance, views=tuple(v for v in args.views.split(",") if v), segments=args.segment or None,
                  mask_poly=json.loads(args.mask_poly) if args.mask_poly else None, hand_stride=args.hand_stride)
    print_summary(rep); print(f"\nreport: {out / 'g0_report.json'}  pictures: {out / 'overlay.jpg'}, {out / 'signals.png'}, {out / 'mask_vi.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
