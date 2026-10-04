"""Known-size metric scale from the red cube, in the SAME units as the MASt3R-SLAM trajectory (HRA_red, 2026-10-03).

Why: on the 10 s right-only approach takes the IMU-VI scale is unobservable (pilot s = 0.084 / 0.020 vs ego p50 0.36-0.50;
displacements came out 3-8 cm).  User decision: per-episode known-size scale, never "assume the approach was N cm".

Input per side: <runs>/<tag>/ss1.kf.npz from run_perframe_resident_kfdump.py -- per keyframe the pointmap X in the
keyframe's camera frame, its confidence C, the RGB MASt3R saw, and the FINAL keyframe Sim3 T_WC = [t, q xyzw, s].
World (= trajectory) lengths are s_k * camera-frame lengths, so every length below is multiplied by s_k.

Per keyframe:
    red mask (HSV, largest blob; NOT eroded: erosion shrank faces 5-17 % in the synthetic test -- flying pixels at the
    silhouette are off every face plane and fall out of the RANSAC inliers instead)
    -> 3D points (confidence-gated) -> sequential RANSAC planes (<= 3 faces), each must be ~orthogonal to the others
    -> per face: minimum-area rectangle of the in-plane points (1 % farthest trimmed) = two edge estimates
       a face counts only if it is square-ish (occlusion by the hand shrinks one side -> rectangle -> rejected),
       is not seen edge-on (angle between normal and view ray < MAX_VIEW_DEG), has enough points and a small residual
    -> L_raw(frame) = median of the accepted face edges (world units);  s_frame = L_real / L_raw
Per episode:  s_cube = median(s_frame) over >= MIN_FRAMES accepted frames with spread (MAD / median) <= MAX_SPREAD,
              else INVALID with the reason.  Every number goes into the QC record (user list 2026-10-03).
"""
from __future__ import annotations
import json
import pathlib
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")   # numpy 2 + macOS Accelerate: spurious
CUBE_EDGE_M = 0.05             # the task cube.  CONFIRM with the operator before freezing the dataset
MIN_MASK_PX = 150              # at the dumped (stride-2) resolution
MIN_FACE_PTS = 60
MAX_RESID_FRAC = 0.06          # plane RMS residual / edge estimate
SQUARE_TOL = 0.25              # face aspect must be within 1 +- 0.25
MAX_VIEW_DEG = 70.0            # face normal vs view ray
ORTHO_TOL_DEG = 20.0           # extra faces must be 90 +- 20 deg from the first
MIN_FRAMES = 3
MAX_SPREAD = 0.15              # MAD / median of s_frame inside an episode
CONF_MIN = 1.5                 # MASt3R confidence (same default C_conf_threshold as the SLAM viewer)


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def red_mask(rgb: np.ndarray, *, erode: int = 0) -> np.ndarray:
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    m = (((h <= 18) | (h >= 170)) & (s >= 150) & (v >= 50)).astype(np.uint8)   # the HRA cube is orange-red: hue ~10, sat ~230
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1: return np.zeros_like(m, bool)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); m = (lab == k).astype(np.uint8)
    if erode: m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    return m.astype(bool)


def fit_plane(P):
    c = P.mean(0); _, sv, vt = np.linalg.svd(P - c, full_matrices=False); return c, vt[2], vt[:2]


def ransac_plane(P, thr, iters=200, rng=None):
    rng = rng or np.random.default_rng(0); best = None
    for _ in range(iters):
        i = rng.choice(len(P), 3, replace=False); n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]])
        if np.linalg.norm(n) < 1e-12: continue
        n /= np.linalg.norm(n); inl = np.abs((P - P[i[0]]) @ n) < thr
        if best is None or inl.sum() > best.sum(): best = inl
    return best


def face_rect(uv, trim=0.01):
    """side lengths of the minimum-area rectangle around the in-plane points, after dropping the `trim` fraction farthest
    from the median.  Orientation-free: a PCA extent is undefined for a square (isotropic) face and reads up to sqrt(2) L."""
    import cv2
    d = np.linalg.norm(uv - np.median(uv, 0), axis=1); keep = d <= np.quantile(d, 1 - trim)
    (_, _), (w, h), _ = cv2.minAreaRect(uv[keep].astype(np.float32)); return np.array([w, h], float)


def faces_of(P, view_dirs, *, max_faces=3):
    """sequential RANSAC faces. P (n,3) camera-frame points (any unit), view_dirs (n,3) unit rays."""
    out, rest = [], np.arange(len(P))
    ext = np.percentile(P, 97, axis=0) - np.percentile(P, 3, axis=0); thr = 0.04 * float(np.linalg.norm(ext)) + 1e-9
    for _ in range(max_faces):
        if len(rest) < MIN_FACE_PTS: break
        inl = ransac_plane(P[rest], thr)
        if inl is None or inl.sum() < MIN_FACE_PTS: break
        idx = rest[inl]; c, n, ax = fit_plane(P[idx]); resid = float(np.sqrt(np.mean(((P[idx] - c) @ n) ** 2)))
        e = face_rect(((P[idx] - c) @ ax.T))
        view = float(np.degrees(np.arccos(min(1.0, abs(float(np.mean(view_dirs[idx], 0) @ n) / max(np.linalg.norm(np.mean(view_dirs[idx], 0)), 1e-9))))))
        out.append(dict(n_pts=int(len(idx)), normal=n, edges=e, resid=resid, view_deg=view)); rest = rest[~inl]
    return out


def frame_scale(X_cam, C, rgb, s_k, K, stride, *, edge_m=CUBE_EDGE_M) -> dict:
    m = red_mask(rgb); rec = dict(mask_px=int(m.sum()))
    if m.sum() < MIN_MASK_PX: return dict(rec, ok=False, why="mask_small")
    good = m & (C >= CONF_MIN) & np.isfinite(X_cam).all(-1)
    P = X_cam[good].astype(np.float64); rec["pts"] = int(len(P))
    if len(P) < MIN_FACE_PTS: return dict(rec, ok=False, why="few_points")
    vd = P / np.linalg.norm(P, axis=1, keepdims=True)
    fs = faces_of(P, vd); acc = []; why = []
    for j, f in enumerate(fs):
        if j and abs(float(f["normal"] @ fs[0]["normal"])) > np.sin(np.radians(ORTHO_TOL_DEG)): why.append("not_orthogonal"); continue
        a, b = sorted(f["edges"]); L = float(np.sqrt(a * b))
        if b / max(a, 1e-12) > 1 + SQUARE_TOL: why.append("occluded_or_partial"); continue
        if f["view_deg"] > MAX_VIEW_DEG: why.append("edge_on"); continue
        if f["resid"] / max(L, 1e-12) > MAX_RESID_FRAC: why.append("residual"); continue
        acc.append(dict(L_cam=L, a=a, b=b, resid_frac=f["resid"] / L, view_deg=f["view_deg"], n_pts=f["n_pts"]))
    rec.update(faces=len(fs), faces_ok=len(acc), reject=why)
    if not acc: return dict(rec, ok=False, why="no_valid_face:" + ",".join(sorted(set(why))) if why else "no_face")
    L_raw = float(s_k * np.median([f["L_cam"] for f in acc]))
    return dict(rec, ok=True, L_raw=L_raw, s_frame=edge_m / L_raw, face_resid_frac=float(np.median([f["resid_frac"] for f in acc])),
                aspect=float(np.median([f["b"] / f["a"] for f in acc])), view_deg=float(np.median([f["view_deg"] for f in acc])))


def episode_scale(kf_npz: pathlib.Path, *, ss1_csv: pathlib.Path | None = None, edge_m=CUBE_EDGE_M, use_keyframes: bool = True) -> dict:
    """keyframes (final Sim3 scale) + tracked-frame red ROIs at full resolution (Sim3 scale of that frame from ss1.csv,
    the same re-expression on the final keyframes the trajectory uses)"""
    z = np.load(kf_npz, allow_pickle=True)
    frames = []
    if use_keyframes and len(z["T_WC"]):
        X, C, I, T = z["X_cam"].astype(np.float32), z["C"].astype(np.float32), z["img"], z["T_WC"]
        for k in range(len(T)):
            r = frame_scale(X[k], C[k], I[k], float(T[k, 7]), z["K"], int(z["stride"]), edge_m=edge_m)
            r.update(src="kf", kf=k, frame_idx=int(z["frame_idx"][k])); frames.append(r)
    if "roi_src" in z.files and len(z["roi_src"]):
        import pandas as pd
        ss = pd.read_csv(ss1_csv if ss1_csv else pathlib.Path(kf_npz).with_name("ss1.csv")).set_index("frame_idx")
        for j, src in enumerate(z["roi_src"]):
            src = int(src)
            if src not in ss.index or str(ss.loc[src, "is_lost"]).lower() == "true": continue
            r = frame_scale(z["roi_X"][j].astype(np.float32), z["roi_C"][j].astype(np.float32), z["roi_img"][j],
                            float(ss.loc[src, "sim3_scale"]), z["K"], 1, edge_m=edge_m)
            r.update(src="roi", frame_idx=src); frames.append(r)
    ok = [f for f in frames if f["ok"]]; s = np.array([f["s_frame"] for f in ok])
    out = dict(n_keyframes=len(frames), n_frames_ok=len(ok), cube_edge_m=edge_m, frames=frames,
               reject_counts={w: sum(1 for f in frames if not f["ok"] and f["why"].startswith(w.split(":")[0]))
                              for w in sorted(set(f["why"].split(":")[0] for f in frames if not f["ok"]))})
    if len(ok) < MIN_FRAMES: return dict(out, valid=False, reason=f"only {len(ok)} keyframes with a measurable cube (< {MIN_FRAMES})")
    med = float(np.median(s)); spread = float(np.median(np.abs(s - med)) / med)
    out.update(s_cube=med, spread=spread, cube_edge_raw=float(edge_m / med),
               fit_residual=float(np.median([f["face_resid_frac"] for f in ok])), num_cube_points=int(np.median([f["pts"] for f in ok])))
    if spread > MAX_SPREAD: return dict(out, valid=False, reason=f"s_frame spread {spread:.2f} > {MAX_SPREAD}")
    return dict(out, valid=True, reason="")


def strip(rec: dict) -> dict:
    """QC row without the per-frame detail"""
    return {k: v for k, v in rec.items() if k != "frames"}
