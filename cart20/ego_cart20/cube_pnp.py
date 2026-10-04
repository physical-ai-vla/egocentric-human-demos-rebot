"""Metric scale of a MASt3R-SLAM trajectory from a STATIC cube of known edge, by PnP (HRA_red, 2026-10-03, user design).

The cube is the calibration object.  Per frame of the SLAM input video (the exact frames ss1.csv indexes):
    KB-fisheye -> pinhole (cv2.fisheye, balance 0 = the camera MASt3R saw: same axes, R = I)
    orange-red blob -> convex hull -> hexagon (3 faces visible) -> 12 cyclic assignments to the cube's silhouette corners
    -> solvePnP, best reprojection with cheirality (the hexagon's "near" corner must be the corner closest to the camera)
    -> t_i = cube centre in the camera frame, METRES.  t_i is invariant to the cube's 48 symmetries, so a corner
       labelling that flips between frames does not matter.
SLAM gives camera->world (R_i, p_i) in arbitrary world units.  The cube does not move, so with k = world units per metre
    c_i = p_i + k * R_i @ t_i      must be the same point for every frame
    k = - sum (p_i - p_mean) . (u_i - u_mean) / sum |u_i - u_mean|^2,   u_i = R_i t_i      (closed form, then trimmed)
    metric scale s = 1 / k  (metres per world unit; the trajectory times s is in metres)
Nothing here uses MASt3R depth, the pointmap, the Sim3 scale column, or the IMU.  QC per episode: valid PnP frames,
reprojection error, scale spread (bootstrap), cube-centre residual in metres.
"""
from __future__ import annotations
import pathlib
import re

import numpy as np

CUBE_EDGE_M = 0.038            # measured by the operator 2026-10-03 (the HRL80 / HRA stacking cube set). 0.05 was a guess
MIN_AREA_PX = 400              # blob area at 960x540 (pinhole)
MIN_SOLIDITY = 0.88
MAX_REPROJ_PX = 3.0
FACE_MAX_VIEW_DEG = 35.0      # single-face model: face normal vs line of sight
MIN_FRAMES = 10             # validity is judged by the bootstrap spread + centre residual gates; 10 frames already over-determine one scalar
MAX_REL_SPREAD = 0.10          # bootstrap (p84 - p16) / 2 / median
MAX_CENTRE_RESID_M = 0.02

# cube silhouette from the octant of the near corner (+,+,+): alternating single / double sign flips, cyclic
_SIL = np.array([(-1, 1, 1), (-1, -1, 1), (1, -1, 1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)], float)
# two visible faces (+z top, +y toward the camera): the silhouette runs over the top-back edge, down the front-right edge,
# along the front-bottom edge and up the front-left edge (measured: the far approach frames see exactly top + front)
_SIL2 = np.array([(-1, -1, 1), (1, -1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1), (-1, 1, 1)], float)
MODELS = {"3face": (_SIL, {(1, 0, 0), (0, 1, 0), (0, 0, 1)}), "2face": (_SIL2, {(0, 1, 0), (0, 0, 1)})}
_FACES = [tuple(int(v) for v in n) for n in np.vstack([np.eye(3), -np.eye(3)])]
_CORNERS = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)


def read_setting(path):
    t = pathlib.Path(path).read_text(); g = lambda k: float(re.search(rf"^{k}:\s*([-\d.eE]+)", t, re.M).group(1))
    K = np.array([[g("Camera1.fx"), 0, g("Camera1.cx")], [0, g("Camera1.fy"), g("Camera1.cy")], [0, 0, 1.0]])
    D = np.array([g(f"Camera1.k{i}") for i in (1, 2, 3, 4)]).reshape(4, 1)
    return K, D, (int(g("Camera.width")), int(g("Camera.height")))


def undistorter(K, D, size):
    import cv2
    P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, np.eye(3), balance=0.0)
    mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, size, cv2.CV_32FC1)
    return P, (lambda img: cv2.remap(img, mx, my, cv2.INTER_LINEAR))


def cube_blob(bgr):
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (((h <= 18) | (h >= 170)) & (s >= 150) & (v >= 50)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1: return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); x, y, w, hh, a = st[k]
    H, W = m.shape
    if a < MIN_AREA_PX or x <= 1 or y <= 1 or x + w >= W - 1 or y + hh >= H - 1: return None     # cut by the border
    return (lab == k).astype(np.uint8)


def hexagon(mask):
    import cv2
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea); hull = cv2.convexHull(c)
    sol = cv2.contourArea(c) / max(cv2.contourArea(hull), 1e-9)
    per = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.08, 36):
        ap = cv2.approxPolyDP(hull, eps * per, True)
        if len(ap) == 6: return sharpen(c.reshape(-1, 2).astype(np.float64), ap.reshape(6, 2).astype(np.float64)), sol
        if len(ap) == 4 and quad is None: quad = ap.reshape(4, 2).astype(np.float64)
        if len(ap) < 4: break
    # [2026-10-03] a top-down approach sees ONE face: a quadrilateral silhouette (measured on HRA ep 150 / 180, where the
    # hexagon models found 0-2 frames).  Corners from the hull vertices (no 6-side sharpening for 4 sides).
    if quad is not None: return quad, sol
    return None, sol


def sharpen(contour, verts, keep=(0.2, 0.8)):
    """corner = intersection of the lines fitted to the two adjacent silhouette sides (middle 60 % of each side's contour
    points).  A polygon vertex taken ON the contour sits inside the true corner: the physical cube edges are rounded and
    the image is blurred, which made the cube look ~10 % small / far (PnP depth above the f*L/side bound, HRL80 s_pnp
    1.13 x s_imu)."""
    n = len(contour); idx0 = [int(np.argmin(np.sum((contour - v) ** 2, 1))) for v in verts]
    order = np.argsort(idx0)                             # walk the vertices in CONTOUR order (the hull may run the other way)
    if len(set(idx0)) < 6: return verts
    res = sharpen_ordered(contour, verts[order], [idx0[j] for j in order], keep)
    out = np.empty_like(verts); out[order] = res; return out


def sharpen_ordered(contour, verts, idx, keep):
    lines = []
    for j in range(6):
        a, b = idx[j], idx[(j + 1) % 6]; seg = contour[a:b + 1] if a <= b else np.vstack([contour[a:], contour[:b + 1]])
        m = len(seg); seg = seg[int(m * keep[0]):max(int(m * keep[1]), int(m * keep[0]) + 2)]
        if len(seg) < 2: return verts
        d = seg - seg.mean(0); _, _, vt = np.linalg.svd(d, full_matrices=False); lines.append((seg.mean(0), vt[0]))
    out = []
    for j in range(6):                                   # vertex j joins side j-1 and side j
        (p1, d1), (p2, d2) = lines[j - 1], lines[j]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6: out.append(verts[j]); continue
        t = np.linalg.solve(A, p2 - p1); q = p1 + t[0] * d1
        out.append(q if np.linalg.norm(q - verts[j]) < 0.25 * np.linalg.norm(verts[(j + 1) % 6] - verts[j - 1]) else verts[j])
    return np.array(out)


def occluded_below(bgr, quad, band=8, dark=90):
    """True if the image band just below the quad's lowest edge is dark: the black gripper jaw covers the face there, so that
    edge is the jaw's outline, not the cube's (measured: these near frames moved the cube-centre residual 0.5 -> 4 cm)."""
    import cv2
    q = quad[np.argsort(quad[:, 1])[-2:]]                    # the two lowest vertices
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); H_, W_ = g.shape
    xs = np.linspace(q[0, 0], q[1, 0], 20); ys = np.linspace(q[0, 1], q[1, 1], 20) + band / 2 + 3
    v = [g[int(min(H_ - 1, max(0, y + d))), int(min(W_ - 1, max(0, x)))] for x, y in zip(xs, ys) for d in range(-band // 2, band // 2)]
    return float(np.median(v)) < dark


def pnp_face(quad, P, edge=CUBE_EDGE_M):
    """one visible square face: solvePnP on the face (z = 0 plane, 4 cyclic x 2 directions), cube centre = face centre
    pushed half an edge along the face normal AWAY from the camera.  -> (t_centre [m], rvec, rms px, "1face") or None"""
    import cv2
    sq = np.array([(-1, 1, 0), (1, 1, 0), (1, -1, 0), (-1, -1, 0)], float) * edge / 2
    best = None
    for direction in (1, -1):
        for sh in range(4):
            obj = np.roll(sq[::direction], sh, axis=0)
            ok, rv, tv = cv2.solvePnP(obj, quad, P, None, flags=cv2.SOLVEPNP_IPPE)
            if not ok or tv.ravel()[2] <= 0: continue
            rv, tv = cv2.solvePnPRefineLM(obj, quad, P, None, rv, tv)
            pr, _ = cv2.projectPoints(obj, rv, tv, P, None); e = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - quad) ** 2, 1))))
            R, _ = cv2.Rodrigues(rv); f = tv.ravel(); n = R[:, 2]
            if float(n @ (-f)) < 0: n = -n                          # outward normal points toward the camera
            # ONE face only when it is seen nearly head-on: an oblique view shows 2-3 faces whose silhouette can still reduce to 4
            # vertices; treating that as a single square biased s by -22 % on HRA ep 1 / 2 (multi-face 0.55 vs "1face" 0.43)
            if float(n @ (-f)) / np.linalg.norm(f) < np.cos(np.radians(FACE_MAX_VIEW_DEG)): continue
            c = f - n * edge / 2
            if best is None or e < best[2]: best = (c, rv.ravel().copy(), e, "1face")
    return best


def pnp_cube(hexpts, P, edge=CUBE_EDGE_M):
    if len(hexpts) == 4: return pnp_face(hexpts, P, edge)
    """best of {3-face, 2-face} x 6 cyclic shifts x 2 directions; a candidate is admissible only if the faces that face
    the camera are exactly the model's visible faces (cheirality / visibility).  -> (t [m], rvec, rms px, model) or None"""
    import cv2
    best = None
    for name, (sil, vis) in MODELS.items():
        for direction in (1, -1):
            seq = sil[::direction]
            for sh in range(6):
                obj = np.roll(seq, sh, axis=0) * edge / 2
                ok, rv, tv = cv2.solvePnP(obj, hexpts, P, None, flags=cv2.SOLVEPNP_SQPNP)
                if not ok: continue
                rv, tv = cv2.solvePnPRefineLM(obj, hexpts, P, None, rv, tv)
                if tv.ravel()[2] <= 0: continue
                R, _ = cv2.Rodrigues(rv); cam = -R.T @ tv.ravel()                 # camera centre in the cube frame
                facing = {f for f in _FACES if float(np.dot(f, cam)) > edge / 2 * (1 + 1e-3)}
                if facing != vis: continue
                pr, _ = cv2.projectPoints(obj, rv, tv, P, None); e = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - hexpts) ** 2, 1))))
                if best is None or e < best[2]: best = (tv.ravel().copy(), rv.ravel().copy(), e, name)
    return best


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def solve_k(p, u):
    """k minimising sum |p_i + k u_i - c|^2 over k, c"""
    dp, du = p - p.mean(0), u - u.mean(0); den = float(np.sum(du * du))
    return -float(np.sum(dp * du)) / den if den > 0 else np.nan


def fit_scale(p, u, *, trim_iter=3, rng=None):
    keep = np.ones(len(p), bool); k = np.nan
    for _ in range(trim_iter):
        k = solve_k(p[keep], u[keep]); c = (p + k * u)[keep].mean(0)
        r = np.linalg.norm(p + k * u - c, axis=1) / max(abs(k), 1e-12)          # metres
        med = np.median(r[keep]); mad = np.median(np.abs(r[keep] - med)) + 1e-9
        new = r <= med + 4 * 1.4826 * mad
        if new.sum() < MIN_FRAMES or np.array_equal(new, keep): break
        keep = new
    k = solve_k(p[keep], u[keep]); c = (p + k * u)[keep].mean(0); r = np.linalg.norm(p + k * u - c, axis=1) / max(abs(k), 1e-12)
    rng = rng or np.random.default_rng(0); idx = np.flatnonzero(keep); boots = []
    for _ in range(200):
        b = rng.choice(idx, len(idx)); kb = solve_k(p[b], u[b])
        if np.isfinite(kb) and kb > 0: boots.append(1 / kb)
    return dict(k=k, s=1 / k if k > 0 else np.nan, keep=keep, centre_resid_m=float(np.sqrt(np.mean(r[keep] ** 2))),
                s_p16=float(np.percentile(boots, 16)) if boots else np.nan, s_p84=float(np.percentile(boots, 84)) if boots else np.nan)


def episode_pnp_scale(video, setting, ss1_csv, *, edge=CUBE_EDGE_M, stride=1, max_frac: float = 1.0) -> dict:
    """max_frac < 1: only the first fraction of the video (validation on stacking takes, where the cube is static only until grasped)"""
    import cv2
    import pandas as pd
    K, D, size = read_setting(setting); P, und = undistorter(K, D, size)
    ss = pd.read_csv(ss1_csv).set_index("frame_idx")
    cap = cv2.VideoCapture(str(video)); i = -1; frames = []; n_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); i_max = int(n_tot * max_frac)
    while True:
        ok, img = cap.read(); i += 1
        if not ok: break
        if i > i_max: break
        if i % stride or i not in ss.index or str(ss.loc[i, "is_lost"]).lower() == "true": continue
        u_img = und(img); m = cube_blob(u_img)
        if m is None: continue
        hx, sol = hexagon(m)
        if hx is None or sol < MIN_SOLIDITY: frames.append(dict(i=i, ok=False, why="no_hexagon" if hx is None else "solidity", area=int(m.sum()))); continue
        if len(hx) == 4 and occluded_below(u_img, hx):
            frames.append(dict(i=i, ok=False, why="occluded", area=int(m.sum()))); continue
        r = pnp_cube(hx, P, edge)
        if r is None: frames.append(dict(i=i, ok=False, why="pnp", area=int(m.sum()))); continue
        t, rv, e, model = r
        frames.append(dict(i=i, ok=e <= MAX_REPROJ_PX, why="" if e <= MAX_REPROJ_PX else "reproj", area=int(m.sum()), reproj_px=e, model=model,
                           dist_m=float(np.linalg.norm(t)), t=t.tolist()))
    cap.release()
    ok = [f for f in frames if f["ok"]]
    out = dict(n_candidate_frames=len(frames), n_valid_pnp=len(ok), cube_edge_m=edge, frames=frames,
               reproj_px_p50=float(np.median([f["reproj_px"] for f in ok])) if ok else None,
               reject_counts={w: sum(1 for f in frames if not f["ok"] and f["why"] == w) for w in sorted(set(f["why"] for f in frames if not f["ok"]))})
    if len(ok) < MIN_FRAMES: return dict(out, valid=False, reason=f"only {len(ok)} valid PnP frames (< {MIN_FRAMES})")
    p = np.array([[ss.loc[f["i"], c] for c in ("x", "y", "z")] for f in ok], float)
    R = np.array([quat_xyzw_to_R([ss.loc[f["i"], c] for c in ("q_x", "q_y", "q_z", "q_w")]) for f in ok])
    u = np.einsum("nij,nj->ni", R, np.array([f["t"] for f in ok]))
    fs = fit_scale(p, u)
    spread = (fs["s_p84"] - fs["s_p16"]) / 2 / fs["s"] if np.isfinite(fs["s"]) else np.inf
    out.update(s_pnp=fs["s"], s_p16=fs["s_p16"], s_p84=fs["s_p84"], rel_spread=float(spread), centre_resid_m=fs["centre_resid_m"],
               n_used=int(fs["keep"].sum()), dist_m_range=[float(min(f["dist_m"] for f in ok)), float(max(f["dist_m"] for f in ok))])
    why = []
    if not np.isfinite(fs["s"]) or fs["s"] <= 0: why.append("non-positive scale")
    if spread > MAX_REL_SPREAD: why.append(f"bootstrap spread {spread:.2f} > {MAX_REL_SPREAD}")
    if fs["centre_resid_m"] > MAX_CENTRE_RESID_M: why.append(f"cube-centre residual {fs['centre_resid_m'] * 100:.1f} cm > {MAX_CENTRE_RESID_M * 100:.0f} cm")
    return dict(out, valid=not why, reason="; ".join(why))
