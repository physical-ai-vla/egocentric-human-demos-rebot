#!/usr/bin/env python3
"""[2026-09-18] Track the HandUMI rigid body directly from the head Orbbec RGB-D. No AprilTag, no hand model.

The target is the WRIST-CAMERA RIGID BODY, not the operator's palm. `camera_to_tcp_v1` is a wrist-camera -> TCP
offset, so a palm frame was always one uncalibrated link short; the body carries the wrist camera, so localizing the
body is localizing exactly what the calibration already describes. It is also the easier target: large, rigid, black
against a white table, in frame whenever it is in use, and it comes with depth.

    RGB-D -> table-plane removal -> depth-band filter -> dark-body segmentation -> connected components
          -> temporal identity association -> per-unit 3D point cloud -> centroid (+ PCA axes)

Translation is closed first, deliberately: the rotation component of camera_to_tcp_v1 was never measured, so a full
SE(3) TCP cannot be honest yet. PCA axes are computed and reported anyway, because their stability is what decides
whether orientation can come from the body at all or has to come from the IMU / MASt3R side.

The decisive sanity check is free and needs no ground truth: **the idle unit resting on the table must not move.**
Its 3D position std over an episode is the detector's own noise floor. Everything else is interpretation; that number
is not.

    .venv/bin/python scripts/handumi_body_track.py --episode <ep> [--save-npz] [--debug-frames 6]
    .venv/bin/python scripts/handumi_body_track.py --session <session> [--limit 5]

[2026-10-12] Hfold laundry-folding setup (fixed overhead Gemini 336, black T-shirt, both units always moving):
    .venv/bin/python scripts/handumi_body_track.py --profile laundry_v1 --episode <ep> \
        [--debug-frames 8 --debug-dir <dir>] [--save-npz] [--json <out.json>]
The default profile is unchanged; see LAUNDRY_V1 below for what differs and why.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handumi_collector.pose.rgbd_io import RgbdEpisode                         # noqa: E402

BAND = (0.25, 0.80)          # working volume in front of the head camera
ABOVE_PLANE = (0.015, 0.45)  # a body sits above the table; below is the table, far above is the operator/background
# Measured 2026-09-18 on episode_000005: above-table pixels are bimodal -- a huge peak at 10-40 (the black body) and a
# tail at 70-160 (grey glove, skin, cables). Stability barely moves across 40..95 (still-window |std| 1.5-3.4 mm), so
# this is not a knife edge; 70 is at or near the minimum for both units.
DARK_MAX = 70                # the HandUMI is black; table, glove and skin are not
MIN_AREA = 700               # px, after opening
GATE_M = 0.18                # a body cannot move this far between two frames at 30 Hz -> association gate
STILL_FRAMES = 30            # 1.0 s at 30 Hz: long enough that a plateau inside a reach cannot masquerade as rest


# ----------------------------------------------------------------------------------------- table plane
def fit_table_plane(ep: RgbdEpisode, n_frames: int = 30) -> tuple[np.ndarray, float, dict]:
    """RANSAC plane over the pooled points of the first frames. The table is the one large flat thing every episode
    contains, and removing it is what turns 'dark pixels' into 'dark pixels that are an object'.

    Returns (normal, d) with normal . X + d = 0, normal oriented so that points ABOVE the table are positive."""
    K = ep.intrinsics.K
    pts = []
    step = max(1, ep.n_frames // n_frames)
    for f in ep.iter_frames(0, min(ep.n_frames, n_frames * step), step):
        d = f.depth_m
        m = (d > BAND[0]) & (d < 1.5)
        ys, xs = np.nonzero(m)
        if len(xs) > 8000:
            sel = np.random.default_rng(0).choice(len(xs), 8000, replace=False)
            ys, xs = ys[sel], xs[sel]
        z = d[ys, xs]
        pts.append(np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], 1))
    P = np.concatenate(pts)
    rng = np.random.default_rng(0)
    best, best_n, best_d = -1, None, None
    for _ in range(300):
        i = rng.choice(len(P), 3, replace=False)
        a, b, c = P[i]
        nrm = np.cross(b - a, c - a)
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm /= nn
        dd = -nrm @ a
        inl = int((np.abs(P @ nrm + dd) < 0.008).sum())
        if inl > best:
            best, best_n, best_d = inl, nrm, dd
    # refit on the inliers, then orient the normal towards the camera (the camera is above the table it is looking at)
    inl = np.abs(P @ best_n + best_d) < 0.008
    Q = P[inl]
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
    nrm = Vt[-1]
    dd = -nrm @ c
    if (nrm @ np.array([0.0, 0.0, 1.0])) > 0:        # +z points away from the camera; the table normal must come back
        nrm, dd = -nrm, -dd
    info = dict(inlier_share=float(inl.mean()), n_points=int(len(P)),
                rms_mm=float(np.sqrt(((Q @ nrm + dd) ** 2).mean()) * 1e3))
    return nrm, float(dd), info


# ----------------------------------------------------------------------------------------- per-frame detection
def detect_bodies(f, K, nrm, pd) -> list[dict]:
    """Every dark, above-table blob in the working band, as a 3D point cloud with a centroid and principal axes."""
    d = f.depth_m
    g = cv2.cvtColor(f.rgb, cv2.COLOR_BGR2GRAY)
    H, W = d.shape
    ys, xs = np.mgrid[0:H, 0:W]
    z = d
    X = (xs - K[0, 2]) * z / K[0, 0]
    Y = (ys - K[1, 2]) * z / K[1, 1]
    height = X * nrm[0] + Y * nrm[1] + z * nrm[2] + pd        # signed distance above the table plane

    m = ((z > BAND[0]) & (z < BAND[1]) & (g < DARK_MAX)
         & (height > ABOVE_PLANE[0]) & (height < ABOVE_PLANE[1])).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        sel = (lab == i) & (z > 0)
        if sel.sum() < 200:
            continue
        P = np.stack([X[sel], Y[sel], z[sel]], 1)
        c3 = np.median(P, 0)                      # median, not mean: one depth outlier must not drag the origin
        Pc = P - c3
        # principal axes, reported so their stability can be judged; NOT yet used as an orientation
        _, sv, Vt = np.linalg.svd(Pc[np.random.default_rng(0).choice(len(Pc), min(len(Pc), 4000), replace=False)],
                                  full_matrices=False)
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        out.append(dict(area=int(stats[i, cv2.CC_STAT_AREA]), n_points=int(sel.sum()), points=P,
                        depth_cov=float(sel.sum() / max(stats[i, cv2.CC_STAT_AREA], 1)),
                        aspect=float(max(bw, bh) / max(min(bw, bh), 1)),
                        elongation=float(sv[0] / max(sv[1], 1e-9)),
                        uv=(float(cen[i][0]), float(cen[i][1])), xyz=c3, axes=Vt, sv=sv / max(sv[0], 1e-9),
                        bbox=(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                              int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))))
    return out


BODY_RADIUS_M = 0.12         # trim radius around the previous centre; chosen by bake-off, see refine()
AREA_RATIO = (0.40, 2.50)    # reported as evidence; the trim is unconditional, so this is not a gate


def refine(det: dict, prev_xyz, prev_area: float) -> dict:
    """Repair a component that has merged with the forearm, instead of rejecting it.

    The forearm is CONNECTED to the body, so no threshold on darkness separates them -- tightening DARK_MAX only eats
    into the body itself. What does separate them is size: the body is 0.14 x 0.11 x 0.08 m and the arm continues well
    past that, so points farther than BODY_RADIUS_M from where the body was last seen are not the body. Repairing
    rather than rejecting is what keeps coverage: a merged frame still carries a perfectly good body inside it.

    The area ratio is the trigger, not the criterion. It only says "this component is no longer the same shape";
    the geometry says which part of it to keep. A cold track has no previous centroid and is left alone."""
    det["repaired"] = False
    det["area_ratio"] = float(det["area"] / prev_area) if prev_area > 0 else float("nan")
    if prev_xyz is None:
        return det                            # cold track: no reference to trim against, take the blob as it is
    # ALWAYS trim, never only on a trigger. Trimming just the suspicious frames would leave the origin defined two
    # different ways -- whole blob here, body-sized neighbourhood there -- and switching between them is itself a step.
    # One definition, applied every frame: the body is what lies within its own extent of where it was last seen.
    # Bake-off, 8 episodes x 2 sides on identical frames and identical identity association (body_center_bakeoff.py):
    #
    #   estimator            >50mm  >100mm  step p95  still med
    #   full-blob centroid      78      15      21.5      1.77
    #   trim r=0.06             52      20      14.1      1.45      <- tight trims cannot follow a real fast move
    #   trim r=0.09             65      11      15.8      1.79         and snap: >100 mm gets WORSE than baseline
    #   trim r=0.12             61       9      18.9      1.79      <- chosen: >100 mm -40 %, >50 mm -22 %
    #   mean-shift bw=0.09     142      18      21.9      2.23      <- REJECTED
    #
    # Mean-shift was the intuitive answer and it is wrong here: the HandUMI is a hollow multi-part mechanism, not one
    # dense lump, so a small bandwidth hops between spurious internal modes (bw 0.05 -> 285 jumps) and a large one
    # converges back to the plain centroid. Density is not what separates the body from the arm; extent is.
    P = det["points"]
    keep = np.linalg.norm(P - prev_xyz, axis=1) < BODY_RADIUS_M
    if keep.sum() < 200:                      # nothing recognisable within reach -> leave the detection as it was
        return det
    det["xyz"] = np.median(P[keep], 0)
    det["n_points_kept"] = int(keep.sum())
    det["repaired"] = bool(keep.sum() < 0.9 * len(P))
    return det


# ----------------------------------------------------------------------------------------- identity
def associate(tracks: dict, dets: list[dict], W: int) -> dict:
    """Two tracks, `left` and `right`, held across frames.

    Association is nearest 3D centroid under a physical gate, never nearest-in-image: two units at the same image
    position but 20 cm apart in depth are not the same object, and the image alone cannot say so. A detection that
    matches no track within the gate does not steal one -- the track goes invalid for that frame, because a label
    that silently jumps to the other hand is worse than a gap.

    Cold start (and re-acquisition after a long gap) uses the left/right spatial prior: the head camera looks
    outward and unmirrored, so the operator's left unit is the one at the smaller image u."""
    assigned = {}
    used = set()
    for side in ("left", "right"):
        t = tracks.get(side)
        if t is None or t["miss"] > 15:
            continue
        cands = [(float(np.linalg.norm(dd["xyz"] - t["xyz"])), j) for j, dd in enumerate(dets) if j not in used]
        cands = [(c, j) for c, j in cands if c < GATE_M]
        if cands:
            c, j = min(cands)
            assigned[side] = j; used.add(j)
    missing = [s for s in ("left", "right") if s not in assigned]
    if missing and len(used) < len(dets):
        free = sorted((j for j in range(len(dets)) if j not in used), key=lambda j: dets[j]["uv"][0])
        # only seed a cold track from the spatial prior; never re-seed a track that is merely occluded this frame
        for side in missing:
            t = tracks.get(side)
            if t is not None and t["miss"] <= 15:
                continue
            if not free:
                break
            j = free.pop(0) if side == "left" else free.pop(-1)
            assigned[side] = j; used.add(j)
    return assigned


def update_tracks(tracks: dict, dets: list[dict], assigned: dict) -> dict:
    """Refine each assigned detection against its own track history, then advance the tracks. Shared by every caller
    so the body-track, the ego16 export and the jump audit can never drift apart."""
    for side in ("left", "right"):
        if side in assigned:
            t = tracks.get(side)
            d = refine(dets[assigned[side]], t["xyz"] if t and t["miss"] == 0 else None,
                       t.get("area", 0.0) if t and t["miss"] == 0 else 0.0)
            tracks[side] = dict(xyz=d["xyz"], area=float(d["area"]), miss=0)
        elif side in tracks:
            tracks[side]["miss"] += 1
    return tracks


def track_episode(ep_path: Path, stream: str | None, debug_frames: int) -> dict:
    ep = RgbdEpisode.load(ep_path, stream=stream)
    K = ep.intrinsics.K
    nrm, pd, plane = fit_table_plane(ep)
    n = ep.n_frames
    xyz = {s: np.full((n, 3), np.nan) for s in ("left", "right")}
    axes = {s: np.full((n, 3, 3), np.nan) for s in ("left", "right")}
    area = {s: np.zeros(n, int) for s in ("left", "right")}
    valid = {s: np.zeros(n, bool) for s in ("left", "right")}
    tracks: dict = {}
    switches = {s: 0 for s in ("left", "right")}
    repaired = {s: 0 for s in ("left", "right")}
    n_dets = []
    dbg = 0
    for f in ep.iter_frames(0, n, 1):
        dets = detect_bodies(f, K, nrm, pd)
        n_dets.append(len(dets))
        a = associate(tracks, dets, f.rgb.shape[1])
        prev_xyz = {s: (tracks[s]["xyz"] if s in tracks and tracks[s]["miss"] == 0 else None) for s in ("left", "right")}
        update_tracks(tracks, dets, a)
        for side in ("left", "right"):
            if side in a:
                d = dets[a[side]]
                if prev_xyz[side] is not None and float(np.linalg.norm(d["xyz"] - prev_xyz[side])) > GATE_M:
                    switches[side] += 1
                repaired[side] += int(d.get("repaired", False))
                xyz[side][f.index] = d["xyz"]
                axes[side][f.index] = d["axes"]
                area[side][f.index] = d["area"]
                valid[side][f.index] = True
        if dbg < debug_frames and len(dets) >= 1:
            vis = f.rgb.copy()
            for side, col in (("left", (0, 200, 255)), ("right", (255, 120, 0))):
                if side in a:
                    x, y, w, h = dets[a[side]]["bbox"]
                    cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
                    cv2.putText(vis, f"{side} {dets[a[side]]['xyz'][2]:.3f}m", (x, max(14, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            Path("/tmp/body_track").mkdir(exist_ok=True)
            cv2.imwrite(f"/tmp/body_track/{ep_path.name}_{f.index:06d}.jpg", vis)
            dbg += 1
    return dict(episode=ep_path.name, n_frames=n, plane=plane, n_dets_mean=float(np.mean(n_dets)),
                # the normal and offset themselves, not just the fit quality: the export needs them to build the
                # world frame, and re-fitting the plane there would be a second, silently divergent estimate.
                plane_normal=nrm, plane_d=pd,
                xyz=xyz, axes=axes, area=area, valid=valid, switches=switches, repaired=repaired, t_ns=ep.t_ns)


# ========================================================================================= profile laundry_v1
# [2026-10-12] Hfold laundry-folding setup: Orbbec fixed overhead, table ~0.90-0.97 m away, BOTH units moving the whole
# time, a BLACK T-shirt on the table that is exactly as dark as the units, the operator's dark hair/sleeves entering
# the bottom/right of the frame, and a black light-stand clamp + floor outside the table at the left edge.
#
# What the default profile got wrong here, and what replaces it:
#   * BAND=(0.25, 0.80) m was tuned for a lower camera. With the table at 0.90 m, the units (0.75-0.88 m) sit at the
#     far edge of / outside the band, while the stand clamp (0.60 m) and the forearm/head sit inside it. -> depth band
#     is replaced by height ABOVE THE TABLE PLANE, restricted to a TABLE ROI (hull of the plane inliers).
#   * darkness alone cannot separate the unit from the shirt -> a unit pixel must stand >= H_MIN above the plane; a
#     flat shirt (< 1-2 cm) never qualifies. A lifted shirt flap still does, so the centre is a trimmed median around
#     the velocity-predicted position, not the whole blob.
#   * "one unit rests" is gone: both tracks use a constant-velocity prediction + gate; cold start/re-acquire by the
#     left/right image prior, and the two tracks are kept from collapsing onto one blob.
LAUNDRY_V1 = dict(
    H_MIN=0.025,          # m above table: flat shirt (<~2 cm) is excluded, unit (>=3 cm) is not
    H_MAX=0.70,           # m above table: a unit lifted to shake the shirt out reaches 0.55-0.62 (0.3 m from the
                          # camera, 163800 f312 / 152646 f1735); the head is removed by the blob rule, not by this cap
    HEAD_H_P50=0.30,      # blob whose median height is above this AND larger than any unit = head/torso
    HEAD_AREA_M2=0.025,
    HEAD_H_TOUCH=9.0,     # (disabled) 'very high AND touching the border' also hit units lifted at the bottom edge
    BG_DELTA=0.02,        # must rise this much above the static background (15th pct height over the episode)
    BG_PCT=15,
    DARK_MAX=70,
    RB_MAX=14,            # R-B: skin and forearm are warm, the black unit and the shirt are neutral
    EDGE_DZ=0.03,         # m depth step between neighbours = flying pixels at an occlusion edge -> cut
    MIN_AREA_M2=0.0012,   # ~3.5 x 3.5 cm metric footprint, after opening
    SEED_AREA_M2=(0.002, 0.045),
    SEED_H_P50=(0.07, 0.70),
    UNIT_AREA_M2=(0.003, 0.03),   # a whole unit seen from above, 128-212 cm^2 measured; shirt piles are 290-510
    UNIT_H_P50=0.07,      # a unit in use stands >= 7 cm up (median); shirt folds that pass H_MIN are 3-5 cm
    H_TRACK_MIN=0.04,     # a track whose trimmed centre sinks below this has slid onto the shirt -> miss
    # Texture = share of pixels with |grad gray| > TEX_GRAD inside an 8 cm (metric) disc around the centre. The unit
    # carries white/red cables, green tips, metal and the hand's edge; a lifted shirt sheet is uniform. Measured on 96
    # hand-labelled unit positions (3 episodes): min 0.08, median 0.22; on the 13 positions where an earlier version
    # sat on the shirt: max 0.06. This is what stops a track from riding a lifted shirt flap.
    TEX_GRAD=80, TEX_R=0.08, TEX_SEED=0.08, TEX_KEEP=0.05,
    R_BODY=0.10,          # trim radius for the trimmed median (the body is 0.14 x 0.11 x 0.08 m)
    GATE=0.15,            # m from the velocity prediction: blobs farther than this are not claimed
    MAX_SHIFT=0.10,       # m: the trimmed centre may not move farther than this from the prediction (no sliding
                          # along a shirt that connects both units)
    SWAP_R=0.12,
    COLD_HALF=0.62,       # cold seeding with no other unit to order against: left needs u < 0.62 W, right u > 0.38 W
    DEFER_R=0.30,         # m: no cold seeding this close to where a briefly-missing other unit was last seen
    MIN_PTS=150,
    LOST=10,              # frames without a match before a track may be re-seeded from anywhere
    MIN_SEP=0.07,         # two tracks closer than this are collapsing onto one object
    VEL_DAMP=0.7,
    # the unit is always IN A HAND: keep only dark pixels within HAND_R (3D, projected) of raised skin. This is what
    # removes the crumpled / lifted shirt far from the hands, which height alone cannot (a fold stands 3-10 cm).
    RB_SKIN=16, ARM_GRAY=100, SKIN_GRAY=(35, 250), SKIN_MIN_M2=0.001, HAND_R=0.15,
)


def _height_map(d: np.ndarray, K: np.ndarray, nrm: np.ndarray, pd: float):
    H, W = d.shape
    ys, xs = np.mgrid[0:H, 0:W]
    X = (xs - K[0, 2]) * d / K[0, 0]
    Y = (ys - K[1, 2]) * d / K[1, 1]
    h = X * nrm[0] + Y * nrm[1] + d * nrm[2] + pd
    return X, Y, h


def laundry_scene_model(ep: RgbdEpisode, nrm: np.ndarray, pd: float, P: dict, n_frames: int = 60) -> dict:
    """Table ROI + static background, both from frames spread over the WHOLE episode (both hands move all the time, so
    no single frame is clean, but every table pixel is uncovered in most frames).

    ROI: pixels that are on the plane (|h| < 12 mm) in >= 40 % of the sampled frames -> largest component -> closed ->
    convex hull. The floor and the light stand are not on the plane, so they fall outside.
    Background: the BG_PCT-th percentile height per pixel. A static object (stand clamp, clutter on the table) stays
    raised in every frame and keeps a high percentile; a unit passing through keeps it at the table."""
    K = ep.intrinsics.K
    step = max(1, ep.n_frames // n_frames)
    hs, on = [], []
    for f in ep.iter_frames(0, ep.n_frames, step):
        d = f.depth_m
        _, _, h = _height_map(d, K, nrm, pd)
        h = np.where(d > 0, h, np.nan).astype(np.float32)
        hs.append(h)
        on.append(np.abs(h) < 0.012)
    Hs = np.stack(hs)
    frac = np.stack(on).mean(0)
    tab = (frac >= 0.4).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(tab, 8)
    roi = np.zeros_like(tab)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        big = (lab == k).astype(np.uint8)
        big = cv2.morphologyEx(big, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.concatenate(cnts))
        cv2.fillConvexPoly(roi, hull, 1)
        roi = cv2.erode(roi, np.ones((5, 5), np.uint8))
    with np.errstate(all="ignore"):
        bg = np.nanpercentile(Hs, P["BG_PCT"], axis=0)
    bg = np.where(np.isfinite(bg), bg, 0.0).astype(np.float32)
    return dict(roi=roi.astype(bool), bg_h=bg, roi_share=float(roi.mean()), n_sampled=len(hs))


def detect_bodies_laundry(f, K, nrm, pd, scene: dict, P: dict, preds: list | None = None) -> tuple[list[dict], dict]:
    """Dark, NEUTRAL, raised (>= H_MIN above plane AND above the static background), inside the table ROI, below the
    head band, not across a depth discontinuity. Returns blobs plus the per-pixel arrays the tracker trims against."""
    d = f.depth_m
    X, Y, h = _height_map(d, K, nrm, pd)
    bgr = f.rgb
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rb = bgr[..., 2].astype(np.int16) - bgr[..., 0].astype(np.int16)
    dz = np.maximum(np.abs(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=1)), np.abs(cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=1)))
    raised = scene["roi"] & (d > 0) & (h > P["H_MIN"]) & (h < P["H_MAX"]) & (h - scene["bg_h"] > P["BG_DELTA"])
    m = (raised & (g < P["DARK_MAX"]) & (rb < P["RB_MAX"]) & (dz < P["EDGE_DZ"])).astype(np.uint8)
    # "skin" = the operator's hand/forearm: warm, or plainly not black (a shadowed hand has R-B of only 5-13, but it is
    # never as dark as the unit or the shirt). The table is excluded by `raised`, the head by H_MAX.
    skin = (raised & (g > P["SKIN_GRAY"][0]) & (g < P["SKIN_GRAY"][1])
            & ((rb >= P["RB_SKIN"]) | (g >= P["ARM_GRAY"]))).astype(np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ns, slab, sst, _ = cv2.connectedComponentsWithStats(skin, 8)
    pix_m2 = (d / K[0, 0]) * (d / K[1, 1])
    if ns > 1:
        a_m2 = np.bincount(slab.ravel(), weights=pix_m2.ravel(), minlength=ns)
        skin = (a_m2[slab] >= P["SKIN_MIN_M2"]) & (slab > 0)
    else:
        skin = np.zeros_like(skin, bool)
    if skin.any():
        dist = cv2.distanceTransform((~skin).astype(np.uint8), cv2.DIST_L2, 5)
        near = (dist < P["HAND_R"] * K[0, 0] / np.maximum(d, 0.2)).astype(np.uint8)
    else:
        near = np.zeros(skin.shape, np.uint8)
    # ... OR near where a live track predicts its unit: a unit lifted towards the camera often hides the whole hand
    for q in preds or []:
        if q is None or q[2] <= 0.1:
            continue
        u, v = _uv_of(q, K)
        r = (P["R_BODY"] + 0.03) * K[0, 0] / q[2]
        if -r < u < d.shape[1] + r and -r < v < d.shape[0] + r:
            cv2.circle(near, (int(u), int(v)), int(r), 1, -1)
    # Two ways in. (1) A blob that on its own already looks like a unit -- unit-sized footprint and standing well
    # off the table -- is kept whole, hand or no hand (the hand is often hidden under the unit or out of frame at the
    # bottom edge). (2) Everything else (large merged shirt+unit blobs, low folds) is kept only near a visible
    # hand/forearm or near a live track's prediction, which is what cuts away the shirt far from the hands.
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n0, lab0, st0, _ = cv2.connectedComponentsWithStats(m, 8)
    unit_like = np.zeros(n0, bool)
    if n0 > 1:
        a0 = np.bincount(lab0.ravel(), weights=pix_m2.ravel(), minlength=n0)
        order = np.argsort(lab0.ravel(), kind="stable")
        bounds = np.searchsorted(lab0.ravel()[order], np.arange(n0 + 1))
        hflat = h.ravel()[order]
        for i in range(1, n0):
            if not (P["UNIT_AREA_M2"][0] <= a0[i] <= P["UNIT_AREA_M2"][1]):
                continue
            h50 = float(np.median(hflat[bounds[i]:bounds[i + 1]]))
            unit_like[i] = P["UNIT_H_P50"] <= h50 <= P["H_MAX"]
    m = ((unit_like[lab0] & (lab0 > 0)) | ((m > 0) & (near > 0))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cen = cv2.connectedComponentsWithStats(m, 8)
    Hh, Ww = d.shape
    out, rejected = [], []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 60:
            continue
        sel = (lab == i) & (d > 0)
        a_m2 = float(pix_m2[sel].sum())
        if a_m2 < P["MIN_AREA_M2"]:
            continue
        hv = h[sel]
        x0, y0, bw, bh = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        touch = (y0 + bh >= Hh - 2) or (x0 + bw >= Ww - 2) or x0 <= 1
        h50 = float(np.median(hv))
        blob = dict(area=int(stats[i, cv2.CC_STAT_AREA]), area_m2=a_m2, h_p50=h50, h_p90=float(np.percentile(hv, 90)),
                    touch=bool(touch), uv=(float(cen[i][0]), float(cen[i][1])), bbox=(x0, y0, bw, bh), label=i,
                    points=np.stack([X[sel], Y[sel], d[sel]], 1), heights=hv)
        # head/torso: high AND bigger than a unit, or very high and entering from the frame border. A unit lifted to
        # the camera at the bottom edge is high and touches the border too, so "touch" alone is not enough.
        if (h50 > P["HEAD_H_P50"] and a_m2 > P["HEAD_AREA_M2"]) or (h50 > P["HEAD_H_TOUCH"] and touch):
            blob["reject"] = "head"; rejected.append(blob); continue
        blob["xyz"] = np.median(blob["points"], 0)
        out.append(blob)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) > P["TEX_GRAD"]
    return out, dict(mask=m, lab=lab, rejected=rejected, skin=skin, edges=gm)


def texture_at(edges: np.ndarray, xyz: np.ndarray, K: np.ndarray, P: dict) -> float:
    """Share of strong-gradient pixels in a TEX_R metric disc around a 3D point (see LAUNDRY_V1)."""
    u, v = _uv_of(xyz, K)
    r = max(4, int(P["TEX_R"] * K[0, 0] / max(float(xyz[2]), 0.2)))
    H, W = edges.shape
    x0, x1, y0, y1 = max(0, int(u) - r), min(W, int(u) + r + 1), max(0, int(v) - r), min(H, int(v) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (xx - u) ** 2 + (yy - v) ** 2 <= r * r
    return float(edges[y0:y1, x0:x1][disc].mean()) if disc.any() else 0.0


def _trimmed_centre(Pts: np.ndarray, start: np.ndarray, R: float, iters: int = 3):
    c = start.copy()
    keep = None
    for _ in range(iters):
        keep = np.linalg.norm(Pts - c, axis=1) < R
        if keep.sum() == 0:
            return None, keep
        c = np.median(Pts[keep], 0)
    return c, keep


def _is_seedable(b: dict, P: dict) -> bool:
    return (P["SEED_AREA_M2"][0] <= b["area_m2"] <= P["SEED_AREA_M2"][1]
            and P["SEED_H_P50"][0] <= b["h_p50"] <= P["SEED_H_P50"][1])


def associate_laundry(tracks: dict, dets: list[dict], P: dict, K: np.ndarray, nrm: np.ndarray, pd: float,
                      edges: np.ndarray | None = None) -> dict:
    """Both units move. Each live track predicts xyz + damped velocity; it claims the points of every blob within
    GATE of the prediction and takes a trimmed median (R_BODY) seeded at the prediction. If both tracks claim the same
    points (units close together, or one blob = unit + shirt + other unit), the points are split by nearest predicted
    centre first. A track with no claim goes invalid -- a gap, never a silent jump to the other hand.

    Cold start / re-acquire (track lost > LOST frames): a compact seedable blob, chosen by the left/right image prior
    (operator's left = smaller u) and never within MIN_SEP of the other live track."""
    res = {}
    live = [s for s in ("left", "right") if s in tracks and tracks[s]["miss"] <= P["LOST"]]
    pred = {s: tracks[s]["xyz"] + tracks[s]["vel"] * min(tracks[s]["miss"] + 1, 3) for s in live}
    claims = {}
    for s in live:
        gate = P["GATE"] + 0.04 * tracks[s]["miss"]
        idx = [j for j, b in enumerate(dets)
               if np.min(np.linalg.norm(b["points"][::7] - pred[s], axis=1)) < gate] if dets else []
        if idx:
            claims[s] = idx
    pts = {}
    for s, idx in claims.items():
        pts[s] = np.concatenate([dets[j]["points"] for j in idx])
    if len(pts) == 2 and set(claims["left"]) & set(claims["right"]):
        # shared blob(s): split the shared points between the two predictions
        shared = sorted(set(claims["left"]) & set(claims["right"]))
        Sp = np.concatenate([dets[j]["points"] for j in shared])
        dl = np.linalg.norm(Sp - pred["left"], axis=1); dr = np.linalg.norm(Sp - pred["right"], axis=1)
        own = {s: [dets[j]["points"] for j in claims[s] if j not in shared] for s in ("left", "right")}
        own["left"].append(Sp[dl <= dr]); own["right"].append(Sp[dr < dl])
        pts = {s: np.concatenate(own[s]) for s in ("left", "right")}
    for s, Pp in pts.items():
        if len(Pp) < P["MIN_PTS"]:
            continue
        c, keep = _trimmed_centre(Pp, pred[s], P["R_BODY"])
        if (c is None or keep.sum() < P["MIN_PTS"] or np.linalg.norm(c - pred[s]) > P["MAX_SHIFT"] + 0.04 * tracks[s]["miss"]
                or float(c @ nrm + pd) < P["H_TRACK_MIN"]
                or (edges is not None and texture_at(edges, c, K, P) < P["TEX_KEEP"])):
            continue
        j_best = max(claims[s], key=lambda j: dets[j]["area"])
        res[s] = dict(xyz=c, n_points=int(keep.sum()), blob=j_best, area=int(keep.sum()), seeded=False, uv=_uv_of(c, K))
    # anti-swap: a track may not walk onto the spot where the OTHER unit was last seen while that one is only
    # briefly missing -- that is the classic hand-over of a label along a shirt connecting both units
    for s in list(res):
        o = "right" if s == "left" else "left"
        if o in res or o not in live:
            continue
        do = float(np.linalg.norm(res[s]["xyz"] - tracks[o]["xyz"]))
        if do < P["SWAP_R"] and do < float(np.linalg.norm(res[s]["xyz"] - pred[s])):
            res.pop(s)
    if len(res) == 2 and np.linalg.norm(res["left"]["xyz"] - res["right"]["xyz"]) < P["MIN_SEP"]:
        # collapse: keep the side whose result is closer to its own prediction
        e = {s: float(np.linalg.norm(res[s]["xyz"] - pred[s])) for s in res}
        res.pop(max(e, key=e.get))
    # seeding
    for s in ("left", "right"):
        if s in res or s in live:
            continue
        other = "right" if s == "left" else "left"
        oxyz = res[other]["xyz"] if other in res else (tracks[other]["xyz"] if other in live else None)
        ou = res[other]["uv"] if other in res else (tracks[other].get("uv") if other in live else None)
        cands = []
        for j, b in enumerate(dets):
            if not _is_seedable(b, P):
                continue
            if oxyz is not None and np.linalg.norm(b["xyz"] - oxyz) < 0.15:
                continue
            if ou is not None and ((s == "left" and b["uv"][0] >= ou[0]) or (s == "right" and b["uv"][0] <= ou[0])):
                continue
            if ou is None and not (b["uv"][0] < P["COLD_HALF"] * 2 * K[0, 2] if s == "left"
                                   else b["uv"][0] > (1 - P["COLD_HALF"]) * 2 * K[0, 2]):
                continue                      # nothing to order against: fall back to the image-half prior
            if other in live and other not in res and np.linalg.norm(b["xyz"] - tracks[other]["xyz"]) < P["DEFER_R"]:
                # the other unit vanished only a few frames ago right here: this blob is far more likely to be IT
                # coming back than this side's unit appearing. Do not seed; let the live track re-acquire it.
                # (the swaps on 163800 f548 and 163312 f614 were exactly this)
                continue
            c, keep = _trimmed_centre(b["points"], b["xyz"], P["R_BODY"])
            if c is None or keep.sum() < P["MIN_PTS"]:
                continue
            if edges is not None and texture_at(edges, c, K, P) < P["TEX_SEED"]:
                continue                      # a smooth dark sheet: lifted shirt, not a unit
            cands.append((j, c, int(keep.sum())))
        if not cands:
            continue
        # with both tracks cold and only one unit-looking blob, do not guess which hand it is
        if oxyz is None and other not in res and len(cands) < 2:
            continue
        j, c, nk = (min(cands, key=lambda t: dets[t[0]]["uv"][0]) if s == "left"
                    else max(cands, key=lambda t: dets[t[0]]["uv"][0]))
        res[s] = dict(xyz=c, n_points=nk, blob=j, area=nk, seeded=True, uv=_uv_of(c, K))
    return res


def _uv_of(xyz: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    return (float(K[0, 0] * xyz[0] / xyz[2] + K[0, 2]), float(K[1, 1] * xyz[1] / xyz[2] + K[1, 2]))


def track_episode_laundry(ep_path: Path, stream: str | None, debug_frames: int, debug_dir: Path | None = None,
                          P: dict = LAUNDRY_V1) -> dict:
    ep = RgbdEpisode.load(ep_path, stream=stream)
    K = ep.intrinsics.K
    nrm, pd, plane = fit_table_plane(ep)
    scene = laundry_scene_model(ep, nrm, pd, P)
    plane["roi_share"] = scene["roi_share"]
    n = ep.n_frames
    sides = ("left", "right")
    xyz = {s: np.full((n, 3), np.nan) for s in sides}
    axes = {s: np.full((n, 3, 3), np.nan) for s in sides}
    area = {s: np.zeros(n, int) for s in sides}
    valid = {s: np.zeros(n, bool) for s in sides}
    height = {s: np.full(n, np.nan) for s in sides}
    seeded = {s: np.zeros(n, bool) for s in sides}
    tracks: dict = {}
    switches = {s: 0 for s in sides}
    reseeds = {s: 0 for s in sides}
    repaired = {s: 0 for s in sides}
    n_dets, n_head = [], []
    for f in ep.iter_frames(0, n, 1):
        preds = [tracks[s]["xyz"] + tracks[s]["vel"] * min(tracks[s]["miss"] + 1, 3)
                 for s in sides if s in tracks and tracks[s]["miss"] <= P["LOST"]]
        dets, aux = detect_bodies_laundry(f, K, nrm, pd, scene, P, preds)
        n_dets.append(len(dets)); n_head.append(len(aux["rejected"]))
        res = associate_laundry(tracks, dets, P, K, nrm, pd, aux["edges"])
        for s in sides:
            if s in res:
                r = res[s]
                t = tracks.get(s)
                if t is not None and t["miss"] <= P["LOST"] and not r["seeded"]:
                    v = (r["xyz"] - t["xyz"]) / (t["miss"] + 1)
                    vel = P["VEL_DAMP"] * v + (1 - P["VEL_DAMP"]) * t["vel"]
                else:
                    vel = np.zeros(3)
                    if t is not None:
                        reseeds[s] += 1
                tracks[s] = dict(xyz=r["xyz"], vel=vel, miss=0, uv=_uv_of(r["xyz"], K))
                seeded[s][f.index] = r["seeded"]
                xyz[s][f.index] = r["xyz"]
                area[s][f.index] = r["area"]
                valid[s][f.index] = True
                height[s][f.index] = float(r["xyz"] @ nrm + pd)
                b = dets[r["blob"]]
                repaired[s] += int(r["n_points"] < 0.9 * len(b["points"]))
            elif s in tracks:
                tracks[s]["miss"] += 1
        # identity-switch proxy 1: the two tracks swap image order (operator's left must stay at smaller u)
        if valid["left"][f.index] and valid["right"][f.index]:
            ul = _uv_of(xyz["left"][f.index], K)[0]; ur = _uv_of(xyz["right"][f.index], K)[0]
            tracks.setdefault("_order", [])
            tracks["_order"].append(ul < ur)
    order = np.array(tracks.get("_order", []), bool)
    # sustained (>= 10 frames) inversions only: hands may briefly cross in depth without being swapped
    inv_events = 0
    if len(order):
        run = 0
        for ok in order:
            run = 0 if ok else run + 1
            if run == 10:
                inv_events += 1
    for s in sides:
        v = valid[s]
        if v.sum() > 1:
            st = np.linalg.norm(np.diff(xyz[s][v], axis=0), axis=1)
            switches[s] = int((st > P["GATE"]).sum())
    tr = dict(episode=ep_path.name, n_frames=n, plane=plane, n_dets_mean=float(np.mean(n_dets)),
              plane_normal=nrm, plane_d=pd, xyz=xyz, axes=axes, area=area, valid=valid, switches=switches,
              repaired=repaired, t_ns=ep.t_ns, profile="laundry_v1", reseeds=reseeds,
              order_inversion_events=inv_events, order_inverted_frames=int((~order).sum()) if len(order) else 0,
              both_valid_frames=int(len(order)), head_reject_frames=int(np.sum(np.array(n_head) > 0)),
              height_above_table=height, seeded=seeded)
    if debug_frames and debug_dir is not None:
        write_debug_laundry(ep, tr, scene, K, nrm, pd, P, debug_frames, debug_dir)
    return tr


def write_debug_laundry(ep, tr, scene, K, nrm, pd, P, n_even: int, out_dir: Path) -> list[Path]:
    """Evenly spaced frames + the worst events (largest jumps, longest gaps). Overlay: table ROI (green outline),
    candidate mask (magenta tint), head-rejected blobs (red), tracks (left=orange, right=blue) with their 3D trail."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = tr["n_frames"]
    picks = {int(i): "even" for i in np.linspace(0, n - 1, n_even + 2)[1:-1].astype(int)}
    for s in ("left", "right"):
        v = tr["valid"][s]; idx = np.nonzero(v)[0]
        if len(idx) > 2:
            st = np.linalg.norm(np.diff(tr["xyz"][s][idx], axis=0), axis=1)
            for k in np.argsort(st)[::-1][:2]:
                if st[k] > 0.05:
                    picks.setdefault(int(idx[k + 1]), f"jump_{s}_{st[k]*1e3:.0f}mm")
        miss = np.nonzero(~v)[0]
        if len(miss):
            picks.setdefault(int(miss[len(miss) // 2]), f"miss_{s}")
    cnts, _ = cv2.findContours(scene["roi"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    written = []
    tag = f"{ep.path.parent.name}_{ep.path.name}"
    for fi in sorted(picks):
        f = next(ep.iter_frames(fi, fi + 1))
        preds = [tr["xyz"][s][fi - 1] for s in ("left", "right") if fi > 0 and tr["valid"][s][fi - 1]]
        dets, aux = detect_bodies_laundry(f, K, nrm, pd, scene, P, preds)
        vis = f.rgb.copy()
        tint = vis.copy(); tint[aux["mask"] > 0] = (255, 0, 255); tint[aux["skin"]] = (0, 255, 255)
        vis = cv2.addWeighted(vis, 0.6, tint, 0.4, 0)
        for b in aux["rejected"]:
            x, y, w, h = b["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis, f"head? h{b['h_p50']:.2f}", (x, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.drawContours(vis, cnts, -1, (0, 220, 0), 1)
        for s, col in (("left", (0, 165, 255)), ("right", (255, 140, 0))):
            P3 = tr["xyz"][s]; v = tr["valid"][s]
            trail = [(_uv_of(P3[k], K)) for k in range(max(0, fi - 30), fi + 1) if v[k]]
            for a, b in zip(trail[:-1], trail[1:]):
                cv2.line(vis, tuple(int(t) for t in a), tuple(int(t) for t in b), col, 1)
            if v[fi]:
                u, w_ = _uv_of(P3[fi], K)
                cv2.circle(vis, (int(u), int(w_)), 9, col, 2)
                cv2.drawMarker(vis, (int(u), int(w_)), col, cv2.MARKER_CROSS, 14, 2)
                hh = tr["height_above_table"][s][fi]
                cv2.putText(vis, f"{s} z{P3[fi][2]:.3f} h{hh*100:.0f}cm", (int(u) + 10, int(w_) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            else:
                cv2.putText(vis, f"{s}: LOST", (10 if s == "left" else 700, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.putText(vis, f"{tag} f{fi} [{picks[fi]}] laundry_v1", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(vis, f"{tag} f{fi} [{picks[fi]}] laundry_v1", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        p = out_dir / f"{tag}_f{fi:05d}_{picks[fi]}.jpg"
        cv2.imwrite(str(p), vis)
        written.append(p)
    return written


# ----------------------------------------------------------------------------------------- QA
def qa(tr: dict) -> dict:
    out = {"episode": tr["episode"], "n_frames": tr["n_frames"], "plane": tr["plane"],
           "detections_per_frame": tr["n_dets_mean"], "sides": {}}
    if tr.get("profile"):
        # laundry_v1 identity evidence: there is no idle unit, so switches are judged by image order + re-seeds
        out.update(profile=tr["profile"], order_inversion_events=tr["order_inversion_events"],
                   order_inverted_frames=tr["order_inverted_frames"], both_valid_frames=tr["both_valid_frames"],
                   both_valid_share=tr["both_valid_frames"] / max(tr["n_frames"], 1),
                   head_reject_frames=tr["head_reject_frames"], reseeds=tr["reseeds"])
    for side in ("left", "right"):
        P, v = tr["xyz"][side], tr["valid"][side]
        r = dict(coverage=float(v.mean()), identity_switches=int(tr["switches"][side]),
                 repaired_frames=int(tr["repaired"][side]),
                 area_px_p50=float(np.median(tr["area"][side][v])) if v.any() else float("nan"))
        gaps = []
        i = 0
        while i < len(v):
            if not v[i]:
                j = i
                while j < len(v) and not v[j]:
                    j += 1
                gaps.append(j - i); i = j
            else:
                i += 1
        r["max_gap_frames"] = int(max(gaps, default=0))
        # THE GATE, and it has to FIND its still window rather than assume one. Std over the whole episode is workspace
        # extent, not noise. The head of the episode is not a safe substitute either: the collector's stillness gate is
        # driven by the IMUs, this profile has `imus: []`, so the gate never ran and nothing guarantees the operator
        # held still before REC. So: slide a window over the track and take the quietest one actually present. Where
        # the unit truly rested, that window's spread is the detector's own noise floor; where it never rested, the
        # number is honestly labelled as an upper bound rather than quietly reported as noise.
        # THE GATE, and it has to FIND its quiet window rather than assume one. Std over the whole episode is
        # workspace extent, not noise. The head of the episode is not a safe substitute either: the collector's
        # stillness gate is driven by the IMUs, this profile has `imus: []`, so the gate never ran and nothing
        # guarantees the operator held still before REC.
        #
        # Ranking by position variance alone would happily pick a short accidental plateau in the middle of a reach,
        # so a candidate is ranked on how slow it is (p95 speed first, median second) and must be gap-free for its
        # whole length -- a window containing a tracking gap says nothing about the detector's noise.
        w = STILL_FRAMES
        cands = []
        for a0 in range(0, len(P) - w + 1):
            if not v[a0:a0 + w].all():
                continue
            Q0 = P[a0:a0 + w]
            sp = np.linalg.norm(np.diff(Q0, axis=0), axis=1) * 1e3          # mm per frame
            ac = np.abs(np.diff(sp)) if len(sp) > 1 else np.zeros(1)
            cands.append((float(np.percentile(sp, 95)), float(np.median(sp)), float(np.percentile(ac, 95)), a0, Q0))
        if cands:
            p95s, med, acc, a0, Q0 = min(cands)
            path = float(np.linalg.norm(np.diff(Q0, axis=0), axis=1).sum()) * 1e3
            r["quiet"] = dict(start=int(a0), end=int(a0 + w), duration_s=w / 30.0,
                              xyz_std_mm=[float(Q0[:, k].std()) * 1e3 for k in range(3)],
                              std_norm_mm=float(np.linalg.norm(Q0.std(0))) * 1e3,
                              speed_mm_p50=med, speed_mm_p95=p95s, accel_mm_p95=acc, path_mm=path,
                              # a window the unit never actually rested in bounds the noise from above, it does not
                              # measure it, and the two must not be averaged together
                              is_rest=bool(path < 15.0), n_candidates=len(cands))
            r["still_xyz_std_norm_mm"] = r["quiet"]["std_norm_mm"]
            r["still_window_is_rest"] = r["quiet"]["is_rest"]
            r["still_window_start"] = int(a0)
            r["still_window_path_mm"] = path
        # single-frame jumps: a body at 30 Hz does not move 50 mm between frames, so these are the detector's own
        # failures, and their RATE is what a downstream filter would have to clean up
        if v.sum() > 5:
            st = np.linalg.norm(np.diff(P[v], axis=0), axis=1) * 1e3
            r["jump_over_50mm"] = int((st > 50).sum())
            r["jump_over_50mm_rate"] = float((st > 50).mean())
            r["jump_over_100mm"] = int((st > 100).sum())
        if v.sum() > 5:
            Q = P[v]
            step = np.linalg.norm(np.diff(P[v], axis=0), axis=1)
            r.update(xyz_std_mm=[float(Q[:, k].std()) * 1e3 for k in range(3)],
                     xyz_std_norm_mm=float(np.linalg.norm(Q.std(0))) * 1e3,
                     path_mm=float(step.sum()) * 1e3,
                     step_mm_p50=float(np.percentile(step, 50)) * 1e3,
                     step_mm_p95=float(np.percentile(step, 95)) * 1e3,
                     step_mm_p99=float(np.percentile(step, 99)) * 1e3,
                     step_mm_max=float(step.max()) * 1e3,
                     depth_range_m=[float(Q[:, 2].min()), float(Q[:, 2].max())])
            # PCA axis stability: angle between each frame's first principal axis and the episode's median one
            A = tr["axes"][side][v]
            fin = np.all(np.isfinite(A.reshape(len(A), 9)), 1)
            if fin.sum() > 5:
                a0 = A[fin][:, 0, :]
                a0 = a0 * np.sign(a0 @ np.median(a0, 0))[:, None]     # PCA axes have no sign; fix it before averaging
                ref = np.median(a0, 0); ref /= np.linalg.norm(ref)
                ang = np.degrees(np.arccos(np.clip(a0 @ ref, -1, 1)))
                r["pca_axis1_dev_deg_p50"] = float(np.percentile(ang, 50))
                r["pca_axis1_dev_deg_p95"] = float(np.percentile(ang, 95))
        if tr.get("profile"):
            # >50 mm steps between ADJACENT frames only (the column above also counts steps across tracking gaps,
            # which for two always-moving hands mostly measures how far the unit went while it was not seen)
            adj = v[1:] & v[:-1]
            if adj.any():
                st1 = np.linalg.norm(P[1:][adj] - P[:-1][adj], axis=1) * 1e3
                r["jump_over_50mm_adjacent"] = int((st1 > 50).sum())
                r["jump_over_50mm_adjacent_rate"] = float((st1 > 50).mean())
            hh = tr["height_above_table"][side][v]
            if len(hh):
                r["height_above_table_cm_p05_p50_p95"] = [float(np.percentile(hh, q)) * 100 for q in (5, 50, 95)]
            r["reseeds"] = int(tr["reseeds"][side])
        out["sides"][side] = r
    # the still unit is the noise floor: whichever side moved least is the one that was resting on the table
    mv = {s: out["sides"][s].get("path_mm", float("inf")) for s in ("left", "right")}
    out["idle_side"] = min(mv, key=mv.get) if np.isfinite(min(mv.values())) else None
    return out


def print_report(rows: list[dict]) -> None:
    print("\n" + "=" * 112)
    print("HANDUMI BODY TRACKING  --  translation first; the idle unit's xyz std is the detector's noise floor")
    print("=" * 112)
    print(f"\n{'episode':>14s} {'plane inl':>10s} {'plane rms':>10s} {'blobs/frm':>10s}")
    for r in rows:
        p = r["plane"]
        print(f"{r['episode'][-11:]:>14s} {p['inlier_share']*100:9.1f}% {p['rms_mm']:9.2f}mm {r['detections_per_frame']:10.1f}")

    print(f"\n{'episode':>14s} {'side':>6s} {'cover':>7s} {'switch':>7s} {'maxgap':>7s} {'area px':>9s} "
          f"{'xyz std mm (x,y,z)':>22s} {'|std|':>8s} {'path':>9s} {'STILL |std|':>11s}")
    for r in rows:
        for side, s in r["sides"].items():
            if "xyz_std_mm" not in s:
                print(f"{r['episode'][-11:]:>14s} {side:>6s} {s['coverage']*100:6.1f}%  -- not enough valid frames --")
                continue
            tag = " (idle)" if side == r.get("idle_side") else ""
            sd = s["xyz_std_mm"]
            print(f"{r['episode'][-11:]:>14s} {side:>6s} {s['coverage']*100:6.1f}% {s['identity_switches']:7d} "
                  f"{s['max_gap_frames']:7d} {s['area_px_p50']:9.0f} "
                  f"{sd[0]:6.1f},{sd[1]:6.1f},{sd[2]:6.1f} {s['xyz_std_norm_mm']:7.1f} {s['path_mm']:8.0f} "
                  f"{s.get('still_xyz_std_norm_mm', float('nan')):9.2f}{tag}")

    print(f"\n{'episode':>14s} {'side':>6s} {'step p50':>9s} {'p95':>8s} {'p99':>8s} {'max':>8s} "
          f"{'depth range m':>16s} {'PCA ax1 dev p50/p95':>22s}")
    for r in rows:
        for side, s in r["sides"].items():
            if "step_mm_p50" not in s:
                continue
            pca = (f"{s.get('pca_axis1_dev_deg_p50', float('nan')):8.1f} /{s.get('pca_axis1_dev_deg_p95', float('nan')):7.1f}"
                   if "pca_axis1_dev_deg_p50" in s else "n/a")
            print(f"{r['episode'][-11:]:>14s} {side:>6s} {s['step_mm_p50']:8.1f} {s['step_mm_p95']:7.1f} "
                  f"{s['step_mm_p99']:7.1f} {s['step_mm_max']:7.1f} "
                  f"{s['depth_range_m'][0]:7.3f}..{s['depth_range_m'][1]:6.3f} {pca:>22s}")

    rest = [(r["episode"], side, s) for r in rows for side, s in r["sides"].items() if s.get("still_window_is_rest")]
    if rest:
        print("\n  quietest windows that were GENUINELY at rest (window path < 15 mm) -- these are the noise floor")
        for epn, side, s in rest:
            q = s["quiet"]
            print(f"    {epn:>18s} {side:>6s}  frames {q['start']:4d}-{q['end']:<4d} ({q['duration_s']:.1f}s)"
                  f"  speed p50/p95 {q['speed_mm_p50']:5.2f}/{q['speed_mm_p95']:5.2f} mm  accel p95 {q['accel_mm_p95']:5.2f}"
                  f"  std {q['xyz_std_mm'][0]:5.2f},{q['xyz_std_mm'][1]:5.2f},{q['xyz_std_mm'][2]:5.2f} -> {q['std_norm_mm']:5.2f} mm")
    other = [(r["episode"], side, s) for r in rows for side, s in r["sides"].items()
             if "still_xyz_std_norm_mm" in s and not s.get("still_window_is_rest")]
    if other:
        print("\n  tracks with NO genuine rest window -- the unit kept moving, so this is an upper bound, not noise")
        for epn, side, s in other:
            q = s["quiet"]
            print(f"    {epn:>18s} {side:>6s}  frames {q['start']:4d}-{q['end']:<4d} ({q['duration_s']:.1f}s)"
                  f"  speed p50/p95 {q['speed_mm_p50']:5.2f}/{q['speed_mm_p95']:5.2f} mm  accel p95 {q['accel_mm_p95']:5.2f}"
                  f"  std {q['xyz_std_mm'][0]:5.2f},{q['xyz_std_mm'][1]:5.2f},{q['xyz_std_mm'][2]:5.2f} -> {q['std_norm_mm']:5.2f} mm")
    print(f"\n{'episode':>14s} {'side':>6s} {'>50mm jumps':>12s} {'rate':>8s} {'>100mm':>8s}")
    for r in rows:
        for side, s in r["sides"].items():
            if "jump_over_50mm" in s:
                print(f"{r['episode'][-11:]:>14s} {side:>6s} {s['jump_over_50mm']:12d} {s['jump_over_50mm_rate']*100:7.2f}% {s['jump_over_100mm']:8d}")
    lr = [r for r in rows if r.get("profile")]
    if lr:
        print(f"\n{'episode':>14s} {'side':>6s} {'>50mm adjacent-frame jumps':>27s} {'rate':>8s} {'height above table cm p5/p50/p95':>34s}")
        for r in lr:
            for side, sd in r["sides"].items():
                if "jump_over_50mm_adjacent" in sd:
                    hq = sd.get("height_above_table_cm_p05_p50_p95", [float("nan")] * 3)
                    print(f"{r['episode'][-11:]:>14s} {side:>6s} {sd['jump_over_50mm_adjacent']:27d} "
                          f"{sd['jump_over_50mm_adjacent_rate']*100:7.2f}% {hq[0]:12.1f} /{hq[1]:5.1f} /{hq[2]:5.1f}")
        print(f"\n{'episode':>14s} {'profile':>11s} {'both valid':>11s} {'L/R order inv (events, frames)':>32s} "
              f"{'reseed L/R':>11s} {'head-rej frm':>13s}")
        for r in lr:
            print(f"{r['episode'][-11:]:>14s} {r['profile']:>11s} {r['both_valid_share']*100:10.1f}% "
                  f"{r['order_inversion_events']:20d}, {r['order_inverted_frames']:9d} "
                  f"{r['reseeds']['left']:5d}/{r['reseeds']['right']:<5d} {r['head_reject_frames']:13d}")
    st = [s["still_xyz_std_norm_mm"] for _e, _s, s in rest]
    cov = [s["coverage"] for r in rows for s in r["sides"].values()]
    sw = [s["identity_switches"] for r in rows for s in r["sides"].values()]
    if st:
        print(f"\nGATE  still-window xyz std   {min(st):.2f} .. {max(st):.2f} mm   over {len(st)} tracks")
        print(f"      coverage               {min(cov)*100:.1f} .. {max(cov)*100:.1f} %")
        print(f"      identity switches      {sum(sw)} total")
        print("      A few mm here means the translation detector is alive. The whole-episode std above is workspace")
        print("      extent, not noise: a unit that legitimately crossed the table reads centimetres and means nothing.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", type=Path)
    g.add_argument("--episode", type=Path)
    ap.add_argument("--stream", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--debug-frames", type=int, default=6, help="annotated frames written to /tmp/body_track/")
    ap.add_argument("--save-npz", action="store_true", help="write EPISODE/derived/body_track/track.npz")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--profile", choices=("default", "laundry_v1"), default="default",
                    help="laundry_v1 = Hfold overhead setup: table ROI + height-above-plane + two moving units")
    ap.add_argument("--debug-dir", type=Path, default=None,
                    help="laundry_v1: where annotated frames go (default /tmp/body_track/); evenly spaced + worst events")
    a = ap.parse_args()

    eps = [a.episode] if a.episode else sorted(p for p in a.session.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if a.limit:
        eps = eps[:a.limit]
    rows = []
    for p in eps:
        try:
            if a.profile == "laundry_v1":
                tr = track_episode_laundry(p, a.stream, a.debug_frames, a.debug_dir or Path("/tmp/body_track"))
            else:
                tr = track_episode(p, a.stream, a.debug_frames)
            rows.append(qa(tr))
            if a.save_npz:
                d = p / "derived" / "body_track"; d.mkdir(parents=True, exist_ok=True)
                np.savez(d / "track.npz", t_ns=tr["t_ns"],
                         xyz_left=tr["xyz"]["left"], xyz_right=tr["xyz"]["right"],
                         valid_left=tr["valid"]["left"], valid_right=tr["valid"]["right"],
                         axes_left=tr["axes"]["left"], axes_right=tr["axes"]["right"],
                         # additive keys only; existing readers ignore them
                         profile=np.array(tr.get("profile", "default")),
                         plane_normal=tr["plane_normal"], plane_d=np.array(tr["plane_d"]),
                         **({"height_left": tr["height_above_table"]["left"],
                             "height_right": tr["height_above_table"]["right"]} if tr.get("profile") else {}))
            print(f"  tracked {p.name}", flush=True)
        except Exception as exc:
            print(f"  {p.name}: FAILED -- {exc}", flush=True)
    if rows:
        print_report(rows)
    if a.json and rows:
        a.json.write_text(json.dumps(rows, indent=1, default=float)); print(f"\nrecord -> {a.json}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
