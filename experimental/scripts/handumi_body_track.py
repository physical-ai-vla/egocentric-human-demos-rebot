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


# ----------------------------------------------------------------------------------------- QA
def qa(tr: dict) -> dict:
    out = {"episode": tr["episode"], "n_frames": tr["n_frames"], "plane": tr["plane"],
           "detections_per_frame": tr["n_dets_mean"], "sides": {}}
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
    a = ap.parse_args()

    eps = [a.episode] if a.episode else sorted(p for p in a.session.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if a.limit:
        eps = eps[:a.limit]
    rows = []
    for p in eps:
        try:
            tr = track_episode(p, a.stream, a.debug_frames)
            rows.append(qa(tr))
            if a.save_npz:
                d = p / "derived" / "body_track"; d.mkdir(parents=True, exist_ok=True)
                np.savez(d / "track.npz", t_ns=tr["t_ns"],
                         xyz_left=tr["xyz"]["left"], xyz_right=tr["xyz"]["right"],
                         valid_left=tr["valid"]["left"], valid_right=tr["valid"]["right"],
                         axes_left=tr["axes"]["left"], axes_right=tr["axes"]["right"])
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
