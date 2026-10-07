#!/usr/bin/env python3
"""[2026-09-19] The held cube's GEOMETRIC centre, from its visible faces and its known size.

The centroid of a cube's visible depth points is not the cube's centre, and the error is not noise: the cube is
held by the gripper, so its attitude follows the body's, and the visible-centroid offset therefore varies
SYSTEMATICALLY with attitude -- aliasing directly with the R(t) @ t_body_tcp term the lever-arm fit is trying to
estimate. Measured on CALIB_bodytcp_20260919_123133, the body-frame offset's z component walks -25.3 -> -0.4 ->
+2.0 -> +44.5 mm across attitude bins that should all report the same constant.

The fix uses geometry instead of mass, and uses no orientation estimate at all, so it cannot alias with one:

    segment the blob's points into planar faces (RANSAC, up to 3)
    each face gives one linear constraint   n_i . c = -d_i - HALF_EDGE
    solve the stacked constraints for c

With two or three faces visible the centre is pinned in as many directions; with one it is pinned only along
that face's normal, and the in-plane position falls back to the face's own centroid, which is reported so a
consumer can tell the two cases apart.

    .venv/bin/python scripts/cube_center_planes.py --episode <ep> --side right
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handumi_collector.pose.rgbd_io import RgbdEpisode                              # noqa: E402
from scripts.handumi_body_track import fit_table_plane, BAND, ABOVE_PLANE           # noqa: E402
from scripts.provenance import stamp                                                # noqa: E402
from scripts.cube_tcp_detect import (SAT_MIN, VAL_MIN, MIN_AREA, NEAR_BODY_M,       # noqa: E402
                                     MIN_DEPTH_PTS, MIN_MID_EXTENT_M, MAX_MAJOR_EXTENT_M,
                                     grip_plateau)

HALF_EDGE = 0.020          # 40 mm cube
PLANE_TOL = 0.004          # inlier band; the depth noise floor on this sensor is ~2 mm rms on a flat surface
MIN_FACE_PTS = 120
MAX_FACES = 3


def fit_planes(P: np.ndarray, rng) -> list[tuple[np.ndarray, float, int]]:
    """RANSAC planes, largest first, removing inliers as it goes. Normals oriented TOWARDS the camera."""
    out, rest = [], P
    for _ in range(MAX_FACES):
        if len(rest) < MIN_FACE_PTS:
            break
        best = (0, None, None)
        for _ in range(120):
            i = rng.choice(len(rest), 3, replace=False)
            a, b, c = rest[i]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n = n / nn
            d = -n @ a
            k = int((np.abs(rest @ n + d) < PLANE_TOL).sum())
            if k > best[0]:
                best = (k, n, d)
        k, n, d = best
        if n is None or k < MIN_FACE_PTS:
            break
        m = np.abs(rest @ n + d) < PLANE_TOL
        Q = rest[m]
        c0 = Q.mean(0)
        _, _, Vt = np.linalg.svd(Q - c0, full_matrices=False)
        n = Vt[-1]
        d = -n @ c0
        if n @ c0 > 0:                      # camera looks down +z; a visible face's normal must point back at it
            n, d = -n, -d
        out.append((n, float(d), int(m.sum())))
        rest = rest[~m]
    return out


def centre_from_faces(faces, fallback: np.ndarray) -> tuple[np.ndarray, int]:
    """Stack n_i . c = -d_i - HALF_EDGE. Under-determined directions are filled from the blob centroid."""
    if not faces:
        return fallback, 0
    A = np.stack([f[0] for f in faces])
    b = np.array([-f[1] - HALF_EDGE for f in faces])
    # least squares with the fallback as a prior on the directions the faces do not constrain
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    c = fallback.copy()
    keep = S > 1e-6
    if keep.any():
        y = (U.T @ (b - A @ fallback))[:len(S)]
        c = c + Vt[:len(S)][keep].T @ (y[keep] / S[keep])
    return c, len(faces)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--side", choices=["left", "right"], required=True)
    a = ap.parse_args()
    npz = a.episode / "derived" / "body_track" / "track_export.npz"
    z = np.load(npz)
    s = a.side
    R_align = z["R_align"]
    t_ns = z["timestamp_ns"]
    body_ok = z[f"{s}_valid"]
    p_body_world = z[f"{s}_body_xyz"]
    win = grip_plateau(a.episode, s)
    in_hold = (t_ns >= win[0]) & (t_ns <= win[1])

    ep = RgbdEpisode.load(a.episode)
    K = ep.intrinsics.K
    nrm, pd, _ = fit_table_plane(ep)
    rng = np.random.default_rng(0)

    n = len(t_ns)
    cen = np.full((n, 3), np.nan)
    geo = np.full((n, 3), np.nan)
    nface = np.zeros(n, int)
    for f in ep.iter_frames(0, n, 1):
        i = f.index
        if not (in_hold[i] and body_ok[i]):
            continue
        pb_cam = R_align.T @ p_body_world[i]
        d = f.depth_m
        hsv = cv2.cvtColor(f.rgb, cv2.COLOR_BGR2HSV)
        ys, xs = np.nonzero((d > BAND[0]) & (d < BAND[1]))
        if len(xs) < 100:
            continue
        zz = d[ys, xs]
        P = np.stack([(xs - K[0, 2]) * zz / K[0, 0], (ys - K[1, 2]) * zz / K[1, 1], zz], 1)
        h = P @ nrm + pd
        col = ((h > ABOVE_PLANE[0]) & (h < ABOVE_PLANE[1])
               & (hsv[ys, xs, 1] > SAT_MIN) & (hsv[ys, xs, 2] > VAL_MIN))
        if col.sum() < 60:
            continue
        mask = np.zeros(d.shape, np.uint8)
        mask[ys[col], xs[col]] = 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        nl, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        best = None
        for k in range(1, nl):
            if stats[k, cv2.CC_STAT_AREA] < MIN_AREA:
                continue
            sel = lab[ys, xs] == k
            if sel.sum() < MIN_DEPTH_PTS:
                continue
            Q = P[sel]
            c = Q.mean(0)
            if np.linalg.norm(c - pb_cam) > NEAR_BODY_M:
                continue
            ext = 2 * np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov((Q - c).T)), 0))[::-1]
            if ext[1] < MIN_MID_EXTENT_M or ext[0] > MAX_MAJOR_EXTENT_M:
                continue
            if best is None or stats[k, cv2.CC_STAT_AREA] > best[0]:
                best = (stats[k, cv2.CC_STAT_AREA], Q)
        if best is None:
            continue
        Q = best[1]
        faces = fit_planes(Q, rng)
        c, k = centre_from_faces(faces, Q.mean(0))
        cen[i] = R_align @ Q.mean(0)
        geo[i] = R_align @ c
        nface[i] = k

    v = np.isfinite(geo).all(1)
    print(f"\n{a.episode.name}  side {s}")
    print(f"  frames with a geometric centre   {v.sum()}")
    print(f"  faces used   1: {(nface[v] == 1).sum()}   2: {(nface[v] == 2).sum()}   3: {(nface[v] == 3).sum()}")
    shift = np.linalg.norm(geo[v] - cen[v], axis=1) * 1e3
    print(f"  centroid -> geometric centre shift  p50 {np.percentile(shift, 50):.1f}  p95 {np.percentile(shift, 95):.1f} mm")

    out = dict(np.load(npz))
    out.update(stamp(**{f"{s}_cube_geo_version": "ransac_planes_half_edge_v1",
                        f"{s}_cube_half_edge_m": float(HALF_EDGE),
                        f"{s}_cube_plane_tol_m": float(PLANE_TOL)}))
    out[f"{s}_p_tcp_geo"] = geo
    out[f"{s}_p_tcp_geo_valid"] = v
    out[f"{s}_p_tcp_geo_faces"] = nface
    np.savez(npz, **out)
    print(f"  -> {s}_p_tcp_geo written into {npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
