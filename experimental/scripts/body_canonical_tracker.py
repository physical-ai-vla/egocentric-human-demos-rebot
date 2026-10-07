#!/usr/bin/env python3
"""[2026-09-19] Rigid body POSITION for the HandUMI: every frame registered to one canonical model, never integrated.

Why this exists: the tracked "body position" was the centroid of the camera-visible surface, and that centroid
slides across the object as the view changes. Measured on CALIB_bodytcp_20260919_123133 -- the body demonstrably
rotated 82 deg (ICP vs IMU r=0.969, 1.7 mm correspondence error) while the body->cube offset direction scattered
7.6 deg in the body frame against 7.1 deg in the world, i.e. applying the verified rotation made it no more
constant. A centroid is not a point on a rigid body, so no lever arm can be fitted through one.

TWO DECISIONS THAT DEFINE THIS FILE:

  1. Registration is to a FIXED canonical model, not to the previous frame. Frame-to-frame ICP accumulates
     drift -- propagating a point that way scored 13.2 mm in the reference bake-off, no better than the centroid
     it was meant to replace. Against a fixed model there is nothing to accumulate.

  2. Registration solves TRANSLATION ONLY. The rotation is already measured, and measured well; letting ICP
     re-solve it would let a 3-DOF answer absorb rotation error as position error and would reintroduce the
     drift that decision 1 removes. So the IMU supplies R and this supplies t, each from the sensor that is
     good at it.

THE CANONICAL FRAME IS THE IMU FRAME. Defining it that way makes `R_imu_body_rp` identity by construction --
the mounting rotation is absorbed into what "body frame" means rather than estimated. The lever arm this
enables is therefore expressed in IMU axes, and anything downstream must use the same convention.

BOOTSTRAP. Building the model needs positions and getting positions needs the model, so the centroid seeds the
first model, blurred by its own ~20 mm error, and each round of registration sharpens it. Rounds are reported
so the convergence is visible rather than assumed.

THE MODEL IS A VISIBILITY-FILTERED CORE, BALANCED BY ATTITUDE. Accumulating every point would bake the forearm,
the glove and the operator's skin into the "body" -- they are in the crop but they move relative to it. A voxel
kept only when it is seen repeatedly is, by construction, the part that is rigidly attached.

But "repeatedly" must be counted over ATTITUDES, not frames. Counted over frames, the first attempt built a
model out of whatever faced the camera at the take's dominant attitude: 194 of 245 frames sat within 15 deg of
the first, so the far side of the body never reached the threshold, and registration then failed exactly where
the body had turned -- registration residual correlated +0.828 with attitude, with 70% of frames under 4 mm
below 15 deg and 0% above 30 deg. Filtering to the well-registered frames left a 18.6 deg span and cond 34.8,
unobservable. So each frame is weighted by the inverse population of its attitude bin: a face seen only from
one side still earns its place in the model, and no side can crowd the others out.

    .venv/bin/python scripts/body_canonical_tracker.py --episode <ep> --side right
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handumi_collector.pose.rgbd_io import RgbdEpisode                              # noqa: E402
from scripts.handumi_body_track import fit_table_plane, BAND, ABOVE_PLANE, DARK_MAX  # noqa: E402
from scripts.provenance import stamp                                                 # noqa: E402

VOXEL_M = 0.005            # 5 mm: finer than the depth noise is wasted, coarser blurs the core
CROP_M = 0.12              # same crop the tracker uses, so the input is not quietly different
PER_FRAME_POINTS = 3000    # subsample kept in memory for the whole episode
# 0.05 chosen by sweep on CALIB_bodytcp_20260919_123133, judged on registration flatness across attitude rather
# than on median alone: 0.05 gives residual p50 3.31 / p95 5.44 mm with corr(residual, attitude) -0.371, against
# +0.828 before attitude balancing. Higher thresholds starve the model of the faces that only appear when the
# body turns (0.20 keeps 236 voxels and registers at 15.8 mm; 0.35 keeps none at all).
VISIBILITY = 0.05          # a voxel must reach this share of the ATTITUDE-BALANCED weight to be called rigid
ATT_BIN_DEG = 15.0         # attitude bin width for that balancing
ICP_ITERS = 20
TRIM = 0.80                # correspondence trim: occlusion changes which part of the model is seen


def voxel_key(P: np.ndarray) -> np.ndarray:
    return np.floor(P / VOXEL_M).astype(np.int64)


def register_translation(P: np.ndarray, tree: cKDTree, t0: np.ndarray) -> tuple[np.ndarray, float]:
    """Translation-only ICP of P onto the model behind `tree`, starting at t0."""
    t = t0.copy()
    err = np.inf
    for _ in range(ICP_ITERS):
        d, j = tree.query(P + t, k=1)
        keep = d <= max(np.quantile(d, TRIM), 1e-4)
        if keep.sum() < 50:
            break
        step = (tree.data[j[keep]] - (P[keep] + t)).mean(0)
        t = t + step
        err = float(d[keep].mean())
        if np.linalg.norm(step) < 1e-5:
            break
    return t, err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--visibility", type=float, default=VISIBILITY)
    ap.add_argument("--recollect", action="store_true", help="ignore the cached clouds and re-read the video")
    a = ap.parse_args()

    npz = a.episode / "derived" / "body_track" / "track_export.npz"
    z = np.load(npz)
    s = a.side
    R_align = z["R_align"]
    valid = z[f"{s}_valid"] & z[f"{s}_quat_valid"]
    cen_world = z[f"{s}_body_xyz"]
    R_wb = Rotation.from_quat(z[f"{s}_quat_world_imu"]).as_matrix()

    ep = RgbdEpisode.load(a.episode)
    K = ep.intrinsics.K
    nrm, pd, _ = fit_table_plane(ep)

    print(f"\n{a.episode.name}  side {s}")
    cache = a.episode / "derived" / "body_track" / f"clouds_{s}.npz"
    if cache.exists() and not a.recollect:
        cz = np.load(cache)
        frames = cz["frames"]
        off = cz["offsets"]
        flat = cz["points"]
        clouds = [flat[off[k]:off[k + 1]] for k in range(len(frames))]
        print(f"  {len(clouds)} frames from cache ({cache.name}); --recollect to rebuild")
        return _run(a, s, npz, z, clouds, np.asarray(frames), cen_world, R_wb, valid)
    print("  collecting body clouds ...", flush=True)
    clouds, frames = [], []
    for f in ep.iter_frames(0, len(valid), 1):
        i = f.index
        if not valid[i]:
            continue
        d = f.depth_m
        g = cv2.cvtColor(f.rgb, cv2.COLOR_BGR2GRAY)
        ys, xs = np.nonzero((d > BAND[0]) & (d < BAND[1]) & (g < DARK_MAX))
        if len(xs) < 300:
            continue
        zz = d[ys, xs]
        P = np.stack([(xs - K[0, 2]) * zz / K[0, 0], (ys - K[1, 2]) * zz / K[1, 1], zz], 1)
        h = P @ nrm + pd
        P = P[(h > ABOVE_PLANE[0]) & (h < ABOVE_PLANE[1])] @ R_align.T      # into the world frame
        P = P[np.linalg.norm(P - cen_world[i], axis=1) < CROP_M]
        if len(P) < 300:
            continue
        if len(P) > PER_FRAME_POINTS:
            P = P[np.random.default_rng(i).choice(len(P), PER_FRAME_POINTS, replace=False)]
        clouds.append(P)
        frames.append(i)
    frames = np.asarray(frames)
    print(f"  {len(clouds)} frames, {sum(len(c) for c in clouds)} points")
    off = np.cumsum([0] + [len(c) for c in clouds])
    np.savez(cache, frames=frames, offsets=off, points=np.concatenate(clouds))
    return _run(a, s, npz, z, clouds, frames, cen_world, R_wb, valid)


def _run(a, s, npz, z, clouds, frames, cen_world, R_wb, valid) -> int:

    pos = cen_world[frames].copy()                    # seed: today's centroid, ~20 mm wrong
    # attitude-balanced frame weights: bins with many frames contribute no more than bins with few
    att = np.degrees((Rotation.from_matrix(R_wb[frames]) * Rotation.from_matrix(R_wb[frames[0]]).inv()).magnitude())
    b = np.floor(att / ATT_BIN_DEG).astype(int)
    _, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
    weight = 1.0 / cnt[inv]
    weight = weight / weight.sum() * len(weight)
    print(f"  attitude bins ({ATT_BIN_DEG:.0f} deg): populations {cnt.tolist()}  -> weights "
          f"{np.round(sorted(set(weight)), 2).tolist()}")
    prev_pos = None
    for rnd in range(a.rounds):
        # --- build the canonical model: de-rotate each cloud into the IMU frame about the current position
        counts: dict[tuple, float] = {}
        sums: dict[tuple, np.ndarray] = {}
        for c, i, p, w in zip(clouds, frames, pos, weight):
            X = (c - p) @ R_wb[i]                     # world -> body(IMU) frame
            seen = set()
            for key, pt in zip(map(tuple, voxel_key(X)), X):
                if key not in seen:                   # one vote per frame per voxel, not one per point
                    counts[key] = counts.get(key, 0.0) + w
                    seen.add(key)
                sums[key] = sums.get(key, 0.0) + np.concatenate([pt, [1.0]])
        need = a.visibility * float(weight.sum())
        model = np.asarray([sums[k][:3] / sums[k][3] for k, n in counts.items() if n >= need])
        if len(model) < 100:
            print(f"  round {rnd}: canonical model has only {len(model)} voxels at visibility "
                  f"{a.visibility:.2f} -- nothing is reliably visible; lower --visibility"); return 1
        tree = cKDTree(model)

        # --- register every frame to that one model, translation only
        errs = np.zeros(len(clouds))
        newpos = np.zeros_like(pos)
        for n, (c, i, p) in enumerate(zip(clouds, frames, pos)):
            X = (c - p) @ R_wb[i]
            t, e = register_translation(X, tree, np.zeros(3))
            newpos[n] = p - R_wb[i] @ t                # the correction, carried back to the world frame
            errs[n] = e
        shift = np.linalg.norm(newpos - pos, axis=1) * 1e3
        pos, prev_pos = newpos, pos
        print(f"  round {rnd}: model {len(model):6d} voxels of {len(counts):7d} seen "
              f"(visibility >= {a.visibility:.0%})   registration residual p50 {1e3*np.median(errs):5.2f} "
              f"p95 {1e3*np.percentile(errs,95):5.2f} mm   position moved p50 {np.median(shift):5.2f} mm")

    out = dict(np.load(npz))
    out.update(stamp(**{f"{s}_canonical_visibility": float(a.visibility),
                        f"{s}_canonical_pose_balanced": True,
                        f"{s}_canonical_rounds": int(a.rounds),
                        f"{s}_canonical_voxel_m": float(VOXEL_M),
                        f"{s}_canonical_att_bin_deg": float(ATT_BIN_DEG),
                        f"{s}_canonical_version": "canonical_union_v1"}))
    rigid = np.full_like(cen_world, np.nan)
    rigid[frames] = pos
    rv = np.zeros(len(valid), bool); rv[frames] = True
    out[f"{s}_body_xyz_rigid"] = rigid
    out[f"{s}_body_rigid_valid"] = rv
    out[f"{s}_body_rigid_residual_mm"] = np.where(rv, 0.0, np.nan)
    out[f"{s}_body_rigid_residual_mm"][frames] = errs * 1e3
    np.savez(npz, **out)

    # --- what changed, against the centroid it replaces
    stepc = np.linalg.norm(np.diff(cen_world[frames], axis=0), axis=1) * 1e3
    stepr = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 1e3
    print("\n  translation smoothness (consecutive frames)")
    print(f"    centroid        p50 {np.median(stepc):5.2f}  p95 {np.percentile(stepc,95):6.2f} mm")
    print(f"    canonical rigid p50 {np.median(stepr):5.2f}  p95 {np.percentile(stepr,95):6.2f} mm")
    print(f"\n  -> {s}_body_xyz_rigid written into {npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
