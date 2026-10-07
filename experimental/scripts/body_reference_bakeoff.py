#!/usr/bin/env python3
"""[2026-09-19] Which point on the HandUMI body is actually fixed to it? Ranked by one rotation-free number.

The lever-arm fit failed on CALIB_bodytcp_20260919_123133 not because of the IMU -- the invariant
||p_cube - p_body|| scatters with correlation +0.04 to attitude, so orientation is exonerated -- but because the
tracked body point is a centroid of the VISIBLE surface, and the visible surface swings 2x in point count as the
unit is handled. A centroid of a changing subset is not a point on the object.

So: propose reference points, and rank them by the one quantity that cannot be contaminated by any rotation
estimate. If the cube is rigidly held and the reference is rigidly attached, d(t) = ||p_cube - p_ref|| is
CONSTANT. Whatever spread it has is the reference's own failure, measured in millimetres.

    A  trimmed centroid r=0.12      what the tracker does today
    B  inner-core centroid r=0.06 / 0.08   does shrinking the trim help, or just cut the support?
    C  bounding-box centre r=0.12   extent-based rather than mass-based: insensitive to which face is dense
    D  geometric median r=0.12      L1 centre: a robust mass centre that ignores a changing tail
    E  ICP-propagated reference     a point rigid BY CONSTRUCTION, carried frame to frame by the cloud's own
                                    motion. Drift accumulates and is reported; the others cannot drift but were
                                    never rigid to begin with.

Orientation is not used anywhere in this file, on purpose.

    .venv/bin/python scripts/body_reference_bakeoff.py --episode <ep> --side right
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handumi_collector.pose.rgbd_io import RgbdEpisode                              # noqa: E402
from scripts.handumi_body_track import fit_table_plane, BAND, ABOVE_PLANE, DARK_MAX  # noqa: E402

ICP_POINTS = 1200
ICP_ITERS = 12


def geometric_median(P: np.ndarray, iters: int = 24) -> np.ndarray:
    """Weiszfeld. The mean moves when the visible tail grows; the L1 centre largely does not."""
    x = P.mean(0)
    for _ in range(iters):
        d = np.linalg.norm(P - x, axis=1)
        w = 1.0 / np.maximum(d, 1e-4)
        xn = (P * w[:, None]).sum(0) / w.sum()
        if np.linalg.norm(xn - x) < 1e-6:
            return xn
        x = xn
    return x


def icp(src: np.ndarray, dst: np.ndarray, iters: int = ICP_ITERS) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-point ICP, own implementation (open3d segfaults under numpy 2 on this machine)."""
    from scipy.spatial import cKDTree
    R, t = np.eye(3), np.zeros(3)
    tree = cKDTree(dst)
    cur = src
    err = np.inf
    for _ in range(iters):
        d, j = tree.query(cur, k=1)
        keep = d < max(np.percentile(d, 80), 1e-3)           # trim the 20% worst: occlusion changes the overlap
        A, B = cur[keep], dst[j[keep]]
        ca, cb = A.mean(0), B.mean(0)
        H = (A - ca).T @ (B - cb)
        U, _, Vt = np.linalg.svd(H)
        S = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
        dR = Vt.T @ S @ U.T
        dt = cb - dR @ ca
        cur = cur @ dR.T + dt
        R, t = dR @ R, dR @ t + dt
        err = float(np.mean(d[keep]))
    return R, t, err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--gap-guard", type=int, default=5)
    ap.add_argument("--min-conf", type=float, default=0.9)
    a = ap.parse_args()

    z = np.load(a.episode / "derived" / "body_track" / "track_export.npz")
    s = a.side
    R_align = z["R_align"]
    vb, vt = z[f"{s}_valid"], z[f"{s}_p_tcp_valid"]
    conf = z[f"{s}_p_tcp_conf"]
    pbw, pcw = z[f"{s}_body_xyz"], z[f"{s}_p_tcp_obs"]
    idx_all = np.arange(len(vb))
    bad = idx_all[~vb]
    gapd = (np.min(np.abs(idx_all[:, None] - bad[None, :]), axis=1) if len(bad)
            else np.full(len(vb), len(vb))).astype(float)
    use = vb & vt & (conf >= a.min_conf) & (gapd > a.gap_guard)
    print(f"\n{a.episode.name}  side {s}   {use.sum()} frames "
          f"(confidence >= {a.min_conf}, more than {a.gap_guard} from a track gap)")

    ep = RgbdEpisode.load(a.episode)
    K = ep.intrinsics.K
    nrm, pd, _ = fit_table_plane(ep)

    names = ["A centroid r=0.12", "B centroid r=0.06", "B centroid r=0.08",
             "C bbox centre r=0.12", "D geometric median r=0.12", "E ICP-propagated"]
    refs = {n: [] for n in names}
    cubes, frames = [], []
    prev_cloud, icp_ref, icp_drift = None, None, 0.0
    lo, hi = int(np.flatnonzero(use)[0]), int(np.flatnonzero(use)[-1]) + 1
    for f in ep.iter_frames(lo, hi, 1):
        i = f.index
        if not use[i]:
            prev_cloud = None                      # a break invalidates the ICP chain; say so rather than bridge it
            continue
        pb = R_align.T @ pbw[i]
        d = f.depth_m
        g = cv2.cvtColor(f.rgb, cv2.COLOR_BGR2GRAY)
        ys, xs = np.nonzero((d > BAND[0]) & (d < BAND[1]) & (g < DARK_MAX))
        if len(xs) < 300:
            prev_cloud = None
            continue
        zz = d[ys, xs]
        P = np.stack([(xs - K[0, 2]) * zz / K[0, 0], (ys - K[1, 2]) * zz / K[1, 1], zz], 1)
        h = P @ nrm + pd
        P = P[(h > ABOVE_PLANE[0]) & (h < ABOVE_PLANE[1])]
        r = np.linalg.norm(P - pb, axis=1)
        core = P[r < 0.12]
        if len(core) < 300:
            prev_cloud = None
            continue
        row = {"A centroid r=0.12": core.mean(0),
               "C bbox centre r=0.12": (core.min(0) + core.max(0)) / 2,
               "D geometric median r=0.12": geometric_median(core[::max(len(core) // 3000, 1)])}
        for rr, nm in ((0.06, "B centroid r=0.06"), (0.08, "B centroid r=0.08")):
            sub = P[r < rr]
            row[nm] = sub.mean(0) if len(sub) >= 150 else np.full(3, np.nan)

        sub = core[::max(len(core) // ICP_POINTS, 1)]
        if prev_cloud is None or icp_ref is None:
            icp_ref = core.mean(0)                 # seed the chain; only its RIGIDITY is under test, not its place
        else:
            Ri, ti, err = icp(prev_cloud, sub)
            icp_ref = Ri @ icp_ref + ti
            icp_drift = max(icp_drift, err)
        row["E ICP-propagated"] = icp_ref.copy()
        prev_cloud = sub

        for n in names:
            refs[n].append(row[n])
        cubes.append(R_align.T @ pcw[i])
        frames.append(i)

    cubes = np.asarray(cubes)
    print(f"\n  {'reference point':>28s} {'n':>5s} {'|d| mean':>9s} {'std':>7s} {'range':>8s} "
          f"{'p95 step':>9s} {'half shift':>11s}")
    scored = []
    for n in names:
        Pref = np.asarray(refs[n])
        ok = np.isfinite(Pref).all(1)
        if ok.sum() < 60:
            print(f"  {n:>28s}   too few frames"); continue
        d = np.linalg.norm(cubes[ok] - Pref[ok], axis=1) * 1e3
        step = np.linalg.norm(np.diff(Pref[ok], axis=0), axis=1) * 1e3
        half = abs(np.median(d[:len(d) // 2]) - np.median(d[len(d) // 2:]))
        print(f"  {n:>28s} {ok.sum():5d} {d.mean():9.1f} {d.std():7.1f} {d.max() - d.min():8.1f} "
              f"{np.percentile(step, 95):9.1f} {half:10.1f} mm")
        scored.append((d.std(), n))
    if icp_drift:
        print(f"\n  ICP mean correspondence error (worst frame): {1e3 * icp_drift:.1f} mm -- E's drift budget")
    scored.sort()
    print(f"\n  best by invariance: {scored[0][1]} at {scored[0][0]:.1f} mm std"
          f"   (today's tracker: {[x for x in scored if x[1].startswith('A')][0][0]:.1f} mm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
