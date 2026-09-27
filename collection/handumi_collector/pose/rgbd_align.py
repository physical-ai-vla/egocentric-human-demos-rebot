"""Is the depth map pointing at the same place as the colour image?

`check_rgbd_ready` already asks whether depth exists, covers enough of the frame and sits at a plausible distance, and
`tools.calibrate_table_frame` cross-checks the table plane against a board solved from colour. Neither notices a depth
map that is spatially *shifted* — the plane is still flat and the coverage is still fine when every object boundary is
four pixels to the left, and four pixels at a metre is a couple of centimetres of cube. The Orbbec aligns depth to
colour in hardware, which is exactly the kind of thing that is either right or quietly wrong.

So: find the edges both images agree should exist, and measure how far apart they land.

    rgb edges          Canny on the colour image
    depth edges        where the depth gradient jumps by more than a few centimetres
    offset             for each depth edge pixel, the distance to the nearest rgb edge

Three numbers come out, and none of them is a threshold. What counts as acceptable depends on the distance to the table
and on how much a pixel is worth there, so the first datasets record these and the limits are set afterwards from what
a known-good rig actually produces. Only a gross offset -- edges that are nowhere near each other -- is a failure on
sight."""
from __future__ import annotations
import numpy as np


def _edges_rgb(rgb: np.ndarray, *, lo: int = 60, hi: int = 160) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.ndim == 3 else rgb
    return cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), lo, hi) > 0


def _edges_depth(depth_m: np.ndarray, *, jump_m: float = 0.03) -> np.ndarray:
    """Depth discontinuities: a step of more than `jump_m` between neighbours, ignoring steps to or from invalid depth
    (a hole boundary is a sensor artefact, not an object edge, and would otherwise dominate the statistic)."""
    d = np.asarray(depth_m, np.float32)
    valid = d > 0
    e = np.zeros(d.shape, bool)
    # Mark both pixels either side of a jump. A one-sided difference puts the depth edge a pixel inside the object while
    # Canny puts the colour edge on the transition itself, and that bias shows up as a systematic 1 px shift in every
    # measurement -- which makes a real 1 px misalignment impossible to distinguish from the operators disagreeing.
    stepx = (np.abs(d[:, 1:] - d[:, :-1]) > jump_m) & valid[:, 1:] & valid[:, :-1]
    e[:, 1:] |= stepx; e[:, :-1] |= stepx
    stepy = (np.abs(d[1:, :] - d[:-1, :]) > jump_m) & valid[1:, :] & valid[:-1, :]
    e[1:, :] |= stepy; e[:-1, :] |= stepy
    return e & valid


def edge_alignment(rgb: np.ndarray, depth_m: np.ndarray, *, roi: tuple[int, int, int, int] | None = None,
                   jump_m: float = 0.03, band_px: int = 3, max_offset_px: float = 25.0) -> dict:
    """Colour/depth edge agreement over one frame. `roi` is (x0, y0, x1, y1) to restrict to the table or a cube.

    Returns median/p95 offset in pixels, the fraction of depth edges that land within `band_px` of a colour edge
    (`overlap`), and how much of the band around colour edges has no valid depth at all -- object boundaries are exactly
    where a depth sensor drops out, and a number for it is worth having before the dataset rather than after.

    `shift_px` is the **correction**: the offset that has to be added to depth-edge pixels to put them on the colour
    edges. Depth content sitting too far right therefore reports a negative dx."""
    import cv2
    if rgb.shape[:2] != np.asarray(depth_m).shape[:2]:
        # Comparing edge maps of different sizes indexes one by the other's coordinates, which produces numbers rather
        # than an error. Depth aligned to colour is the same grid by definition; anything else is a different question.
        return dict(error=f"colour is {rgb.shape[1]}x{rgb.shape[0]} and depth is "
                          f"{np.asarray(depth_m).shape[1]}x{np.asarray(depth_m).shape[0]} — not the same grid, so there "
                          f"is no per-pixel alignment to measure")
    e_rgb, e_dep = _edges_rgb(rgb), _edges_depth(depth_m, jump_m=jump_m)
    d = np.asarray(depth_m, np.float32)
    if roi:
        x0, y0, x1, y1 = roi
        m = np.zeros(e_rgb.shape, bool); m[y0:y1, x0:x1] = True
        e_rgb, e_dep, d = e_rgb & m, e_dep & m, d * m

    out: dict = dict(n_rgb_edge=int(e_rgb.sum()), n_depth_edge=int(e_dep.sum()))
    if out["n_rgb_edge"] < 50 or out["n_depth_edge"] < 50:
        out["error"] = "too few edges to compare (a featureless scene, or depth with no structure in it)"
        return out

    # Distance from every pixel to the nearest colour edge, then read it off at the depth edges.
    dist = cv2.distanceTransform((~e_rgb).astype(np.uint8), cv2.DIST_L2, 3)
    off = dist[e_dep]
    off = np.minimum(off, max_offset_px)              # a depth edge with no colour edge anywhere is capped, not dropped
    out.update(median_offset_px=round(float(np.median(off)), 3),
               p95_offset_px=round(float(np.percentile(off, 95)), 3),
               overlap=round(float((off <= band_px).mean()), 4))

    # The distance distribution alone is blind to the failure it most needs to see. Slide the depth map sideways and an
    # edge that runs the same way simply glides along itself: a clean 8 px shift of a square came out as a median of
    # 0.9 px, because its horizontal edges outnumber the vertical ones that actually moved. So estimate the shift
    # directly -- the offset that best lands depth edges on colour edges -- and report the leftover separately.
    ys, xs = np.nonzero(e_dep)
    best = (0, 0, float(np.mean(off)))
    r = int(min(max_offset_px, 15))
    for dy in range(-r, r + 1):
        yy = ys + dy
        ky = (yy >= 0) & (yy < dist.shape[0])
        for dx in range(-r, r + 1):
            xx = xs + dx
            k = ky & (xx >= 0) & (xx < dist.shape[1])
            if k.sum() < 20: continue
            c = float(np.mean(np.minimum(dist[yy[k], xx[k]], max_offset_px)))
            if c < best[2]: best = (dx, dy, c)
    dx, dy, _ = best
    yy, xx = ys + dy, xs + dx
    k = (yy >= 0) & (yy < dist.shape[0]) & (xx >= 0) & (xx < dist.shape[1])
    res = np.minimum(dist[yy[k], xx[k]], max_offset_px)
    out.update(shift_px=[int(dx), int(dy)], shift_magnitude_px=round(float(np.hypot(dx, dy)), 3),
               residual_median_px=round(float(np.median(res)), 3),
               residual_overlap=round(float((res <= band_px).mean()), 4))

    # Invalid depth in a band around the colour edges: the dropout that object boundaries cause.
    band = cv2.dilate(e_rgb.astype(np.uint8), np.ones((2 * band_px + 1, 2 * band_px + 1), np.uint8)) > 0
    if roi: band &= m
    out["invalid_depth_near_edges"] = round(float((d[band] <= 0).mean()), 4) if band.sum() else None
    out["invalid_depth_overall"] = round(float((d <= 0).mean()), 4)
    return out


def summarise(per_frame: list[dict], *, gross_offset_px: float = 12.0) -> dict:
    """Aggregate across frames and say only what can be said without a calibrated threshold: a gross offset is a
    failure, everything else is a number and, where it looks off, a warning."""
    usable = [f for f in per_frame if "median_offset_px" in f]
    out: dict = dict(frames=len(per_frame), usable=len(usable), problems=[], warnings=[])
    if not usable:
        out["warnings"].append("no frame had enough colour and depth structure to compare")
        return out
    med = float(np.median([f["median_offset_px"] for f in usable]))
    p95 = float(np.median([f["p95_offset_px"] for f in usable]))
    ov = float(np.median([f["overlap"] for f in usable]))
    shift = float(np.median([f["shift_magnitude_px"] for f in usable]))
    sx = int(np.median([f["shift_px"][0] for f in usable])); sy = int(np.median([f["shift_px"][1] for f in usable]))
    resid = float(np.median([f["residual_median_px"] for f in usable]))
    inv = float(np.median([f["invalid_depth_near_edges"] for f in usable if f.get("invalid_depth_near_edges") is not None] or [np.nan]))
    out.update(median_offset_px=round(med, 3), p95_offset_px=round(p95, 3), overlap=round(ov, 4),
               shift_px=[sx, sy], shift_magnitude_px=round(shift, 3), residual_median_px=round(resid, 3),
               invalid_depth_near_edges=None if np.isnan(inv) else round(inv, 4))
    # Gate on the estimated shift, which is what misalignment actually is, rather than on a distance distribution that
    # a translation along an edge does not move.
    if shift > gross_offset_px:
        out["problems"].append(f"depth sits {shift:.0f} px ({sx:+d}, {sy:+d}) from colour — gross misalignment, "
                               f"not calibration error")
    elif shift > gross_offset_px / 4:
        out["warnings"].append(f"depth appears shifted by {shift:.0f} px ({sx:+d}, {sy:+d}) — worth watching; "
                               f"no calibrated limit yet")
    if ov < 0.3:
        out["warnings"].append(f"only {ov:.0%} of depth edges land near a colour edge")
    return out
