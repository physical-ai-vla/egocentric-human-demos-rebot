#!/usr/bin/env python3
"""[2026-09-19] The held cube's 3D centroid: the ONLY observation of the TCP in a calibration take.

`body_tcp_fit` solves p_tcp(t) = p_body(t) + R_world_body(t) @ t_body_tcp. The body comes from the RGB-D
tracker and the rotation from the IMU; this supplies the left-hand side. Without it the fit has no equation,
which is exactly what stopped CALIB_bodytcp_20260919_121002.

WHY SIZE AND NOT COLOUR. Near the body there are reliably two coloured blobs, measured over 30 sampled frames
of CALIB_bodytcp_20260919_123133: the cube at ~2270 px (3D extent 25x21x9 mm) and slivers of the operator's
bare hand showing between the black straps at ~283 px (12x5x2 mm). An 8x area gap separates them without ever
naming a colour, so a take with a red or purple cube needs no change here. Hue is reported as a DIAGNOSTIC --
if the chosen blob's hue wanders mid-take, something other than the cube was picked.

The distractors on the table (other cubes, 0.30-0.50 m away) are excluded by proximity to the tracked body,
not by colour either.

FRAMES USED: only those inside the grip plateau, because a cube is the grasp point only while it is held. The
plateau is found the same way `calib_take_qa.grasp_check` finds it, so the two cannot disagree about when the
take had an object in the jaw.

    .venv/bin/python scripts/cube_tcp_detect.py --episode <ep> --side right [--debug-frames 6]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handumi_collector.pose.rgbd_io import RgbdEpisode                         # noqa: E402
from handumi_collector.pose.episode_io import RawEpisode                       # noqa: E402
from scripts.handumi_body_track import fit_table_plane, BAND, ABOVE_PLANE      # noqa: E402
from scripts.provenance import stamp                                           # noqa: E402

SAT_MIN, VAL_MIN = 80, 60      # coloured: not the white table, not the black body, not the dark background
MIN_AREA = 800                 # px; measured gap is cube 2270 vs hand-sliver 283
NEAR_BODY_M = 0.20             # the held cube is at the gripper; the table's spare cubes are 0.30-0.50 m away
MIN_DEPTH_PTS = 300            # a centroid from a handful of depth pixels is a guess with a decimal point
# 3D compactness. A cube is thick in its two largest directions; the hand slivers that show between the black
# straps are not. Measured 2-sigma extents: cube 25 x 21 x 9 mm, sliver 12 x 5 x 2 mm -- so the MIDDLE extent
# separates them 4x while elongation alone does not (cube p90 8.8 vs sliver p90 9.5, fully overlapping).
MIN_MID_EXTENT_M = 0.012
MAX_MAJOR_EXTENT_M = 0.10      # anything longer than this is a forearm or a merged blob, not a 4 cm cube
CONTINUITY_M = 0.10            # a held cube cannot jump this far between two frames at 30 Hz
PLATEAU_BAND, OFF_STOP, MIN_HOLD_S = 25.0, 40.0, 3.0


def grip_plateau(ep_path: Path, side: str, hz: float = 100.0) -> tuple[int, int] | None:
    """(t_start_ns, t_end_ns) of the longest stretch with the jaw parked clear of its hard stop."""
    raw = RawEpisode.load(ep_path)
    g = raw.grip.get(side)
    if g is None:
        return None
    r = np.asarray(g.raw_position, float)
    t = np.asarray(g.t_ns, np.int64)
    lo = r.min()
    best, start, out = 0, 0, None
    for i in range(1, len(r) + 1):
        if i == len(r) or abs(r[i] - r[start]) > PLATEAU_BAND or r[i] < lo + OFF_STOP:
            if r[start] >= lo + OFF_STOP and (i - start) > best:
                best, out = i - start, (int(t[start]), int(t[i - 1]))
            start = i
        elif r[i] < lo + OFF_STOP:
            start = i
    return out if best >= MIN_HOLD_S * hz else None


def detect(ep_path: Path, side: str, debug_frames: int = 0) -> dict:
    z = np.load(ep_path / "derived" / "body_track" / "track_export.npz")
    t_ns = z["timestamp_ns"]
    p_body_world = z[f"{side}_body_xyz"]
    body_ok = z[f"{side}_valid"]
    R_align = z["R_align"]

    win = grip_plateau(ep_path, side)
    if win is None:
        raise RuntimeError(f"{ep_path.name}: no grip plateau for {side} -- nothing was held, so there is no TCP")
    in_hold = (t_ns >= win[0]) & (t_ns <= win[1])

    ep = RgbdEpisode.load(ep_path)
    K = ep.intrinsics.K
    nrm, pd, plane = fit_table_plane(ep)

    n = len(t_ns)
    p_tcp = np.full((n, 3), np.nan)
    valid = np.zeros(n, bool)
    conf_all = np.zeros(n, float)
    hues, areas, dists = [], [], []
    last, miss = None, 0
    dbg = 0
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
        m = (h > ABOVE_PLANE[0]) & (h < ABOVE_PLANE[1])
        col = m & (hsv[ys, xs, 1] > SAT_MIN) & (hsv[ys, xs, 2] > VAL_MIN)
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
            if sel.sum() < MIN_DEPTH_PTS:              # depth support, not just pixel count in the mask
                continue
            Q = P[sel]
            c = Q.mean(0)
            dist = float(np.linalg.norm(c - pb_cam))
            if dist > NEAR_BODY_M:
                continue
            ext = 2.0 * np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov((Q - c).T)), 0))[::-1]
            if ext[1] < MIN_MID_EXTENT_M or ext[0] > MAX_MAJOR_EXTENT_M:
                continue                                # not cube-shaped: a sliver, or a merged blob
            # Temporal continuity dominates size: once the cube has been seen, the blob that continues its
            # track is the cube even on a frame where a bigger blob appears. Ranking by area alone picked a
            # hand sliver on 10% of frames of CALIB_bodytcp_20260919_123133 -- caught by the hue diagnostic.
            cont = None if last is None else float(np.linalg.norm(c - last))
            if cont is not None and cont > CONTINUITY_M:
                continue
            score = (0.0 if cont is None else -cont, int(stats[k, cv2.CC_STAT_AREA]))
            if best is None or score > best[0]:
                best = (score, c, int(stats[k, cv2.CC_STAT_AREA]), dist,
                        float(np.median(hsv[ys[sel], xs[sel], 0])), k, ext, int(sel.sum()), cont)
        if best is None:
            miss += 1
            if miss > 15:                               # lost for half a second: allow a fresh acquisition
                last = None
            continue
        _, c, area, dist, hue, k, ext, npts, cont = best
        last, miss = c, 0
        # Confidence: how far inside each gate this detection sits. Low values mark frames a consumer may drop
        # without re-running the detector.
        conf = float(min(1.0, area / 2400.0) * min(1.0, ext[1] / 0.020) * min(1.0, npts / 800.0)
                     * (1.0 if cont is None else float(np.clip(1.0 - cont / CONTINUITY_M, 0.0, 1.0))))
        p_tcp[i] = R_align @ c
        valid[i] = True
        conf_all[i] = conf
        hues.append(hue); areas.append(area); dists.append(dist)
        if dbg < debug_frames:
            vis = f.rgb.copy()
            vis[lab == k] = (0.5 * vis[lab == k] + np.array([0, 0, 127])).astype(np.uint8)
            for p, colr, tag in ((pb_cam, (0, 255, 255), "body"), (c, (0, 0, 255), f"cube {area}px")):
                u = int(K[0, 0] * p[0] / p[2] + K[0, 2]); vv = int(K[1, 1] * p[1] / p[2] + K[1, 2])
                cv2.circle(vis, (u, vv), 8, colr, 2)
                cv2.putText(vis, tag, (u + 10, vv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colr, 1)
            out = Path("/tmp/cube_tcp"); out.mkdir(exist_ok=True)
            cv2.imwrite(str(out / f"{ep_path.name}_{i:06d}.jpg"), vis)
            dbg += 1

    return dict(p_tcp=p_tcp, valid=valid, conf=conf_all, plane=plane, hold_frames=int(in_hold.sum()),
                hues=np.array(hues), areas=np.array(areas), dists=np.array(dists),
                hold_window_s=(win[1] - win[0]) / 1e9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--debug-frames", type=int, default=6)
    a = ap.parse_args()
    r = detect(a.episode, a.side, a.debug_frames)
    v = r["valid"]
    print(f"\n{a.episode.name}  side {a.side}")
    print(f"  grip plateau           {r['hold_window_s']:.1f} s  ({r['hold_frames']} depth frames inside it)")
    print(f"  cube found             {v.sum()} frames ({100 * v.sum() / max(r['hold_frames'], 1):.1f}% of the hold)")
    if v.sum():
        print(f"  blob area px           p10 {np.percentile(r['areas'], 10):.0f}  p50 {np.percentile(r['areas'], 50):.0f}"
              f"  p90 {np.percentile(r['areas'], 90):.0f}")
        print(f"  distance to body       p10 {np.percentile(r['dists'], 10):.3f}  p50 {np.percentile(r['dists'], 50):.3f}"
              f"  p90 {np.percentile(r['dists'], 90):.3f} m")
        # hue is a diagnostic, never a rule: a stable hue means the same object was picked throughout
        print(f"  confidence             p10 {np.percentile(r['conf'][v], 10):.2f}  p50 {np.percentile(r['conf'][v], 50):.2f}")
        spread = np.percentile(r["hues"], 90) - np.percentile(r["hues"], 10)
        print(f"  hue (diagnostic)       p10 {np.percentile(r['hues'], 10):.0f}  p50 {np.percentile(r['hues'], 50):.0f}"
              f"  p90 {np.percentile(r['hues'], 90):.0f}   spread {spread:.0f}"
              + ("   <-- MORE THAN ONE OBJECT WAS PICKED" if spread > 25 else ""))

    npz = a.episode / "derived" / "body_track" / "track_export.npz"
    d = dict(np.load(npz))
    d.update(stamp(**{f"{a.side}_cube_version": "centroid_size_gated_v1",
                      f"{a.side}_cube_min_area_px": int(MIN_AREA),
                      f"{a.side}_cube_near_body_m": float(NEAR_BODY_M)}))
    d[f"{a.side}_p_tcp_obs"] = r["p_tcp"]
    d[f"{a.side}_p_tcp_valid"] = v
    d[f"{a.side}_p_tcp_conf"] = r["conf"]
    np.savez(npz, **d)
    print(f"\n  -> {a.side}_p_tcp_obs written into {npz}")
    return 0 if v.sum() > 100 else 1


if __name__ == "__main__":
    sys.exit(main())
