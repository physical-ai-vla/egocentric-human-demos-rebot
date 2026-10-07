#!/usr/bin/env python
"""Camera intrinsics from a ChArUco board — fisheye for a wrist, pinhole for the head.

    python scripts/charuco_calibrate.py --side left --hardware handumi_rgbd --save-dir ~/calib/left --views 30
    python scripts/charuco_calibrate.py --side head --hardware temporal_sync_calib \
        --charuco configs/charuco_wrist.yaml --views 25 --save-dir ~/calib/head
    python scripts/charuco_calibrate.py --side left --images '~/calib/left/*.png'      (offline, no provenance)

`--side head` calibrates the centre camera with cv2.calibrateCamera and writes head_c922_intrinsics_vNNN.yaml. That
is a separate namespace on purpose: head_mount_v001..v005 are the Orbbec-era TABLE-FRAME solves, not intrinsics, and
they stay as legacy; fisheye_<side> is a Kannala-Brandt file the pose pipeline reads for a wrist. Forcing
Kannala-Brandt on a normal lens fits a 4-parameter equidistant projection to a lens that is not equidistant and hides
the error in D, where the reprojection residual absorbs it.

Pass --charuco explicitly. The default configs/charuco.yaml describes a 5x7 @30 mm board; the board on this desk is
the 6x8 @31 mm one in configs/charuco_wrist.yaml, and solving the wrong geometry yields a confident wrong scale.

Composed from what the repo already has, not reimplemented:
    ego_collector.camera.calibration   CharucoSpec (configs/charuco.yaml), detect_charuco, pose_diversity_ok
    handumi_collector.devices          camera selection BY PRODUCT NAME, frozen UVC preset with read-back
    handumi_collector.tools            the 3x3 + near/mid/far coverage accounting
    handumi_collector.pose.calibration write_fisheye with provenance
The correspondences come from ChArUco; the model is chosen per camera. The camera is opened at a MEASURED OpenCV
index, never at its position in the device listing -- those two disagreed (reversed) on this host on 2026-09-15.

It lives in scripts/ on purpose: handumi_collector is a provably fiducial-free package and
tests/handumi/test_pose_pipeline.py enforces that by import, by source line AND by filename. A ChArUco board is a
legitimate lens target rather than a world anchor, but it is still marker code, so the dependency runs one way only —
this script imports the package, never the reverse.
"""
from __future__ import annotations
import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego_collector.camera.calibration import CharucoSpec, detect_charuco                  # noqa: E402
from handumi_collector.pose.calibration import write_fisheye                              # noqa: E402
from handumi_collector.tools.calibrate_fisheye import (DIST_BINS, DIST_MID, DIST_NEAR,    # noqa: E402
                                                       GRID, MIN_CORNERS_PER_VIEW,
                                                       coverage_of, coverage_report)

# The distance bins above were measured on a 160 deg lens, where the 186x248 mm board spans 14-32 % of the frame.
# A C922 is far narrower, so the same board at the same distances fills much more: across 25 real head views the span
# ran 0.195-0.396 (median 0.319), which puts 21 of them in "near" and leaves "far" empty and unreachable -- the board
# has to go beyond the distance where it still shows 12 corners to get under 0.18. Gating on an unreachable bin
# rejects every take, which is the mistake the fisheye constants already carry a comment about.
DIST_NEAR_PINHOLE, DIST_MID_PINHOLE = 0.33, 0.25

# Focal length is separated from board DISTANCE only by how much the perspective foreshortening varies between poses.
# Coverage of the image and of the distance bins does not deliver that, and this tool used to check nothing else: 50
# real head views passed both gates while every single one sat between 18.5 and 23.7 degrees of tilt -- the shape a
# board lying flat on the table makes when only its position moves. fx then came out anywhere between 455 and 549
# depending on which half of the views were used, with the reprojection error falling the whole time. So account for
# tilt as well, and say out loud which tilt band is still missing.
TILT_BINS = ("flat", "tilted", "steep")
TILT_TILTED, TILT_STEEP = 15.0, 35.0      # degrees between the board normal and the optical axis
NOMINAL_F = 450.0                         # only to solve a pose for the HUD; the calibration never uses it


def tilt_of(object_points: np.ndarray, image_points: np.ndarray, size: tuple[int, int]) -> float:
    """Angle between the board normal and the optical axis, under a nominal focal length. A HUD number, not a
    measurement: it needs no calibration to be right to a few degrees, which is all the binning asks of it."""
    w, h = size
    K = np.array([[NOMINAL_F, 0, w / 2], [0, NOMINAL_F, h / 2], [0, 0, 1]], np.float64)
    ok, rv, _ = cv2.solvePnP(object_points.reshape(-1, 3).astype(np.float64),
                             image_points.reshape(-1, 1, 2).astype(np.float64), K, np.zeros(5))
    if not ok:
        return 0.0
    R, _ = cv2.Rodrigues(rv)
    return float(np.degrees(np.arccos(min(1.0, abs(float(R[2, 2]))))))


def tilt_bin(deg: float) -> str:
    return "steep" if deg >= TILT_STEEP else "tilted" if deg >= TILT_TILTED else "flat"


def bins_for(model: str) -> tuple[float, float]:
    return (DIST_NEAR_PINHOLE, DIST_MID_PINHOLE) if model == "pinhole" else (DIST_NEAR, DIST_MID)


def coverage_of_model(corners: np.ndarray, size: tuple[int, int], model: str) -> tuple[int, int, str]:
    """coverage_of() with the distance thresholds that belong to this lens."""
    w, h = size
    c = corners.reshape(-1, 2)
    cx, cy = c.mean(axis=0)
    gr = min(int(cy / h * GRID), GRID - 1); gc = min(int(cx / w * GRID), GRID - 1)
    span = max(np.ptp(c[:, 0]), np.ptp(c[:, 1])) / max(w, h)
    near, mid = bins_for(model)
    return gr, gc, ("near" if span > near else "mid" if span > mid else "far")


def coverage_from_images(images: list[np.ndarray], spec: CharucoSpec, model: str) -> dict:
    """Coverage of a set of saved frames, so an offline re-solve records it instead of leaving it blank."""
    cov = dict(grid=[[0] * GRID for _ in range(GRID)], distance={d: 0 for d in DIST_BINS},
               tilt={t: 0 for t in TILT_BINS})
    for im in images:
        size = (im.shape[1], im.shape[0])
        det = detect_charuco(im, spec, min_corners=MIN_CORNERS_PER_VIEW)
        if det is None:
            continue
        gr, gc, db = coverage_of_model(det.image_points.reshape(-1, 2), size, model)
        cov["grid"][gr][gc] += 1; cov["distance"][db] += 1
        cov["tilt"][tilt_bin(tilt_of(det.object_points, det.image_points, size))] += 1
    return cov


def calibrate_fisheye_from_charuco(images: list[np.ndarray], spec: CharucoSpec, *,
                                   min_corners: int = MIN_CORNERS_PER_VIEW) -> dict:
    """Views contribute however many corners they show — the reason to use a marker board at all: on a fisheye the
    distortion is decided at the frame EDGE, where a plain checkerboard never fits whole."""
    obj, img_pts, size, per_view = [], [], None, []
    for im in images:
        size = (im.shape[1], im.shape[0])
        det = detect_charuco(im, spec, min_corners=min_corners)
        per_view.append(0 if det is None else det.count)
        if det is None:
            continue
        obj.append(det.object_points.reshape(1, -1, 3).astype(np.float64))
        img_pts.append(det.image_points.reshape(1, -1, 2).astype(np.float64))
    if len(obj) < 8:
        raise RuntimeError(
            f"only {len(obj)} views with >= {min_corners} ChArUco corners (need >= 8); per-view counts were "
            f"{per_view}. A view showing a handful of corners constrains the edge distortion almost not at all: bring "
            f"the board closer so it fills more of the frame, or print a larger one (the 150x210 mm board spans only "
            f"~19% of this lens's frame).")
    K = np.eye(3); D = np.zeros((4, 1))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    rms, K, D, _, _ = cv2.fisheye.calibrate(obj, img_pts, size, K, D, flags=flags,
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7))
    used = [n for n in per_view if n >= min_corners]
    return dict(K=K, D=D.reshape(-1), image_size=size, rms_px=float(rms), views=len(obj), corners_per_view=per_view,
                corners_median=float(np.median(used)) if used else 0.0, total_corners=int(sum(used)),
                min_corners_per_view=min_corners)


def calibrate_pinhole_from_charuco(images: list[np.ndarray], spec: CharucoSpec, *,
                                   min_corners: int = MIN_CORNERS_PER_VIEW) -> dict:
    """Same correspondences, the model a NORMAL lens needs. The C922 is not a fisheye: forcing Kannala-Brandt on it
    fits a 4-parameter equidistant projection to a lens that is not equidistant, and the error goes into D where the
    reprojection residual can absorb it."""
    obj, img_pts, size, per_view = [], [], None, []
    for im in images:
        size = (im.shape[1], im.shape[0])
        det = detect_charuco(im, spec, min_corners=min_corners)
        per_view.append(0 if det is None else det.count)
        if det is None:
            continue
        obj.append(det.object_points.reshape(-1, 3).astype(np.float32))
        img_pts.append(det.image_points.reshape(-1, 1, 2).astype(np.float32))
    if len(obj) < 8:
        raise RuntimeError(f"only {len(obj)} views with >= {min_corners} ChArUco corners (need >= 8); "
                           f"per-view counts were {per_view}")
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img_pts, size, None, None,
                                          criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7))
    used = [n for n in per_view if n >= min_corners]
    return dict(K=K, D=D.reshape(-1), image_size=size, rms_px=float(rms), views=len(obj), corners_per_view=per_view,
                corners_median=float(np.median(used)) if used else 0.0, total_corners=int(sum(used)),
                min_corners_per_view=min_corners)


def grab_live(side: str, hardware: str, spec: CharucoSpec, views: int, *, save_dir: Path | None = None,
              model: str = "fisheye"):
    """Same frozen-state guarantees as the package's checkerboard tool: camera by NAME, preset applied and verified,
    delivered format enforced, coverage accounted. Only the detector differs."""
    from handumi_collector.config import load_config
    from handumi_collector.devices.camera import list_video_devices, probe_index_map, resolve_camera_index, verify_identity
    from handumi_collector.devices.uvc_controls import apply_preset, find_uvc_util

    stream = "head" if side == "head" else f"{side}_wrist"
    cfg = next((c for c in load_config(hardware=hardware).hardware.cameras if c.name == stream), None)
    if cfg is None:
        raise SystemExit(f"profile {hardware!r} has no camera {stream!r}")
    listing = list_video_devices()
    identity = verify_identity(cfg, listing)
    # NOT resolve_camera_index(): a position in that listing is not an OpenCV index. Measured 2026-09-15, ffmpeg
    # listed the three cameras in exactly the reverse of the order OpenCV opens them, so calibrating "by name" this
    # way would have fitted the intrinsics of a different camera onto this one's file.
    want = identity.get("listing_name") or cfg.match_name
    measured = probe_index_map([want]) if want else {}
    idx = measured.get(want, resolve_camera_index(cfg, listing))
    print(f"camera {want!r}: OpenCV index {idx}"
          + ("" if want in measured else "  (NOT measured - fell back to the listing position)"))
    preset = []
    if cfg.uvc_controls:
        # uvc-util selects by FULL product name; cfg.match_name is a listing substring ("C922" vs "C922 Pro Stream
        # Webcam") and addressing uvc-util with it fails outright.
        uvc_name = identity.get("listing_name") or cfg.match_name
        preset = apply_preset(uvc_name, dict(cfg.uvc_controls), binary=find_uvc_util(cfg.uvc_util))
        bad = [r for r in preset if not r["applied"]]
        if bad:
            raise SystemExit("frozen imaging preset did not apply: "
                             + "; ".join(f"{r['control']} ({r['reason']})" for r in bad)
                             + " — calibrating under a different exposure than the data takes makes the intrinsics "
                               "untraceable, so this is fatal")
        print("imaging preset applied and verified: " + ", ".join(f"{r['control']}={r['readback']}" for r in preset))

    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*(cfg.fourcc or "MJPG")))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    imgs, size, t0 = [], None, time.time()
    frame_ns: list[int] = []      # every delivered frame, so the capture rate is measured rather than assumed
    cov = dict(grid=[[0] * GRID for _ in range(GRID)], dist={d: 0 for d in DIST_BINS},
               tilt={t: 0 for t in TILT_BINS})
    while len(imgs) < views:
        ok, f = cap.read()
        if ok and f is not None:
            frame_ns.append(time.monotonic_ns())
        if not ok or f is None:
            if time.time() - t0 > 10:
                cap.release(); raise SystemExit(f"camera {cfg.match_name!r} opened but returned no frame")
            continue
        if size is None:
            size = (f.shape[1], f.shape[0])
            if size != (cfg.width, cfg.height):
                cap.release()
                raise SystemExit(f"camera delivers {size[0]}x{size[1]}, the profile freezes {cfg.width}x{cfg.height} — "
                                 "intrinsics are tied to the resolution, so this is fatal")
        det = detect_charuco(f, spec, min_corners=4)
        n = 0 if det is None else det.count
        view = f.copy()
        if det is not None:
            for x, y in det.image_points.reshape(-1, 2):
                cv2.circle(view, (int(x), int(y)), 6, (0, 255, 255), -1)
        cells, dists = coverage_report(cov)
        h, w = view.shape[:2]
        for i in range(1, GRID):
            cv2.line(view, (w * i // GRID, 0), (w * i // GRID, h), (60, 60, 60), 1)
            cv2.line(view, (0, h * i // GRID), (w, h * i // GRID), (60, 60, 60), 1)
        for r in range(GRID):
            for cc in range(GRID):
                col = (0, 200, 0) if cov["grid"][r][cc] else (0, 0, 220)
                cv2.putText(view, str(cov["grid"][r][cc]), (w * cc // GRID + 12, h * r // GRID + 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        good = n >= MIN_CORNERS_PER_VIEW
        cv2.putText(view, f"CHARUCO {spec.squares_x}x{spec.squares_y}: "
                          + (f"{n} corners - SPACE to grab" if good else
                             f"{n} corners - need >= {MIN_CORNERS_PER_VIEW}, bring the board CLOSER"),
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 255, 0) if good else (0, 0, 255), 3)
        if det is not None:
            tdeg = tilt_of(det.object_points, det.image_points, size)
            tb_now = tilt_bin(tdeg)
            tcol = (0, 220, 220) if cov["tilt"][tb_now] == 0 else (230, 230, 230)
            cv2.putText(view, f"tilt {tdeg:4.0f} deg -> {tb_now.upper()}"
                              + ("  (still missing - GRAB IT)" if cov["tilt"][tb_now] == 0 else "")
                              + f"   have {dict(cov['tilt'])}",
                        (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tcol, 2)
            _, _, db_now = coverage_of_model(det.image_points.reshape(-1, 2), size, model)
            pts = det.image_points.reshape(-1, 2)
            span_now = float(np.ptp(pts[:, 0])) / size[0]
            need = (0, 220, 220) if cov["dist"][db_now] == 0 else (230, 230, 230)
            cv2.putText(view, f"span {span_now:.0%} -> {db_now.upper()} bin"
                              + ("  (still missing - GRAB IT)" if cov["dist"][db_now] == 0 else ""),
                        (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.8, need, 2)
        cv2.putText(view, f"views {len(imgs)}/{views}   R=reset  Q=done", (20, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.putText(view, f"missing cells {cells or 'none'}   missing distances {dists or 'none'}", (20, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2)
        cv2.imshow(f"charuco fisheye calibration [{side}]", cv2.resize(view, (1280, 720)))
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            if not good:
                print(f"  SPACE ignored: {n} corners (need >= {MIN_CORNERS_PER_VIEW})")
            else:
                gr, gc, db = coverage_of_model(det.image_points.reshape(-1, 2), size, model)
                tb = tilt_bin(tilt_of(det.object_points, det.image_points, size))
                cov["grid"][gr][gc] += 1; cov["dist"][db] += 1; cov["tilt"][tb] += 1
                imgs.append(f)
                if save_dir is not None:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_dir / f"{side}_{len(imgs):03d}_r{gr}c{gc}_{db}.png"), f)
                print(f"  view {len(imgs):2d}  {n:2d} corners  cell r{gr}c{gc}  distance {db}  tilt {tb}")
        elif k == ord("r"):
            imgs.clear(); cov = dict(grid=[[0] * GRID for _ in range(GRID)], dist={d: 0 for d in DIST_BINS},
                                     tilt={t: 0 for t in TILT_BINS})
            print("  reset")
        elif k == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()
    d = np.diff(np.asarray(frame_ns, np.float64)) / 1e6 if len(frame_ns) > 2 else np.zeros(0)
    fps_stats = (dict(frames=len(frame_ns), measured_fps=round(1000.0 / float(np.median(d)), 2),
                      interval_ms=dict(p50=round(float(np.median(d)), 2), p95=round(float(np.percentile(d, 95)), 2),
                                       max=round(float(d.max()), 2)))
                 if len(d) else dict(frames=len(frame_ns)))
    print(f"capture rate: {fps_stats.get('measured_fps', '?')} fps over {fps_stats['frames']} delivered frames "
          f"(requested {cfg.fps})")
    return imgs, dict(fps_measured=fps_stats, opencv_index=idx,
                      product_name=identity.get("listing_name"), match_name=cfg.match_name,
                      reported_serial=(identity.get("observed_serials") or [None])[0],
                      resolution=[cfg.width, cfg.height], fps=cfg.fps, fourcc=cfg.fourcc,
                      imaging_preset={r["control"]: r["readback"] for r in preset},
                      coverage=dict(grid=cov["grid"], distance=cov["dist"]))


def estimate_valid_radius(imgs, spec, K, D, image_size, *, n_subsets: int = 8, frac: float = 0.7,
                          max_spread_deg: float = 0.3, seed: int = 0) -> dict:
    """Largest image radius at which the distortion model is actually SUPPORTED BY DATA, not extrapolated.

    A 160 deg lens is calibrated from a board that never reaches the image corners, so D beyond the outermost observed
    corner is an extrapolation of a 9th-order polynomial -- confidently wrong, with nothing in the reprojection error to
    reveal it.  Refitting on random view subsets exposes that: where the data constrains the model the refits agree,
    and where it does not they fan out.  The returned radius is the largest one whose bearing angle stays within
    `max_spread_deg` across refits, capped at the outermost corner ever observed.  Consumers mask beyond it (see
    VirtualPinholeView) so no feature is ever built on an extrapolated ray."""
    rng = np.random.default_rng(seed)
    K = np.asarray(K, float); W, H = image_size
    cx, cy = K[0, 2], K[1, 2]
    r_corner = float(np.hypot(max(cx, W - cx), max(cy, H - cy)))

    obs = []
    for im in imgs:
        det = detect_charuco(im, spec, min_corners=MIN_CORNERS_PER_VIEW)
        if det is not None:
            q = det.image_points.reshape(-1, 2)
            obs.append(np.hypot(q[:, 0] - cx, q[:, 1] - cy))
    r_observed = float(np.concatenate(obs).max()) if obs else 0.0

    def theta_at(Ki, Di, r):
        th = np.linspace(0, np.pi / 2, 40001)
        k1, k2, k3, k4 = np.asarray(Di, float).ravel()[:4]
        return np.degrees(np.interp(r, Ki[0, 0] * (th + k1 * th**3 + k2 * th**5 + k3 * th**7 + k4 * th**9), th))

    fits = []
    for _ in range(n_subsets):
        idx = rng.choice(len(imgs), max(3, int(len(imgs) * frac)), replace=False)
        try:
            r = calibrate_fisheye_from_charuco([imgs[j] for j in idx], spec)
            fits.append((np.asarray(r["K"], float), np.asarray(r["D"], float)))
        except Exception:
            continue
    if len(fits) < 3:
        return dict(valid_radius_px=round(r_observed, 1), r_observed_px=round(r_observed, 1),
                    r_corner_px=round(r_corner, 1), method="observed-only (too few subset refits to measure spread)",
                    n_subset_refits=len(fits))

    radii = np.arange(200, r_corner + 25, 25.0)
    base_th = [theta_at(np.asarray(K, float), np.asarray(D, float), r) for r in radii]
    spreads, worst = [], []
    for r, tb in zip(radii, base_th):
        th = np.array([theta_at(Ki, Di, r) for Ki, Di in fits])
        spreads.append(float(np.std(th)))
        worst.append(float(np.abs(th - tb).max()))
    # Contiguous prefix, not max(): the spread is not monotonic, and a dip back under the tolerance at a large radius
    # says nothing about the radii that failed below it.
    r_stable = 0.0
    for r, sp in zip(radii, spreads):
        if sp > max_spread_deg:
            break
        r_stable = float(r)
    valid = float(min(r_stable, r_observed))
    return dict(valid_radius_px=round(valid, 1), r_observed_px=round(r_observed, 1),
                r_stable_px=round(r_stable, 1), r_corner_px=round(r_corner, 1),
                masked_frame_fraction=round(float(
                    (np.hypot(*np.meshgrid(np.arange(W) - cx, np.arange(H) - cy)[::-1]) > valid).mean()), 4),
                spread_deg_by_radius={int(r): round(sp, 3) for r, sp in zip(radii, spreads)},
                max_deviation_deg_by_radius={int(r): round(wd, 3) for r, wd in zip(radii, worst)},
                # what the angular disagreement costs in the thing we actually care about
                worst_lateral_mm_at_0p5m_by_radius={int(r): round(float(np.tan(np.radians(wd)) * 500.0), 1)
                                                    for r, wd in zip(radii, worst)},
                max_spread_deg=max_spread_deg, n_subset_refits=len(fits), subset_fraction=frac,
                method="subset-refit bearing spread, capped at outermost observed corner")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True, choices=["left", "right", "head"],
                    help="wrist side, or `head` for the centre camera (stream `head`)")
    ap.add_argument("--model", default=None, choices=["fisheye", "pinhole"],
                    help="lens model; default fisheye for a wrist, pinhole for the head")
    ap.add_argument("--out", default=None,
                    help="calibration file stem; default fisheye_<side> for a wrist, head_c922_intrinsics for the head")
    ap.add_argument("--hardware", default="handumi_rgbd")
    ap.add_argument("--charuco", type=Path, default=Path("configs/charuco.yaml"))
    ap.add_argument("--images", default=None, help="offline: calibrate from saved frames (no camera provenance)")
    ap.add_argument("--views", type=int, default=30)
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--allow-gaps", action="store_true")
    a = ap.parse_args(argv)

    spec = CharucoSpec.from_yaml(a.charuco) if a.charuco.exists() else CharucoSpec()
    corners = (spec.squares_x - 1) * (spec.squares_y - 1)
    print(f"board {spec.squares_x}x{spec.squares_y}, square {spec.square_length_m*1e3:g} mm, "
          f"marker {spec.marker_length_m*1e3:g} mm, {spec.dictionary}, legacy_pattern={spec.legacy_pattern} "
          f"-> {corners} corners   (from {a.charuco})")

    model = a.model or ("pinhole" if a.side == "head" else "fisheye")
    near, mid = bins_for(model)
    print(f"lens model {model}; distance bins near > {near:.0%} of the frame, mid > {mid:.0%}")

    if a.images:
        paths = sorted(glob.glob(str(Path(a.images).expanduser())))
        imgs = [im for im in (cv2.imread(p) for p in paths) if im is not None]
        meta = dict(source_images=a.images, n_images=len(imgs),
                    coverage=coverage_from_images(imgs, spec, model))
        print(f"offline: {len(imgs)} images — camera provenance comes from the session that saved them, not from here")
    else:
        imgs, meta = grab_live(a.side, a.hardware, spec, a.views, save_dir=a.save_dir, model=model)

    cov = meta.get("coverage")
    if cov:
        miss_c = [f"r{r}c{c}" for r in range(GRID) for c in range(GRID) if not cov["grid"][r][c]]
        miss_d = [d for d in DIST_BINS if not cov["distance"][d]]
        miss_t = [t for t in TILT_BINS if not (cov.get("tilt") or {}).get(t)]
        if miss_t and not a.allow_gaps:
            raise SystemExit(
                f"tilt coverage incomplete — missing {miss_t} (flat < {TILT_TILTED:.0f} deg, tilted "
                f"{TILT_TILTED:.0f}-{TILT_STEEP:.0f}, steep >= {TILT_STEEP:.0f}). Focal length is separated from board "
                f"distance only by how the foreshortening CHANGES between poses; 50 views all at ~20 deg left fx "
                f"anywhere between 455 and 549. Hold the board up and tilt it, or --allow-gaps (recorded in the file).")
        if (miss_c or miss_d) and not a.allow_gaps:
            raise SystemExit(f"coverage incomplete — missing cells {miss_c}, missing distances {miss_d} "
                             f"(near > {near:.0%} of the frame, mid > {mid:.0%}). An uncovered frame edge "
                             "produces a confidently wrong D on a 160 deg lens; add views there, or --allow-gaps to "
                             "accept it (recorded in the file).")
        meta["coverage_complete"] = not (miss_c or miss_d or miss_t)

    fn = calibrate_pinhole_from_charuco if model == "pinhole" else calibrate_fisheye_from_charuco
    r = fn(imgs, spec)
    vr = {}
    if model == "fisheye":       # a valid-radius mask only means something where D is a 9th-order extrapolation
        print("  measuring the radius out to which the model is supported by data (subset refits)...")
        vr = estimate_valid_radius(imgs, spec, r["K"], r["D"], r["image_size"])
    from handumi_collector.config import load_config
    from handumi_collector.pose.calibration import save_versioned
    extra = dict(meta)
    if vr:
        extra.update(valid_radius_px=vr["valid_radius_px"], valid_radius=vr)
    model_name = ("cv2.calibrateCamera (pinhole, plumb_bob)" if model == "pinhole"
                  else "cv2.fisheye.calibrate (kannala_brandt)")
    extra.update(rms_px=round(r["rms_px"], 4), views=r["views"], corners_median=r["corners_median"],
                 total_corners_used=r["total_corners"], corners_per_view=r["corners_per_view"],
                 min_corners_per_view=r["min_corners_per_view"], board=spec.to_dict(),
                 board_corners=corners, charuco_config=str(a.charuco),
                 model=model_name, calibrated_by="scripts/charuco_calibrate.py",
                 distance_bins=dict(near_gt=near, mid_gt=mid),
                 hardware_profile=a.hardware,
                 calibration_version=load_config(hardware=a.hardware).hardware.calibration_version)
    source = (f"charuco {spec.squares_x}x{spec.squares_y} "
              f"{spec.square_length_m*1e3:g}/{spec.marker_length_m*1e3:g}mm, "
              f"{r['views']} views, rms {r['rms_px']:.3f}px")
    if model == "pinhole":
        # Its own namespace: head_mount_v001..v005 are the Orbbec-era table-frame solves and stay untouched as legacy,
        # and fisheye_<side> is a Kannala-Brandt file the pose pipeline reads for a wrist.
        stem = a.out or "head_c922_intrinsics"
        p = save_versioned(stem, dict(schema="handumi_pinhole_intrinsics/v1", distortion_model="plumb_bob", side=a.side,
                                      K=[float(x) for x in np.asarray(r["K"]).reshape(-1)],
                                      D=[float(x) for x in np.asarray(r["D"]).reshape(-1)],
                                      image_size=[int(r["image_size"][0]), int(r["image_size"][1])],
                                      source=source, **extra))
    else:
        p = write_fisheye(a.side, r["K"], r["D"], r["image_size"], source=source, extra=extra)
    print(f"\nwrote {p}")
    print(f"  model    {model_name}")
    print(f"  rms      {r['rms_px']:.3f} px over {r['views']} views at {r['image_size'][0]}x{r['image_size'][1]}")
    print(f"  corners  median {r['corners_median']:.1f}/view, {r['total_corners']} used in total")
    if vr:
        print(f"  valid r  {vr['valid_radius_px']:.0f} px of {vr.get('r_corner_px', 0):.0f} px to the image corner "
              f"({vr.get('masked_frame_fraction', 0):.1%} of the frame masked as extrapolated)")
    rms = r["rms_px"]
    print(f"  verdict  " + ("GOOD (< 0.5 px)" if rms < 0.5 else "USABLE (0.5-0.8 px)" if rms < 0.8 else
                            "MARGINAL (0.8-1.0 px)" if rms < 1.0 else "REJECT (> 1 px) - reshoot"))
    print(f"  camera   {extra.get('product_name')!r} serial {extra.get('reported_serial')!r}")
    print(f"  imaging  {extra.get('imaging_preset')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
