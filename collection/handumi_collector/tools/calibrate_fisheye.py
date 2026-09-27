"""Arducam fisheye intrinsics from a plain CHECKERBOARD (cv2.findChessboardCornersSB + cv2.fisheye.calibrate).

A lens calibration target, not a fiducial world anchor; nothing here is imported by the pose pipeline. Writes
configs/calibration/fisheye_<side>_vNNN.yaml (kannala_brandt K, D, image_size) plus the imaging state the intrinsics
are only valid under.

    python -m handumi_collector.tools.calibrate_fisheye --side left --hardware handumi_rgbd
    python -m handumi_collector.tools.calibrate_fisheye --side left --images 'calib/left/*.png'   (offline, no camera)

Live: SPACE grabs a view when the board is found, Q finishes, R resets. The HUD shows which of the nine image regions
and which of the three distances still have no view — on a 160 deg fisheye a centre-only set produces a confidently
wrong D, so the tool refuses to calibrate until the frame is covered (--allow-gaps to override, recorded in the file).

Three things this does that a bare --index call cannot, and they are the difference between usable intrinsics and
silently wrong ones:
  * selects the camera BY PRODUCT NAME through the hardware profile, because the AVFoundation enumeration order is not
    stable and an index can point at the other wrist or at the Orbbec's RGB node;
  * applies and verifies the profile's frozen UVC preset first, so the calibration is measured under the same exposure
    and gain as the data takes, and refuses if a control did not take;
  * records camera identity, serial, resolution, fps and the whole imaging preset into the intrinsics file, so a live
    device can be checked against it later.
"""
from __future__ import annotations
import argparse
import glob
import time
from pathlib import Path
import cv2
import numpy as np
from ..pose.calibration import write_fisheye

GRID = 3                      # 3x3 image regions; a 160 deg lens needs the edges, not only the centre

# Distance bins by the board's apparent size, as a fraction of the frame's long side. Set from MEASUREMENT, not from
# what a normal lens would give: on this 160 deg fisheye a 150x210 mm ChArUco board spans 14-32% of the frame across
# 34 real views (median 19%), and 45% is simply unreachable without pushing inside the lens's focus distance. Gating on
# an unreachable bin rejects every take, which is what the first two calibration sessions ran into.
DIST_BINS = ("near", "mid", "far")
DIST_NEAR, DIST_MID = 0.26, 0.18

# What actually decides fisheye calibration quality is how many corners each view contributes, not how far away the
# board was: D is determined at the frame edge, and a view showing 5 of 24 corners constrains almost nothing there.
MIN_CORNERS_PER_VIEW = 12


def calibrate(images: list[np.ndarray], cols: int, rows: int, square_m: float) -> dict:
    objp = np.zeros((1, cols * rows, 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    obj, img_pts, size = [], [], None
    for im in images:
        g = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        size = (g.shape[1], g.shape[0])
        ok, c = cv2.findChessboardCornersSB(g, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            obj.append(objp); img_pts.append(c.reshape(1, -1, 2).astype(np.float64))
    if len(obj) < 8:
        raise RuntimeError(f"only {len(obj)} usable views (need >= 8)")
    K = np.eye(3); D = np.zeros((4, 1))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    rms, K, D, _, _ = cv2.fisheye.calibrate(obj, img_pts, size, K, D, flags=flags,
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7))
    return dict(K=K, D=D.reshape(-1), image_size=size, rms_px=float(rms), views=len(obj))


def coverage_of(corners: np.ndarray, size: tuple[int, int]) -> tuple[int, int, str]:
    """(grid row, grid col, distance bin) of one detected board, from its centroid and apparent size."""
    w, h = size
    c = corners.reshape(-1, 2)
    cx, cy = c.mean(axis=0)
    gr = min(int(cy / h * GRID), GRID - 1); gc = min(int(cx / w * GRID), GRID - 1)
    span = max(np.ptp(c[:, 0]), np.ptp(c[:, 1])) / max(w, h)   # board size as a fraction of the frame
                                                              # (ndarray.ptp() was removed in numpy 2)
    return gr, gc, ("near" if span > DIST_NEAR else "mid" if span > DIST_MID else "far")


def coverage_report(cov: dict) -> tuple[list[str], list[str]]:
    cells = [f"r{r}c{c}" for r in range(GRID) for c in range(GRID) if not cov["grid"][r][c]]
    dists = [d for d in DIST_BINS if not cov["dist"][d]]
    return cells, dists


PROBE_SIZES = [(c, r) for c in range(4, 15) for r in range(3, 12) if c > r]


def probe_board(gray: np.ndarray) -> list[tuple[int, int]]:
    """Which (cols, rows) INNER-corner counts this frame actually detects. `--cols/--rows` are inner corners, not
    squares, and a 10x7-square board is 9x6 — getting that wrong means SPACE silently never grabs, which is the one
    failure this tool used to give no feedback about at all."""
    hits = []
    for c, r in PROBE_SIZES:
        ok, _ = cv2.findChessboardCornersSB(gray, (c, r))
        if ok:
            hits.append((c, r))
    return hits


def grab_live(side: str, hardware: str | None, cols: int, rows: int, views: int, *,
              save_dir: Path | None = None, probe: bool = False,
              target: "CharucoTarget | None" = None) -> tuple[list[np.ndarray], dict]:
    from ..config import load_config
    from ..devices.camera import list_video_devices, resolve_camera_index, verify_identity
    from ..devices.uvc_controls import apply_preset, find_uvc_util

    stream = f"{side}_wrist"
    cfg = None
    if hardware:
        hw = load_config(hardware=hardware).hardware
        cfg = next((c for c in hw.cameras if c.name == stream), None)
        if cfg is None:
            raise SystemExit(f"profile {hardware!r} has no camera {stream!r}")
    if cfg is None:
        raise SystemExit("--hardware is required for live capture: the camera must be selected by product name, not by "
                         "index (the AVFoundation enumeration order is not stable)")

    listing = list_video_devices()
    identity = verify_identity(cfg, listing)
    idx = resolve_camera_index(cfg, listing)
    preset = []
    if cfg.uvc_controls:
        preset = apply_preset(cfg.match_name, dict(cfg.uvc_controls), binary=find_uvc_util(cfg.uvc_util))
        bad = [r for r in preset if not r["applied"]]
        if bad:
            raise SystemExit("frozen imaging preset did not apply: " +
                             "; ".join(f"{r['control']} ({r['reason']})" for r in bad) +
                             " — calibrating under a different exposure than the data takes makes the intrinsics "
                             "untraceable, so this is fatal")
        print("imaging preset applied and verified: " + ", ".join(f"{r['control']}={r['readback']}" for r in preset))

    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*(cfg.fourcc or "MJPG")))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    imgs: list[np.ndarray] = []
    cov = dict(grid=[[0] * GRID for _ in range(GRID)], dist={d: 0 for d in DIST_BINS})
    size = None
    t0 = time.time()
    if probe:
        for _ in range(40):
            ok, f = cap.read()
            if ok and f is not None:
                g0 = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                chk, mk = probe_board(g0), probe_markers(g0)
                print(f"probe: plain-checkerboard geometries {chk or 'NONE'}")
                print(f"probe: marker dictionaries {[(h['dictionary'], h['n_markers']) for h in mk] or 'NONE'}")
                if mk:
                    print(f"  -> ChArUco/marker board detected ({mk[0]['dictionary']}, {mk[0]['n_markers']} markers). "
                          "findChessboardCornersSB cannot use it.")
                elif chk and (cols, rows) not in chk:
                    print(f"  -> the board in view is {chk[0][0]}x{chk[0][1]} inner corners, not {cols}x{rows}")
                break
    while len(imgs) < views:
        ok, f = cap.read()
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
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        view = f.copy()
        if target is not None:
            ip, _op, _ids = target.detect(g)
            found = ip is not None
            c = None if ip is None else ip.reshape(-1, 1, 2).astype(np.float32)
            label = f"CHARUCO {target.squares[0]}x{target.squares[1]} {target.name}"
            if found:
                for x, y in ip:
                    cv2.circle(view, (int(x), int(y)), 6, (0, 255, 255), -1)
        else:
            found, c = cv2.findChessboardCornersSB(g, (cols, rows))
            label = f"BOARD {cols}x{rows}"
            if found:
                cv2.drawChessboardCorners(view, (cols, rows), c, found)
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
        cv2.putText(view, f"{label}: " + (f"FOUND {0 if c is None else len(c)} corners - press SPACE"
                                          if found else "NOT FOUND - SPACE does nothing"),
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0) if found else (0, 0, 255), 3)
        cv2.putText(view, f"views {len(imgs)}/{views}   SPACE=grab  P=probe board size  R=reset  Q=done", (20, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.putText(view, f"missing cells {cells or 'none'}   missing distances {dists or 'none'}", (20, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2)
        cv2.imshow(f"fisheye calibration [{side}]", cv2.resize(view, (1280, 720)))
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            if not found:
                print(f"  SPACE ignored: no board detected in this frame ({label}). Press P to find out what this "
                      "board actually is, or check the board geometry arguments.")
            else:
                gr, gc, db = coverage_of(c, size)
                cov["grid"][gr][gc] += 1; cov["dist"][db] += 1
                imgs.append(f)
                if save_dir is not None:      # saved immediately: an aborted session must not lose the grabs
                    save_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_dir / f"{side}_{len(imgs):03d}_r{gr}c{gc}_{db}.png"), f)
                print(f"  view {len(imgs):2d}  cell r{gr}c{gc}  distance {db}"
                      + (f"  saved" if save_dir is not None else ""))
        elif k == ord("p"):
            chk = probe_board(g)
            mk = probe_markers(g)
            print(f"  probe: plain-checkerboard geometries {chk or 'NONE'}")
            print(f"  probe: marker dictionaries {[(h['dictionary'], h['n_markers']) for h in mk] or 'NONE'}")
            if mk:
                print(f"  -> this is a ChArUco/marker board ({mk[0]['dictionary']}, {mk[0]['n_markers']} markers, "
                      f"ids {mk[0]['ids']}). findChessboardCornersSB cannot use it — that is why SPACE does nothing.")
            elif chk and (cols, rows) not in chk:
                print(f"  -> re-run with --cols {chk[0][0]} --rows {chk[0][1]}")
        elif k == ord("r"):
            imgs.clear(); cov = dict(grid=[[0] * GRID for _ in range(GRID)], dist={d: 0 for d in DIST_BINS})
            print("  reset")
        elif k == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()
    meta = dict(product_name=identity.get("listing_name"), match_name=cfg.match_name,
                reported_serial=(identity.get("observed_serials") or [None])[0],
                resolution=[cfg.width, cfg.height], fps=cfg.fps, fourcc=cfg.fourcc,
                imaging_preset={r["control"]: r["readback"] for r in preset},
                coverage=dict(grid=cov["grid"], distance=cov["dist"]))
    return imgs, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True, choices=["left", "right"])
    ap.add_argument("--hardware", default="handumi_rgbd", help="profile that names the camera and freezes its imaging")
    ap.add_argument("--images", default=None, help="offline: calibrate from saved images instead of a live camera")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--views", type=int, default=30)
    ap.add_argument("--allow-gaps", action="store_true", help="calibrate despite uncovered regions (recorded in the file)")
    ap.add_argument("--save-dir", type=Path, default=None,
                    help="write every grabbed frame here immediately, so an aborted session is not lost and can be "
                         "re-processed with --images")
    ap.add_argument("--probe", action="store_true",
                    help="report which board geometry is actually detected before starting (also on P at any time)")
    a = ap.parse_args(argv)

    meta: dict = {}
    if a.images:
        imgs = [cv2.imread(p) for p in sorted(glob.glob(a.images))]
        imgs = [i for i in imgs if i is not None]
        meta = dict(source_images=a.images, n_images=len(imgs))
    else:
        imgs, meta = grab_live(a.side, a.hardware, a.cols, a.rows, a.views, save_dir=a.save_dir, probe=a.probe)


    cov = meta.get("coverage")
    if cov:
        cells = [f"r{r}c{c}" for r in range(GRID) for c in range(GRID) if not cov["grid"][r][c]]
        dists = [d for d in DIST_BINS if not cov["distance"][d]]
        if (cells or dists) and not a.allow_gaps:
            raise SystemExit(f"coverage incomplete — missing cells {cells}, missing distances {dists}. On a 160 deg "
                             "fisheye an uncovered frame edge produces a confidently wrong D; add views there, or pass "
                             "--allow-gaps to accept it (the gap is then recorded in the intrinsics file).")
        meta["coverage_complete"] = not (cells or dists)

    r = calibrate(imgs, a.cols, a.rows, a.square_mm / 1000)
    board_meta = dict(kind="checker", cols=a.cols, rows=a.rows, square_mm=a.square_mm)
    from ..config import load_config
    extra = dict(meta)
    extra.update(rms_px=round(r["rms_px"], 4), views=r["views"], board=board_meta,
                 hardware_profile=a.hardware,
                 calibration_version=(load_config(hardware=a.hardware).hardware.calibration_version if a.hardware else None))
    p = write_fisheye(a.side, r["K"], r["D"], r["image_size"],
                      source=f"checkerboard {a.cols}x{a.rows} {a.square_mm}mm, {r['views']} views, rms {r['rms_px']:.3f}px",
                      extra=extra)
    print(f"\nwrote {p}")
    print(f"  rms {r['rms_px']:.3f} px over {r['views']} views at {r['image_size'][0]}x{r['image_size'][1]}")
    print(f"  camera {extra.get('product_name')!r} serial {extra.get('reported_serial')!r}")
    print(f"  imaging {extra.get('imaging_preset')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
