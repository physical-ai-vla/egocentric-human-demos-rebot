"""Robot wrist C922 intrinsics in the ACTUAL recording mode (640×480) with a plain checkerboard → written as the per-side
K_target / D_target of the next configs/calibration/virtual_wrist_vNNN.yaml (the other side and rpy are carried over).

    python -m handumi_collector.tools.calibrate_c922 --side left --index 1 [--width 640 --height 480] [--cols 9 --rows 6 --square-mm 25] [--views 25]
    python -m handumi_collector.tools.calibrate_c922 --side right --images 'calib/right_wrist_c922/*.png'
Live: SPACE grabs a view when the board is found, Q finishes. Move the board over the whole frame incl. corners, tilt it."""
from __future__ import annotations
import argparse
import glob
import cv2
import numpy as np
from ..pose.calibration import save_versioned
from ..pose.virtual_view import load_virtual_wrist, write_virtual_wrist


def fov_deg(K, size) -> tuple[float, float]:
    """(HFOV, VFOV) in degrees of a pinhole camera with intrinsics K at `size` = (w, h)."""
    K = np.asarray(K, float); w, h = int(size[0]), int(size[1])
    return (float(np.degrees(2 * np.arctan(w / (2 * K[0, 0])))), float(np.degrees(2 * np.arctan(h / (2 * K[1, 1])))))


def calibrate_pinhole(images: list[np.ndarray], cols: int, rows: int, square_m: float) -> dict:
    objp = np.zeros((cols * rows, 3), np.float32); objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    obj, img_pts, size = [], [], None
    for im in images:
        g = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); size = (g.shape[1], g.shape[0])
        ok, c = cv2.findChessboardCornersSB(g, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok: obj.append(objp); img_pts.append(c.astype(np.float32))
    if len(obj) < 8: raise RuntimeError(f"only {len(obj)} usable views (need >= 8)")
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img_pts, size, None, None)
    return dict(K=K, D=D.reshape(-1)[:5], image_size=size, rms_px=float(rms), views=len(obj))


def grab_views(index: int, width: int, height: int, cols: int, rows: int, n_views: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION); cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); cap.set(3, width); cap.set(4, height)
    imgs = []
    while len(imgs) < n_views:
        ok, f = cap.read()
        if not ok: continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); found, c = cv2.findChessboardCornersSB(g, (cols, rows)); view = f.copy()
        if found: cv2.drawChessboardCorners(view, (cols, rows), c, found)
        cv2.putText(view, f"{f.shape[1]}x{f.shape[0]}  views {len(imgs)}/{n_views}  SPACE=grab  Q=done", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("C922 calibration", view); k = cv2.waitKey(1) & 0xFF
        if k == ord(" ") and found: imgs.append(f)
        if k == ord("q"): break
    cap.release(); cv2.destroyAllWindows(); return imgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--side", required=True, choices=["left", "right"]); ap.add_argument("--index", type=int, default=None); ap.add_argument("--images", default=None)
    ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=480); ap.add_argument("--cols", type=int, default=9); ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square-mm", type=float, default=25.0); ap.add_argument("--views", type=int, default=25); ap.add_argument("--cal-dir", default=None)
    a = ap.parse_args(argv)
    imgs = [cv2.imread(p) for p in sorted(glob.glob(a.images))] if a.images else grab_views(a.index, a.width, a.height, a.cols, a.rows, a.views)
    r = calibrate_pinhole(imgs, a.cols, a.rows, a.square_mm / 1000)
    if tuple(r["image_size"]) != (a.width, a.height): print(f"WARNING: images are {r['image_size']}, not the requested {a.width}x{a.height}")
    import time
    from pathlib import Path
    from ..devices.camera import list_video_devices
    cal_dir = Path(a.cal_dir) if a.cal_dir else None
    hfov, vfov = fov_deg(r["K"], r["image_size"])
    listing = [] if a.images else list_video_devices()
    device = (f"images:{a.images}" if a.images else (listing[a.index] if a.index is not None and a.index < len(listing) else f"index {a.index}"))
    # standalone record with full provenance (the pipeline reads virtual_wrist_vNNN.yaml; this is the audit trail)
    rec = save_versioned(f"robot_{a.side}_c922_{r['image_size'][0]}", dict(
        schema="handumi_robot_wrist_c922/v1", side=a.side, K=r["K"].tolist(), D=r["D"].tolist(), resolution=list(r["image_size"]),
        rms_reprojection_error_px=r["rms_px"], hfov_deg=round(hfov, 2), vfov_deg=round(vfov, 2), views=r["views"],
        device_identity=device, capture_backend=("file" if a.images else "AVFoundation/MJPG via cv2.VideoCapture"),
        board=f"checkerboard {a.cols}x{a.rows} {a.square_mm}mm", timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z")), cal_dir=cal_dir)
    _, cur = load_virtual_wrist(cal_dir=cal_dir)
    K = dict(cur["K_target"]); D = dict(cur["D_target"]); K[a.side] = r["K"]; D[a.side] = r["D"]
    p = write_virtual_wrist(K, r["image_size"], D_target=D, rpy_deg=cur["rpy_deg"], cal_dir=cal_dir,
                            source=f"{a.side}: {rec.stem} (robot wrist C922 measured at {r['image_size'][0]}x{r['image_size'][1]}, "
                                   f"checkerboard {a.cols}x{a.rows} {a.square_mm}mm, {r['views']} views, rms {r['rms_px']:.3f}px); other side carried over")
    K_ = r["K"]
    print(f"\n{a.side.upper()} C922 ({r['image_size'][0]}x{r['image_size'][1]}, {device})")
    print(f"HFOV: {hfov:.2f} deg\nVFOV: {vfov:.2f} deg\nfx:   {K_[0, 0]:.2f}\nfy:   {K_[1, 1]:.2f}\ncx:   {K_[0, 2]:.2f}\ncy:   {K_[1, 2]:.2f}\nRMS:  {r['rms_px']:.3f} px")
    print(f"D:    {np.round(r['D'], 5).tolist()}\nviews: {r['views']}")
    print(f"\nwrote {rec.name} (record) and {p.name} (pipeline target)")
    print(f"provisional HFOV in virtual_wrist_v001 was 49.1 deg -> measured {hfov:.2f} deg (delta {hfov - 49.1:+.2f})"); return 0


if __name__ == "__main__": raise SystemExit(main())
