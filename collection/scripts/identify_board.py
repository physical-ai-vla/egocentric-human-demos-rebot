"""Which calibration board is actually on the desk?

    sudo .venv/bin/python scripts/identify_board.py --live
    .venv/bin/python scripts/identify_board.py --image shot.png

Records disagree about the board that was used -- 5x7 @30/15 mm appears with DICT_5X5_50 in one note and DICT_5X5_100
in configs/charuco.yaml, and the only PDF actually printed into ~/calib is the 6x8 @31/23 mm wrist board. Guessing the
dictionary is the worst option available: the wrong one detects nothing at all, which is indistinguishable from a bad
frame, bad lighting or a board too far away. So show the board to the camera once and let every candidate try.

Lives in scripts/ deliberately. `handumi_collector` is held to "no AprilTag / ArUco / ChArUco anywhere in the production
package" by tests/handumi/test_pose_pipeline.py, and a board identification run is calibration-time tooling, not part of
the package the rule protects -- the same reason the ChArUco detector it reuses already lives outside it."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ego_collector.camera.calibration import CharucoSpec, detect_charuco    # noqa: E402  (outside the guarded package)

CANDIDATES = [
    ("5x7 @30/15 DICT_5X5_50", CharucoSpec(5, 7, 0.030, 0.015, "DICT_5X5_50")),
    ("5x7 @30/15 DICT_5X5_100", CharucoSpec(5, 7, 0.030, 0.015, "DICT_5X5_100")),
    ("6x8 @31/23 DICT_5X5_100 (board_wrist_A4)", CharucoSpec(6, 8, 0.031, 0.023, "DICT_5X5_100")),
    ("6x8 @31/23 DICT_5X5_50", CharucoSpec(6, 8, 0.031, 0.023, "DICT_5X5_50")),
]


def probe(img, K=None, D=None) -> list[dict]:
    import cv2
    out = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    for name, spec in CANDIDATES:
        row: dict = dict(name=name, spec=spec)
        board = spec.board()
        det = cv2.aruco.ArucoDetector(board.getDictionary())
        corners, ids, _ = det.detectMarkers(gray)
        row["markers"] = 0 if ids is None else int(len(ids))
        row["max_id"] = -1 if ids is None else int(np.max(ids))
        d = detect_charuco(img, spec, min_corners=4)
        row["charuco_corners"] = 0 if d is None else d.count
        # Reprojection is the part that separates "a few markers happened to match" from "this is the board": a wrong
        # geometry still produces corners, but they will not sit on a single rigid plane pose.
        if d is not None and d.count >= 6 and K is not None:
            ok, rvec, tvec = cv2.solvePnP(d.object_points, d.image_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                proj, _ = cv2.projectPoints(d.object_points, rvec, tvec, K, D)
                err = np.linalg.norm(proj.reshape(-1, 2) - d.image_points.reshape(-1, 2), axis=1)
                row["reproj_rms_px"] = round(float(np.sqrt(np.mean(err ** 2))), 3)
                row["distance_m"] = round(float(np.linalg.norm(tvec)), 3)
        out.append(row)
    # A plain checkerboard would be the other possibility, and costs one call to rule out.
    for cols, rows in ((4, 6), (5, 7), (6, 8), (8, 6), (9, 6)):
        found, _ = cv2.findChessboardCornersSB(gray, (cols, rows))
        if found:
            out.append(dict(name=f"plain checkerboard {cols}x{rows} inner corners", markers=0,
                            charuco_corners=cols * rows, spec=None))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--live", action="store_true", help="grab from the head Orbbec (needs sudo on macOS)")
    g.add_argument("--image", type=Path)
    ap.add_argument("--save", type=Path, default=None, help="--live: keep the frame here")
    a = ap.parse_args(argv)
    import cv2

    K = D = None
    if a.image:
        img = cv2.imread(str(a.image))
        if img is None: raise SystemExit(f"{a.image}: unreadable")
    else:
        from handumi_collector.config import CameraCfg
        from handumi_collector.devices.camera import OrbbecCamera
        import time
        cam = OrbbecCamera(CameraCfg("head_depth", "aux_depth", backend="orbbec", width=848, height=480, fps=30))
        cam.open(); cam.start()
        intr = cam.intrinsics
        if intr:
            K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64); D = np.zeros(5)
        img = None
        for _ in range(60):
            time.sleep(0.1)
            f = cam.buffer.latest()
            if f is not None: img = f.image; break
        cam.stop()
        if img is None: raise SystemExit("no frame from the camera")
        if a.save: cv2.imwrite(str(a.save), img); print(f"saved {a.save}")

    print(f"frame {img.shape[1]}x{img.shape[0]}" + (f"   intrinsics fx {K[0,0]:.1f}" if K is not None else "   (no intrinsics: reprojection not checked)"))
    rows = probe(img, K, D)
    print(f"\n{'candidate':44s}{'markers':>9s}{'corners':>9s}{'reproj px':>11s}{'dist m':>9s}")
    print("-" * 82)
    for r in rows:
        print(f"{r['name']:44s}{r['markers']:>9d}{r['charuco_corners']:>9d}"
              f"{r.get('reproj_rms_px', float('nan')):>11.3f}{r.get('distance_m', float('nan')):>9.3f}")
    best = max(rows, key=lambda r: (r["charuco_corners"], r["markers"]))
    print()
    if best["charuco_corners"] < 6:
        print("nothing detected. The board may be too small in frame, out of focus, or badly lit — move it closer,\n"
              "fill more of the view, and try again before concluding anything about which board it is.")
        return 1
    print(f"-> {best['name']}")
    if best.get("spec") is not None:
        s = best["spec"]
        print(f"   squares {s.squares_x}x{s.squares_y}  square {s.square_length_m*1000:.0f} mm  "
              f"marker {s.marker_length_m*1000:.0f} mm  dictionary {s.dictionary}")
    # A smaller predefined dictionary is the first N entries of the larger one, so a board whose marker ids all fall
    # inside the smaller detects identically in both. When that is why two candidates tie, the records disagreeing about
    # the dictionary simply does not matter -- say so, rather than asking for a better frame that cannot break the tie.
    ties = [r for r in rows if r is not best and r.get("spec") is not None and r["charuco_corners"] == best["charuco_corners"]
            and best.get("spec") is not None
            and (r["spec"].squares_x, r["spec"].squares_y, r["spec"].square_length_m) ==
                (best["spec"].squares_x, best["spec"].squares_y, best["spec"].square_length_m)]
    dict_only = [r for r in ties if r["spec"].dictionary != best["spec"].dictionary]
    if dict_only:
        sizes = {r["spec"].dictionary: int(r["spec"].dictionary.rsplit("_", 1)[-1]) for r in dict_only + [best]}
        smallest = min(sizes, key=sizes.get)
        print(f"   dictionary is not determined here, and does not need to be: the board's marker ids go up to "
              f"{best['max_id']}, inside every candidate, so {' and '.join(sorted(sizes))} detect identically.\n"
              f"   Record the smallest that covers it ({smallest}) and the disagreement in the notes stops mattering.")
    elif ties:
        print("   ! another candidate with different geometry detects just as well — take a frame with the board larger in view")
    return 0


if __name__ == "__main__":
    sys.exit(main())
