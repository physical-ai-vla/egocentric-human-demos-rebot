"""Fixed head-camera mount: where it sits, and the table frame it sees.

    python -m handumi_collector.tools.calibrate_table_frame --from-episode EP --rows 6 --cols 9 --square-mm 25 \
        --height-cm 82 --edge-cm 35 --pitch-deg 38 --write
    sudo .venv/bin/python -m handumi_collector.tools.calibrate_table_frame --live --frames 20 --rows 6 --cols 9 --square-mm 25

Lay a plain checkerboard flat on the table where the cubes will be, take a handful of views without moving the camera,
and this solves the board pose in the camera and inverts it. The board defines the table frame: origin at its first
corner, X and Y along its axes in the table plane, Z out of the surface, right-handed. `T_table_camera` then carries any
RGB-D point into table coordinates, which is what makes cube positions comparable across sessions and convertible to a
robot frame later.

A checkerboard, not a tag. The no-fiducial rule
(`tests/handumi/test_pose_pipeline.py::test_no_fiducial_dependency_in_handumi_collector`) is about the runtime pose
pipeline depending on markers; a calibration target that leaves nothing to detect at recording time is the same thing
`fisheye_<side>` is already solved from.

Two things a transform alone will not give you, so both are recorded beside it: the hand-measured mount geometry, which
is what lets someone physically restore the holder after it comes off, and — with `--remount` — the spread of the
solution across remounts, which is how much of the calibration actually survives that."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from ..pose.se3 import inv_T, make_T, mean_pose, rotation_angle_deg


def board_points(rows: int, cols: int, square_m: float) -> np.ndarray:
    """Inner-corner grid on Z=0, in the order cv2.findChessboardCorners returns."""
    g = np.zeros((rows * cols, 3), np.float64)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return g


class CheckerboardDetector:
    """The default target: a plain checkerboard, found with cv2 and nothing else.

    `calibrate_table_frame` takes a detector rather than hard-coding this one because the board that happens to be on
    this desk is a ChArUco, and `handumi_collector` is held to "no AprilTag / ArUco / ChArUco anywhere in the production
    package" by tests/handumi/test_pose_pipeline.py. The rule is about the runtime pose pipeline depending on markers,
    and a one-off calibration is not that -- but the way to honour it is to keep the marker code outside the package and
    inject it, which is exactly what tools.calibrate_fisheye already does. See scripts/calibrate_table_frame_charuco.py."""

    def __init__(self, rows: int, cols: int, square_m: float) -> None:
        self.rows, self.cols, self.square_m = rows, cols, square_m
        self.obj = board_points(rows, cols, square_m)
        self.name = f"checkerboard {cols}x{rows} @{square_m*1000:.0f} mm"

    def __call__(self, bgr: np.ndarray):
        import cv2
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        c = find_corners(g, self.rows, self.cols)
        return None if c is None else (c, self.obj)


def find_corners(gray: np.ndarray, rows: int, cols: int) -> np.ndarray | None:
    import cv2
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, c = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok: return None
    return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)).reshape(-1, 2)


def solve_view(corners_uv: np.ndarray, obj: np.ndarray, K: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, float]:
    """(T_camera_board, rms reprojection px) for one view."""
    import cv2
    ok, rvec, tvec = cv2.solvePnP(obj, corners_uv, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise RuntimeError("solvePnP failed")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners_uv) ** 2, axis=1))))
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec.reshape(3)), rms


MAX_VIEW_SPREAD_MM = 5.0
MAX_VIEW_SPREAD_DEG = 1.0


def board_moved(res: dict) -> str | None:
    """Why the views cannot be averaged, or None. The table frame is defined by where the board is: its origin is a
    corner of it and its X axis runs along it. Turning the board between views therefore defines a second frame rather
    than giving a second look at the first, and the mean of the two is neither -- a half turn of this board shows up as
    155 mm and 180 deg of spread. Diversity is what lens intrinsics need; an extrinsic wants the target nailed down."""
    sp = res["view_spread"]
    if sp["translation_mm"] <= MAX_VIEW_SPREAD_MM and sp["rotation_deg"] <= MAX_VIEW_SPREAD_DEG: return None
    return (f"the views disagree by {sp['translation_mm']:.1f} mm and {sp['rotation_deg']:.1f} deg, far past "
            f"{MAX_VIEW_SPREAD_MM:.0f} mm / {MAX_VIEW_SPREAD_DEG:.0f} deg — the board or the camera moved between them, "
            f"and averaging across that gives a frame that matches neither. Grab every view without touching either.")


def table_frame(views: list[tuple[np.ndarray, float]]) -> dict:
    """Average the per-view board poses and report how much they disagree — the spread is the honest error bar, and a
    camera that did not move should produce views that agree to a fraction of a millimetre."""
    Ts = np.stack([T for T, _ in views])
    T_cam_board = mean_pose(Ts)
    d_t = np.linalg.norm(Ts[:, :3, 3] - T_cam_board[:3, 3], axis=1)
    d_r = [rotation_angle_deg(T[:3, :3], T_cam_board[:3, :3]) for T in Ts]
    return dict(T_table_camera=inv_T(T_cam_board), T_camera_board=T_cam_board, n_views=len(views),
                rms_reproj_px=float(np.mean([r for _, r in views])),
                view_spread=dict(translation_mm=round(float(np.max(d_t)) * 1000, 3),
                                 rotation_deg=round(float(np.max(d_r)), 4)))


def depth_plane_check(depth_m: np.ndarray, K: np.ndarray, corners_uv: np.ndarray, T_camera_board: np.ndarray) -> dict:
    """Fit a plane to the depth inside the board outline and compare it with the board pose solved from colour.

    This is the one cross-check RGB-D allows that colour alone cannot: the board is flat and lying on the table, so the
    depth map's own plane must agree with it. A tilt between the two is depth-to-colour misalignment or a wrong depth
    scale; a large residual is a table the depth sensor does not see flatly, which is worth knowing before the cubes
    ever arrive. Returns metres and degrees; the caller decides what is acceptable."""
    u, v = corners_uv[:, 0], corners_uv[:, 1]
    u0, u1, v0, v1 = int(u.min()), int(np.ceil(u.max())), int(v.min()), int(np.ceil(v.max()))
    patch = depth_m[max(v0, 0):v1 + 1, max(u0, 0):u1 + 1]
    ys, xs = np.mgrid[max(v0, 0):max(v0, 0) + patch.shape[0], max(u0, 0):max(u0, 0) + patch.shape[1]]
    m = np.isfinite(patch) & (patch > 0)
    if m.sum() < 100: return dict(error="too few valid depth pixels inside the board", valid_fraction=float(m.mean()))
    z = patch[m].astype(np.float64)
    x = (xs[m] - K[0, 2]) / K[0, 0] * z
    y = (ys[m] - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], 1)
    c = P.mean(0)
    # Smallest eigenvector of the 3x3 covariance, not an SVD of the point matrix: a board patch is tens of thousands of
    # pixels and np.linalg.svd would build a full N x N left-singular matrix for a normal that needs three numbers.
    n = np.linalg.eigh((P - c).T @ (P - c))[1][:, 0]
    if n @ c > 0: n = -n                                  # point it back towards the camera
    resid = np.abs((P - c) @ n)
    board_n = T_camera_board[:3, 2]
    if board_n @ c > 0: board_n = -board_n
    return dict(plane_point=c.tolist(), plane_normal=n.tolist(),
                valid_fraction=round(float(m.mean()), 4), n_points=int(m.sum()),
                plane_rms_mm=round(float(np.sqrt(np.mean(resid ** 2))) * 1000, 3),
                plane_p95_mm=round(float(np.percentile(resid, 95)) * 1000, 3),
                tilt_vs_board_deg=round(float(np.degrees(np.arccos(np.clip(abs(n @ board_n), -1, 1)))), 3),
                mean_distance_m=round(float(np.linalg.norm(c)), 4))


def mount_geometry(T_table_camera: np.ndarray) -> dict:
    """Height, tilt and heading of the camera, read out of the transform rather than off a tape measure."""
    axis = T_table_camera[:3, 2]                       # the camera's optical axis, in table coordinates
    # Roll is rotation about the optical axis: how far the picture is turned from level with the table, and the same
    # number whatever the camera is pointing at. Two wrong versions preceded this one. Built from the camera's +Y it
    # read 170 deg for a rig 4.5 deg off level -- +Y is image *down* in the OpenCV convention. Taken as the image x
    # axis tipping out of the table plane it was well behaved but shrank with pitch, reporting 7.1 deg for a camera
    # rolled 10 deg at 45 deg of pitch, because that is a projection rather than the angle itself.
    # "Level right" is forward x up, not up x forward. Getting that backwards flips the reference by half a turn, which
    # is how a rig 9.7 deg off level came out at 170.3 deg -- and ray-casting image rows onto the table said plainly
    # that the camera was upright: the top of the frame lands 55 cm from the camera's footprint and the bottom 6 cm.
    level_right = np.cross(axis, [0.0, 0.0, 1.0])              # horizontal, perpendicular to where the camera looks
    n = np.linalg.norm(level_right)
    if n < 1e-9:                                               # straight down: every direction is level, roll is free
        roll = 0.0
    else:
        level_right = level_right / n
        x = T_table_camera[:3, 0]
        roll = float(np.degrees(np.arctan2(np.dot(np.cross(level_right, x), axis), np.dot(level_right, x))))
    return dict(height_cm=round(float(T_table_camera[2, 3]) * 100, 1),
                board_distance_cm=round(float(np.linalg.norm(T_table_camera[:2, 3])) * 100, 1),
                pitch_deg=round(float(np.degrees(np.arcsin(np.clip(-axis[2], -1, 1)))), 1),
                yaw_deg=round(float(np.degrees(np.arctan2(axis[1], axis[0]))), 1),
                roll_deg=round(roll, 1))


def fuse_plane(T_camera_board: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Replace the board's plane with the depth map's, keeping the board's origin and in-plane rotation.

    A board is a small, sometimes slightly bowed piece of paper; the table is the whole scene. Solving orientation from
    a plane fitted to thousands of depth pixels is far better conditioned than solving it from one board pose, but depth
    cannot say where the origin is or which way X points -- a plane has three degrees of freedom and a frame needs six.
    So take roll, pitch and height from the depth plane, and origin and yaw from the board."""
    n = np.asarray(plane_normal, np.float64); n = n / np.linalg.norm(n)
    if n @ T_camera_board[:3, 2] < 0: n = -n                  # agree with the board on which way is up
    o = T_camera_board[:3, 3]
    o = o - ((o - np.asarray(plane_point, np.float64)) @ n) * n     # drop the board origin onto the plane
    x = T_camera_board[:3, 0] - (T_camera_board[:3, 0] @ n) * n     # and its X axis into it
    nx = np.linalg.norm(x)
    if nx < 1e-9: raise ValueError("board X axis is parallel to the plane normal")
    x = x / nx
    return make_T(np.stack([x, np.cross(n, x), n], axis=1), o)


def z_out_of_the_table(T_table_camera: np.ndarray) -> np.ndarray:
    """Force the table frame's Z to point out of the surface, which operationally means towards the camera.

    Whether solvePnP returns the board with its normal facing the camera or away from it depends on the target and the
    ordering of its object points, and the fused plane inherits whichever it got. The first real calibration came out at
    height -48.3 cm and pitch -62.5 deg -- a camera under the table looking up -- from a solve whose reprojection was
    0.36 px and whose depth plane agreed with the board to 0.46 deg. The numbers were right and the frame was upside
    down, so pin it here instead of leaving it to the target: a holder above a table is the only case there is, and a
    camera at negative height is that frame, not that mount.

    The correction is a half turn about X, which keeps the in-plane X axis the board defined and stays right-handed."""
    T = np.asarray(T_table_camera, np.float64)
    if T[2, 3] >= 0: return T
    return np.diag([1.0, -1.0, -1.0, 1.0]) @ T


def _intrinsics_from_file(which: str) -> tuple[np.ndarray, np.ndarray, str]:
    """K, D from a versioned calibration file, for a camera that does not report its own.

    The Orbbec handed its intrinsics over with every capture, so this tool read them out of the episode. A C922 does
    not: its K has to be measured once and carried, and a table frame solved with a guessed focal length is a silent
    metric-scale error in every table coordinate downstream -- which is exactly what this tool refuses to allow."""
    from ..pose.calibration import load_versioned
    stem, _, ver = which.partition(":")
    # load_versioned's `which` is the FULL file stem, not the bare version.
    v, d = load_versioned(stem, f"{stem}_{ver}" if ver else "latest", None)
    if not d:
        from ..pose.calibration import versions
        have = [q.stem for q in versions(stem, None)]
        raise SystemExit(f"no calibration {stem!r} version {ver or 'latest'} in configs/calibration/"
                         + (f" — have {have}" if have else " — none at all"))
    if str(d.get("status", "")).upper() == "VOID":
        raise SystemExit(f"{v} is marked VOID: {d.get('voided_reason', '')}")
    K = np.asarray(d["K"], np.float64).reshape(3, 3)
    D = np.asarray(d["D"], np.float64).reshape(-1)
    note = f"{v} (rms {d.get('rms_px')}, {d.get('views')} views, status {d.get('status', 'unset')})"
    print(f"intrinsics: {note}")
    if str(d.get("status", "")).upper() == "PROVISIONAL":
        print("  NOTE: provisional intrinsics — the table frame inherits their scale uncertainty")
    return K, D, note


def _intrinsics_from_episode(ep: Path, stream: str) -> tuple[np.ndarray, np.ndarray, float]:
    d = json.loads((ep / f"{stream}_depth" / "intrinsics.json").read_text())
    i = d.get("intrinsics", d)
    K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], np.float64)
    scale = float(d.get("depth_scale", 1.0)) / 1000.0 if d.get("depth_scale", 0) > 1 else float(d.get("depth_scale", 1e-3))
    return K, np.zeros(5), scale


def main(argv=None, detector=None) -> int:
    """`detector` is a callable (bgr) -> (image_points Nx2, object_points Nx3) | None. Default: a plain checkerboard."""
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-episode", help="recorded episode; uses its head stream and its own intrinsics")
    src.add_argument("--live", action="store_true", help="grab from the head camera now (needs sudo on macOS)")
    ap.add_argument("--stream", default="head_depth", help="stream / camera name; `head` for the C922 rig")
    ap.add_argument("--intrinsics", default=None, metavar="STEM[:vNNN]",
                    help="versioned calibration file to take K and D from (e.g. head_c922_intrinsics). Required for "
                         "any camera that does not report its own intrinsics, which is every plain RGB camera.")
    ap.add_argument("--hardware", default=None,
                    help="hardware profile to open the live camera through, so it gets the profile's frozen UVC "
                         "preset and a MEASURED OpenCV index instead of a listing position")
    ap.add_argument("--out-stem", default=None,
                    help="calibration file stem to write (default head_mount; use head_mount_c922 for the C922 rig)")
    ap.add_argument("--rows", type=int, default=None, help="inner corners down (checkerboard only)")
    ap.add_argument("--cols", type=int, default=None, help="inner corners across (checkerboard only)")
    ap.add_argument("--square-mm", type=float, default=None, help="checkerboard only")
    ap.add_argument("--board-name", default=None, help="what to record as the board when a detector is injected")
    ap.add_argument("--plane-from-depth", action="store_true",
                    help="take roll/pitch/height from a plane fitted to the depth map and only origin/yaw from the board")
    ap.add_argument("--frames", type=int, default=20, help="--live only")
    ap.add_argument("--max-views", type=int, default=30)
    for name, helptext in (("height-cm", "camera height above the table"), ("edge-cm", "distance from the table edge"),
                           ("pitch-deg", "downward tilt"), ("yaw-deg", ""), ("roll-deg", "")):
        ap.add_argument(f"--{name}", type=float, default=None, help=helptext)
    ap.add_argument("--remount", action="store_true", help="append this solve to the latest calibration's repeatability record")
    ap.add_argument("--write", action="store_true", help="write configs/calibration/head_mount_vNNN.yaml")
    ap.add_argument("--notes", default="")
    a = ap.parse_args(argv)

    import cv2
    if detector is None:
        if None in (a.rows, a.cols, a.square_mm):
            raise SystemExit("--rows, --cols and --square-mm are required for the default checkerboard target")
        detector = CheckerboardDetector(a.rows, a.cols, a.square_mm / 1000.0)
    views: list[tuple[np.ndarray, float]] = []
    last_uv = last_T = None
    depth_report: dict | None = None
    K = D = None

    # Loose stills are deliberately not an input: they carry no intrinsics, and a guessed focal length is a silent
    # metric-scale error in every table coordinate downstream. Both sources here bring the camera's own.
    intr_note = None
    if a.from_episode:
        ep = Path(a.from_episode).expanduser()
        if a.intrinsics:
            K, D, intr_note = _intrinsics_from_file(a.intrinsics); depth_unit = 1e-3
        else:
            K, D, depth_unit = _intrinsics_from_episode(ep, a.stream)
        cap = cv2.VideoCapture(str(ep / f"{a.stream}.mp4"))
        ddir = ep / f"{a.stream}_depth"
        i = 0
        while len(views) < a.max_views:
            ok, img = cap.read()
            if not ok: break
            r = detector(img)
            if r is not None:
                c, obj = r
                T, rms = solve_view(c, obj, K, D); views.append((T, rms)); last_uv, last_T = c, T
                dp = ddir / f"{i:06d}.png"
                if depth_report is None and dp.exists():
                    depth_m = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(np.float64) * depth_unit
                    depth_report = depth_plane_check(depth_m, K, c, T)
            i += 1
        cap.release()
    elif a.intrinsics:
        # Plain RGB camera: K and D come from the calibration file, the picture comes from the profile's camera. Going
        # through the profile is what supplies the frozen UVC preset (the head's focus lock) and a MEASURED OpenCV
        # index -- a listing position is not an index on this host, and opening the wrong camera here would produce a
        # table frame for a camera that is not the one recording.
        import time as _t
        from ..config import load_config
        from ..devices.manager import DeviceManager
        if not a.hardware:
            raise SystemExit("--intrinsics needs --hardware too, so the camera is opened the way the collector opens it")
        hw = load_config(hardware=a.hardware).hardware
        hw.cameras = [c for c in hw.cameras if c.name == a.stream]
        hw.imus, hw.grippers = [], []
        if not hw.cameras:
            raise SystemExit(f"profile {a.hardware!r} has no camera {a.stream!r}")
        mgr = DeviceManager(hw); mgr.build(); mgr.connect_all()
        if mgr.errors:
            raise SystemExit(f"camera {a.stream!r} did not open: {mgr.errors}")
        cam = mgr.cameras[a.stream]
        K, D, intr_note = _intrinsics_from_file(a.intrinsics)
        depth_unit = 1e-3
        print(f"grabbing up to {a.frames} frames — hold still")
        seen = 0
        while seen < a.frames and len(views) < a.max_views:
            _t.sleep(0.15)
            f = cam.latest()
            if f is None: continue
            seen += 1
            r = detector(f.image)
            if r is None: continue
            c, obj = r
            T, rms = solve_view(c, obj, K, D); views.append((T, rms)); last_uv, last_T = c, T
            print(f"\r  {len(views)} views", end="", flush=True)
        mgr.close_all(); print()
    else:
        from ..devices.camera import OrbbecCamera
        from ..config import CameraCfg
        cam = OrbbecCamera(CameraCfg(a.stream, "aux_depth", backend="orbbec", width=848, height=480, fps=30))
        cam.open(); cam.start()
        import time as _t
        intr = cam.intrinsics
        if not intr: cam.stop(); raise SystemExit("the camera reported no intrinsics — see check_rgbd_ready --live")
        K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64); D = np.zeros(5)
        depth_unit = float(cam.depth_scale or 1.0) / 1000.0
        print(f"grabbing up to {a.frames} frames — hold still")
        seen = 0
        while seen < a.frames and len(views) < a.max_views:
            _t.sleep(0.15)
            f = cam.buffer.latest()
            if f is None: continue
            seen += 1
            r = detector(f.image)
            if r is None: continue
            c, obj = r
            T, rms = solve_view(c, obj, K, D); views.append((T, rms)); last_uv, last_T = c, T
            if depth_report is None and f.depth is not None:
                depth_report = depth_plane_check(np.asarray(f.depth, np.float64) * depth_unit, K, c, T)
            print(f"\r  {len(views)} views", end="", flush=True)
        cam.stop(); print()

    if len(views) < 3:
        print(f"only {len(views)} usable views — need at least 3 (is the whole board visible and in focus?)", file=sys.stderr)
        return 1

    res = table_frame(views)
    moved = board_moved(res)
    if moved:
        print(f"REFUSING: {moved}", file=sys.stderr)
        return 1
    T = z_out_of_the_table(res["T_table_camera"])
    if a.plane_from_depth:
        if not depth_report or "plane_point" not in depth_report:
            print("--plane-from-depth: no usable depth plane was measured", file=sys.stderr); return 1
        fused = fuse_plane(res["T_camera_board"], depth_report["plane_point"], depth_report["plane_normal"])
        tilt = depth_report.get("tilt_vs_board_deg")
        print(f"plane taken from the depth map ({depth_report['n_points']} points, rms {depth_report['plane_rms_mm']:.2f} mm); "
              f"it disagreed with the board by {tilt:.2f} deg, which is the correction applied")
        T = z_out_of_the_table(inv_T(fused))
    print(f"views {res['n_views']}   rms reprojection {res['rms_reproj_px']:.3f} px   target {getattr(detector, 'name', 'injected')}")
    print(f"view spread: {res['view_spread']['translation_mm']:.3f} mm, {res['view_spread']['rotation_deg']:.4f} deg"
          " — a camera that did not move should agree to well under a millimetre")
    print("T_table_camera (camera expressed in the table frame):")
    for row in T: print("   " + "  ".join(f"{v:9.5f}" for v in row))
    print(f"camera sits {T[2,3]*100:.1f} cm above the table plane, {np.linalg.norm(T[:2,3])*100:.1f} cm from the board origin")
    if depth_report:
        print("depth cross-check:", json.dumps(depth_report))

    if a.write:
        from ..pose.calibration import load_head_mount, write_head_mount
        geometry = {k: getattr(a, k.replace("-", "_")) for k in ("height_cm", "edge_cm", "pitch_deg", "yaw_deg", "roll_deg")}
        geometry = {k: v for k, v in geometry.items() if v is not None}
        if not geometry:
            print("refusing to write without any mount geometry: a transform alone cannot restore a holder that comes off.\n"
                  "  pass at least --height-cm and --edge-cm", file=sys.stderr)
            return 1
        rep = None
        if a.remount:
            _, prev = load_head_mount(stem=a.out_stem or "head_mount")
            hist = list((prev.get("repeatability") or {}).get("solves") or [])
            if prev.get("T_table_camera"): hist.append(np.asarray(prev["T_table_camera"], float).tolist())
            hist.append(T.tolist())
            Ts = np.array(hist)
            m = mean_pose(Ts)
            rep = dict(n_remounts=len(hist), solves=hist,
                       translation_mm=round(float(np.max(np.linalg.norm(Ts[:, :3, 3] - m[:3, 3], axis=1))) * 1000, 2),
                       rotation_deg=round(float(max(rotation_angle_deg(t[:3, :3], m[:3, :3]) for t in Ts)), 3))
        board = dict(rows=a.rows, cols=a.cols, square_mm=a.square_mm, name=a.board_name or getattr(detector, "name", None),
                     plane_from_depth=bool(a.plane_from_depth))
        p = write_head_mount(T, geometry=geometry, board={k: v for k, v in board.items() if v is not None},
                             rms_reproj_px=res["rms_reproj_px"], camera=a.stream, depth_check=depth_report,
                             repeatability=rep, view_spread=res["view_spread"], notes=a.notes,
                             source=a.from_episode or "live", stem=a.out_stem or "head_mount",
                             intrinsics_source=intr_note)
        print(f"wrote {p}")
    else:
        print("(dry run — pass --write, with --height-cm/--edge-cm, to store it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
