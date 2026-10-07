"""Live view for setting up the fixed head camera and solving the table frame.

    sudo .venv/bin/python scripts/table_frame_ui.py --edge-cm 35

Lay the ChArUco board flat where the cubes will go and watch. Everything the calibration needs is drawn on the frame as
it happens -- whether the board is seen, how the camera sits above the table, whether the depth map agrees with it, and
whether depth is pointing where colour is -- so the mount can be adjusted against real numbers instead of adjusted,
recorded, and checked afterwards.

Most of what looked like hand measurement is not. Camera height, pitch, yaw and roll all fall out of the board pose, and
reading them off a live overlay is both easier and more consistent than a tape measure: they are then true by
construction rather than true to within how the tape was held. The exception is the distance from the table edge, which
the board cannot know because its origin is wherever it was put -- that one is `--edge-cm`, and it exists so someone can
physically restore the mount later.

Keys: SPACE grab a view   D depth view   W write the calibration   R reset   Q quit"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ego_collector.camera.calibration import CharucoSpec, detect_charuco     # noqa: E402  (outside the guarded package)
from handumi_collector.pose.rgbd_align import edge_alignment                 # noqa: E402
from handumi_collector.pose.se3 import inv_T                                 # noqa: E402
from handumi_collector.tools.calibrate_table_frame import (board_moved, depth_plane_check, fuse_plane,  # noqa: E402
                                                          mount_geometry, solve_view, table_frame,
                                                          z_out_of_the_table)

GREEN, RED, WHITE, AMBER = (120, 255, 120), (90, 90, 255), (255, 255, 255), (80, 200, 255)


def draw(view, lines, x=12, y=26, scale=0.52):
    for text, col in lines:
        cv2.putText(view, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
        y += int(24 * scale / 0.52)
    return view


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--squares-x", type=int, default=6); ap.add_argument("--squares-y", type=int, default=8)
    ap.add_argument("--square-mm", type=float, default=31.0); ap.add_argument("--marker-mm", type=float, default=23.0)
    ap.add_argument("--dictionary", default="DICT_5X5_50")
    ap.add_argument("--edge-cm", type=float, default=None, help="distance from the table edge — the one thing the board cannot tell you")
    ap.add_argument("--stream", default="head_depth")
    ap.add_argument("--notes", default="")
    a = ap.parse_args(argv)

    global cv2
    import cv2
    from handumi_collector.config import CameraCfg
    from handumi_collector.devices.camera import OrbbecCamera

    spec = CharucoSpec(a.squares_x, a.squares_y, a.square_mm / 1000.0, a.marker_mm / 1000.0, a.dictionary)
    cam = OrbbecCamera(CameraCfg(a.stream, "aux_depth", backend="orbbec", width=848, height=480, fps=30))
    cam.open(); cam.start()
    intr = cam.intrinsics
    if not intr:
        cam.stop(); raise SystemExit("the camera reported no intrinsics — see check_rgbd_ready --live")
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
    D = np.zeros(5); depth_unit = float(cam.depth_scale or 1.0) / 1000.0

    views: list[tuple[np.ndarray, float]] = []
    show_depth = False
    align = depth_rep = None
    frame_i = 0
    win = "table frame — SPACE grab   D depth   W write   R reset   Q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 1272, 720)

    try:
        while True:
            f = cam.buffer.latest()
            if f is None:
                if cv2.waitKey(30) & 0xFF == ord("q"): break
                continue
            frame_i += 1
            rgb = f.image
            depth_m = np.asarray(f.depth, np.float64) * depth_unit if f.depth is not None else None

            det = detect_charuco(rgb, spec, min_corners=6)
            T_cb = rms = None
            if det is not None:
                uv = det.image_points.reshape(-1, 2).astype(np.float64)
                obj = det.object_points.reshape(-1, 3).astype(np.float64)
                T_cb, rms = solve_view(uv, obj, K, D)
                if depth_m is not None and frame_i % 10 == 0:
                    depth_rep = depth_plane_check(depth_m, K, uv, T_cb)
            if depth_m is not None and frame_i % 20 == 0:
                align = edge_alignment(rgb, depth_m)

            view = rgb.copy()
            if show_depth and depth_m is not None:
                d = np.clip(depth_m, 0.2, 1.5)
                vis = cv2.applyColorMap(((d - 0.2) / 1.3 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                vis[depth_m <= 0] = (40, 40, 40)
                view = cv2.addWeighted(view, 0.35, vis, 0.65, 0)
            if det is not None:
                for p in det.image_points.reshape(-1, 2).astype(int):
                    cv2.circle(view, tuple(p), 3, GREEN, -1)
                cv2.drawFrameAxes(view, K, D, cv2.Rodrigues(T_cb[:3, :3])[0], T_cb[:3, 3], a.square_mm / 1000.0 * 3)

            lines = [(f"board: {det.count if det else 0} corners"
                      + (f"   reproj {rms:.3f} px" if rms is not None else "   — not detected"),
                      GREEN if det is not None else RED)]
            if T_cb is not None:
                g = mount_geometry(z_out_of_the_table(inv_T(T_cb)))
                lines.append((f"camera  height {g['height_cm']:6.1f} cm   pitch {g['pitch_deg']:+6.1f} deg   "
                              f"yaw {g['yaw_deg']:+6.1f}   roll {g['roll_deg']:+6.1f}", WHITE))
                lines.append((f"        {g['board_distance_cm']:.1f} cm horizontally from the board origin", WHITE))
            if depth_rep and "plane_rms_mm" in depth_rep:
                tilt = depth_rep["tilt_vs_board_deg"]
                lines.append((f"depth plane  rms {depth_rep['plane_rms_mm']:.1f} mm   tilt vs board {tilt:+.2f} deg   "
                              f"valid {depth_rep['valid_fraction']:.0%}", GREEN if tilt < 3 else AMBER))
            if align and "shift_px" in align:
                sx, sy = align["shift_px"]
                lines.append((f"rgb/depth  shift ({sx:+d},{sy:+d}) px   overlap {align['overlap']:.0%}",
                              GREEN if align["shift_magnitude_px"] <= 3 else AMBER))
            if len(views) >= 2:
                sp = table_frame(views)["view_spread"]
                moved = board_moved(dict(view_spread=sp))
                lines.append((f"views grabbed {len(views)}   spread {sp['translation_mm']:.1f} mm / "
                              f"{sp['rotation_deg']:.2f} deg" + ("   W to write" if not moved and len(views) >= 3 else ""),
                              RED if moved else (GREEN if len(views) >= 3 else AMBER)))
                if moved:
                    lines.append(("DO NOT MOVE THE BOARD between views — this is an extrinsic, not lens intrinsics.", RED))
                    lines.append(("R to reset and start again with it nailed down.", RED))
            else:
                lines.append((f"views grabbed {len(views)}   need 3, board must not move between them", AMBER))
            if a.edge_cm is None:
                lines.append(("--edge-cm not given: writing is refused, a transform cannot restore a mount", AMBER))
            draw(view, lines)
            cv2.imshow(win, view)

            k = cv2.waitKey(20) & 0xFF
            if k == ord("q"): break
            if k == ord("d"): show_depth = not show_depth
            if k == ord("r"): views.clear()
            if k == ord(" ") and T_cb is not None: views.append((T_cb, rms))
            if k == ord("w"):
                if len(views) < 3: print("need at least 3 grabbed views"); continue
                if a.edge_cm is None: print("pass --edge-cm to write"); continue
                res = table_frame(views)
                moved = board_moved(res)
                if moved:
                    print(f"refusing to write: {moved}"); continue
                T = z_out_of_the_table(res["T_table_camera"])
                if depth_rep and "plane_point" in depth_rep:
                    T = z_out_of_the_table(inv_T(fuse_plane(res["T_camera_board"], depth_rep["plane_point"], depth_rep["plane_normal"])))
                    print(f"plane taken from depth; it disagreed with the board by {depth_rep['tilt_vs_board_deg']:+.2f} deg")
                g = mount_geometry(T); g["edge_cm"] = a.edge_cm
                from handumi_collector.pose.calibration import write_head_mount
                p = write_head_mount(T, geometry=g, rms_reproj_px=res["rms_reproj_px"], camera=a.stream,
                                     board=dict(name=f"charuco {a.squares_x}x{a.squares_y} "
                                                     f"{a.square_mm:.0f}/{a.marker_mm:.0f}mm {a.dictionary}",
                                                plane_from_depth=bool(depth_rep and "plane_point" in depth_rep)),
                                     depth_check={k2: v for k2, v in (depth_rep or {}).items()
                                                  if k2 not in ("plane_point", "plane_normal")},
                                     view_spread=res["view_spread"], notes=a.notes, source="table_frame_ui --live")
                print(f"wrote {p}")
                print(f"  height {g['height_cm']} cm   pitch {g['pitch_deg']} deg   yaw {g['yaw_deg']}   roll {g['roll_deg']}")
                print(f"  views {res['n_views']}  rms {res['rms_reproj_px']:.3f} px  spread "
                      f"{res['view_spread']['translation_mm']:.2f} mm / {res['view_spread']['rotation_deg']:.3f} deg")
    finally:
        cam.stop(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
