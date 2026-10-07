#!/usr/bin/env python3
"""Calibrate the head-mounted tracking camera (Arducam B0202, 160° fisheye) and the world tag map.

::

    # ① intrinsics (fisheye model) -> outputs/calibration/spatial.yaml  cameras.head
    handumi calibrate head-camera intrinsics --views 20

    # ② world map: ChArUco board on the table defines the world/table frame
    #    (+X right, +Y away, +Z up, origin = board centre); fixed tags 100..103
    #    around the workspace are measured relative to it -> outputs/calibration/world_map.yaml
    handumi calibrate head-camera world-map --tag-ids 100,101,102,103 --tag-size 0.08 --frames 40

    # ②' no board: world frame := tag 100's frame
    handumi calibrate head-camera world-map --tag-ids 100,101,102,103 --tag-size 0.08 --anchor-tag 100

Redo ② whenever a world tag is moved. Head-camera motion is otherwise removed
per frame from the world tags, so nothing here is "per session".
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from handumi.calibration.spatial import (
    CameraIntrinsics,
    CharucoBoardSpec,
    board_from_table_pose,
    calibrate_fisheye,
    calibrate_pinhole,
    detect_charuco,
    draw_detection,
    estimate_board_pose,
    load_yaml,
    new_spatial_calibration,
    write_yaml,
)
from handumi.cameras.usb import make_camera_device
from handumi.config import DEFAULT_RIG_CONFIG, load_rig_config
from handumi.robots.utils import mat_to_pose7, pose7_to_mat

log = logging.getLogger("handumi.calibrate_head_camera")
DEFAULT_SPATIAL = Path("outputs/calibration/spatial.yaml")
DEFAULT_WORLD_MAP = Path("outputs/calibration/world_map.yaml")
HEAD_CAMERA = "head"
MIN_CORNERS = 12


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def head_camera_spec(rig: dict) -> dict:
    spec = (rig.get("cameras") or {}).get(HEAD_CAMERA)
    if not isinstance(spec, dict):
        raise SystemExit(f"configs/rig.yaml has no cameras.{HEAD_CAMERA} entry.")
    return {
        "id": spec.get("index_or_path", 0),
        "type": spec.get("type", "opencv"),
        "width": int(spec.get("width", 1920)),
        "height": int(spec.get("height", 1080)),
        "fps": int(spec.get("fps", 30)),
    }


def load_head_intrinsics(spatial_path: Path = DEFAULT_SPATIAL) -> CameraIntrinsics:
    spatial = load_yaml(spatial_path)
    cam = (spatial.get("cameras") or {}).get(HEAD_CAMERA)
    if not isinstance(cam, dict):
        raise SystemExit(f"{spatial_path} has no cameras.{HEAD_CAMERA}; run `handumi calibrate head-camera intrinsics`.")
    return CameraIntrinsics.from_dict(cam)


def _loop_frames(camera, *, title: str, on_frame, max_frames: int, interval_s: float):
    """Show frames; call ``on_frame(image_rgb) -> (accepted: bool, overlay_bgr)`` at most every ``interval_s``."""
    accepted = 0
    last = 0.0
    last_seq = None
    while accepted < max_frames:
        s = camera.sample_at(None)
        if s.sequence == last_seq:
            time.sleep(0.004)
            continue
        last_seq = s.sequence
        now = time.monotonic()
        ok, overlay = on_frame(s.image, now - last >= interval_s)
        if ok:
            accepted += 1
            last = now
        cv2.putText(overlay, f"{accepted}/{max_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow(title, overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyWindow(title)
    return accepted


def cmd_intrinsics(args: argparse.Namespace) -> int:
    rig = load_rig_config(args.rig_config)
    board = CharucoBoardSpec.from_dict((rig.get("spatial_calibration") or {}).get("board"))
    spec = head_camera_spec(rig)
    camera = make_camera_device(spec)
    detections = []

    def on_frame(image_rgb, may_capture):
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        det = detect_charuco(bgr, board, min_corners=MIN_CORNERS)
        if det is not None and may_capture:
            detections.append(det)
            print(f"  view {len(detections)} ({det.count} corners)")
            return True, draw_detection(bgr, det)
        return False, draw_detection(bgr, det)

    camera.connect()
    try:
        print(f"Head intrinsics: show the board in {args.views} varied views (centre, edges, tilted, near/far). q aborts.")
        _loop_frames(camera, title="head intrinsics", on_frame=on_frame, max_frames=args.views, interval_s=args.interval_s)
    finally:
        camera.disconnect()
    if len(detections) < 10:
        raise SystemExit(f"Only {len(detections)} views; need >= 10.")
    size = (spec["width"], spec["height"])
    intrinsics = calibrate_fisheye(HEAD_CAMERA, detections, size) if args.model == "fisheye" else calibrate_pinhole(HEAD_CAMERA, detections, size)
    print(f"{args.model}: RMS {intrinsics.rms_px:.3f} px, mean {intrinsics.mean_error_px:.3f} px over {intrinsics.views} views")
    if intrinsics.mean_error_px > args.max_mean_error_px:
        raise SystemExit(f"Mean reprojection {intrinsics.mean_error_px:.3f} px > {args.max_mean_error_px}; not saved.")
    spatial = load_yaml(args.spatial) if args.spatial.exists() else new_spatial_calibration(board)
    spatial.setdefault("cameras", {})[HEAD_CAMERA] = intrinsics.to_dict()
    spatial["updated_at"] = _now_iso()
    write_yaml(args.spatial, spatial)
    print(f"Saved cameras.{HEAD_CAMERA} -> {args.spatial}")
    return 0


def cmd_world_map(args: argparse.Namespace) -> int:
    from handumi.tracking.apriltag import AprilTagDetector, BundleConfig, CameraGeometry, calibrate_world_map, solve_single_tag

    rig = load_rig_config(args.rig_config)
    board = CharucoBoardSpec.from_dict((rig.get("spatial_calibration") or {}).get("board"))
    intrinsics = load_head_intrinsics(args.spatial)
    geometry = CameraGeometry.for_intrinsics(intrinsics)
    det_intr = geometry.detection_intrinsics()
    dictionary = BundleConfig.from_yaml(args.bundles).dictionary if args.bundles.exists() else "DICT_APRILTAG_36h11"
    detector = AprilTagDetector(dictionary)
    tag_ids = [int(v) for v in args.tag_ids.split(",") if v.strip()]
    tag_sizes = {i: float(args.tag_size) for i in tag_ids}
    board_table = np.linalg.inv(pose7_to_mat(board_from_table_pose(board)))  # T_table_board
    frames: list[tuple[list, np.ndarray]] = []
    use_board = args.anchor_tag is None

    def on_frame(image_rgb, may_capture):
        prepared = geometry.prepare(image_rgb)
        bgr = cv2.cvtColor(prepared, cv2.COLOR_RGB2BGR)
        dets = detector.detect(prepared)
        world_dets = [d for d in dets if d.id in tag_sizes]
        for d in world_dets:
            cv2.polylines(bgr, [d.corners.astype(np.int32).reshape(-1, 1, 2)], True, (60, 220, 60), 2)
        cam_world = None
        if use_board:
            det = detect_charuco(bgr, board, min_corners=MIN_CORNERS)
            bgr = draw_detection(bgr, det)
            if det is not None:
                try:
                    cam_board, err = estimate_board_pose(det, det_intr)
                    if err <= args.max_reproj_px:
                        cam_world = mat_to_pose7(pose7_to_mat(cam_board) @ np.linalg.inv(board_table))  # T_cam_table
                except ValueError:
                    cam_world = None
        else:
            anchor = next((d for d in world_dets if d.id == args.anchor_tag), None)
            if anchor is not None:
                try:
                    cam_world, err = solve_single_tag(anchor, tag_sizes[args.anchor_tag], geometry)
                    if err > args.max_reproj_px:
                        cam_world = None
                except ValueError:
                    cam_world = None
        ok = may_capture and cam_world is not None and len(world_dets) >= 1
        if ok:
            frames.append((world_dets, cam_world))
            print(f"  frame {len(frames)}: world tags {sorted(d.id for d in world_dets)}")
        cv2.putText(bgr, f"ref={'board' if use_board else 'tag'+str(args.anchor_tag)} {'OK' if cam_world is not None else '--'}  tags={sorted(d.id for d in world_dets)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return ok, bgr

    camera = make_camera_device(head_camera_spec(rig))
    camera.connect()
    try:
        print(f"World map: keep the reference ({'board' if use_board else 'tag ' + str(args.anchor_tag)}) and as many world tags as possible in view; move the head around. Need {args.frames} frames.")
        _loop_frames(camera, title="world map", on_frame=on_frame, max_frames=args.frames, interval_s=args.interval_s)
    finally:
        camera.disconnect()
    if len(frames) < 5:
        raise SystemExit(f"Only {len(frames)} frames; need >= 5.")
    frame_name = "table" if use_board else f"tag{args.anchor_tag}"
    world, metrics = calibrate_world_map(frames, tag_sizes, geometry, frame_name=frame_name, anchor_tag=args.anchor_tag, max_reproj_px=args.max_reproj_px)
    for tid, m in metrics.items():
        print(f"  tag {tid}: {m}")
    bad = [k for k, m in metrics.items() if m.get("translation_rms_mm", 0.0) > args.max_rms_mm]
    if bad:
        raise SystemExit(f"World-tag residual > {args.max_rms_mm} mm for {bad}; not saved.")
    world.write_yaml(args.output)
    print(f"Saved world map ({len(world.tag_ids)} tags, frame={frame_name}) -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    sub = parser.add_subparsers(dest="command", required=True)
    intr = sub.add_parser("intrinsics", help="Head camera intrinsics (fisheye by default).")
    intr.add_argument("--views", type=int, default=20)
    intr.add_argument("--interval-s", type=float, default=1.5)
    intr.add_argument("--model", choices=("fisheye", "pinhole"), default="fisheye")
    intr.add_argument("--max-mean-error-px", type=float, default=1.0)
    intr.set_defaults(func=cmd_intrinsics)
    wm = sub.add_parser("world-map", help="Measure fixed world tags in the table (or anchor-tag) frame.")
    wm.add_argument("--tag-ids", default="100,101,102,103")
    wm.add_argument("--tag-size", type=float, default=0.08, help="World tag black-square size (m).")
    wm.add_argument("--anchor-tag", type=int, default=None, help="Use this tag's frame as world instead of the ChArUco board.")
    wm.add_argument("--frames", type=int, default=40)
    wm.add_argument("--interval-s", type=float, default=0.3)
    wm.add_argument("--max-reproj-px", type=float, default=1.5)
    wm.add_argument("--max-rms-mm", type=float, default=5.0)
    wm.add_argument("--bundles", type=Path, default=Path("configs/calibration/apriltag/bundles.yaml"))
    wm.add_argument("--output", type=Path, default=DEFAULT_WORLD_MAP)
    wm.set_defaults(func=cmd_world_map)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
