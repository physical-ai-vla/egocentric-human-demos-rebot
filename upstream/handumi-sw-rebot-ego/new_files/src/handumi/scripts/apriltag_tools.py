#!/usr/bin/env python3
"""AprilTag backend utilities: print tags, preview head tracking, calibrate UMI bundle geometry.

::

    handumi tracking apriltag print --out outputs/tags                  # UMI tags 10,11,20,21 + world tags
    handumi tracking apriltag preview                                   # live: world / left / right overlay
    handumi tracking apriltag bundle-calib --side left --frames 40      # fit T_anchor_tag for the 2nd face
    handumi tracking apriltag solve-image frame.png                     # offline sanity check on one image
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.cameras.usb import make_camera_device
from handumi.config import DEFAULT_RIG_CONFIG, load_rig_config
from handumi.robots.utils import IDENTITY_POSE7
from handumi.scripts.setup.calibrate_head_camera import DEFAULT_SPATIAL, head_camera_spec, load_head_intrinsics
from handumi.tracking.apriltag import (
    DEFAULT_BUNDLES_PATH,
    DEFAULT_WORLD_MAP_PATH,
    AprilTagDetector,
    BundleConfig,
    CameraGeometry,
    WorldMap,
    build_provider_from_rig,
    calibrate_bundle_geometry,
    draw_results,
    process_frame,
)

log = logging.getLogger("handumi.apriltag_tools")


def _identity_cal() -> ControllerTcpCalibration:
    return ControllerTcpCalibration(left=IDENTITY_POSE7.copy(), right=IDENTITY_POSE7.copy())


def _write_tag(det: AprilTagDetector, tag_id: int, size_m: float, label: str, out_dir: Path, dpi: int) -> Path:
    px_per_m = dpi / 0.0254
    tag_px = int(round(size_m * px_per_m))
    border = int(round(tag_px * 0.25))
    marker = det.marker_image(tag_id, tag_px)
    canvas = np.full((tag_px + 2 * border, tag_px + 2 * border), 255, dtype=np.uint8)
    canvas[border : border + tag_px, border : border + tag_px] = marker
    cv2.putText(canvas, f"{label} id{tag_id} {size_m*1000:.0f}mm", (5, canvas.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    path = out_dir / f"tag36h11_{tag_id:03d}_{label}_{int(round(size_m*1000))}mm.png"
    cv2.imwrite(str(path), canvas)
    print(f"{path}  ({tag_px}px black square @ {dpi} dpi = {size_m*1000:.1f} mm; print at 100%)")
    return path


def cmd_print(args: argparse.Namespace) -> int:
    config = BundleConfig.from_yaml(args.bundles)
    det = AprilTagDetector(config.dictionary)
    args.out.mkdir(parents=True, exist_ok=True)
    for side, bundle in config.bundles.items():
        for spec in bundle.tags:
            _write_tag(det, spec.id, spec.size_m, side, args.out, args.dpi)
    for tid in [int(v) for v in args.world_ids.split(",") if v.strip()]:
        _write_tag(det, tid, args.world_size, "world", args.out, args.dpi)
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    provider = build_provider_from_rig(args.rig_config, _identity_cal(), spatial_path=args.spatial, bundles_path=args.bundles, world_map_path=args.world_map)
    provider.start()
    last_print = time.monotonic()
    try:
        while True:
            image, result = provider.latest_frame()
            sample = provider.latest()
            if image is None or result is None:
                time.sleep(0.02)
                continue
            shown = draw_results(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), result, provider.geometry)
            now = time.monotonic()
            if now - last_print >= 1.0:
                head = sample.hmd_pose[:3]
                print(
                    f"world={int(sample.hmd_tracked)} head@{np.round(head, 3)}  "
                    f"L trk={int(sample.left_tracked)} {np.round(sample.left_controller_pose[:3], 3)}  "
                    f"R trk={int(sample.right_tracked)} {np.round(sample.right_controller_pose[:3], 3)}  "
                    f"frames={provider.frames_processed} err={provider.last_error}"
                )
                last_print = now
            cv2.imshow("AprilTag head tracking (q quits)", shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        provider.stop()
        cv2.destroyAllWindows()
    return 0


def cmd_bundle_calib(args: argparse.Namespace) -> int:
    rig = load_rig_config(args.rig_config)
    geometry = CameraGeometry.for_intrinsics(load_head_intrinsics(args.spatial))
    config = BundleConfig.from_yaml(args.bundles)
    bundle = config.bundles[args.side]
    det = AprilTagDetector(config.dictionary)
    camera = make_camera_device(head_camera_spec(rig))
    camera.connect()
    frames = []
    last_seq = None
    others = [t.id for t in bundle.tags if t.id != bundle.anchor_tag]
    print(f"Hold the {args.side} HandUMI so anchor tag {bundle.anchor_tag} AND {others} are visible; vary orientation. Need {args.frames} frames.")
    try:
        while len(frames) < args.frames:
            s = camera.sample_at(None)
            if s.sequence == last_seq:
                time.sleep(0.005)
                continue
            last_seq = s.sequence
            prepared = geometry.prepare(s.image)
            dets = det.detect(prepared)
            ids = {d.id for d in dets}
            shown = cv2.cvtColor(prepared, cv2.COLOR_RGB2BGR)
            for d in dets:
                cv2.polylines(shown, [d.corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            cv2.putText(shown, f"{len(frames)}/{args.frames} ids={sorted(ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("bundle calibration", shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if bundle.anchor_tag in ids and len(ids & set(bundle.tag_ids)) >= 2:
                frames.append(dets)
                time.sleep(args.interval_s)
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()
    if len(frames) < 5:
        raise SystemExit(f"Only {len(frames)} co-visible frames; need >= 5.")
    fixed, metrics = calibrate_bundle_geometry(frames, bundle, geometry, max_reproj_px=args.max_reproj_px)
    for tag_id, m in metrics.items():
        print(f"  tag {tag_id}: {m}")
    bad = [k for k, m in metrics.items() if m.get("translation_rms_mm", 0) > args.max_rms_mm]
    if bad:
        raise SystemExit(f"Bundle residual too large for tags {bad} (> {args.max_rms_mm} mm); not saved.")
    new_bundles = dict(config.bundles)
    new_bundles[args.side] = fixed
    out = replace(config, bundles=new_bundles).to_dict()
    out["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["metrics"] = {**((yaml.safe_load(args.bundles.read_text()) or {}).get("metrics") or {}), args.side: metrics}
    args.bundles.write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"Saved {args.side} bundle geometry -> {args.bundles}")
    return 0


def cmd_solve_image(args: argparse.Namespace) -> int:
    geometry = CameraGeometry.for_intrinsics(load_head_intrinsics(args.spatial))
    config = BundleConfig.from_yaml(args.bundles)
    world = WorldMap.from_yaml(args.world_map) if args.world_map.exists() else None
    det = AprilTagDetector(config.dictionary)
    image = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2RGB)
    result = process_frame(image, detector=det, config=config, geometry=geometry, world_map=world)
    print(f"detected ids: {sorted(d.id for d in result.detections)}  corner_space={result.corner_space}")
    per_side = {"world": result.world, **result.hands}
    for side, r in per_side.items():
        if r is None:
            continue
        if r.solve is None:
            print(f"  {side}: not solved ({len(r.detections)} tags)")
        else:
            print(f"  {side}: tags={r.solve.tag_ids} reproj={r.solve.reproj_px:.2f}px T_cam_anchor={np.round(r.solve.camera_from_anchor, 4)}")
    wfc = result.world_from_camera
    if wfc is not None:
        print(f"  head pose in world: {np.round(wfc, 4)}")
    if args.out:
        cv2.imwrite(str(args.out), draw_results(cv2.cvtColor(geometry.prepare(image), cv2.COLOR_RGB2BGR), result, geometry))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    p.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    p.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLES_PATH)
    p.add_argument("--world-map", type=Path, default=DEFAULT_WORLD_MAP_PATH)
    sub = p.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("print", help="Write printable tag PNGs (UMI bundles + world tags).")
    pr.add_argument("--out", type=Path, default=Path("outputs/tags"))
    pr.add_argument("--dpi", type=int, default=300)
    pr.add_argument("--world-ids", default="100,101,102,103")
    pr.add_argument("--world-size", type=float, default=0.08)
    pr.set_defaults(func=cmd_print)
    pv = sub.add_parser("preview", help="Live world/left/right overlay from the head camera.")
    pv.set_defaults(func=cmd_preview)
    bc = sub.add_parser("bundle-calib", help="Fit T_anchor_tag for the non-anchor faces of one hand.")
    bc.add_argument("--side", choices=("left", "right"), required=True)
    bc.add_argument("--frames", type=int, default=40)
    bc.add_argument("--interval-s", type=float, default=0.25)
    bc.add_argument("--max-reproj-px", type=float, default=1.0)
    bc.add_argument("--max-rms-mm", type=float, default=3.0)
    bc.set_defaults(func=cmd_bundle_calib)
    si = sub.add_parser("solve-image", help="Detect + solve on a saved image.")
    si.add_argument("image", type=Path)
    si.add_argument("--out", type=Path, default=None)
    si.set_defaults(func=cmd_solve_image)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
