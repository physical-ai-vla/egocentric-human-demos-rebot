from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from ego_collector.camera.calibration import CharucoSpec, calibrate_pinhole, detect_charuco, draw_detection, pose_diversity_ok
from ego_collector.camera.capture import CameraCapture, CaptureConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ego_collector calibrate-camera", description="ChArUco intrinsics for the head camera -> configs/camera/<name>.yaml")
    p.add_argument("--camera", default="0")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--name", default="c922")
    p.add_argument("--charuco", type=Path, default=Path("configs/charuco.yaml"))
    p.add_argument("--views", type=int, default=25)
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between auto-captures")
    p.add_argument("--max-mean-error-px", type=float, default=0.8)
    p.add_argument("--rational", action="store_true", help="8-coefficient rational distortion model")
    p.add_argument("--output", type=Path, default=None, help="Default configs/camera/<name>.yaml")
    p.add_argument("--from-video", type=Path, default=None, help="Calibrate from a recorded video instead of the live camera")
    return p


def _frames_live(args):
    cfg = CaptureConfig(index=int(args.camera) if str(args.camera).isdigit() else args.camera, width=args.width, height=args.height, fps=args.fps)
    with CameraCapture(cfg) as cam:
        while True:
            f = cam.get(timeout=1.0)
            if f is not None:
                yield f.image


def _frames_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, img = cap.read()
        if not ok:
            return
        yield img


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    spec = CharucoSpec.from_yaml(args.charuco) if args.charuco.exists() else CharucoSpec()
    out_path = args.output or Path("configs/camera") / f"{args.name}.yaml"
    accepted = []
    last = 0.0
    size = None
    frames = _frames_video(args.from_video) if args.from_video else _frames_live(args)
    print(f"Show the {spec.squares_x}x{spec.squares_y} ChArUco board in {args.views} varied views (centre, corners, tilted, near, far). q aborts.")
    for img in frames:
        size = (img.shape[1], img.shape[0])
        det = detect_charuco(img, spec)
        now = time.monotonic()
        auto = args.from_video is not None or now - last >= args.interval
        if det is not None and auto and pose_diversity_ok(det, accepted):
            accepted.append(det)
            last = now
            print(f"  view {len(accepted)}/{args.views} ({det.count} corners)")
        shown = draw_detection(img, det)
        cv2.putText(shown, f"{len(accepted)}/{args.views}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("calibrate-camera", cv2.resize(shown, None, fx=0.5, fy=0.5))
        if (cv2.waitKey(1) & 0xFF) == ord("q") or len(accepted) >= args.views:
            break
    cv2.destroyAllWindows()
    if size is None or len(accepted) < 10:
        raise SystemExit(f"only {len(accepted)} views; need >= 10")
    intr = calibrate_pinhole(accepted, size, camera_name=args.name, rational=args.rational)
    intr.extra["charuco"] = spec.to_dict()
    print(f"RMS {intr.rms_reprojection_error:.3f} px, mean {intr.mean_reprojection_error:.3f} px, {intr.num_views} views")
    print(f"K = fx {intr.fx:.1f} fy {intr.fy:.1f} cx {intr.cx:.1f} cy {intr.cy:.1f}; dist {intr.distortion_coefficients.round(4).tolist()}")
    if intr.mean_reprojection_error > args.max_mean_error_px:
        raise SystemExit(f"mean reprojection {intr.mean_reprojection_error:.3f} > {args.max_mean_error_px}; not saved (more/better views)")
    intr.save(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
