from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ego_collector.camera.capture import CaptureConfig
from ego_collector.recording.recorder import Recorder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ego_collector record", description="Record raw egocentric episodes (head.mp4 + timestamps + metadata).")
    p.add_argument("--camera", default="0", help="OpenCV index or device path; 'orbbec' = Orbbec Gemini 336 RGB-D (color->head.mp4 + aligned depth/*.png; needs sudo on macOS)")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--task", required=True, help="Task id, e.g. single_cube_pick_place")
    p.add_argument("--instruction", default=None, help="Language instruction (default: task id)")
    p.add_argument("--layout", default="", help="Collection-plan layout id (e.g. L03); stored in metadata, changeable with I")
    p.add_argument("--order", default="", help="Collection-plan stacking order (e.g. RBP); stored in metadata")
    p.add_argument("--operator", default="op01")
    p.add_argument("--root", type=Path, default=Path("datasets/raw"))
    p.add_argument("--camera-calibration", default="configs/camera/c922.yaml")
    p.add_argument("--tag-family", default=None)
    p.add_argument("--no-preview", action="store_true", help="Headless; requires --duration")
    p.add_argument("--duration", type=float, default=None, help="Record one episode for N seconds and exit")
    p.add_argument("--start-delay", type=float, default=2.0, help="Warm-up before the episode starts (C922 auto-exposure settles in ~1 s)")
    b = p.add_argument_group("timed batch collection")
    b.add_argument("--batch", type=int, default=None, help="Auto-record N episodes: record-s of recording, rest-s pause (reset the scene), repeat. R starts the batch, D discards+retakes, Q quits.")
    b.add_argument("--record-s", type=float, default=10.0)
    b.add_argument("--rest-s", type=float, default=10.0)
    b.add_argument("--orders", default="RBP,RPB,BRP,BPR,PRB,PBR", help="Order cycle for the batch (episode i gets orders[i %% len]); '' disables")
    b.add_argument("--colors", default="red,blue,purple", help="Cube colors matching the order initials")
    p.add_argument("--preview-scale", type=float, default=0.5)
    p.add_argument("--threaded-capture", action="store_true", help="Grab frames on a background thread even in the preview UI (default on macOS: main thread, avoids AVFoundation/HighGUI deadlocks)")
    p.add_argument("--diag", action="store_true", help="Print loop / grab / show timings every second (freeze diagnosis)")
    m = p.add_argument_group("live monitors (preview only, never on the recording path)")
    m.add_argument("--monitor-tags", dest="monitor_tags", action="store_true", default=True)
    m.add_argument("--no-monitor-tags", dest="monitor_tags", action="store_false")
    m.add_argument("--monitor-hands", action="store_true", help="Also run MediaPipe on a downscaled preview (~6 Hz)")
    m.add_argument("--wrist-extrinsics", type=Path, default=Path("configs/wrist_extrinsics.yaml"))
    m.add_argument("--world-map", type=Path, default=Path("configs/world_tag_map.yaml"))
    m.add_argument("--world", action="store_true", help="Count world tags in the live ALL-REQUIRED ratio (you put the 4 tags down)")
    m.add_argument("--backend", choices=("auto", "pupil", "opencv"), default="auto")
    m.add_argument("--quad-decimate", type=float, default=2.0, help="pupil-apriltags speed/accuracy knob for the live monitor")
    m.add_argument("--hand-model", type=Path, default=Path("models/hand_landmarker.task"))
    return p


def build_monitors(args):
    from ego_collector.recording.live_monitor import LiveHandMonitor, LiveTagMonitor
    from ego_collector.tracking.detector import make_detector
    from ego_collector.tracking.wrist_pose import WristExtrinsics

    tag_monitor = hand_monitor = None
    if args.monitor_tags:
        wr = WristExtrinsics.from_yaml(args.wrist_extrinsics) if args.wrist_extrinsics.exists() else WristExtrinsics.default("tag36h11")
        wm = None
        if args.world and args.world_map.exists():
            from ego_collector.tracking.world_pose import WorldTagMap

            wm = WorldTagMap.from_yaml(args.world_map)
        kwargs = {"quad_decimate": args.quad_decimate} if args.backend != "opencv" else {}
        try:
            det = make_detector(wr.tag_family, args.backend, **kwargs)
        except TypeError:
            det = make_detector(wr.tag_family, args.backend)
        intr = focal = None
        cal = Path(args.camera_calibration)
        if cal.exists():
            from ego_collector.camera.intrinsics import CameraIntrinsics

            intr = CameraIntrinsics.load(cal)
            focal = intr.fx
        tag_monitor = LiveTagMonitor(
            None, det,
            left_id=wr.sides["left"].tag_id, right_id=wr.sides["right"].tag_id,
            world_ids=wm.ids if wm is not None else (),
            wrist_size_m=wr.sides["left"].tag_size_m, focal_px=focal,
            extrinsics=wr, intr=intr, world_map=wm,  # live wrist pose + grip width on the panel (needs the camera calibration)
        )
    if args.monitor_hands:
        hand_monitor = LiveHandMonitor(None, model_path=args.hand_model)
    return tag_monitor, hand_monitor


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    index: int | str = int(args.camera) if str(args.camera).isdigit() else args.camera
    interactive = args.duration is None
    threaded = (not interactive) or args.threaded_capture or sys.platform != "darwin"
    from ego_collector.recording.recorder import BatchTimer

    batch = BatchTimer(args.batch, args.record_s, args.rest_s) if args.batch else None
    batch_orders = [o.strip().upper() for o in args.orders.split(",") if o.strip()] if (args.batch and args.orders) else []
    colors = {c.strip()[0].upper(): c.strip() for c in args.colors.split(",") if c.strip()}
    rec = Recorder(
        raw_root=args.root,
        capture=CaptureConfig(index=index, width=args.width, height=args.height, fps=args.fps, fourcc=args.fourcc, threaded=threaded),
        diag=args.diag,
        task=args.task,
        instruction=args.instruction or args.task,
        layout=args.layout,
        order=args.order.upper(),
        operator_id=args.operator,
        camera_calibration=args.camera_calibration,
        tag_family=args.tag_family,
        preview=not args.no_preview,
        preview_scale=args.preview_scale,
        batch=batch,
        batch_orders=batch_orders,
        colors=colors,
    )
    if not args.no_preview:
        rec.tag_monitor, rec.hand_monitor = build_monitors(args)
        rec.show_tags = rec.tag_monitor is not None
        rec.show_hands = rec.hand_monitor is not None
    if args.duration is not None:
        out = rec.run_timed(args.duration, start_delay_s=args.start_delay)
        print(out)
    else:
        if args.no_preview:
            raise SystemExit("--no-preview needs --duration (no keyboard without a window)")
        for path in rec.run_interactive():
            print(path)


if __name__ == "__main__":
    main()
