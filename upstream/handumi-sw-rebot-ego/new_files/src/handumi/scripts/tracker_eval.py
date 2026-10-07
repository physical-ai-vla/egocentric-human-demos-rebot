#!/usr/bin/env python3
"""Phase-2 EEF tracker acceptance: capture a Quest pose stream, then grade it.

Milestone 1 of the reBot ego-bimanual pipeline is *not* a dataset — it is
"left/right HandUMI TCP trajectories are stable 6-DoF in the table frame".
This tool produces the numbers that decide that gate.

Workflow
--------
::

    # 1. Static jitter: clamp one HandUMI to the table, record 30 s.
    handumi tracking eval capture --seconds 30 --output outputs/eval/static.csv
    handumi tracking eval jitter outputs/eval/static.csv

    # 2. Return-to-point: press Enter each time the tip is back on point A.
    handumi tracking eval capture --output outputs/eval/return.csv
    handumi tracking eval return outputs/eval/return.csv --side left

    # 3. Left/right agreement: touch both tips on one dimple, press Enter, repeat.
    handumi tracking eval capture --output outputs/eval/agree.csv
    handumi tracking eval agreement outputs/eval/agree.csv

Add ``--session-calibration outputs/calibration/session.yaml`` to capture in
the table frame (then ``table-z`` is meaningful) and ``--robot rebot_b601`` /
``--tcp-calibration <yaml>`` to evaluate the *TCP* rather than the bare
controller anchor. Without a TCP calibration the ``lt_*``/``rt_*`` columns
equal the controller columns.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

from handumi.calibration.control_tcp import (
    ControllerTcpCalibration,
    calibration_path_for_robot_device,
    load_controller_tcp_calibration,
)
from handumi.calibration.spatial import session_table_from_device
from handumi.config import DEFAULT_RIG_CONFIG, load_rig_config
from handumi.eval.tracker_eval import (
    CaptureBuilder,
    Thresholds,
    TrackerCapture,
    dump_json,
    format_agreement,
    format_jitter,
    format_return,
    format_table_z,
    left_right_agreement,
    return_to_point,
    static_jitter,
    table_z,
)
from handumi.robots.utils import IDENTITY_POSE7

log = logging.getLogger("handumi.tracker_eval")


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class _MarkListener(threading.Thread):
    """Enter on stdin => new mark id, active for ``window_s`` seconds."""

    def __init__(self, window_s: float) -> None:
        super().__init__(daemon=True)
        self.window_s = window_s
        self._lock = threading.Lock()
        self._mark = 0
        self._until = 0.0
        self.stop_requested = False

    def run(self) -> None:
        while not self.stop_requested:
            line = sys.stdin.readline()
            if line == "":
                return
            if line.strip().lower() in {"q", "quit", "stop"}:
                self.stop_requested = True
                return
            with self._lock:
                self._mark += 1
                self._until = time.monotonic() + self.window_s
                print(f"  mark {self._mark} (hold {self.window_s:.1f}s)", flush=True)

    def current(self) -> int:
        with self._lock:
            return self._mark if time.monotonic() < self._until else 0

    @property
    def count(self) -> int:
        with self._lock:
            return self._mark


def _load_tcp_calibration(args: argparse.Namespace) -> ControllerTcpCalibration:
    path = args.tcp_calibration
    if path is None and args.robot:
        try:
            path, why = calibration_path_for_robot_device(args.robot, args.device)
            log.info("Controller->TCP calibration: %s", why)
        except Exception as exc:  # robot config missing/broken must not block V0.1
            log.warning("Could not resolve TCP calibration for %s: %s", args.robot, exc)
            path = None
    if path is not None and Path(path).exists():
        cal = load_controller_tcp_calibration(Path(path))
        log.info("Loaded controller->TCP calibration from %s", path)
        return cal
    if path is not None:
        log.warning("TCP calibration %s not found; using identity (TCP == controller).", path)
    else:
        log.info("No TCP calibration: TCP columns will equal controller columns.")
    return ControllerTcpCalibration(left=IDENTITY_POSE7.copy(), right=IDENTITY_POSE7.copy())


def _build_tracker(args: argparse.Namespace, calibration: ControllerTcpCalibration):
    # Reuse the recorder's provider factory so eval and recording see the same frames.
    from handumi.scripts.record import build_tracker

    ns = argparse.Namespace(
        device=args.device,
        rig_config=args.rig_config,
        quest_ip=args.quest_ip,
        tcp_port=args.tcp_port,
        sync_port=args.sync_port,
        pico_mode="mandos",
        pico_wifi=False,
        pico_adb=True,
        skip_adb_check=True,
    )
    tracker = build_tracker(ns, calibration, reset_workspace_on_x=False)
    if args.session_calibration is not None:
        table_from_device = session_table_from_device(Path(args.session_calibration))
        if hasattr(tracker, "set_workspace_from_device_pose"):
            tracker.set_workspace_from_device_pose(table_from_device, locked=True)
            log.info("Workspace locked to table frame from %s", args.session_calibration)
        else:
            log.warning("Tracker %s cannot lock a table frame; ignoring session calibration.", args.device)
    return tracker


def cmd_capture(args: argparse.Namespace) -> int:
    calibration = _load_tcp_calibration(args)
    tracker = _build_tracker(args, calibration)
    builder = CaptureBuilder()
    marks = _MarkListener(args.mark_window_s)
    poll_dt = 1.0 / args.poll_hz
    deadline = time.monotonic() + args.seconds if args.seconds > 0 else float("inf")
    last_seq = None
    last_print = 0.0
    print(
        "Capturing. Press Enter to drop a mark, type q + Enter (or Ctrl+C) to finish."
        + (f" Auto-stop after {args.seconds:.0f}s." if args.seconds > 0 else "")
    )
    tracker.start()
    if not args.no_marks:
        marks.start()
    try:
        while time.monotonic() < deadline and not marks.stop_requested:
            sample = tracker.latest()
            seq = int(sample.sequence)
            if sample.streaming and seq != last_seq:
                last_seq = seq
                builder.add(
                    t_ns=int(sample.aligned_time_ns or sample.pc_monotonic_ns or time.monotonic_ns()),
                    mark=marks.current(),
                    streaming=bool(sample.streaming),
                    clock_synced=bool(sample.clock_synced),
                    left_tracked=bool(sample.left_tracked),
                    right_tracked=bool(sample.right_tracked),
                    left_controller=sample.left_controller_pose,
                    right_controller=sample.right_controller_pose,
                    left_tcp=sample.left_tcp_pose,
                    right_tcp=sample.right_tcp_pose,
                )
            now = time.monotonic()
            if now - last_print > 0.5:
                last_print = now
                lp = sample.left_tcp_pose[:3]
                rp = sample.right_tcp_pose[:3]
                print(
                    f"\r n={len(builder):6d} marks={marks.count:3d} conn={int(sample.connected)} "
                    f"stream={int(sample.streaming)} sync={int(sample.clock_synced)} | "
                    f"L trk={int(sample.left_tracked)} [{lp[0]:+.3f} {lp[1]:+.3f} {lp[2]:+.3f}] "
                    f"R trk={int(sample.right_tracked)} [{rp[0]:+.3f} {rp[1]:+.3f} {rp[2]:+.3f}]   ",
                    end="",
                    flush=True,
                )
            time.sleep(poll_dt)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        tracker.stop()
    capture = builder.build()
    if len(capture) == 0:
        print("No samples captured (Quest never streamed). Check IP/ports and the app.")
        return 2
    capture.write_csv(Path(args.output))
    print(
        f"Wrote {len(capture)} samples ({capture.duration_s:.1f}s @ {capture.sample_rate_hz:.1f} Hz, "
        f"{len(capture.mark_ids())} marks) -> {args.output}"
    )
    return 0


# ---------------------------------------------------------------------------
# Analysis subcommands
# ---------------------------------------------------------------------------


def _thresholds(args: argparse.Namespace) -> Thresholds:
    try:
        rig = load_rig_config(args.rig_config)
    except Exception:
        rig = {}
    return Thresholds.from_rig(rig)


def _finish(report, args: argparse.Namespace, text: str) -> int:
    print(text)
    if args.json:
        dump_json(report, Path(args.json))
        print(f"JSON -> {args.json}")
    return 0 if report.status == "PASS" else (1 if report.status == "WARN" else 2)


def cmd_jitter(args: argparse.Namespace) -> int:
    cap = TrackerCapture.read_csv(Path(args.csv))
    report = static_jitter(cap, kind=args.kind, thresholds=_thresholds(args))
    return _finish(report, args, format_jitter(report))


def cmd_return(args: argparse.Namespace) -> int:
    cap = TrackerCapture.read_csv(Path(args.csv))
    report = return_to_point(cap, side=args.side, kind=args.kind, thresholds=_thresholds(args))
    return _finish(report, args, format_return(report))


def cmd_agreement(args: argparse.Namespace) -> int:
    cap = TrackerCapture.read_csv(Path(args.csv))
    report = left_right_agreement(cap, mode=args.mode, kind=args.kind, thresholds=_thresholds(args))
    return _finish(report, args, format_agreement(report))


def cmd_table_z(args: argparse.Namespace) -> int:
    cap = TrackerCapture.read_csv(Path(args.csv))
    report = table_z(cap, kind=args.kind, thresholds=_thresholds(args))
    return _finish(report, args, format_table_z(report))


def cmd_summary(args: argparse.Namespace) -> int:
    cap = TrackerCapture.read_csv(Path(args.csv))
    print(
        f"{args.csv}: {len(cap)} samples, {cap.duration_s:.1f}s @ {cap.sample_rate_hz:.1f} Hz, "
        f"marks={cap.mark_ids()}, tracked L={cap.tracked['left'].mean()*100:.1f}% "
        f"R={cap.tracked['right'].mean()*100:.1f}%, synced={cap.clock_synced.mean()*100:.1f}%"
    )
    for side in ("left", "right"):
        p = cap.tcp[side][cap.tracked[side]]
        if len(p):
            lo, hi = p[:, :3].min(axis=0), p[:, :3].max(axis=0)
            print(f"  {side} TCP bbox min={np.round(lo,3)} max={np.round(hi,3)} (m)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_analysis(p: argparse.ArgumentParser) -> None:
    p.add_argument("csv", type=Path)
    p.add_argument("--kind", choices=("tcp", "controller"), default="tcp")
    p.add_argument("--json", type=Path, default=None, help="Also write the report as JSON.")
    p.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="Record a Quest pose stream to CSV with operator marks.")
    cap.add_argument("--output", type=Path, default=Path("outputs/eval/tracker_capture.csv"))
    cap.add_argument("--seconds", type=float, default=0.0, help="Auto-stop after N seconds (0 = until q/Ctrl+C).")
    cap.add_argument("--poll-hz", type=float, default=200.0)
    cap.add_argument("--mark-window-s", type=float, default=0.5)
    cap.add_argument("--no-marks", action="store_true", help="Do not read stdin (for non-interactive runs).")
    cap.add_argument("--device", choices=("meta", "pico", "apriltag"), default="meta")
    cap.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    cap.add_argument("--quest-ip", default=None)
    cap.add_argument("--tcp-port", type=int, default=None)
    cap.add_argument("--sync-port", type=int, default=None)
    cap.add_argument("--session-calibration", type=Path, default=None, help="Lock the table frame (session.yaml).")
    cap.add_argument("--robot", default=None, help="Resolve the controller->TCP file from configs/robots/<robot>.yaml.")
    cap.add_argument("--tcp-calibration", type=Path, default=None, help="Explicit controller->TCP yaml.")
    cap.set_defaults(func=cmd_capture)

    j = sub.add_parser("jitter", help="Static jitter: per-axis sigma, translation/rotation RMS.")
    _add_common_analysis(j)
    j.set_defaults(func=cmd_jitter)

    r = sub.add_parser("return", help="Return-to-point error across marks.")
    _add_common_analysis(r)
    r.add_argument("--side", choices=("left", "right"), required=True)
    r.set_defaults(func=cmd_return)

    a = sub.add_parser("agreement", help="Left/right tip separation at shared points.")
    _add_common_analysis(a)
    a.add_argument("--mode", choices=("simultaneous", "sequential"), default="simultaneous")
    a.set_defaults(func=cmd_agreement)

    z = sub.add_parser("table-z", help="Tip z at marks (needs session calibration).")
    _add_common_analysis(z)
    z.set_defaults(func=cmd_table_z)

    s = sub.add_parser("summary", help="Print basic capture statistics.")
    _add_common_analysis(s)
    s.set_defaults(func=cmd_summary)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
