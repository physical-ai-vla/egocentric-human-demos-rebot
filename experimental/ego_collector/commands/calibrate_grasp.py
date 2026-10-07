"""Grasp calibration: the operator alternates OPEN pinch and CLOSED pinch in front of the camera.

Live mode (default): a window shows the landmarks; hold the pose and press O (open) or C (closed)
to record ~1 s of samples; repeat 3+ times each; S saves configs/grasp_calibration.yaml.
Offline mode: --from-episode <processed episode> --open a:b --closed c:d (seconds) uses hand_pose.parquet.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ego_collector.camera.capture import CameraCapture, CaptureConfig
from ego_collector.hands.grasp import GraspCalibration, aperture_features
from ego_collector.hands.mediapipe_tracker import DEFAULT_MODEL, HandLandmarker
from ego_collector.io.parquet import read_parquet, stack_column
from ego_collector.recording.episode import EpisodePaths

DEFAULT_OUT = Path("configs/grasp_calibration.yaml")


def _from_episode(args) -> GraspCalibration:
    paths = EpisodePaths(args.from_episode)
    hp = read_parquet(paths.hand_pose)
    t = (hp["timestamp_ns"].to_numpy() - hp["timestamp_ns"].iloc[0]) / 1e9

    def window(spec):
        a, b = spec.split(":")
        return (t >= float(a)) & (t <= float(b))

    open_s, closed_s = {}, {}
    for side in ("left", "right"):
        lm = stack_column(hp, f"{side}_landmarks_2d", 42).reshape(len(hp), 21, 2)
        vis = hp[f"{side}_hand_visible"].to_numpy(bool)
        norm = np.array([aperture_features(l)["aperture_norm"] if v else np.nan for l, v in zip(lm, vis)])
        open_s[side] = norm[window(args.open)]
        closed_s[side] = norm[window(args.closed)]
    return GraspCalibration.from_samples(open_s, closed_s)


def _live(args) -> GraspCalibration:
    lm = HandLandmarker(args.model)
    cfg = CaptureConfig(index=int(args.camera) if str(args.camera).isdigit() else args.camera, width=args.width, height=args.height, fps=30)
    samples = {"open": {"left": [], "right": []}, "closed": {"left": [], "right": []}}
    mode, mode_until = None, 0.0
    t0 = time.monotonic()
    with CameraCapture(cfg) as cam:
        while True:
            f = cam.get(timeout=1.0)
            if f is None:
                continue
            res = lm.detect(f.image, int((time.monotonic() - t0) * 1000))
            shown = f.image.copy()
            now = time.monotonic()
            if mode is not None and now > mode_until:
                mode = None
            for side, color in (("left", (255, 140, 40)), ("right", (40, 160, 255))):
                r = res[side]
                if not r.visible:
                    continue
                feat = aperture_features(r.landmarks_2d, r.landmarks_3d)
                for p in r.landmarks_2d:
                    cv2.circle(shown, (int(p[0]), int(p[1])), 3, color, -1)
                cv2.line(shown, tuple(r.landmarks_2d[4].astype(int)), tuple(r.landmarks_2d[8].astype(int)), (255, 255, 255), 2)
                cv2.putText(shown, f"{side} aperture_norm {feat['aperture_norm']:.2f}", (10, 70 if side == "left" else 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if mode is not None:
                    samples[mode][side].append(feat["aperture_norm"])
            counts = {m: {s: len(v) for s, v in d.items()} for m, d in samples.items()}
            cv2.putText(shown, f"{'RECORDING ' + mode.upper() if mode else 'O=open  C=closed  S=save  Q=quit'}  {counts}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("calibrate-grasp", cv2.resize(shown, None, fx=0.6, fy=0.6))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("o"), ord("O")):
                mode, mode_until = "open", now + args.hold
            elif key in (ord("c"), ord("C")):
                mode, mode_until = "closed", now + args.hold
            elif key in (ord("s"), ord("S"), ord("q"), ord("Q")):
                break
    cv2.destroyAllWindows()
    lm.close()
    return GraspCalibration.from_samples({s: np.array(v) for s, v in samples["open"].items()}, {s: np.array(v) for s, v in samples["closed"].items()})


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector calibrate-grasp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="0")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--hold", type=float, default=1.0, help="Seconds of samples per key press")
    p.add_argument("--from-episode", type=Path, default=None)
    p.add_argument("--open", default=None, help="a:b seconds of OPEN pinch (offline)")
    p.add_argument("--closed", default=None, help="a:b seconds of CLOSED pinch (offline)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    cal = _from_episode(args) if args.from_episode else _live(args)
    cal.save(args.output)
    print(f"left  open {cal.left_open:.3f} closed {cal.left_closed:.3f}\nright open {cal.right_open:.3f} closed {cal.right_closed:.3f}\n-> {args.output}")


if __name__ == "__main__":
    main()
