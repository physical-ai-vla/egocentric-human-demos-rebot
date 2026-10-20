"""Performance benchmarks (2026-10-14, spec 38): p50 / p90 / p99, never only the mean.

    detector  fingertip student on a recorded wrist video          (~/finger_distill/.venv/bin/python, ultralytics 8.4)
    ahrs      MahonyAhrs.update per sample (capacity vs the 200 Hz IMU)
    ik        ArmIk.solve per arm per tick, pyroki rebot_b601      (~/xvla-mac/bin/python)
    tick      DualArmController.tick incl. IK, mock bridge          (~/xvla-mac/bin/python for pyroki; toy otherwise)
Usage: python -m visual_teleop.tools.benchmark {detector,ahrs,ik,tick} [--video PATH] [--n N]"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np


def pct(x) -> dict:
    x = np.asarray(x, np.float64); return dict(n=int(len(x)), p50=float(np.percentile(x, 50)), p90=float(np.percentile(x, 90)),
                                               p99=float(np.percentile(x, 99)), max=float(x.max()))


def bench_detector(video: str, n: int) -> dict:
    import cv2
    from ..vision.fingertip_detector import StudentYoloDetector
    det = StudentYoloDetector(); cap = cv2.VideoCapture(str(video)); frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    for f in frames[:5]: det.detect(f)                                   # warm-up (MPS graph build)
    ms, hits = [], 0
    for f in frames:
        t = time.perf_counter(); r = det.detect(f); ms.append((time.perf_counter() - t) * 1e3); hits += int(r.detected)
    return dict(stage="fingertip_detector_student", ms=pct(ms), fps_p50=1000 / np.percentile(ms, 50), detected_frac=hits / max(1, len(frames)),
                frame_shape=list(frames[0].shape) if frames else None)


def bench_ahrs(n: int) -> dict:
    from ..imu.ahrs import MahonyAhrs
    a = MahonyAhrs(); rng = np.random.default_rng(0)
    for k in range(300): a.update(k * 5000, [0.0, 0, 0], [0, 0, 9.81])
    us = []
    for k in range(300, 300 + n):
        g = rng.normal(0, 0.3, 3); acc = np.array([0, 0, 9.81]) + rng.normal(0, 0.1, 3)
        t = time.perf_counter(); a.update(k * 5000, g, acc); us.append((time.perf_counter() - t) * 1e6)
    return dict(stage="ahrs_update", us=pct(us), max_rate_hz_p99=1e6 / np.percentile(us, 99))


def bench_ik(n: int) -> dict:
    from scipy.spatial.transform import Rotation
    from .. import config as vcfg
    from ..robot.ik import ArmIk, pyroki_pair
    c = vcfg.load(); S = pyroki_pair(); out = {}
    for side, s in S.items():
        ik = ArmIk.from_cfg(c, s); q = s.q_full[s.idx6].astype(np.float64); T0 = s.fk(q); ms = []
        for k in range(10): ik.solve(q, T0)
        for k in range(n):
            T = T0.copy(); T[:3, 3] += 0.03 * np.array([np.sin(k / 20), np.cos(k / 25) - 1, np.sin(k / 30)])
            T[:3, :3] = T0[:3, :3] @ Rotation.from_rotvec([0, 0, 0.3 * np.sin(k / 40)]).as_matrix()
            t = time.perf_counter(); v = ik.solve(q, T); ms.append((time.perf_counter() - t) * 1e3)
            if v.ok: q = v.q
        out[side] = pct(ms)
    return dict(stage="ik_pyroki_per_arm", ms=out)


def bench_tick(n: int, ik: str) -> dict:
    from .. import config as vcfg
    from ..build import build_controller
    from ..fusion.wrist_pose_fusion import HandEstimate, HandStatus
    from ..gripper.pinch_detector import PinchState
    from ..robot.bridge import MockBridge
    from ..robot.ik import ToyIk, toy_limits, pyroki_pair
    from ..safety.deadman import ScriptedDeadman
    from ego_teleop.transforms.se3 import make_T
    c = vcfg.load(overrides={"control": {"arms_enabled": {"left": True, "right": True}}})
    if ik == "pyroki":
        S = pyroki_pair(); br = MockBridge(q={s: S[s].q_full[S[s].idx6].astype(np.float64) for s in S}); jl = None
    else:
        S = {s: ToyIk() for s in ("left", "right")}; jl = {s: toy_limits() for s in S}
        br = MockBridge(q={"left": np.array([0.3, 0.2, 0.15, 0, 0, 0.0]), "right": np.array([0.3, -0.2, 0.15, 0, 0, 0.0])})
    ctl = build_controller(c, bridge=br, solvers=S, deadman=ScriptedDeadman(), joint_limits=jl)
    H = lambda s, x: HandEstimate(s, 0, HandStatus.TRACKING, [], make_T(np.eye(3), [x, 0, 0]), None, None, 1.0, "imu", 0.0)
    ctl.initialise({s: H(s, 0.0) for s in ("left", "right")}); P = {s: PinchState(False, "OK", 60.0, 0.0) for s in ("left", "right")}
    ms = []
    for k in range(n):
        x = 0.02 * np.sin(k / 30); t = time.perf_counter(); ctl.tick({s: H(s, x) for s in ("left", "right")}, P, int(k * 1e9 / 30))
        ms.append((time.perf_counter() - t) * 1e3)
    return dict(stage=f"controller_tick_{ik}", ms=pct(ms[10:]), counters=ctl.counters)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["detector", "ahrs", "ik", "tick"]); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--video", default="~/finger_distill/rec_01.mp4"); ap.add_argument("--ik", default="toy", choices=["toy", "pyroki"])
    a = ap.parse_args()
    from pathlib import Path
    r = {"detector": lambda: bench_detector(str(Path(a.video).expanduser()), a.n), "ahrs": lambda: bench_ahrs(a.n * 10),
         "ik": lambda: bench_ik(a.n), "tick": lambda: bench_tick(a.n, a.ik)}[a.what]()
    print(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
