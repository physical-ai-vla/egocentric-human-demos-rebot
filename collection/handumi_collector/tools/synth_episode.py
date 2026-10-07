"""Synthetic raw episode generator (no hardware): writes a complete raw episode directory — head.mp4, left/right_wrist.mp4
rendered from a 3D point scene along a known TCP trajectory (HOME still → manipulation → HOME still), sensors.mcap with
IMU (gyro/accel consistent with the trajectory + bias/noise, Teensy-style device clock), grip, frame_meta, events.json,
episode_meta.json — plus ground_truth.json. Used by tests and to exercise every pose backend before hardware exists.

    python -m handumi_collector.tools.synth_episode OUT_DIR --seconds 8 --camera-imu-offset-ms 12
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from ..collector.mcap_writer import SensorMcapWriter
from ..collector.video_writer import VideoStreamWriter
from ..collector.timeline import ClockAnchor
from ..devices.base import GripperSample, ImuSample
from ..pose.se3 import Ts_to_pose7, inv_T, make_T
from ..pose.backends.mock import home_manip_home_trajectory
from ..pose.episode_io import write_json

W, H = 640, 480
K_PIN = np.array([[400.0, 0, W / 2 - 0.5], [0, 400.0, H / 2 - 0.5], [0, 0, 1]])
INTRINSICS = dict(model="pinhole", K=K_PIN, D=np.zeros(4), image_size=(W, H))


def render_points(T_wc: np.ndarray, P: np.ndarray, K=K_PIN, size=(W, H)) -> np.ndarray:
    T_cw = inv_T(T_wc); Xc = (T_cw[:3, :3] @ P.T + T_cw[:3, 3:4]).T
    m = Xc[:, 2] > 0.1; uv = K @ Xc[m].T; uv = (uv[:2] / uv[2]).T
    img = np.zeros((size[1], size[0], 3), np.uint8); img[:] = (20, 20, 20)
    for (u, v), c in zip(uv, (P[m, 0] * 90 + 128).astype(int)):
        if 2 <= u < size[0] - 2 and 2 <= v < size[1] - 2: cv2.circle(img, (int(round(u)), int(round(v))), 2, (255, int(c) % 255, 255 - int(c) % 255), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def synth_episode(out: Path, *, seconds: float = 8.0, fps: int = 30, imu_hz: int = 400, still_s: float = 1.5, camera_imu_offset_ms: float = 0.0,
                  gyro_bias_dps=(0.4, -0.3, 0.2), gyro_noise_dps: float = 0.05, accel_noise: float = 0.02, seed: int = 0, order: str = "RBP",
                  home_events: bool = True, sides=("left", "right"), codec: str = "libx264") -> dict:
    rng = np.random.default_rng(seed); out = Path(out); out.mkdir(parents=True, exist_ok=True); (out / ".incomplete").touch()
    t_start = 10_000_000_000; t_stop = t_start + int(seconds * 1e9)
    anchor = ClockAnchor(t_start, int(time.time_ns()), t_start)
    trajs = {s: home_manip_home_trajectory(t_start, t_stop, still_s=still_s, amp_m=0.2 if s == "left" else 0.15, rot_deg=40 if s == "left" else -30) for s in sides}
    T_off = {s: make_T(Rotation.from_euler("y", 0.3 if s == "left" else -0.3).as_matrix(), [0.3 if s == "left" else -0.3, 0.1, 0.0]) for s in sides}
    P = np.c_[rng.uniform(-1.5, 1.5, 3000), rng.uniform(-1.0, 1.0, 3000), rng.uniform(0.6, 2.5, 3000)]
    streams = ["head", *[f"{s}_wrist" for s in sides]]
    mcap = SensorMcapWriter(out / "sensors.mcap", streams=streams)
    mcap.sync(t_start, dict(kind="episode_start", session_anchor=anchor.to_dict(), t_start_monotonic_ns=t_start, synthetic=True))
    vids = {st: VideoStreamWriter(out / f"{st}.mp4", width=W, height=H, fps=fps, codec=codec, bitrate_kbps=3000) for st in streams}
    n_frames = int(seconds * fps); gt = {s: [] for s in sides}
    for i in range(n_frames):
        t = t_start + int(i * 1e9 / fps) + int(rng.normal(0, 0.3e6))            # host capture jitter ~0.3 ms
        img_h = np.full((H, W, 3), 60, np.uint8); cv2.putText(img_h, f"{order} {i}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        vf = vids["head"].put(img_h); mcap.frame_meta("head", i, vf, t, 0)
        for s in sides:
            T = T_off[s] @ trajs[s](t); gt[s].append((t, T))
            vf = vids[f"{s}_wrist"].put(render_points(T, P)); mcap.frame_meta(f"{s}_wrist", i, vf, t, 0)
    # IMU: body rates from the trajectory (finite differences), accel = gravity in body + linear accel; camera==IMU frame here
    g_w = np.array([0, 0, -9.81]); n_imu = int(seconds * imu_hz); bias = np.radians(gyro_bias_dps)
    for s in sides:
        dev0 = 1_000_000 + (0 if s == "left" else 777_777)
        for k in range(n_imu):
            t_true = t_start + int(k * 1e9 / imu_hz)                              # true sample time on the camera clock
            dt = 1.0 / imu_hz; Ta, Tb = T_off[s] @ trajs[s](t_true), T_off[s] @ trajs[s](t_true + int(dt * 1e9))
            Tm = T_off[s] @ trajs[s](t_true - int(dt * 1e9))
            w = (Rotation.from_matrix(Tm[:3, :3]).inv() * Rotation.from_matrix(Tb[:3, :3])).as_rotvec() / (2 * dt)   # central difference, centred at t_true
            a_w = (Tb[:3, 3] - 2 * Ta[:3, 3] + Tm[:3, 3]) / dt ** 2
            a_b = Ta[:3, :3].T @ (a_w - g_w)
            gyro = w + bias + rng.normal(0, np.radians(gyro_noise_dps), 3); acc = a_b + rng.normal(0, accel_noise, 3)
            dev_us = dev0 + int(k * 1e6 / imu_hz)
            host = t_true - int(camera_imu_offset_ms * 1e6) + int(abs(rng.normal(0.4e6, 0.25e6)))   # receive latency (one-sided) ; offset: camera clock later by offset
            mcap.imu(ImuSample(s, k, dev_us, host, *acc.tolist(), *gyro.tolist(), 30.0))
        for k in range(int(seconds * 100)):
            t = t_start + int(k * 1e7); u = (t - t_start) / (t_stop - t_start)
            val = float(np.clip(1.0 - 0.9 * np.sin(np.pi * u) ** 2, 0, 1))
            mcap.gripper(GripperSample(s, k, t, int(1000 + 1000 * val), int(1000 + 1000 * val) % 4096, val, None))
    events = []
    if home_events:
        for kind, t in (("home_leave", t_start + int(still_s * 1e9)), ("home_return", t_stop - int(still_s * 1e9))):
            events.append(dict(t_ns=t, t_rel_s=(t - t_start) / 1e9, kind=kind, device="operator", detail={})); mcap.event(t, kind, "operator", {})
    mcap.sync(t_stop, dict(kind="episode_stop", t_stop_monotonic_ns=t_stop))
    for v in vids.values(): v.close()
    mcap.close()
    (out / "events.json").write_text(json.dumps(events, indent=1))
    meta = dict(schema="handumi_episode_meta/v1", episode_dir=out.name, order=order, instruction=f"synthetic {order}", status="KEEP", quality="PASS", notes="synthetic",
                t_start_monotonic_ns=t_start, t_stop_monotonic_ns=t_stop, duration_s=seconds, hw_event_in_episode=False, synthetic=True,
                streams={st: dict(frames=n_frames) for st in streams}, sensor_messages=dict(mcap.counts), n_events=len(events), event_kinds=sorted({e["kind"] for e in events}))
    (out / "episode_meta.json").write_text(json.dumps(meta, indent=1)); (out / ".incomplete").unlink()
    gt_out = dict(camera_imu_offset_ms=camera_imu_offset_ms, gyro_bias_dps=list(gyro_bias_dps), still_s=still_s, intrinsics=dict(model="pinhole", K=K_PIN.tolist(), D=[0, 0, 0, 0], image_size=[W, H]),
                  sides={s: dict(t_ns=[t for t, _ in gt[s]], pose7=Ts_to_pose7(np.array([T for _, T in gt[s]])).tolist(), T_offset=T_off[s].tolist()) for s in sides})
    write_json(gt_out, out / "ground_truth.json")
    return gt_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--camera-imu-offset-ms", type=float, default=0.0); ap.add_argument("--no-home-events", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    g = synth_episode(Path(a.out), seconds=a.seconds, camera_imu_offset_ms=a.camera_imu_offset_ms, home_events=not a.no_home_events, seed=a.seed)
    print("wrote", a.out, "offset_ms", g["camera_imu_offset_ms"]); return 0


if __name__ == "__main__": raise SystemExit(main())
