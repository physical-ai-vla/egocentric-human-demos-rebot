"""Measure the camera<->IMU temporal offset per side from processed episodes (visual angular speed from the pose backend
vs |gyro|), and print the pose.yaml `offsets:` block to paste. Never assumed zero.

    python -m handumi_collector.tools.camera_imu_offset SESSION_OR_EPISODE [--backend opencv_vo] [--max-offset-ms 100]"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from ..config import DEFAULT_CONFIG_DIR, load_pose_cfg
from ..pose.episode_io import RawEpisode, derived_dir, read_table
from ..pose.se3 import poses7_to_T
from ..pose.timing import estimate_camera_imu_offset, fit_device_to_host, imu_host_times_ns
from .pose_process import episodes_under


def offset_for_episode(ep_path: Path, side: str, backend: str, *, max_offset_ms: float = 100.0) -> dict:
    ep = RawEpisode.load(ep_path)
    cam = read_table(derived_dir(ep_path, backend) / f"{side}_camera_pose")
    ok = cam["valid"].values.astype(bool)
    t = cam["t_ns"].values[ok]; P = cam[["x", "y", "z", "qx", "qy", "qz", "qw"]].values[ok]
    imu = ep.imu[side]; clock = fit_device_to_host(imu.device_us, imu.host_ns)
    t_imu = imu_host_times_ns(imu, clock, 0)                     # raw clock mapping, offset 0 => result IS the offset
    qa = json.loads((derived_dir(ep_path, backend) / "pose_qa.json").read_text())
    bias_dps = qa["sides"].get(side, {}).get("gyro_bias_dps")   # still-window bias from the processed episode, when detected
    r = estimate_camera_imu_offset(t, poses7_to_T(P)[:, :3, :3], t_imu, imu.gyro, max_offset_ms=max_offset_ms,
                                   gyro_bias=None if bias_dps is None else np.radians(bias_dps))
    r.update(episode=ep_path.name, side=side); return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--backend", default=None); ap.add_argument("--max-offset-ms", type=float, default=100.0)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR / "pose.yaml")); ap.add_argument("--min-confidence", type=float, default=0.2)
    a = ap.parse_args(argv); cfg = load_pose_cfg(a.config); backend = a.backend or cfg.backend
    res = {s: [] for s in cfg.sides}
    for ep in episodes_under(Path(a.path)):
        for s in cfg.sides:
            try: r = offset_for_episode(ep, s, backend, max_offset_ms=a.max_offset_ms)
            except FileNotFoundError: print(f"{ep.name} {s}: no processed poses (run tools.pose_process first)"); continue
            print(f"{ep.name} {s}: offset {r.get('offset_ms')} ms  conf {r.get('confidence', 0):.2f}  {r.get('reason', '')}")
            if r.get("offset_ms") is not None and r.get("confidence", 0) >= a.min_confidence: res[s].append(r["offset_ms"])
    block = {"measured": all(len(v) > 0 for v in res.values())}
    for s, v in res.items():
        block[f"{s}_camera_imu_offset_ms"] = float(np.median(v)) if v else None
        if v: print(f"{s}: median {np.median(v):.1f} ms over {len(v)} episodes (std {np.std(v):.1f})")
    print("\npaste into configs/handumi/pose.yaml:\noffsets:")
    for k, v in block.items(): print(f"  {k}: {json.dumps(v)}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
