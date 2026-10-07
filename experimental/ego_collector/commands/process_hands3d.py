"""Offline WiLoR 3D hand pose over recorded episodes -> pose parquets + overlay video + plots.

    ego-collector process-hands3d <ep...> [--overlay] [--plots] [--stride 1] [--device auto]

Writes hands3d/wilor_pose.parquet (raw MANO track) and tracking/{camera_pose,wrist_pose,
finger_pose}.parquet in the head-frame schema, so the rest of the pipeline is unchanged:

    ego-collector generate-actions <ep> --frame head --grasp-source tags
    ego-collector qa <ep>          # with require_world: false, sides: right in configs/qa.yaml
    ego-collector tracking-test <ep> --side right

--overlay renders hands3d/overlay.mp4 (skeleton + wrist + thumb-index + grip mm) - watch this
FIRST. --plots writes hands3d/trajectories.png (wrist xyz / rotation / aperture / confidence).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ego_collector.recording.episode import EpisodePaths


def plot_trajectories(paths: EpisodePaths, side: str = "right") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation

    from ego_collector.io.parquet import read_parquet

    df = read_parquet(paths.hands3d_pose)
    t = (df["timestamp_ns"].to_numpy() - df["timestamp_ns"].iloc[0]) / 1e9
    vis = df[f"{side}_visible"].to_numpy(bool)
    P = ("x", "y", "z", "qx", "qy", "qz", "qw")
    pose = df[[f"{side}_wrist_cam_{k}" for k in P]].to_numpy(dtype=np.float64)
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for j, lab in enumerate("xyz"):
        axes[0].plot(t, np.where(vis, pose[:, j], np.nan), label=f"wrist {lab} (cam)")
    axes[0].set_ylabel("m"), axes[0].legend(loc="upper right"), axes[0].set_title(f"{paths.root.name}  {side} wrist (camera frame) - watch for depth jumps")
    ang = np.full(len(df), np.nan)
    idx = np.flatnonzero(vis)
    for a, b in zip(idx[:-1], idx[1:]):
        ra, rb = Rotation.from_quat(pose[a, 3:7]), Rotation.from_quat(pose[b, 3:7])
        dt = max(t[b] - t[a], 1e-6)
        ang[b] = np.degrees((ra.inv() * rb).magnitude()) / dt
    axes[1].plot(t, ang, color="tab:purple")
    axes[1].set_ylabel("deg/s"), axes[1].set_title("wrist rotation speed - spikes = orientation flips")
    ap = df[f"{side}_aperture_m"].to_numpy(dtype=np.float64) * 1000
    axes[2].plot(t, np.where(vis, ap, np.nan), color="tab:green")
    axes[2].set_ylabel("mm"), axes[2].set_title("thumb-index aperture - should clearly separate open vs pinch")
    conf = df[f"{side}_confidence"].to_numpy(dtype=np.float64)
    axes[3].plot(t, conf, color="tab:gray")
    axes[3].plot(t, vis.astype(float), color="tab:red", alpha=0.4, label="visible")
    axes[3].set_ylabel("conf / visible"), axes[3].set_xlabel("s"), axes[3].legend(loc="upper right")
    fig.tight_layout()
    paths.hands3d_plots.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.hands3d_plots, dpi=110)
    plt.close(fig)
    return paths.hands3d_plots


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector process-hands3d", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", type=Path, nargs="+")
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame (quick quality preview)")
    p.add_argument("--hand-conf", type=float, default=0.3)
    p.add_argument("--focal-px", type=float, default=1400.0, help="Real camera focal in px (C922@1080p ~1400); fixes wilor depth scale")
    p.add_argument("--image-size", default="1920x1080")
    p.add_argument("--overlay", action="store_true", help="Write hands3d/overlay.mp4")
    p.add_argument("--plots", action="store_true", help="Write hands3d/trajectories.png")
    p.add_argument("--side", default="right", choices=("left", "right"), help="Side for --plots")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)
    from ego_collector.hands3d.wilor import WilorTracker, process_hands3d

    w, hgt = (int(v) for v in args.image_size.split("x"))
    tracker = WilorTracker(device=args.device, hand_conf=args.hand_conf, focal_px=args.focal_px, image_size=(w, hgt))
    for ep in args.episodes:
        paths = EpisodePaths(ep)
        s = process_hands3d(paths, tracker, stride=args.stride, progress=not args.no_progress, overlay=paths.hands3d_overlay if args.overlay else None)
        line = f"{ep.name}: L {s['left_hand_visible_fraction']*100:.1f}%  R {s['right_hand_visible_fraction']*100:.1f}% visible ({s['frames']} frames, stride {s['stride']})"
        if args.overlay:
            line += f"  overlay -> {paths.hands3d_overlay}"
        if args.plots:
            line += f"  plots -> {plot_trajectories(paths, args.side)}"
        print(line)


if __name__ == "__main__":
    main()
