"""3D wrist trajectories + grasp signals: Rerun when installed, Matplotlib fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ego_collector.io.parquet import read_parquet
from ego_collector.recording.episode import EpisodePaths

POSE = ("x", "y", "z", "qx", "qy", "qz", "qw")


def _load(paths: EpisodePaths) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    cam = read_parquet(paths.camera_pose) if paths.camera_pose.exists() else None
    wrist = read_parquet(paths.wrist_pose) if paths.wrist_pose.exists() else None
    actions = read_parquet(paths.pseudo_actions) if paths.pseudo_actions.exists() else None
    return cam, wrist, actions


def plot_matplotlib(paths: EpisodePaths, out: Path | None = None, show: bool = True) -> Path | None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam, wrist, actions = _load(paths)
    if wrist is None:
        raise SystemExit("run process-tags first")
    t = (wrist["timestamp_ns"].to_numpy() - wrist["timestamp_ns"].iloc[0]) / 1e9
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for side, color in (("left", "tab:orange"), ("right", "tab:blue")):
        m = wrist[f"{side}_wrist_valid"].to_numpy(dtype=bool)
        xyz = wrist[[f"{side}_wrist_x", f"{side}_wrist_y", f"{side}_wrist_z"]].to_numpy()
        ax.plot(xyz[m, 0], xyz[m, 1], xyz[m, 2], color=color, label=f"{side} wrist ({m.mean()*100:.0f}% valid)")
    if cam is not None:
        m = cam["world_pose_valid"].to_numpy(dtype=bool)
        xyz = cam[["head_x", "head_y", "head_z"]].to_numpy()
        ax.plot(xyz[m, 0], xyz[m, 1], xyz[m, 2], color="tab:green", alpha=0.6, label=f"head ({m.mean()*100:.0f}% valid)")
    ax.set_xlabel("X right [m]"); ax.set_ylabel("Y away [m]"); ax.set_zlabel("Z up [m]")
    ax.set_title(paths.root.name); ax.legend(loc="upper left", fontsize=8)
    ax2 = fig.add_subplot(2, 2, 2)
    for side, color in (("left", "tab:orange"), ("right", "tab:blue")):
        m = wrist[f"{side}_wrist_valid"].to_numpy(dtype=bool)
        ax2.plot(t, np.where(m, wrist[f"{side}_wrist_z"], np.nan), color=color, label=f"{side} z")
    ax2.set_ylabel("wrist z [m]"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(2, 2, 4, sharex=ax2)
    if actions is not None:
        ta = (actions["timestamp_ns"].to_numpy() - wrist["timestamp_ns"].iloc[0]) / 1e9
        for side, color in (("left", "tab:orange"), ("right", "tab:blue")):
            if f"{side}_grasp" in actions:
                ax3.plot(ta, actions[f"{side}_grasp"], color=color, label=f"{side} grasp")
        ax3.set_ylim(-0.05, 1.05)
    else:
        ax3.text(0.5, 0.5, "no pseudo actions yet (run generate-actions)", ha="center", transform=ax3.transAxes)
    ax3.set_xlabel("time [s]"); ax3.set_ylabel("grasp (1=open)"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out


def log_rerun(paths: EpisodePaths, *, spawn: bool = True, with_video: bool = True) -> None:
    import cv2
    import rerun as rr

    cam, wrist, actions = _load(paths)
    if wrist is None:
        raise SystemExit("run process-tags first")
    rr.init(f"ego_collector/{paths.root.name}", spawn=spawn)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    t0 = int(wrist["timestamp_ns"].iloc[0])
    cap = cv2.VideoCapture(str(paths.video)) if with_video else None
    for i, w in wrist.iterrows():
        rr.set_time("t", duration=(int(w["timestamp_ns"]) - t0) / 1e9)
        rr.set_time("frame", sequence=int(w["frame_index"]))
        for side, color in (("left", (255, 140, 40)), ("right", (40, 160, 255))):
            if bool(w[f"{side}_wrist_valid"]):
                p = np.array([w[f"{side}_wrist_{k}"] for k in POSE])
                rr.log(f"world/{side}_wrist", rr.Transform3D(translation=p[:3], rotation=rr.Quaternion(xyzw=p[3:]), axis_length=0.05))
                rr.log(f"world/{side}_trail", rr.Points3D([p[:3]], colors=[color], radii=0.004))
        if cam is not None:
            c = cam.iloc[i]
            if bool(c["world_pose_valid"]):
                p = np.array([c[f"head_{k}"] for k in POSE])
                rr.log("world/head", rr.Transform3D(translation=p[:3], rotation=rr.Quaternion(xyzw=p[3:]), axis_length=0.08))
        if actions is not None and i < len(actions):
            for side in ("left", "right"):
                g = actions.iloc[i].get(f"{side}_grasp", np.nan)
                if np.isfinite(g):
                    rr.log(f"grasp/{side}", rr.Scalars(float(g)))
        if cap is not None:
            ok, img = cap.read()
            if ok:
                rr.log("camera/rgb", rr.Image(cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)))
    if cap is not None:
        cap.release()


__all__ = ["log_rerun", "plot_matplotlib"]
