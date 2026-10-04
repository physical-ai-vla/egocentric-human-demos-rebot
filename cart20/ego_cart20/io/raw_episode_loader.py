"""Raw ego episode format (source-independent; written by a source exporter such as sources/handumi_export.py).

<raw_root>/<episode_id>/raw_episode.json
    episode_id, source, task, instruction, stack_order, clock ("capture_monotonic_ns"),
    videos {camera: absolute mp4 path}, tool_frame (how the hand TCP axes were fixed), gripper_calibration, provenance
<raw_root>/<episode_id>/raw_episode.npz      (per arm, own clock; LEFT and RIGHT are separate streams)
    {arm}_t_ns int64 (n)      {arm}_position (n,3) m      {arm}_quaternion (n,4) wxyz
    {arm}_valid bool (n)      {arm}_gripper (n)  normalized aperture, 0 = closed, 1 = open (calibrated, NOT clipped yet ok)
    cam_{camera}_t_ns int64   cam_{camera}_frame int64  (video frame index in the camera's mp4)
Raw data is preserved untouched; processing never writes into a raw episode directory.
"""
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np
from ..config import ARMS
from ..geometry.transforms import pose_to_T

CAMERAS = ("head", "left_wrist", "right_wrist")     # view 0 / 1 / 2


@dataclass
class RawArm:
    t_s: np.ndarray
    T: np.ndarray
    valid: np.ndarray
    gripper: np.ndarray


@dataclass
class RawEpisode:
    episode_id: str
    meta: dict
    arms: dict
    cams: dict = field(default_factory=dict)        # camera -> (t_s, frame_index)
    t0_ns: int = 0                                  # clock offset subtracted from every stream


REQUIRED_META = ("episode_id", "instruction", "stack_order", "task")


def load_raw_episode(ep_dir):
    ep_dir = pathlib.Path(ep_dir)
    meta = json.load(open(ep_dir / "raw_episode.json")); z = np.load(ep_dir / "raw_episode.npz")
    miss = [k for k in REQUIRED_META if not meta.get(k)]
    if miss: raise ValueError(f"{ep_dir}: raw_episode.json missing {miss}")
    t0 = int(min(int(z[f"{a}_t_ns"][0]) for a in ARMS))
    arms = {}
    for a in ARMS:
        t = (z[f"{a}_t_ns"].astype(np.int64) - t0) / 1e9
        arms[a] = RawArm(t, pose_to_T(z[f"{a}_position"], z[f"{a}_quaternion"]), z[f"{a}_valid"].astype(bool), z[f"{a}_gripper"].astype(np.float64))
    cams = {c: ((z[f"cam_{c}_t_ns"].astype(np.int64) - t0) / 1e9, z[f"cam_{c}_frame"].astype(np.int64))
            for c in CAMERAS if f"cam_{c}_t_ns" in z.files}
    return RawEpisode(meta["episode_id"], meta, arms, cams, t0)


def save_raw_episode(ep_dir, meta, arrays):
    """used by source exporters; refuses to overwrite"""
    ep_dir = pathlib.Path(ep_dir); ep_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(ep_dir / "raw_episode.npz", **arrays)
    json.dump(meta, open(ep_dir / "raw_episode.json", "w"), indent=1)
