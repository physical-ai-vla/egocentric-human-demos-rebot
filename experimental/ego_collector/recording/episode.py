"""Episode directory layout (LEVEL 0 raw + derived sub-folders).

::

    datasets/raw/episode_000001/
    ├── head.mp4                   raw video, source of truth (never rewritten)
    ├── frame_timestamps.parquet   frame_index, timestamp_ns (time.monotonic_ns at capture)
    ├── metadata.json
    ├── tracking/  apriltags.parquet camera_pose.parquet wrist_pose.parquet
    ├── hands/     hand_pose.parquet
    ├── actions/   pseudo_actions.parquet (world frame) pseudo_actions_head.parquet (head/camera frame) bimanual_features.parquet
    └── qa/        report.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EPISODE_RE = re.compile(r"^episode_(\d{6})$")


@dataclass(frozen=True)
class EpisodePaths:
    root: Path

    @property
    def video(self) -> Path:
        return self.root / "head.mp4"

    @property
    def timestamps(self) -> Path:
        return self.root / "frame_timestamps.parquet"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def apriltags(self) -> Path:
        return self.root / "tracking" / "apriltags.parquet"

    @property
    def camera_pose(self) -> Path:
        return self.root / "tracking" / "camera_pose.parquet"

    @property
    def wrist_pose(self) -> Path:
        return self.root / "tracking" / "wrist_pose.parquet"

    @property
    def finger_pose(self) -> Path:
        return self.root / "tracking" / "finger_pose.parquet"

    @property
    def hand_pose(self) -> Path:
        return self.root / "hands" / "hand_pose.parquet"

    @property
    def hands3d_pose(self) -> Path:
        return self.root / "hands3d" / "wilor_pose.parquet"

    @property
    def hands3d_overlay(self) -> Path:
        return self.root / "hands3d" / "overlay.mp4"

    @property
    def hands3d_plots(self) -> Path:
        return self.root / "hands3d" / "trajectories.png"

    @property
    def pseudo_actions(self) -> Path:
        return self.root / "actions" / "pseudo_actions.parquet"

    @property
    def pseudo_actions_head(self) -> Path:
        return self.root / "actions" / "pseudo_actions_head.parquet"

    @property
    def bimanual_features(self) -> Path:
        return self.root / "actions" / "bimanual_features.parquet"

    @property
    def qa_report(self) -> Path:
        return self.root / "qa" / "report.json"

    @property
    def curation_dir(self) -> Path:
        return self.root / "curation"

    @property
    def episode_id(self) -> int:
        m = EPISODE_RE.match(self.root.name)
        return int(m.group(1)) if m else -1

    def read_metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata.read_text()) if self.metadata.exists() else {}

    def write_metadata(self, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata.write_text(json.dumps(data, indent=2, sort_keys=True))

    def update_metadata(self, **fields: Any) -> dict[str, Any]:
        data = self.read_metadata()
        data.update(fields)
        self.write_metadata(data)
        return data


def episode_dir_name(episode_id: int) -> str:
    return f"episode_{episode_id:06d}"


def next_episode_id(raw_root: Path) -> int:
    raw_root = Path(raw_root)
    ids = [int(m.group(1)) for p in raw_root.glob("episode_*") if (m := EPISODE_RE.match(p.name))]
    return (max(ids) + 1) if ids else 1


def list_episodes(raw_root: Path) -> list[EpisodePaths]:
    raw_root = Path(raw_root)
    return [EpisodePaths(p) for p in sorted(raw_root.glob("episode_*")) if EPISODE_RE.match(p.name)]


__all__ = ["EpisodePaths", "episode_dir_name", "list_episodes", "next_episode_id"]
