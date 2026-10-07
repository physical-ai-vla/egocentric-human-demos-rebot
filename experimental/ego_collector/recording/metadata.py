"""Episode metadata (LEVEL 0). Outcome labels are added after recording; nothing is deleted."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

OUTCOMES = ("unknown", "success", "failure", "partial")
DATASET_TIERS = ("unprocessed", "action_labelled", "video_only")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EpisodeMetadata:
    episode_id: int
    task: str
    instruction: str
    operator_id: str = "op01"
    camera: str = "Logitech C922"
    resolution: list[int] = field(default_factory=lambda: [1920, 1080])
    fps_target: int = 30
    outcome: str = "unknown"
    recovery: bool = False
    dataset_tier: str = "unprocessed"
    recorded_at: str = field(default_factory=now_iso)
    camera_index: int = 0
    camera_calibration: str | None = None
    tag_family: str | None = None
    notes: str = ""
    layout: str = ""  # workspace layout id from the collection plan (e.g. L03)
    order: str = ""  # stacking order code from the collection plan (e.g. RBP)
    # filled in by the recorder when the episode is closed
    frame_count: int = 0
    dropped_frames: int = 0
    duration_s: float = 0.0
    fps_measured: float = 0.0
    video_codec: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeMetadata":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


__all__ = ["DATASET_TIERS", "OUTCOMES", "EpisodeMetadata", "now_iso"]
