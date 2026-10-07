"""Typed config for the RGB-D rigid-body tracker (configs/handumi/depth_pose.yaml).

Lives here rather than in config.py so the frozen collector config path is untouched. Every threshold the pipeline uses
is in the YAML — nothing is hard-coded in a code path, and the acceptance numbers are explicitly PROVISIONAL until the
first real pilot distribution is measured (§16)."""
from __future__ import annotations
import dataclasses as dc
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from ..config import REPO_ROOT, DEFAULT_CONFIG_DIR


@dataclass
class DepthPoseCfg:
    backend: str = "icp"
    sides: list[str] = field(default_factory=lambda: ["left", "right"])
    stream: str | None = None                 # depth stream name; None = the episode's only *_depth stream
    mesh: dict = field(default_factory=lambda: {"left": "assets/handumi/left_tracking_body.obj",
                                                "right": "assets/handumi/right_tracking_body.obj"})
    fps: int = 30
    z_range_m: list = field(default_factory=lambda: [0.10, 2.5])
    edge_trim_s: float = 0.15        # trimmed off each end of an operator-marked HOME window (settling)
    qa: dict = field(default_factory=lambda: {
        "valid_ratio": {"pass": 0.95, "warn": 0.90},
        "jumps": {"translation_mm": 50.0, "rotation_deg": 20.0},
        "static_jitter": {"pass_translation_mm": 5.0, "warn_translation_mm": 10.0,
                          "pass_rotation_deg": 2.0, "warn_rotation_deg": 5.0},
        "static_detect": {"translation_mm_per_frame": 1.5, "rotation_deg_per_frame": 0.5, "min_duration_s": 0.5},
        "lost": {"reject_if_lost_longer_than_s": 1.0},
        "depth_residual": {"warn_mm": 6.0, "reject_mm": 12.0},
        "home_return": {"pass": {"translation_mm": 15.0, "rotation_deg": 5.0},
                        "warn": {"translation_mm": 35.0, "rotation_deg": 10.0}},
    })
    backend_options: dict = field(default_factory=dict)

    def mesh_path(self, side: str) -> Path | None:
        p = self.mesh.get(side)
        if not p:
            return None
        p = Path(p)
        return p if p.is_absolute() else (REPO_ROOT / p)

    def options_for(self, backend: str) -> dict:
        return dict(self.backend_options.get(backend, {}))


def load_depth_pose_cfg(path: str | Path | None = None) -> DepthPoseCfg:
    path = Path(path) if path else DEFAULT_CONFIG_DIR / "depth_pose.yaml"
    if not path.exists():
        return DepthPoseCfg()
    d = yaml.safe_load(path.read_text()) or {}
    names = {f.name for f in dc.fields(DepthPoseCfg)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    return DepthPoseCfg(**d)
