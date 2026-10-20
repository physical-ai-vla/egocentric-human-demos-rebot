"""Motion scaling (2026-10-14, spec 17). Applied to the robot-frame body delta AFTER the adapter, so sx/sy/sz name robot TCP
axes. Rotation scale multiplies the delta angle (axis kept). Scale 1 = identity; never negative (signs belong to the adapter)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from ego_teleop.transforms.se3 import make_T


@dataclass(frozen=True)
class MotionScaling:
    translation: tuple = (1.0, 1.0, 1.0)
    rotation: float = 1.0

    def __post_init__(self) -> None:
        if any(s < 0 for s in self.translation) or self.rotation < 0: raise ValueError("scales must be >= 0 (signs belong to the adapter)")

    @classmethod
    def from_cfg(cls, d: dict) -> "MotionScaling":
        t = d["translation_scale"]; t = (float(t),) * 3 if np.isscalar(t) else tuple(float(v) for v in t)
        return cls(t, float(d["rotation_scale"]))

    def apply(self, D: np.ndarray) -> np.ndarray:
        D = np.asarray(D, np.float64)
        R = D[:3, :3] if self.rotation == 1.0 else Rotation.from_rotvec(Rotation.from_matrix(D[:3, :3]).as_rotvec() * self.rotation).as_matrix()
        return make_T(R, np.asarray(self.translation) * D[:3, 3])
