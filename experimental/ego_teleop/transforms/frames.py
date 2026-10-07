"""Coordinate frames and THE one place where the human<->robot basis change lives.

Frames (names used in configs, logs and calibration files):
    W   VIO world (arbitrary per session)    C   wrist fisheye camera     I   IMU
    H   human wrist control frame            RB  reBot base frame         RE  reBot end-effector (TCP) frame
    AH  Aero Hand base frame
T_A_B = pose of frame B expressed in frame A (4x4).

Rule: nowhere else in the repository may an axis be swapped or a sign inverted between human and robot. Every such
conversion goes through HumanRobotFrameMapper, configured by configs/ego_teleop/frames.yaml (or a versioned
human_robot_frame_vNNN.yaml). Translation scale lives here too (never in calibration files)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from .se3 import make_T

FRAMES = ("W", "C", "I", "H", "RB", "RE", "AH")
_AXES = {"x": 0, "y": 1, "z": 2}


def basis_matrix(axis_map, sign) -> np.ndarray:
    """M such that v_robot = M @ v_human. Row i (robot axis i) picks human axis axis_map[i] with sign sign[i].
    axis_map must be a permutation of x,y,z and every sign +-1, so M is orthogonal (det +1 = rotation, -1 = mirror)."""
    axis_map = tuple(str(a).lower() for a in axis_map); sign = tuple(float(s) for s in sign)
    if len(axis_map) != 3 or sorted(axis_map) != ["x", "y", "z"]:
        raise ValueError(f"axis_map must be a permutation of x,y,z; got {axis_map}")
    if len(sign) != 3 or any(abs(s) != 1.0 for s in sign):
        raise ValueError(f"sign must be three values of +1/-1; got {sign}")
    M = np.zeros((3, 3))
    for i, (a, s) in enumerate(zip(axis_map, sign)): M[i, _AXES[a]] = s
    return M


@dataclass(frozen=True)
class FrameMapperConfig:
    translation_axis_map: tuple = ("x", "y", "z")   # robot axis i  <-  human axis map[i]
    translation_sign: tuple = (1.0, 1.0, 1.0)
    rotation_axis_map: tuple | None = None            # default: same basis change as translation
    rotation_sign: tuple | None = None
    translation_scale: tuple = (1.0, 1.0, 1.0)        # s_xyz, applied to robot-frame delta translation
    rotation_scale: float = 1.0                       # scales the delta rotation angle (1.0 = 1:1)

    @classmethod
    def from_dict(cls, d: dict | None) -> "FrameMapperConfig":
        d = dict(d or {})
        for k in ("translation_axis_map", "translation_sign", "rotation_axis_map", "rotation_sign", "translation_scale"):
            if d.get(k) is not None: d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FrameMapperConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    def to_dict(self) -> dict:
        return dict(translation_axis_map=list(self.translation_axis_map), translation_sign=list(self.translation_sign),
                    rotation_axis_map=list(self.rotation_axis_map) if self.rotation_axis_map else None,
                    rotation_sign=list(self.rotation_sign) if self.rotation_sign else None,
                    translation_scale=list(self.translation_scale), rotation_scale=self.rotation_scale)


class HumanRobotFrameMapper:
    """Maps a human wrist *delta* (expressed in the human wrist frame H at anchor time) to a robot EEF delta
    (expressed in the robot EEF frame RE at anchor time): t_R = s ⊙ (M_t t_H), R_R = M_r R_H M_rᵀ (conjugation,
    which keeps composition: map(A·B) = map(A)·map(B) when scale is 1). A mirror (det M_r = -1, e.g. human left
    hand driving the right arm) still yields a proper rotation."""

    def __init__(self, cfg: FrameMapperConfig | None = None) -> None:
        self.cfg = cfg or FrameMapperConfig()
        self.M_t = basis_matrix(self.cfg.translation_axis_map, self.cfg.translation_sign)
        rmap = self.cfg.rotation_axis_map or self.cfg.translation_axis_map
        rsign = self.cfg.rotation_sign or self.cfg.translation_sign
        self.M_r = basis_matrix(rmap, rsign)
        self.scale_t = np.asarray(self.cfg.translation_scale, np.float64).reshape(3)
        if np.any(self.scale_t < 0): raise ValueError("translation_scale must be >= 0 (use sign for direction)")
        self.scale_r = float(self.cfg.rotation_scale)
        self.is_mirror = bool(np.linalg.det(self.M_r) < 0)

    def map_translation(self, delta_xyz) -> np.ndarray:
        return self.scale_t * (self.M_t @ np.asarray(delta_xyz, np.float64).reshape(3))

    def map_rotation(self, delta_R) -> np.ndarray:
        R = self.M_r @ np.asarray(delta_R, np.float64)[:3, :3] @ self.M_r.T
        if self.scale_r != 1.0:
            R = Rotation.from_rotvec(Rotation.from_matrix(R).as_rotvec() * self.scale_r).as_matrix()
        return R

    def map_delta(self, Delta_H) -> np.ndarray:
        D = np.asarray(Delta_H, np.float64)
        return make_T(self.map_rotation(D[:3, :3]), self.map_translation(D[:3, 3]))

    def describe(self) -> dict:
        return dict(**self.cfg.to_dict(), mirror=self.is_mirror)
