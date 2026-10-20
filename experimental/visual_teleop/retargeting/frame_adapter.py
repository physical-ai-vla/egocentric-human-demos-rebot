"""THE coordinate adapter: human virtual-TCP body delta -> robot IK-TCP body delta (2026-10-14, spec 19).

    Delta_R = M . Delta_H . M^T   (rotation, conjugation)        t_R = M . t_H   (translation)

M is a signed permutation (ego_teleop.transforms.frames.basis_matrix), configured per side in teleop_rebot.yaml
retarget.adapter. No other module may swap an axis or flip a sign between human and robot; scaling is a separate step
(motion_scaling.py) so the adapter's unit tests (identity, +-X/Y/Z, +-90 deg about each axis) test geometry only.
The reBot IK frame is the pyroki rebot_b601 TCP (FK frame). Dataset-TCP axes differ by Ry(90) -- that conversion lives ONLY in
ego_teleop/transforms/umi_robot_frame.py and is applied at export, never here."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ego_teleop.transforms.frames import basis_matrix
from ego_teleop.transforms.se3 import make_T


@dataclass(frozen=True)
class FrameAdapter:
    axis_map: tuple = ("x", "y", "z")
    sign: tuple = (1.0, 1.0, 1.0)

    @property
    def M(self) -> np.ndarray: return basis_matrix(self.axis_map, self.sign)

    @property
    def is_mirror(self) -> bool: return bool(np.linalg.det(self.M) < 0)

    @classmethod
    def from_cfg(cls, d: dict) -> "FrameAdapter": return cls(tuple(d["axis_map"]), tuple(float(s) for s in d["sign"]))

    def map_delta(self, D_H: np.ndarray) -> np.ndarray:
        D = np.asarray(D_H, np.float64); M = self.M
        if not np.all(np.isfinite(D)): raise ValueError("adapter: non-finite delta")
        return make_T(M @ D[:3, :3] @ M.T, M @ D[:3, 3])

    def unmap_delta(self, D_R: np.ndarray) -> np.ndarray:
        D = np.asarray(D_R, np.float64); M = self.M
        return make_T(M.T @ D[:3, :3] @ M, M.T @ D[:3, 3])
