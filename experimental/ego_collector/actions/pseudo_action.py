"""Bimanual pseudo actions from wrist poses + grasp signals (generic BimanualActionState).

The same interface will be fed by HandUMI later (EEF tag + jaw encoder), so nothing here
knows about hand landmarks: it only consumes per-frame pose7 + grasp per side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ego_collector.tracking.transforms import delta_pose, pose7_to_T

POSE = ("x", "y", "z", "qx", "qy", "qz", "qw")
DELTA = ("dx", "dy", "dz", "drx", "dry", "drz")
SIDES = ("left", "right")


@dataclass
class BimanualActionState:
    """One timestep of the generic human/robot action state (LEVEL 1)."""

    timestamp_ns: int
    left_pose: np.ndarray | None  # pose7 world frame, None if invalid
    right_pose: np.ndarray | None
    left_grasp: float  # 0 closed .. 1 open, NaN if unknown
    right_grasp: float
    validity: dict[str, bool] = field(default_factory=dict)

    def pose(self, side: str) -> np.ndarray | None:
        return self.left_pose if side == "left" else self.right_pose

    def grasp(self, side: str) -> float:
        return self.left_grasp if side == "left" else self.right_grasp


def states_from_arrays(timestamps_ns: np.ndarray, poses: dict[str, np.ndarray], valid: dict[str, np.ndarray], grasp: dict[str, np.ndarray]) -> list[BimanualActionState]:
    out = []
    for i in range(len(timestamps_ns)):
        out.append(
            BimanualActionState(
                timestamp_ns=int(timestamps_ns[i]),
                left_pose=poses["left"][i] if valid["left"][i] else None,
                right_pose=poses["right"][i] if valid["right"][i] else None,
                left_grasp=float(grasp["left"][i]),
                right_grasp=float(grasp["right"][i]),
                validity={"left_pose": bool(valid["left"][i]), "right_pose": bool(valid["right"][i]), "left_grasp": bool(np.isfinite(grasp["left"][i])), "right_grasp": bool(np.isfinite(grasp["right"][i]))},
            )
        )
    return out


def pseudo_action_table(states: list[BimanualActionState], *, horizon: int = 1) -> dict[str, np.ndarray]:
    """Absolute states + per-step deltas (t -> t+horizon). Rows keep the input length;
    the last ``horizon`` rows (and rows next to an invalid pose) have NaN deltas."""
    n = len(states)
    cols: dict[str, np.ndarray] = {"timestamp_ns": np.array([s.timestamp_ns for s in states], dtype=np.int64)}
    for side in SIDES:
        valid = np.array([s.pose(side) is not None for s in states], dtype=bool)
        poses = np.full((n, 7), np.nan)
        for i, s in enumerate(states):
            if s.pose(side) is not None:
                poses[i] = s.pose(side)
        cols[f"{side}_valid"] = valid
        for j, k in enumerate(POSE):
            cols[f"{side}_{k}"] = poses[:, j]
        cols[f"{side}_grasp"] = np.array([s.grasp(side) for s in states], dtype=np.float64)
        deltas = np.full((n, 6), np.nan)
        for i in range(n - horizon):
            if valid[i] and valid[i + horizon]:
                deltas[i] = delta_pose(pose7_to_T(poses[i]), pose7_to_T(poses[i + horizon]))
        for j, k in enumerate(DELTA):
            cols[f"{side}_{k}"] = deltas[:, j]
        cols[f"{side}_delta_valid"] = np.isfinite(deltas[:, 0])
    cols["action_valid"] = cols["left_delta_valid"] & cols["right_delta_valid"] & np.isfinite(cols["left_grasp"]) & np.isfinite(cols["right_grasp"])
    return cols


def action14(cols: dict[str, np.ndarray]) -> np.ndarray:
    """(N, 14) = [left dx dy dz drx dry drz grasp, right ...] with NaN where not available."""
    parts = []
    for side in SIDES:
        parts += [cols[f"{side}_{k}"] for k in DELTA] + [cols[f"{side}_grasp"]]
    return np.stack(parts, axis=1)


__all__ = ["DELTA", "POSE", "SIDES", "BimanualActionState", "action14", "pseudo_action_table", "states_from_arrays"]
