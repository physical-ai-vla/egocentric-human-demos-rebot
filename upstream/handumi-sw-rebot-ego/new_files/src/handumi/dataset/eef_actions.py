"""Robot-independent end-effector (TCP) state/action representations.

Phase 7 of the reBot ego-bimanual pipeline: the raw HandUMI dataset stores the
*controller* poses (workspace/table frame) and jaw widths; nothing robot
specific. This module derives a training-ready **EEF dataset** from it without
touching any robot kinematics:

``observation.state`` (16)::

    left_tcp  x y z qx qy qz qw   right_tcp x y z qx qy qz qw   left_grip right_grip

``action`` (``delta``, 14)::

    left  dx dy dz drx dry drz grip   right dx dy dz drx dry drz grip

where ``[d*]`` is the *local-frame* SE(3) increment ``inv(T_t) · T_{t+h}``
expressed as translation + rotation-vector (radians), and ``grip`` is the jaw
opening at ``t+h``. ``action`` (``absolute``, 16) is simply ``state[t+h]``.

The raw capture is never modified: swapping the action representation is a
re-run of ``handumi convert-eef`` on the same raw dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from handumi.calibration.control_tcp import (
    ControllerTcpCalibration,
    apply_controller_tcp_calibration,
)
from handumi.dataset.raw import (
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
)

ActionRepr = Literal["delta", "absolute"]
GripperUnits = Literal["normalized", "m"]

POSE7_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")
EEF_STATE_NAMES: tuple[str, ...] = (
    *[f"left_tcp.{n}" for n in POSE7_NAMES],
    *[f"right_tcp.{n}" for n in POSE7_NAMES],
    "left_gripper",
    "right_gripper",
)
EEF_DELTA_ACTION_NAMES: tuple[str, ...] = (
    *[f"left.{n}" for n in ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")],
    *[f"right.{n}" for n in ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")],
)
EEF_ABSOLUTE_ACTION_NAMES: tuple[str, ...] = EEF_STATE_NAMES
EEF_STATE_SEMANTICS = "table_tcp_pose7_pair_plus_gripper"


@dataclass(frozen=True)
class EefEpisode:
    """One derived episode (rows already aligned: ``action[t]`` maps ``t -> t+h``)."""

    states: np.ndarray  # (T-h, 16)
    actions: np.ndarray  # (T-h, 14 | 16)
    left_tcp: np.ndarray  # (T, 7) full-length TCP trajectory (table frame)
    right_tcp: np.ndarray  # (T, 7)
    action_repr: ActionRepr
    horizon: int

    @property
    def action_names(self) -> tuple[str, ...]:
        return EEF_DELTA_ACTION_NAMES if self.action_repr == "delta" else EEF_ABSOLUTE_ACTION_NAMES


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def tcp_trajectories(
    raw_states: np.ndarray, calibration: ControllerTcpCalibration | None
) -> tuple[np.ndarray, np.ndarray]:
    """Controller pose7 pair -> TCP pose7 pair (identity calibration if ``None``)."""
    raw_states = np.asarray(raw_states, dtype=np.float32)
    left = raw_states[:, LEFT_POSE_SLICE]
    right = raw_states[:, RIGHT_POSE_SLICE]
    if calibration is None:
        return _continuous(left), _continuous(right)
    return apply_controller_tcp_calibration(left, right, calibration)


def gripper_columns(
    raw_states: np.ndarray, *, units: GripperUnits, max_width_m: float
) -> np.ndarray:
    """(T, 2) jaw openings. Raw widths are metres; ``normalized`` -> [0, 1]."""
    widths = np.asarray(raw_states, dtype=np.float32)[:, [LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX]]
    if units == "m":
        return widths
    if max_width_m <= 0:
        raise ValueError("max_width_m must be positive for normalized gripper units")
    return np.clip(widths / np.float32(max_width_m), 0.0, 1.0).astype(np.float32)


def eef_state(left_tcp: np.ndarray, right_tcp: np.ndarray, grippers: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.asarray(left_tcp, np.float32), np.asarray(right_tcp, np.float32), np.asarray(grippers, np.float32)],
        axis=1,
    )


def local_delta(poses7: np.ndarray, horizon: int = 1) -> np.ndarray:
    """(T-h, 6) local-frame increments ``inv(T_t) · T_{t+h}`` as [dxyz, rotvec]."""
    poses7 = np.asarray(poses7, dtype=np.float64)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(poses7) <= horizon:
        return np.zeros((0, 6), dtype=np.float32)
    cur, nxt = poses7[:-horizon], poses7[horizon:]
    r_cur = Rotation.from_quat(cur[:, 3:7])
    r_nxt = Rotation.from_quat(nxt[:, 3:7])
    dpos_world = nxt[:, :3] - cur[:, :3]
    dpos_local = r_cur.inv().apply(dpos_world)
    drot = (r_cur.inv() * r_nxt).as_rotvec()
    return np.concatenate([dpos_local, drot], axis=1).astype(np.float32)


def apply_local_delta(pose7: np.ndarray, delta6: np.ndarray) -> np.ndarray:
    """Inverse of :func:`local_delta` for one step: ``T_t · Exp(delta)``."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    r = Rotation.from_quat(pose7[3:7])
    pos = pose7[:3] + r.apply(np.asarray(delta6[:3], dtype=np.float64))
    rot = r * Rotation.from_rotvec(np.asarray(delta6[3:6], dtype=np.float64))
    return np.concatenate([pos, rot.as_quat()]).astype(np.float32)


def delta_actions(left_tcp: np.ndarray, right_tcp: np.ndarray, grippers: np.ndarray, horizon: int = 1) -> np.ndarray:
    dl = local_delta(left_tcp, horizon)
    dr = local_delta(right_tcp, horizon)
    g = np.asarray(grippers, dtype=np.float32)[horizon:]
    return np.concatenate([dl, g[:, :1], dr, g[:, 1:2]], axis=1).astype(np.float32)


def absolute_actions(states16: np.ndarray, horizon: int = 1) -> np.ndarray:
    return np.asarray(states16, dtype=np.float32)[horizon:]


def build_eef_episode(
    raw_states: np.ndarray,
    *,
    calibration: ControllerTcpCalibration | None,
    action_repr: ActionRepr = "delta",
    horizon: int = 1,
    gripper_units: GripperUnits = "normalized",
    gripper_max_width_m: float = 0.08,
) -> EefEpisode:
    """Raw 16-D HandUMI states -> aligned EEF (state, action) arrays."""
    raw_states = np.asarray(raw_states, dtype=np.float32)
    if raw_states.ndim != 2 or raw_states.shape[1] != 16:
        raise ValueError(f"expected raw states (T, 16), got {raw_states.shape}")
    if len(raw_states) <= horizon:
        raise ValueError(f"episode has {len(raw_states)} frames; need > horizon={horizon}")
    left, right = tcp_trajectories(raw_states, calibration)
    grip = gripper_columns(raw_states, units=gripper_units, max_width_m=gripper_max_width_m)
    states = eef_state(left, right, grip)
    if action_repr == "delta":
        actions = delta_actions(left, right, grip, horizon)
    elif action_repr == "absolute":
        actions = absolute_actions(states, horizon)
    else:
        raise ValueError(f"unknown action_repr {action_repr!r}")
    return EefEpisode(
        states=states[:-horizon],
        actions=actions,
        left_tcp=left,
        right_tcp=right,
        action_repr=action_repr,
        horizon=horizon,
    )


def action_feature(action_repr: ActionRepr) -> dict:
    names = EEF_DELTA_ACTION_NAMES if action_repr == "delta" else EEF_ABSOLUTE_ACTION_NAMES
    return {"dtype": "float32", "shape": [len(names)], "names": list(names)}


def _continuous(poses: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so consecutive samples stay on one hemisphere."""
    out = np.asarray(poses, dtype=np.float32).copy()
    for i in range(1, len(out)):
        if float(np.dot(out[i - 1, 3:7], out[i, 3:7])) < 0.0:
            out[i, 3:7] *= -1.0
    return out


__all__ = [
    "EEF_ABSOLUTE_ACTION_NAMES",
    "EEF_DELTA_ACTION_NAMES",
    "EEF_STATE_NAMES",
    "EEF_STATE_SEMANTICS",
    "EefEpisode",
    "absolute_actions",
    "action_feature",
    "apply_local_delta",
    "build_eef_episode",
    "delta_actions",
    "eef_state",
    "gripper_columns",
    "local_delta",
    "tcp_trajectories",
]
