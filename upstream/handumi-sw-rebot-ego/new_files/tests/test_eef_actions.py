from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.dataset.eef_actions import (
    EEF_DELTA_ACTION_NAMES,
    EEF_STATE_NAMES,
    apply_local_delta,
    build_eef_episode,
    local_delta,
)
from handumi.robots.utils import IDENTITY_POSE7


def _traj(T=40, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)
    pos = np.stack([0.2 + 0.1 * np.sin(2 * np.pi * t), 0.1 * t, 0.05 + 0.02 * np.cos(2 * np.pi * t)], axis=1)
    rot = Rotation.from_euler("xyz", np.stack([0.3 * t, 0.2 * np.sin(3 * t), 1.0 * t], axis=1))
    return np.concatenate([pos, rot.as_quat()], axis=1).astype(np.float32)


def _raw_states(T=40):
    left = _traj(T, 0)
    right = _traj(T, 1)
    right[:, 1] -= 0.4
    widths = np.stack([np.linspace(0.08, 0.0, T), np.linspace(0.0, 0.08, T)], axis=1)
    return np.concatenate([left, right, widths], axis=1).astype(np.float32)


def test_local_delta_roundtrip():
    poses = _traj()
    d = local_delta(poses, horizon=1)
    assert d.shape == (39, 6)
    for i in range(39):
        rec = apply_local_delta(poses[i], d[i])
        np.testing.assert_allclose(rec[:3], poses[i + 1, :3], atol=1e-5)
        dq = (Rotation.from_quat(rec[3:]).inv() * Rotation.from_quat(poses[i + 1, 3:])).magnitude()
        assert dq < 1e-5


def test_local_delta_is_expressed_in_local_frame():
    # Body rotated 90 deg about z, moving +x in world -> local delta is -y... check sign
    q = Rotation.from_euler("z", 90, degrees=True).as_quat()
    p0 = np.array([0, 0, 0, *q]); p1 = np.array([0.1, 0, 0, *q])
    d = local_delta(np.stack([p0, p1]))
    np.testing.assert_allclose(d[0, :3], [0.0, -0.1, 0.0], atol=1e-6)
    np.testing.assert_allclose(d[0, 3:], 0.0, atol=1e-6)


def test_build_eef_episode_delta_shapes_and_alignment():
    raw = _raw_states(40)
    ep = build_eef_episode(raw, calibration=None, action_repr="delta", horizon=1, gripper_max_width_m=0.08)
    assert ep.states.shape == (39, 16) and ep.actions.shape == (39, 14)
    assert len(EEF_STATE_NAMES) == 16 and len(EEF_DELTA_ACTION_NAMES) == 14
    # identity calibration -> TCP == controller
    np.testing.assert_allclose(ep.states[:, :7], raw[:-1, :7], atol=1e-6)
    # gripper normalized and taken at t+h
    np.testing.assert_allclose(ep.states[:, 14], raw[:-1, 14] / 0.08, atol=1e-6)
    np.testing.assert_allclose(ep.actions[:, 6], raw[1:, 14] / 0.08, atol=1e-6)
    np.testing.assert_allclose(ep.actions[:, 13], raw[1:, 15] / 0.08, atol=1e-6)
    # delta integrates state[t] to state[t+1]
    for i in range(0, 39, 7):
        rec = apply_local_delta(ep.states[i, :7], ep.actions[i, :6])
        np.testing.assert_allclose(rec[:3], ep.left_tcp[i + 1, :3], atol=1e-5)


def test_build_eef_episode_absolute_and_horizon():
    raw = _raw_states(40)
    ep = build_eef_episode(raw, calibration=None, action_repr="absolute", horizon=5, gripper_units="m")
    assert ep.states.shape == (35, 16) and ep.actions.shape == (35, 16)
    np.testing.assert_allclose(ep.actions, np.concatenate([raw[5:, :14], raw[5:, 14:16]], axis=1), atol=1e-6)


def test_tcp_calibration_is_applied():
    raw = _raw_states(10)
    offset = np.array([0.0, 0.0, 0.1, 0, 0, 0, 1], dtype=np.float32)  # 10 cm along controller z
    cal = ControllerTcpCalibration(left=offset, right=offset.copy())
    ep = build_eef_episode(raw, calibration=cal)
    z_axis = Rotation.from_quat(raw[0, 3:7]).apply([0, 0, 0.1])
    np.testing.assert_allclose(ep.left_tcp[0, :3], raw[0, :3] + z_axis, atol=1e-5)


def test_rejects_short_episode():
    with pytest.raises(ValueError):
        build_eef_episode(_raw_states(3), calibration=None, horizon=3)


def test_write_eef_dataset_smoke(tmp_path: Path):
    from handumi.scripts.convert_eef import write_eef_dataset

    raw = _raw_states(30)
    source_root = tmp_path / "raw"
    (source_root / "meta").mkdir(parents=True)
    source_info = {"codebase_version": "v3.0", "fps": 30, "features": {}, "handumi": {"recording_device": "meta"}}
    (source_root / "meta" / "info.json").write_text(json.dumps(source_info))
    episodes = [
        (0, "Stack red, blue, green.", build_eef_episode(raw, calibration=None)),
        (1, "Stack blue, red, green.", build_eef_episode(raw[::-1].copy(), calibration=None)),
    ]
    out = tmp_path / "eef"
    write_eef_dataset(
        output_root=out,
        source_root=source_root,
        source_info=source_info,
        episodes=episodes,
        fps=30,
        action_repr="delta",
        horizon=1,
        gripper_units="normalized",
        gripper_max_width_m=0.08,
        tcp_calibration_metadata={"source": "identity"},
    )
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [16]
    assert info["features"]["action"]["shape"] == [14]
    assert info["features"]["action"]["names"] == list(EEF_DELTA_ACTION_NAMES)
    assert info["total_episodes"] == 2 and info["total_frames"] == 58
    assert info["handumi"]["eef_actions"]["action_repr"] == "delta"
    import pandas as pd

    df = pd.concat([pd.read_parquet(f) for f in sorted((out / "data").rglob("*.parquet"))])
    assert len(df) == 58
    assert len(df["action"].iloc[0]) == 14 and len(df["observation.state"].iloc[0]) == 16
    tasks = pd.read_parquet(out / "meta" / "tasks.parquet")
    assert len(tasks) == 2
