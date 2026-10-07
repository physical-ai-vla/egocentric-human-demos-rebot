from __future__ import annotations

import dataclasses

import numpy as np

from handumi.feetech.gripper import GripperWidths
from handumi.feetech.telemetry import parse_status_block, telemetry_features, telemetry_frame


def _block(position=2200, speed=-120, load=350, voltage=0x7B, temp=41, moving=1, current=-46):
    def s15(v):
        return (abs(v) | 0x8000) if v < 0 else v

    def lohi(v):
        return [v & 0xFF, (v >> 8) & 0xFF]

    present = lohi(position) + lohi(s15(speed)) + lohi(s15(load)) + [voltage, temp, 0, 0, moving, 0, 0] + lohi(s15(current))
    goal = lohi(2100) + [0] * 12  # 42..55: Goal_Position + 12 config bytes
    return goal + present


def test_parse_status_block_signs_and_units():
    t = parse_status_block(_block())
    assert t["goal_position"] == 2100 and t["position"] == 2200
    assert parse_status_block(_block()[14:])["goal_position"] == -1  # legacy 15-byte block
    assert t["speed"] == -120 and t["load"] == 350 and t["load_percent"] == 35.0
    assert abs(t["voltage_v"] - 12.3) < 1e-9 and t["temperature_c"] == 41 and t["moving"] == 1
    assert t["current_raw"] == -46 and abs(t["current_ma"] - (-46 * 6.5)) < 1e-9


def test_telemetry_frame_matches_features_and_handles_missing():
    feats = telemetry_features()
    widths = dataclasses.replace(GripperWidths.zero(), left_telemetry=parse_status_block(_block()), right_telemetry=None)
    frame = telemetry_frame(widths)
    assert set(frame) == set(feats)
    assert frame["observation.feetech.left_current_ma"][0] == np.float32(-46 * 6.5)
    assert frame["observation.feetech.left_telemetry_ok"][0] == 1
    assert frame["observation.feetech.right_telemetry_ok"][0] == 0
    assert np.isnan(frame["observation.feetech.right_current_ma"][0])
    for key, spec in feats.items():
        assert frame[key].shape == tuple(spec["shape"]) and str(frame[key].dtype) == spec["dtype"]
