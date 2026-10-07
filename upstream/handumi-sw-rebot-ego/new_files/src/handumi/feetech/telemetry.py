"""Feetech STS3215 status-block telemetry (position, speed, load, voltage, temperature, current).

The jaw servo is passive in HandUMI (the human squeezes it), so there is no
goal position to log; everything else in the "present" block is kept raw so a
contact proxy can be derived later (see :mod:`handumi.dataset.derived`).

Memory map (STS/SMS series, one contiguous read from address 42, 29 bytes)::

    42-43 Goal_Position      44-45 Goal_Time        46-47 Goal_Speed
    48    Torque_Limit ...   55    Lock
    56-57 Present_Position   58-59 Present_Speed    60-61 Present_Load
    62    Present_Voltage    63    Present_Temperature
    64    Async_Write_Flag   65    Status            66    Moving
    67-68 (reserved)         69-70 Present_Current

Speed/load/current are 15-bit magnitudes with bit 15 as the sign.
Units: voltage 0.1 V, current 6.5 mA, load 0.1 % of stall torque.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

STATUS_BLOCK_ADDR = 42
STATUS_BLOCK_LEN = 29
_PRESENT_OFFSET = 56 - STATUS_BLOCK_ADDR
CURRENT_MA_PER_UNIT = 6.5
VOLTAGE_V_PER_UNIT = 0.1

TELEMETRY_FLOAT_KEYS: tuple[str, ...] = ("goal_position", "speed", "load", "current_ma", "voltage_v", "temperature_c")


def _u16(lo: int, hi: int) -> int:
    return (int(hi) << 8) | int(lo)


def _signed15(raw: int) -> int:
    return -(raw & 0x7FFF) if raw & 0x8000 else raw


def parse_status_block(data: Sequence[int]) -> dict[str, Any]:
    """Decode the status block (29 B from addr 42, or legacy 15 B from addr 56)."""
    raw = [int(b) & 0xFF for b in data]
    if len(raw) >= STATUS_BLOCK_LEN:
        goal_position = _u16(raw[0], raw[1])
        d = raw[_PRESENT_OFFSET : _PRESENT_OFFSET + 15]
    elif len(raw) >= 15:
        goal_position = -1
        d = raw[:15]
    else:
        raise ValueError(f"status block needs {STATUS_BLOCK_LEN} (or 15) bytes, got {len(raw)}")
    position = _u16(d[0], d[1])
    speed = _signed15(_u16(d[2], d[3]))
    load = _signed15(_u16(d[4], d[5]))
    voltage_raw = d[6]
    temperature = d[7]
    moving = d[10]
    current = _signed15(_u16(d[13], d[14]))
    return {
        "goal_position": goal_position,
        "position": position,
        "speed": speed,
        "load": load,
        "load_percent": load / 10.0,
        "voltage_raw": voltage_raw,
        "voltage_v": voltage_raw * VOLTAGE_V_PER_UNIT,
        "temperature_c": temperature,
        "moving": moving,
        "status": d[9],
        "current_raw": current,
        "current_ma": current * CURRENT_MA_PER_UNIT,
    }


def telemetry_features() -> dict[str, Any]:
    features: dict[str, Any] = {}
    for side in ("left", "right"):
        for key in TELEMETRY_FLOAT_KEYS:
            features[f"observation.feetech.{side}_{key}"] = {"dtype": "float32", "shape": (1,), "names": None}
        features[f"observation.feetech.{side}_telemetry_ok"] = {"dtype": "int64", "shape": (1,), "names": None}
    return features


def telemetry_frame(widths: Any) -> dict[str, np.ndarray]:
    """Per-row columns from ``GripperWidths.{left,right}_telemetry`` (NaN when absent)."""
    frame: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        tele: Mapping[str, Any] | None = getattr(widths, f"{side}_telemetry", None)
        ok = bool(tele)
        for key in TELEMETRY_FLOAT_KEYS:
            value = float(tele[key]) if ok and key in tele else float("nan")
            frame[f"observation.feetech.{side}_{key}"] = np.array([value], dtype=np.float32)
        frame[f"observation.feetech.{side}_telemetry_ok"] = np.array([int(ok)], dtype=np.int64)
    return frame


__all__ = [
    "CURRENT_MA_PER_UNIT",
    "STATUS_BLOCK_ADDR",
    "STATUS_BLOCK_LEN",
    "TELEMETRY_FLOAT_KEYS",
    "parse_status_block",
    "telemetry_features",
    "telemetry_frame",
]
