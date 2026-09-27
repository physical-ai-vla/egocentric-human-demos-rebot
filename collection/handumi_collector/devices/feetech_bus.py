"""Minimal Feetech (STS3215) bus + status-block parser. Vendored/trimmed from handumi-sw feetech/{bus,telemetry}.py
(branch rebot-ego, uncommitted working tree, 2026-09-09). Only the read path the jaw encoder needs."""
from __future__ import annotations
import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Sequence

STATUS_BLOCK_ADDR = 42
STATUS_BLOCK_LEN = 29
_PRESENT_OFFSET = 56 - STATUS_BLOCK_ADDR
_PRESENT_POSITION_ADDR = 56
CURRENT_MA_PER_UNIT = 6.5
VOLTAGE_V_PER_UNIT = 0.1
_RETRYABLE = (IndexError, OSError)


def _u16(lo: int, hi: int) -> int:
    return (int(hi) << 8) | int(lo)


def _signed15(raw: int) -> int:
    return -(raw & 0x7FFF) if raw & 0x8000 else raw


def parse_status_block(data: Sequence[int]) -> dict[str, Any]:
    raw = [int(b) & 0xFF for b in data]
    if len(raw) >= STATUS_BLOCK_LEN:
        goal = _u16(raw[0], raw[1]); d = raw[_PRESENT_OFFSET:_PRESENT_OFFSET + 15]
    elif len(raw) >= 15:
        goal = -1; d = raw[:15]
    else:
        raise ValueError(f"status block needs {STATUS_BLOCK_LEN} (or 15) bytes, got {len(raw)}")
    return {
        "goal_position": goal, "position": _u16(d[0], d[1]), "speed": _signed15(_u16(d[2], d[3])),
        "load": _signed15(_u16(d[4], d[5])), "load_percent": _signed15(_u16(d[4], d[5])) / 10.0,
        "voltage_v": d[6] * VOLTAGE_V_PER_UNIT, "temperature_c": d[7], "status": d[9], "moving": d[10],
        "current_ma": _signed15(_u16(d[13], d[14])) * CURRENT_MA_PER_UNIT,
    }


def _install_reliable_packet_timeout(port: Any) -> None:
    def set_packet_timeout(self: Any, packet_length: int) -> None:
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = self.tx_time_per_byte * float(packet_length) + self.tx_time_per_byte * 3.0 + 50.0
    port.setPacketTimeout = MethodType(set_packet_timeout, port)


def _ok(comm: Any, sdk: Any, error: Any) -> bool:
    """Communication succeeded. The servo's status error byte (bit0 voltage, bit2 temperature, bit3 current, bit4 angle,
    bit5 overload) does NOT invalidate the returned data — a HandUMI servo on a 12.2 V supply answers every packet with 0x01;
    the bits are recorded on the bus (`last_error_bits`) and surfaced as telemetry / a UI warning instead."""
    return comm == getattr(sdk, "COMM_SUCCESS", 0) or comm == 0


ERROR_BITS = {0x01: "voltage", 0x02: "sensor", 0x04: "temperature", 0x08: "current", 0x10: "angle", 0x20: "overload"}


def describe_error_bits(bits: int) -> str:
    return ",".join(n for b, n in ERROR_BITS.items() if bits & b) or ""


@dataclass
class FeetechBus:
    port: str
    baudrate: int = 1_000_000
    protocol_version: int = 0

    def __post_init__(self) -> None:
        self._sdk = None; self._port = None; self._packet = None
        self.last_error_bits: int = 0

    def open(self) -> None:
        import scservo_sdk as sdk
        port = sdk.PortHandler(self.port)
        _install_reliable_packet_timeout(port)
        if not port.setBaudRate(self.baudrate):
            port.closePort(); raise RuntimeError(f"could not open Feetech port {self.port} @ {self.baudrate}")
        self._sdk, self._port, self._packet = sdk, port, sdk.PacketHandler(self.protocol_version)

    def close(self) -> None:
        if self._port is not None:
            try: self._port.closePort()
            except Exception: pass
        self._sdk = self._port = self._packet = None

    def reset_io(self) -> None:
        if self._port is None: return
        clear = getattr(self._port, "clearPort", None)
        if callable(clear): clear()
        if hasattr(self._port, "is_using"): self._port.is_using = False

    def ping(self, servo_id: int) -> bool:
        """True only when the servo actually answers (a 1-byte read; the SDK's ping() reports success without a reply)."""
        try:
            _, comm, err = self._packet.read1ByteTxRx(self._port, int(servo_id), _PRESENT_POSITION_ADDR)
        except _RETRYABLE:
            self.reset_io(); return False
        if _ok(comm, self._sdk, err): self.last_error_bits = int(err); return True
        return False

    def read_position(self, servo_id: int, *, retries: int = 1, retry_delay_s: float = 0.02) -> int:
        last = "no response"
        for attempt in range(retries + 1):
            try:
                v, comm, err = self._packet.read2ByteTxRx(self._port, int(servo_id), _PRESENT_POSITION_ADDR)
            except _RETRYABLE as exc:
                self.reset_io(); last = f"{type(exc).__name__}"
            else:
                if _ok(comm, self._sdk, err): self.last_error_bits = int(err); return int(v)
                last = f"comm={comm} err={err}"
            if attempt < retries: time.sleep(retry_delay_s)
        raise RuntimeError(f"Feetech read_position id={servo_id} failed ({last})")

    def read_status_block(self, servo_id: int, *, retries: int = 1, retry_delay_s: float = 0.02) -> dict[str, Any]:
        last = "no response"
        for attempt in range(retries + 1):
            try:
                data, comm, err = self._packet.readTxRx(self._port, int(servo_id), STATUS_BLOCK_ADDR, STATUS_BLOCK_LEN)
            except _RETRYABLE as exc:
                self.reset_io(); last = f"{type(exc).__name__}"
            else:
                if _ok(comm, self._sdk, err) and data is not None and len(data) >= STATUS_BLOCK_LEN:
                    self.last_error_bits = int(err); d = parse_status_block(data); d["error_bits"] = int(err); return d
                last = f"comm={comm} err={err} bytes={len(data) if data else 0}"
            if attempt < retries: time.sleep(retry_delay_s)
        raise RuntimeError(f"Feetech read_status_block id={servo_id} failed ({last})")
