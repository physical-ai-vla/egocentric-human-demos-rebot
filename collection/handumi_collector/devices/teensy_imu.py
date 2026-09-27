"""Teensy 4.1 + ICM42688P IMU over USB CDC. Protocol: firmware/teensy_imu/PROTOCOL.md (framed binary, CRC-16/CCITT).
Recovers from partial packets, garbage bytes, CRC failures, USB disconnect (auto-reconnect), sequence gaps and device
timestamp wraps (reported as events, raw values kept). Raw sensor axes are stored as-is (no L/R flips here)."""
from __future__ import annotations
import logging
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from .base import DeviceStatus, ImuSample, RateMeter, SampleBuffer, now_ns, resolve_serial_port
from ..config import ImuCfg

log = logging.getLogger("handumi.imu")
MAGIC = b"\xa5\x5a"
TYPE_IMU, TYPE_STATUS, TYPE_LOG = 0x01, 0x02, 0x03
IMU_FMT = "<IQ7f"                        # seq, t_us, ax ay az gx gy gz temp
IMU_LEN = struct.calcsize(IMU_FMT)       # 40
STATUS_FMT = "<QIIIBBH"
STATUS_LEN = struct.calcsize(STATUS_FMT)  # 24
SEQ_MOD = 1 << 32
# `host_receive_ns` is stamped once per read(), so every sample in one read shares it and this timeout IS the host-clock
# quantisation. Measured on a 200 Hz board, 25 s each: 50 ms -> 14.543 ms clock-fit residual (= 50/sqrt(12), quantisation,
# not USB latency), 5 ms -> 1.054 ms. `estimate_camera_imu_offset` searches in 0.5 ms steps, so the old value was the
# limit on how precisely the camera<->IMU offset could be measured. See docs/handumi_collector/IMU_BRINGUP.md.
READ_TIMEOUT_S = 0.005
DRAIN_S = 0.35         # long enough for the device ring to arrive; the byte cap in _drain bounds it too
SERIAL_PLACEHOLDER = "REQUIRED_SET_ME"   # what an unconfigured profile carries; opening on it would guess the hand
TX_RING_PACKETS = 256                    # mirrors firmware/teensy_imu/config.h; only used to express high-water as a fraction
G0 = 9.80665


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode_packet(ptype: int, payload: bytes) -> bytes:
    body = bytes([ptype, len(payload)]) + payload
    return MAGIC + body + struct.pack("<H", crc16_ccitt(body))


def encode_imu(seq: int, t_us: int, acc, gyr, temp_c: float) -> bytes:
    return encode_packet(TYPE_IMU, struct.pack(IMU_FMT, seq % SEQ_MOD, t_us, *acc, *gyr, temp_c))


def encode_status(t_us: int, rate_hz: int, emitted: int, dropped: int, fw=(1, 0)) -> bytes:
    return encode_packet(TYPE_STATUS, struct.pack(STATUS_FMT, t_us, rate_hz, emitted, dropped, fw[0], fw[1], 0))


class PacketParser:
    """Incremental byte-stream parser; resynchronises on the magic word; counts CRC failures and resyncs."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self.crc_errors = 0
        self.resyncs = 0

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self._buf += data
        out: list[tuple[int, bytes]] = []
        while True:
            i = self._buf.find(MAGIC)
            if i < 0:
                if len(self._buf) > 1: self._buf = self._buf[-1:]
                return out
            if i > 0:
                self.resyncs += 1; del self._buf[:i]
            if len(self._buf) < 4: return out
            ptype, plen = self._buf[2], self._buf[3]
            total = 4 + plen + 2
            if len(self._buf) < total: return out
            body = bytes(self._buf[2:4 + plen])
            crc = struct.unpack_from("<H", self._buf, 4 + plen)[0]
            if crc16_ccitt(body) == crc:
                out.append((ptype, body[2:])); del self._buf[:total]
            else:
                self.crc_errors += 1; del self._buf[:2]


def decode_imu(side: str, payload: bytes, host_ns: int) -> ImuSample:
    seq, t_us, ax, ay, az, gx, gy, gz, temp = struct.unpack(IMU_FMT, payload[:IMU_LEN])
    return ImuSample(side, seq, t_us, host_ns, ax, ay, az, gx, gy, gz, temp)


@dataclass
class ImuQuality:
    hz: float = 0.0
    loss_ratio: float = 0.0          # lost / expected over the window (from seq gaps)
    crc_errors: int = 0
    resyncs: int = 0
    max_gap_ms: float = 0.0          # largest device-time gap seen in the window
    age_ms: float | None = None
    seq_gaps: int = 0
    ts_wraps: int = 0
    reconnects: int = 0
    connected: bool = False

    def grade(self, th: dict) -> str:
        """GREEN / YELLOW / RED from config thresholds (collector.yaml imu_quality)."""
        if not self.connected or self.age_ms is None: return "RED"
        g, y = th.get("green", {}), th.get("yellow", {})
        if self.hz >= g.get("min_hz", 380) and self.loss_ratio <= g.get("max_loss", 0.001) and self.age_ms <= g.get("max_age_ms", 50): return "GREEN"
        if self.hz >= y.get("min_hz", 300) and self.loss_ratio <= y.get("max_loss", 0.01) and self.age_ms <= y.get("max_age_ms", 250): return "YELLOW"
        return "RED"


class SeqTracker:
    """Detects sequence gaps (uint32 wrap-aware) and device-timestamp regressions; keeps loss stats over a sliding window."""

    def __init__(self, window: int = 2000) -> None:
        self.prev_seq: int | None = None
        self.prev_t_us: int | None = None
        self._events: deque[tuple[int, int]] = deque(maxlen=window)    # (received=1, lost=n)
        self._got = self._lost = 0                                     # running window sums; see loss_ratio
        self.seq_gaps = 0
        self.ts_wraps = 0
        self.max_gap_us = 0
        self.resync = True          # see update(): the first discontinuity after a connect is a ring boundary

    def update(self, seq: int, t_us: int) -> tuple[int, bool]:
        """Returns (lost_samples_before_this, timestamp_regressed).

        The first packet after a connect is NOT compared against anything. The Teensy keeps its own ring and
        `reset_input_buffer()` does not reach it, so opening the port delivers a stale block first and then jumps to
        the present: one discontinuity of several thousand, which is a ring boundary and not a loss. Counted, it put
        both IMUs at 95 % loss on the cockpit for the ten seconds it takes to evict from the window, which is how an
        operator learns to ignore a health display."""
        if self.resync:
            self.resync = False
            self.prev_seq, self.prev_t_us = seq, t_us
            self._events.append((1, 0)); self._got += 1
            return 0, False
        lost, wrapped = 0, False
        if self.prev_seq is not None:
            lost = (seq - self.prev_seq - 1) % SEQ_MOD
            if lost > SEQ_MOD // 2: lost = 0                    # duplicate / reorder: not a loss
            if lost: self.seq_gaps += 1
            dt = t_us - self.prev_t_us
            if dt < 0: wrapped = True; self.ts_wraps += 1
            elif lost == 0: self.max_gap_us = max(self.max_gap_us, dt)
        self.prev_seq, self.prev_t_us = seq, t_us
        if len(self._events) == self._events.maxlen:                   # about to evict: take it out of the sums first
            r0, l0 = self._events[0]
            self._got -= r0; self._lost -= l0
        self._events.append((1, lost))
        self._got += 1; self._lost += lost
        return lost, wrapped

    def loss_ratio(self) -> float:
        """Read two integers, never walk the window.

        This used to sum the deque on every call, which meant iterating a container the reader thread appends to --
        `RuntimeError: deque mutated during iteration`. The odds per call are small and the exposure is not: a 3 h
        unattended Allan run died 46 minutes in, and `imu_monitor --log`, the documented soak tool, asks twice a second.
        Running sums remove the race rather than narrowing it, and turn an O(window) call into two loads."""
        got, lost = self._got, self._lost
        return lost / (got + lost) if (got + lost) else 0.0

    def reset_window(self) -> None:
        self.max_gap_us = 0


class TeensyImu:
    def __init__(self, cfg: ImuCfg, *, buffer_s: float = 5.0) -> None:
        self.cfg = cfg
        self.name = f"imu_{cfg.side}"
        self.required = cfg.required
        self.buffer: SampleBuffer[ImuSample] = SampleBuffer(int(cfg.rate_hz * buffer_s))
        self.port: str | None = None
        self.serial_number: str | None = None
        self._ser = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._rate = RateMeter(256)
        self._status = DeviceStatus(self.name)
        self.parser = PacketParser()
        self.seq = SeqTracker()
        self.last_status: dict | None = None
        self.last_device_log: str | None = None
        self.ring_high_water = 0
        self.last_sample: ImuSample | None = None
        self.reconnects = 0
        self._exclude: tuple[str, ...] = ()
        self.on_event = None     # callback(kind:str, detail:dict, t_ns:int) set by the recorder/UI

    # ------------------------------------------------------------- lifecycle
    def open(self, *, exclude: tuple[str, ...] = ()) -> None:
        """Fails closed on identity: which hand a board belongs to is a claim, and a wrong one silently mislabels a dataset."""
        import serial
        from serial.tools import list_ports
        self._exclude = exclude
        want = (self.cfg.serial_number or "").strip()
        if not want or want == SERIAL_PLACEHOLDER:
            raise RuntimeError(f"{self.name}: serial_number is {self.cfg.serial_number!r} — LEFT/RIGHT identity must come from the "
                               f"USB serial, never from port enumeration order. Run `imu_logger --list` and set it in the profile.")
        matches = [p.device for p in list_ports.comports() if p.serial_number and want in p.serial_number]
        if len(matches) > 1:                       # matching is by substring, so a short serial can name two boards
            raise RuntimeError(f"{self.name}: serial {want!r} matches {len(matches)} ports ({', '.join(matches)}) — use the full serial")
        self.port = matches[0] if matches else None
        if self.port is None:
            raise RuntimeError(f"{self.name}: no serial port matches serial={want!r} glob={self.cfg.port_glob}")
        if self.port in exclude:
            raise RuntimeError(f"{self.name}: serial {want!r} resolves to {self.port}, already taken by another device")
        for p in list_ports.comports():
            if p.device == self.port: self.serial_number = p.serial_number
        self._ser = serial.Serial(self.port, self.cfg.baud, timeout=READ_TIMEOUT_S)
        self._ser.reset_input_buffer()
        self._drain()
        self._status.connected = True; self._status.error = None
        self._status.detail.update(port=self.port, serial_number=self.serial_number)

    def _drain(self, seconds: float = DRAIN_S) -> int:
        """Throw away the block the Teensy already had queued, before anything starts counting.

        `reset_input_buffer()` clears the HOST's buffer; the device keeps its own ring and hands over whatever was
        sitting in it, so the stream opens with a stale block that is internally contiguous and then jumps to the
        present. That jump is a ring boundary, and counting it as loss put both IMUs at 95 % on the cockpit for the
        ten seconds it took to leave the window -- which teaches an operator to disbelieve the health display, right
        before the pilot where it is the only thing watching. The host reads far faster than 200 Hz, so a short
        discard consumes the backlog and leaves the stream at the present."""
        # Bounded by TIME and by BYTES, and not by in_waiting: the stale block is not sitting in the host buffer when
        # the port opens, it arrives over the following tens of milliseconds (measured as a 512 Hz burst against a
        # 200 Hz stream), so in_waiting reads 0 and a queue-driven drain gives up before the block shows up. The byte
        # cap is what keeps this from eating a stream indefinitely -- the device ring is 256 packets, so a few times
        # that is generous and still finite.
        cap = 8 * TX_RING_PACKETS * 40
        n, quiet, t0 = 0, 0, time.monotonic()
        while time.monotonic() - t0 < seconds and n < cap:
            try: got = len(self._ser.read(4096))
            except Exception: break
            n += got
            quiet = quiet + 1 if got == 0 else 0
            if quiet >= 10: break          # a real board at 200 Hz is never silent for 50 ms; a dry source is
        try: self._ser.reset_input_buffer()
        except Exception: pass
        self.parser = PacketParser(); self.seq.resync = True
        log.info("%s: drained %d stale bytes from the device ring", self.name, n)
        return n

    def start(self) -> None:
        if self._thread: return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True); self._thread.start()
        self._status.running = True; self._status.running_since_ns = now_ns()

    def stop(self) -> None:
        self._stop.set()
        if self._thread: self._thread.join(2.0); self._thread = None
        self._status.running = False

    def close(self) -> None:
        self.stop()
        if self._ser is not None:
            try: self._ser.close()
            except Exception: pass
        self._ser = None; self._status.connected = False

    # ------------------------------------------------------------- status
    def quality(self) -> ImuQuality:
        st = self._status
        return ImuQuality(hz=self._rate.hz(), loss_ratio=self.seq.loss_ratio(), crc_errors=self.parser.crc_errors,
                          resyncs=self.parser.resyncs, max_gap_ms=self.seq.max_gap_us / 1e3, age_ms=st.age_ms,
                          seq_gaps=self.seq.seq_gaps, ts_wraps=self.seq.ts_wraps, reconnects=self.reconnects, connected=st.connected)

    def status(self) -> DeviceStatus:
        q = self.quality()
        self._status.rate_hz = q.hz
        self._status.detail.update(crc_errors=q.crc_errors, resyncs=q.resyncs, loss_ratio=round(q.loss_ratio, 5),
                                   seq_gaps=q.seq_gaps, ts_wraps=q.ts_wraps, reconnects=self.reconnects)
        return self._status

    def _emit(self, kind: str, detail: dict, t_ns: int) -> None:
        if self.on_event:
            try: self.on_event(kind, detail, t_ns)
            except Exception: pass

    # ------------------------------------------------------------- data path
    def handle(self, ptype: int, payload: bytes, host_ns: int) -> None:
        if ptype == TYPE_IMU and len(payload) >= IMU_LEN:
            s = decode_imu(self.cfg.side, payload, host_ns)
            lost, wrapped = self.seq.update(s.seq, s.device_timestamp_us)
            if lost: self._emit("imu_seq_gap", dict(lost=lost, seq=s.seq), host_ns)
            if wrapped: self._emit("imu_timestamp_wrap", dict(seq=s.seq, t_us=s.device_timestamp_us), host_ns)
            self.buffer.append(s); self.last_sample = s
            self._rate.tick(host_ns); self._status.last_sample_ns = host_ns; self._status.error = None
        elif ptype == TYPE_STATUS and len(payload) >= STATUS_LEN:
            t_us, rate, emitted, dropped, fmaj, fmin, hiwater = struct.unpack(STATUS_FMT, payload[:STATUS_LEN])
            # The device reports the deepest the ring got since the last status packet and then resets it, so the max
            # across a recording is that recording's high-water. fw 1.0 sent zero here, which reads as "never queued".
            self.ring_high_water = max(self.ring_high_water, int(hiwater))
            self.last_status = dict(t_us=t_us, sample_rate_hz=rate, emitted=emitted, dropped=dropped, fw=f"{fmaj}.{fmin}",
                                    ring_high_water=int(hiwater))
            self._status.detail.update(device=self.last_status)
        elif ptype == TYPE_LOG:
            # The firmware's only way to say why it is silent ("ICM42688P not found on SPI"). Keep the newest one on the
            # device so a tool reporting "0 samples" can say the reason instead of leaving it to the log file.
            text = payload.decode("utf-8", "replace")
            self.last_device_log = text
            log.info("%s: %s", self.name, text)
            self._emit("device_log", dict(text=text), host_ns)

    def _reconnect(self) -> bool:
        try:
            if self._ser is not None:
                try: self._ser.close()
                except Exception: pass
            self._ser = None; self._status.connected = False
            self.open(exclude=self._exclude)
            self.reconnects += 1; self.parser = PacketParser()
            self.seq.resync = True        # a reconnect delivers the device ring's stale block first, same as an open
            self._emit("device_reconnect", dict(port=self.port), now_ns())
            return True
        except Exception as exc:
            self._status.error = f"disconnected: {exc}"
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._ser is None:
                if not self._reconnect(): self._stop.wait(0.5)
                continue
            try:
                data = self._ser.read(4096)
            except Exception as exc:
                self._status.error = f"usb error: {exc}"; self._status.connected = False
                self._emit("device_error", dict(error=str(exc)), now_ns())
                self._ser = None; self._stop.wait(0.2); continue
            if not data: continue
            host_ns = now_ns()
            for ptype, payload in self.parser.feed(data):
                self.handle(ptype, payload, host_ns)


def list_teensy_ports() -> list[dict]:
    """Every serial port with USB identity, for `imu_monitor --list` and the UI identity banner."""
    from serial.tools import list_ports
    out = []
    for p in list_ports.comports():
        out.append(dict(device=p.device, serial_number=p.serial_number, vid=p.vid, pid=p.pid, manufacturer=p.manufacturer,
                        product=p.product, location=p.location))
    return out
