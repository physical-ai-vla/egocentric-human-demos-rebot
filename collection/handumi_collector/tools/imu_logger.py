"""Standalone Teensy IMU -> session directory, for board bring-up and soak tests (no hardware.yaml, one board).

    python -m handumi_collector.tools.imu_logger --list
    python -m handumi_collector.tools.imu_logger --side right --seconds 60
    python -m handumi_collector.tools.validate_session data/imu_right_20260914_120000

`imu_monitor --log` is the tool for a configured two-hand rig; this one talks to whatever single Teensy is plugged in,
which is what a freshly flashed board needs, and it leaves behind a self-describing session instead of a bare CSV:

    data/imu_<side>_<stamp>/imu.csv          side,seq,device_timestamp_us,host_receive_ns,ax,ay,az,gx,gy,gz,temp_c
                           /metadata.json    identity, firmware status, and the `devices.imu_stats` health block
                           /session.log      the status lines, plus every gap / CRC / reconnect event with a timestamp

The CSV columns are the episode schema (`pose.episode_io.ImuArrays`) in SI units, so `imu_allan` and the pose tools read
a bring-up log and a recorded episode the same way. Framing, sequence accounting and the host clock all come from
`devices.teensy_imu` / `devices.base` — this module adds no second parser and no second notion of time.

Wire format is auto-detected. Binary (PROTOCOL.md) is the production path. A CSV-text `t_us,ax,ay,az,gx,gy,gz` stream is
also accepted so a bring-up sketch can be logged before the real firmware is flashed; it is a field adapter onto the one
schema, never a second pipeline — it has no CRC, no device sequence and no clock guarantees, and `metadata.json` says so."""
from __future__ import annotations
import argparse
import csv
import json
import math
import platform
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from ..devices.base import now_ns
from ..devices.imu_stats import stream_stats
from ..devices.teensy_imu import (G0, MAGIC, TX_RING_PACKETS, PacketParser, SeqTracker, STATUS_FMT, STATUS_LEN,
                                  TYPE_IMU, TYPE_LOG, TYPE_STATUS, decode_imu, list_teensy_ports)

PJRC_VID = 0x16C0          # Teensy USB vendor id — the only reliable way to tell it from the servo bus adapters
D2R = math.pi / 180.0
TEXT_FIELDS = 7            # t_us,ax,ay,az,gx,gy,gz
DRAIN_S = 0.5              # discard the device-side backlog before recording; see drain() below
CSV_COLUMNS = ["side", "seq", "device_timestamp_us", "host_receive_ns", "ax", "ay", "az", "gx", "gy", "gz", "temp_c"]


def teensies_without_a_serial_port() -> list[dict]:
    """PJRC boards macOS has enumerated but that expose no CDC port — i.e. flashed with a USB Type other than Serial.

    A RawHID/MIDI/HID build is a working, powered, enumerated Teensy that `serial.tools.list_ports` cannot see at all, so
    it looks exactly like a dead board, a power-only cable or a bad hub port. Reading it out of the IOKit registry turns
    a USB debugging session into one line of output. macOS only; returns nothing anywhere else."""
    import plistlib
    import subprocess
    try:
        raw = subprocess.run(["ioreg", "-p", "IOUSB", "-a", "-l", "-w0"], capture_output=True, timeout=10).stdout
        tree = plistlib.loads(raw)
    except Exception:
        return []
    have = {p["serial_number"] for p in list_teensy_ports() if p["serial_number"]}
    found: list[dict] = []

    def walk(node):
        if isinstance(node, list):
            for c in node: walk(c)
            return
        if not isinstance(node, dict): return
        if node.get("idVendor") == PJRC_VID:
            sn = str(node.get("USB Serial Number", "") or "")
            if sn and sn not in have and not any(f["serial_number"] == sn for f in found):
                found.append(dict(serial_number=sn, product=str(node.get("USB Product Name", "?")), pid=node.get("idProduct")))
        walk(node.get("IORegistryEntryChildren") or [])

    walk(tree)
    return found


def serial_for_side(hardware: str, side: str) -> str | None:
    """The serial this profile assigns to `side`, so `--side left` finds its own board once two are plugged in."""
    try:
        from ..config import DEFAULT_CONFIG_DIR, load_config
        for imu in load_config(str(DEFAULT_CONFIG_DIR), hardware=hardware).hardware.imus:
            if imu.side == side and (imu.serial_number or "").strip() not in ("", "REQUIRED_SET_ME"):
                return imu.serial_number.strip()
    except Exception:
        pass
    return None


def pick_port(serial_number: str | None) -> str:
    """The one PJRC port, or the one whose serial number matches. Anything ambiguous is the user's to resolve."""
    ports = list_teensy_ports()
    if serial_number:
        m = [p for p in ports if p["serial_number"] and serial_number in p["serial_number"]]
        if not m: raise SystemExit(f"no serial port with serial number {serial_number!r} (see --list)")
        return m[0]["device"]
    teensy = [p for p in ports if p["vid"] == PJRC_VID]
    if not teensy: raise SystemExit("no Teensy (PJRC vid 0x16c0) found — plug it in, or pass --port / --serial-number (see --list)")
    if len(teensy) > 1:
        raise SystemExit("several Teensys: " + ", ".join(f"{p['device']} (serial {p['serial_number']})" for p in teensy) + " — pass --serial-number")
    return teensy[0]["device"]


def sniff(ser, seconds: float = 1.5) -> tuple[str, bytes]:
    """Read up to `seconds` of bytes and decide 'binary' | 'text'; returns (kind, bytes_read) so nothing is discarded."""
    buf = bytearray(); t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        buf += ser.read(4096)
        if MAGIC in buf and len(buf) >= 64: return "binary", bytes(buf)
        if buf.count(b"\n") >= 3: break
    if MAGIC in buf: return "binary", bytes(buf)
    if not buf:
        raise SystemExit("no bytes from the board in 1.5 s — is the Arduino IDE Serial Monitor still holding the port? "
                         "(check with: lsof /dev/cu.usbmodem*)")
    if any(l.count(b",") == TEXT_FIELDS - 1 for l in bytes(buf).split(b"\n")): return "text", bytes(buf)
    raise SystemExit(f"unrecognised stream ({len(buf)} B), first 64 bytes: {bytes(buf[:64]).hex(' ')}")


def drain(ser, seconds: float = DRAIN_S) -> int:
    """Read and discard, so recording starts on a live sample. Returns the bytes thrown away.

    `reset_input_buffer()` empties the host's buffer but not the Teensy's: its ring holds up to TX_RING_PACKETS samples
    taken while nobody was reading, and it drops the *newest* when full, so the packets waiting there are the oldest.
    Recording them opens every session with a stale block and one enormous sequence gap — measured once as 233 stale
    samples followed by a 6357-sample jump, which read as 34 % loss when the board had in fact lost nothing."""
    n = 0; t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        n += len(ser.read(4096))
    ser.reset_input_buffer()
    return n


class TextParser:
    """Bring-up-sketch adapter: `t_us,ax,ay,az,gx,gy,gz` lines to the episode schema. Banner/header lines are skipped and
    the sequence number is synthesised, so it detects nothing the device did not tell us — drops are invisible here."""

    def __init__(self, units: str) -> None:
        self._buf = bytearray(); self.seq = 0; self.skipped = 0; self.crc_errors = 0; self.resyncs = 0
        self.a_scale = G0 if units == "g_dps" else 1.0
        self.g_scale = D2R if units == "g_dps" else 1.0

    def feed(self, data: bytes) -> list[tuple]:
        self._buf += data; out = []
        *lines, rest = self._buf.split(b"\n")
        self._buf = bytearray(rest)
        for raw in lines:
            parts = raw.decode("utf-8", "replace").strip().split(",")
            if len(parts) != TEXT_FIELDS: self.skipped += 1; continue
            try: t_us, v = int(float(parts[0])), [float(x) for x in parts[1:]]
            except ValueError: self.skipped += 1; continue
            self.seq += 1
            out.append((self.seq, t_us, v[0] * self.a_scale, v[1] * self.a_scale, v[2] * self.a_scale,
                        v[3] * self.g_scale, v[4] * self.g_scale, v[5] * self.g_scale))
        return out


class Session:
    """data/imu_<side>_<stamp>/ — the CSV is streamed, the log is append-only, metadata.json is written on exit."""

    def __init__(self, path: Path, flush_every: int) -> None:
        self.path = path; self.path.mkdir(parents=True, exist_ok=True)
        self._csv_fh = open(self.path / "imu.csv", "w", newline=""); self._w = csv.writer(self._csv_fh)
        self._w.writerow(CSV_COLUMNS)
        self._log_fh = open(self.path / "session.log", "a")
        self.flush_every = flush_every; self.n = 0

    def log(self, msg: str) -> None:
        self._log_fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds')} {msg}\n"); self._log_fh.flush()

    def write(self, side, seq, dev_us, host_ns, ax, ay, az, gx, gy, gz, temp) -> None:
        self._w.writerow([side, seq, dev_us, host_ns, ax, ay, az, gx, gy, gz, temp])
        self.n += 1
        if self.n % self.flush_every == 0: self._csv_fh.flush()

    def finish(self, meta: dict) -> None:
        self._csv_fh.flush(); self._csv_fh.close()
        (self.path / "metadata.json").write_text(json.dumps(meta, indent=1, sort_keys=False) + "\n")
        self.log(f"session closed: {self.n} samples"); self._log_fh.close()

    def arrays(self) -> dict:
        """Read back from the CSV rather than keeping every row: a 3 h dual run at 200 Hz is 4.3 M samples, and holding
        them as Python tuples costs well over a gigabyte — enough to lose the whole recording at the last step. Shaped
        empties when nothing arrived, because a side that recorded no samples is a result to report, not a crash."""
        self._csv_fh.flush()
        cols = ["seq", "device_timestamp_us", "host_receive_ns", "ax", "ay", "az", "gx", "gy", "gz"]
        if self.n == 0:
            a = np.empty((0, 9), np.float64)
        else:
            try:
                import pandas as pd
                a = pd.read_csv(self.path / "imu.csv", usecols=cols)[cols].to_numpy(np.float64)
            except Exception:
                import csv as _csv
                with open(self.path / "imu.csv", newline="") as fh:
                    a = np.array([[float(r[c]) for c in cols] for r in _csv.DictReader(fh)], np.float64).reshape(-1, 9)
        return dict(seq=a[:, 0], device_us=a[:, 1], host_ns=a[:, 2], accel=a[:, 3:6], gyro=a[:, 6:9])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data", help="session root; a data/imu_<side>_<stamp>/ directory is created under it")
    ap.add_argument("--session", default=None, help="write into this exact directory instead")
    ap.add_argument("--port", default=None, help="serial device (default: the single PJRC port)")
    ap.add_argument("--serial-number", default=None, help="pick the board by USB serial number instead")
    ap.add_argument("--hardware", default="handumi_v1", help="profile consulted for this side's serial when neither --port nor --serial-number is given")
    ap.add_argument("--side", default="right", choices=["left", "right"], help="written to the side column; identity is yours to get right")
    ap.add_argument("--baud", type=int, default=2000000, help="ignored by USB CDC, but a text sketch on a real UART needs the right value")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until Ctrl-C)")
    ap.add_argument("--expected-rate", type=float, default=None, help="gate rate (default: the firmware status packet, else 200)")
    ap.add_argument("--accel-fs-g", type=float, default=8.0, help="configured accel full scale, for the saturation headroom report")
    ap.add_argument("--text-units", default="g_dps", choices=["g_dps", "si"], help="units a bring-up sketch prints (converted to SI)")
    ap.add_argument("--flush-every", type=int, default=200, help="flush the CSV every N samples")
    ap.add_argument("--drain", type=float, default=DRAIN_S,
                    help="discard this many seconds of stream before recording, to clear the device-side ring backlog")
    ap.add_argument("--read-timeout", type=float, default=0.005,
                    help="serial read timeout: every sample in one read() shares a host stamp, so this sets the host-clock quantisation")
    ap.add_argument("--list", action="store_true", help="list serial ports with USB identity and exit")
    a = ap.parse_args(argv)
    if a.list:
        teensy = []
        for p in list_teensy_ports():
            is_t = p["vid"] == PJRC_VID
            if is_t: teensy.append(p)
            print(f"{'TEENSY' if is_t else '      '}  {p['device']:32s} serial={p['serial_number']} "
                  f"vid={p['vid'] and hex(p['vid'])} pid={p['pid'] and hex(p['pid'])} product={p['product']}")
        if teensy:
            print("\nconfigs/handumi/hardware_handumi_v1.yaml — identity is the serial, never the port order:")
            for p in teensy:
                print(f'  - {{side: <left|right>, backend: teensy, port_glob: "/dev/cu.usbmodem*", serial_number: "{p["serial_number"]}", '
                      f'baud: 2000000, rate_hz: 200, required: true}}')
            if len(teensy) < 2: print(f"  ({len(teensy)} Teensy with a serial port; a two-hand rig needs two)")
        for t in teensies_without_a_serial_port():
            pid = f"0x{t['pid']:04x}" if isinstance(t["pid"], int) else t["pid"]
            print(f"\n!! Teensy serial {t['serial_number']} is enumerated as {t['product']!r} (pid {pid}) and exposes NO serial port.")
            print("   It was built with a USB Type other than Serial, so macOS creates no /dev/cu.usbmodem node for it and")
            print("   pyserial cannot see it. Reflash it with USB Type: Serial:")
            print("     arduino-cli board list        # find its usb:<id>")
            print("     arduino-cli upload -b teensy:avr:teensy41:usb=serial,speed=600 -p usb:<id> firmware/teensy_imu")
        return 0

    import serial
    want = a.serial_number or serial_for_side(a.hardware, a.side)
    port = a.port or pick_port(want)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    sess = Session(Path(a.session) if a.session else Path(a.out_dir) / f"imu_{a.side}_{stamp}", a.flush_every)
    ser = serial.Serial(port, a.baud, timeout=a.read_timeout); ser.reset_input_buffer()
    kind, _ = sniff(ser)
    discarded = drain(ser, a.drain)
    sess.log(f"open {port} baud={a.baud} wire={kind} side={a.side}; discarded {discarded} B of device backlog in {a.drain:g} s")
    print(f"{port} -> {sess.path}  ({kind} stream" + (f", {a.text_units} in" if kind == "text" else "") + ")   Ctrl-C to stop")
    if kind == "text":
        print("  ! bring-up sketch format: no CRC and no device sequence, so drops are undetectable — flash the binary firmware for real logs")

    parser = PacketParser() if kind == "binary" else TextParser(a.text_units)
    seqt = SeqTracker(); fw: dict = {}; fw_at_start = (0, 0); ring_hi = 0
    t_wall = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t_start = now_ns(); t_print = time.monotonic(); n_print = 0; last = None
    prev_crc = prev_skip = 0
    try:
        data = b""
        while True:
            if data:
                host_ns = now_ns()
                if kind == "binary":
                    for ptype, payload in parser.feed(data):
                        if ptype == TYPE_IMU:
                            s = decode_imu(a.side, payload, host_ns)
                            lost, wrapped = seqt.update(s.seq, s.device_timestamp_us)
                            if lost: sess.log(f"seq gap: {lost} sample(s) lost before seq {s.seq}")
                            if wrapped: sess.log(f"device timestamp regressed at seq {s.seq}")
                            sess.write(s.side, s.seq, s.device_timestamp_us, s.host_receive_ns, s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.temperature_c)
                            last = (s.ax, s.ay, s.az, s.gx, s.gy, s.gz)
                        elif ptype == TYPE_STATUS:
                            t_us, rate, emitted, dropped, fmaj, fmin, hiwater = struct.unpack(STATUS_FMT, payload[:STATUS_LEN])
                            ring_hi = max(ring_hi, int(hiwater))     # each packet carries the last period's peak, then resets
                            if not fw:
                                sess.log(f"firmware {fmaj}.{fmin}, device rate {rate} Hz, counters at start: emitted={emitted} dropped={dropped}")
                                fw_at_start = (int(emitted), int(dropped))
                            # The device counters are cumulative since boot, and the ring drops every sample taken while no
                            # host was reading — so the session only owns the delta. Reporting the raw total reads as alarming.
                            fw = dict(version=f"{fmaj}.{fmin}", device_rate_hz=int(rate),
                                      samples_emitted=int(emitted) - fw_at_start[0], samples_dropped=int(dropped) - fw_at_start[1],
                                      cumulative_since_boot=dict(emitted=int(emitted), dropped=int(dropped)),
                                      ring_high_water=ring_hi, ring_capacity=TX_RING_PACKETS,
                                      ring_high_water_percent=round(100.0 * ring_hi / TX_RING_PACKETS, 2))
                        elif ptype == TYPE_LOG:
                            sess.log("device: " + payload.decode("utf-8", "replace"))
                else:
                    for seq, t_us, ax, ay, az, gx, gy, gz in parser.feed(data):
                        seqt.update(seq, t_us)                     # the row's own number: `parser.seq` has already run to the end of the chunk
                        sess.write(a.side, seq, t_us, host_ns, ax, ay, az, gx, gy, gz, "")
                        last = (ax, ay, az, gx, gy, gz)
            now = time.monotonic()
            if now - t_print >= 1.0:
                hz = (sess.n - n_print) / (now - t_print); t_print, n_print = now, sess.n
                acc = last[:3] if last else (0, 0, 0); gyr = last[3:] if last else (0, 0, 0)
                if kind == "binary":
                    if parser.crc_errors != prev_crc: sess.log(f"crc errors now {parser.crc_errors} (+{parser.crc_errors - prev_crc})"); prev_crc = parser.crc_errors
                    tail = f"  crc_err {parser.crc_errors}  resync {parser.resyncs}"
                else:
                    if parser.skipped != prev_skip: prev_skip = parser.skipped
                    tail = f"  unparsed {parser.skipped}"
                print(f"\r{sess.n:9d} samples  {hz:6.1f} Hz  loss {seqt.loss_ratio()*100:5.2f}%  gaps {seqt.seq_gaps}{tail}"
                      f"  |a| {math.sqrt(sum(x * x for x in acc)) / G0:5.3f} g"
                      f"  gyro {gyr[0]/D2R:+7.1f} {gyr[1]/D2R:+7.1f} {gyr[2]/D2R:+7.1f} dps", end="", flush=True)
            if a.seconds and (now_ns() - t_start) / 1e9 >= a.seconds: break
            data = ser.read(4096)
    except KeyboardInterrupt:
        print()
    finally:
        ser.close()
    t_stop = now_ns()

    expected = a.expected_rate or fw.get("device_rate_hz") or 200.0
    stats = stream_stats(**sess.arrays(), expected_rate_hz=expected, accel_fs_g=a.accel_fs_g,
                         crc_errors=parser.crc_errors, resyncs=parser.resyncs) if sess.n else dict(n_samples=0)
    meta = dict(tool="handumi_collector.tools.imu_logger", schema_version=1, created_utc=t_wall,
                side=a.side, port=port, serial_number=next((p["serial_number"] for p in list_teensy_ports() if p["device"] == port), None),
                wire_format=kind, text_units=a.text_units if kind == "text" else None,
                drops_detectable=(kind == "binary"), firmware=fw or None,
                host=dict(clock="time.monotonic_ns", platform=platform.platform(), python=platform.python_version()),
                t_start_monotonic_ns=t_start, t_stop_monotonic_ns=t_stop,
                accel_fs_g=a.accel_fs_g, csv_columns=CSV_COLUMNS, stats=stats)
    if kind != "binary": meta["warning"] = "bring-up text stream: no CRC, synthesised sequence — sample drops are not detectable"
    sess.finish(meta)
    print(f"\n{sess.path}: {sess.n} samples, {stats.get('duration_s', 0)} s, {stats.get('rate_hz_mean', 0)} Hz mean, "
          f"drop {stats.get('drops', {}).get('drop_rate', 0)*100:.3f}%"
          + (f", crc {parser.crc_errors}, resync {parser.resyncs}" if kind == "binary" else f", unparsed lines {parser.skipped}"))
    if not sess.n:
        print("no samples decoded — check the sketch output format and --text-units", file=sys.stderr); return 1
    print(f"validate:  python -m handumi_collector.tools.validate_session {sess.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
