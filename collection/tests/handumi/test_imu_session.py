"""The bring-up logger, the stats block and the PASS/FAIL gates, over synthetic streams.

The point of these is that the standalone logger reads the *production* binary protocol byte-for-byte and lands on the
same schema an episode uses, so nothing downstream can tell the two apart — plus that the gates actually fail when the
stream is bad, which is the only reason to have gates."""
import csv
import json
import re
import struct
import time

import numpy as np
import pytest

from handumi_collector.devices.imu_stats import seq_losses, stream_stats
from handumi_collector.devices.teensy_imu import PacketParser, SEQ_MOD, TYPE_IMU, decode_imu, encode_imu, encode_status
from handumi_collector.tools import validate_session as vs
from handumi_collector.tools.imu_logger import CSV_COLUMNS, TextParser, sniff

RATE = 200.0
DT_US = int(1e6 / RATE)


def synth_stream(n=1200, *, drop_every=0, t0_us=1_000_000):
    """Production-format bytes plus the (seq, t_us) the host should recover — gravity on z, a slow gyro sweep."""
    out, expect = bytearray(), []
    seq = 0
    for i in range(n):
        seq += 1
        if drop_every and i % drop_every == drop_every - 1:
            continue                                   # emitted by the device, lost on the wire: seq keeps counting
        t_us = t0_us + i * DT_US
        acc = (0.02, -0.01, 9.80665)
        gyr = (0.001 * np.sin(i / 50), 0.0, 0.0)
        out += encode_imu(seq, t_us, acc, gyr, 31.0)
        expect.append((seq, t_us))
        if i % int(RATE) == 0:
            out += encode_status(t_us, int(RATE), seq, 0, fw=(1, 0))
    return bytes(out), expect


def parse_all(stream, chunk=64):
    p = PacketParser()
    got = []
    for i in range(0, len(stream), chunk):
        for ptype, payload in p.feed(stream[i:i + chunk]):
            if ptype == TYPE_IMU:
                got.append(decode_imu("right", payload, host_ns=0))
    return p, got


def stats_from(samples, **kw):
    a = np.array([[s.seq, s.device_timestamp_us, s.host_receive_ns, s.ax, s.ay, s.az, s.gx, s.gy, s.gz] for s in samples])
    kw.setdefault("expected_rate_hz", RATE)
    return stream_stats(seq=a[:, 0], device_us=a[:, 1], host_ns=None, accel=a[:, 3:6], gyro=a[:, 6:9], **kw)


# ---------------------------------------------------------------- wire compatibility

def test_logger_reads_production_protocol_byte_for_byte():
    stream, expect = synth_stream(n=400)
    p, got = parse_all(stream)
    assert p.crc_errors == 0 and p.resyncs == 0
    assert [(s.seq, s.device_timestamp_us) for s in got] == expect
    assert abs(got[0].az - 9.80665) < 1e-5              # SI on the wire, SI in the row


def test_sniff_tells_binary_from_bring_up_text():
    class FakeSerial:
        def __init__(self, data): self._d = data
        def read(self, n): d, self._d = self._d[:n], self._d[n:]; return d

    binary, _ = synth_stream(n=50)
    assert sniff(FakeSerial(binary))[0] == "binary"
    text = b"# WHO_AM_I=0x47\n" + b"".join(f"{i*5000},0.0,0.0,1.0,0.1,0.2,0.3\n".encode() for i in range(10))
    assert sniff(FakeSerial(text))[0] == "text"


def test_text_adapter_converts_to_si_and_skips_banners():
    p = TextParser("g_dps")
    rows = p.feed(b"# DEVICE=ICM42688P\nt_us,ax,ay,az\n1000,0.0,0.0,1.0,0.0,0.0,90.0\nnot,a,row\n")
    assert len(rows) == 1 and p.skipped == 3
    seq, t_us, ax, ay, az, gx, gy, gz = rows[0]
    assert (seq, t_us) == (1, 1000) and abs(az - 9.80665) < 1e-6 and abs(gz - np.pi / 2) < 1e-6


def test_text_adapter_numbers_every_row_in_a_chunk():
    """Each row carries its own sequence number. Using the parser's counter after the call gave every row in the chunk
    the same number, which the sequence tracker then read as a burst of losses at each chunk boundary."""
    p = TextParser("si")
    chunk = b"".join(f"{i * 5000},0,0,9.80665,0,0,0\n".encode() for i in range(1, 6))
    assert [r[0] for r in p.feed(chunk)] == [1, 2, 3, 4, 5]
    assert [r[0] for r in p.feed(b"30000,0,0,9.80665,0,0,0\n")] == [6]


# ---------------------------------------------------------------- sequence accounting

@pytest.mark.parametrize("seq,lost,gaps", [
    ([1, 2, 3, 4], 0, 0),
    ([1, 2, 4, 5], 1, 1),
    ([1, 2, 6, 7, 9], 4, 2),
    ([SEQ_MOD - 2, SEQ_MOD - 1, 0, 1], 0, 0),                 # uint32 wrap is not a loss
    ([SEQ_MOD - 2, 1], 2, 1),                                 # wrap with a real gap across it
    ([5, 4, 6], 1, 1),                                        # a reorder is charged as one loss, never as 2^32 of them
])
def test_seq_losses_is_wrap_aware(seq, lost, gaps):
    assert seq_losses(np.array(seq, np.int64)) == (lost, gaps)


def test_stats_measure_rate_and_drops():
    _, got = parse_all(synth_stream(n=1200)[0])
    st = stats_from(got, accel_fs_g=8.0)
    assert abs(st["rate_hz_mean"] - RATE) < 0.5
    assert st["drops"]["lost_samples"] == 0 and st["crc_errors"] == 0
    assert abs(st["interval_ms"]["median"] - 5.0) < 1e-6
    assert 0.99 < st["accel_g"]["norm_median"] < 1.01
    assert st["accel_g"]["headroom_used"] < 0.2
    json.dumps(st)                                   # metadata.json is written from this dict: no numpy scalars allowed

    _, dropped = parse_all(synth_stream(n=1200, drop_every=10)[0])
    st_d = stats_from(dropped)
    assert st_d["drops"]["lost_samples"] == 119          # 120 dropped; the last one has no successor to reveal it
    assert st_d["drops"]["drop_rate"] == pytest.approx(0.1, abs=1e-3)


# ---------------------------------------------------------------- session round trip + gates

def write_session(tmp_path, samples, *, name="imu_right_test", expected_rate=RATE, accel_fs_g=8.0, drops_detectable=True):
    d = tmp_path / name
    d.mkdir()
    with open(d / "imu.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for s in samples:
            w.writerow(["right", s.seq, s.device_timestamp_us, s.host_receive_ns, s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.temperature_c])
    (d / "metadata.json").write_text(json.dumps(dict(accel_fs_g=accel_fs_g, drops_detectable=drops_detectable,
                                                     stats=dict(expected_rate_hz=expected_rate, crc_errors=0, resyncs=0))))
    return d


def host_stamped(samples, *, jitter_us=120, seed=0):
    """Give each sample a plausible host_receive_ns: nominal mapping plus one-sided USB latency."""
    rng = np.random.default_rng(seed)
    out = []
    for s in samples:
        lat = abs(rng.normal(0, jitter_us)) * 1000
        out.append(type(s)(s.side, s.seq, s.device_timestamp_us, int(s.device_timestamp_us * 1000 + lat),
                           s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.temperature_c))
    return out


def test_good_static_session_passes_every_gate(tmp_path):
    _, got = parse_all(synth_stream(n=2000)[0])
    d = write_session(tmp_path, host_stamped(got))
    stats, prov = vs.load_session(d, None)
    rows = vs.gates(stats, prov, static=True)
    assert [n for n, ok, _ in rows if not ok] == []
    assert dict((n, det) for n, _, det in rows)["rate"].startswith("200.0")


def test_drop_gate_catches_what_the_rate_gate_cannot(tmp_path):
    """2 % of samples lost on the wire moves the mean rate to ~196 Hz, still inside the +/-2.5 % rate band.

    This is the whole argument for carrying a device sequence number: timing alone looks healthy here."""
    _, got = parse_all(synth_stream(n=2000, drop_every=50)[0])
    d = write_session(tmp_path, host_stamped(got))
    stats, prov = vs.load_session(d, None)
    rows = dict((n, (ok, det)) for n, ok, det in vs.gates(stats, prov, static=True))
    assert rows["rate"][0] is True, rows["rate"][1]
    assert rows["drops"][0] is False and stats["drops"]["drop_rate"] == pytest.approx(0.02, abs=2e-3)


def test_half_rate_firmware_fails_the_rate_gate(tmp_path):
    """The ODR trap: a board that actually runs 100 Hz while the config claims 200 must not pass."""
    _, got = parse_all(synth_stream(n=800)[0])
    half = got[::2]
    d = write_session(tmp_path, host_stamped(half))
    stats, prov = vs.load_session(d, None)
    failed = [n for n, ok, _ in vs.gates(stats, prov, static=True) if not ok]
    assert "rate" in failed and "jitter_p99" in failed


def test_text_session_cannot_claim_zero_drops(tmp_path):
    _, got = parse_all(synth_stream(n=2000)[0])
    d = write_session(tmp_path, host_stamped(got), drops_detectable=False)
    stats, prov = vs.load_session(d, None)
    drops = [(ok, det) for n, ok, det in vs.gates(stats, prov, static=True) if n == "drops"][0]
    assert drops[0] is False and "not detectable" in drops[1]


def test_saturating_accel_fails_the_headroom_gate(tmp_path):
    stream, _ = synth_stream(n=1200)
    p, got = parse_all(stream)
    hot = [type(s)(s.side, s.seq, s.device_timestamp_us, s.host_receive_ns, 7.9 * 9.80665, s.ay, s.az,
                   s.gx, s.gy, s.gz, s.temperature_c) for s in got]
    d = write_session(tmp_path, host_stamped(hot))
    stats, prov = vs.load_session(d, None)
    failed = [n for n, ok, _ in vs.gates(stats, prov, static=True) if not ok]
    assert "accel_headroom" in failed


def test_clock_fit_recovers_the_us_to_ns_mapping(tmp_path):
    _, got = parse_all(synth_stream(n=2000)[0])
    samples = host_stamped(got)
    a = np.array([[s.seq, s.device_timestamp_us, s.host_receive_ns, s.ax, s.ay, s.az, s.gx, s.gy, s.gz] for s in samples])
    st = stream_stats(seq=a[:, 0], device_us=a[:, 1], host_ns=a[:, 2], accel=a[:, 3:6], gyro=a[:, 6:9], expected_rate_hz=RATE)
    assert abs(st["clock_fit"]["slope_ns_per_us"] - 1000.0) < 0.5
    assert st["clock_fit"]["residual_std_ms"] < 1.0


# ---------------------------------------------------------------- host-clock quantisation

class FakePort:
    def __init__(self, device, serial_number): self.device, self.serial_number = device, serial_number


def fake_bus(monkeypatch, ports, opened=None):
    """Patch the serial layer so open() can be exercised without hardware."""
    import handumi_collector.devices.teensy_imu as ti

    class FakeSerial:
        def __init__(self, port, baud, timeout=None):
            if opened is not None: opened.update(port=port, baud=baud, timeout=timeout)
        def reset_input_buffer(self): pass

    monkeypatch.setattr("serial.Serial", FakeSerial)
    monkeypatch.setattr("serial.tools.list_ports.comports", lambda: list(ports))
    return ti


def test_driver_opens_the_port_with_the_short_read_timeout(monkeypatch):
    """The episode logger path must carry the same host-stamp precision the bring-up logger does."""
    from handumi_collector.config import ImuCfg
    opened = {}
    ti = fake_bus(monkeypatch, [FakePort("/dev/fake", "20540160")], opened)
    ti.TeensyImu(ImuCfg("right", backend="teensy", rate_hz=200, serial_number="20540160")).open()
    assert opened["port"] == "/dev/fake"
    assert opened["timeout"] == ti.READ_TIMEOUT_S <= 0.01


# ---------------------------------------------------------------- LEFT/RIGHT identity

def test_open_refuses_to_guess_the_hand_from_port_order(monkeypatch):
    """Two identical boards on /dev/cu.usbmodem* differ only by serial. Without one, whichever enumerates first would
    become LEFT — a swap that silently mislabels every episode and cannot be detected after the fact."""
    from handumi_collector.config import ImuCfg
    ti = fake_bus(monkeypatch, [FakePort("/dev/a", "20540160"), FakePort("/dev/b", "20540199")])
    for missing in (None, "", "REQUIRED_SET_ME"):
        with pytest.raises(RuntimeError, match="never from port enumeration order"):
            ti.TeensyImu(ImuCfg("left", backend="teensy", serial_number=missing)).open()


def test_open_refuses_an_ambiguous_partial_serial(monkeypatch):
    """Matching is by substring, so a truncated serial can name two boards."""
    from handumi_collector.config import ImuCfg
    ti = fake_bus(monkeypatch, [FakePort("/dev/a", "20540160"), FakePort("/dev/b", "205401601")])
    with pytest.raises(RuntimeError, match="matches 2 ports"):
        ti.TeensyImu(ImuCfg("right", backend="teensy", serial_number="2054016")).open()


def test_open_refuses_a_port_another_device_already_took(monkeypatch):
    from handumi_collector.config import ImuCfg
    ti = fake_bus(monkeypatch, [FakePort("/dev/a", "20540160")])
    with pytest.raises(RuntimeError, match="already taken"):
        ti.TeensyImu(ImuCfg("right", backend="teensy", serial_number="20540160")).open(exclude=("/dev/a",))


@pytest.mark.parametrize("left,right", [("20540160", "20540160"), ("2054016", "20540160")])
def test_manager_refuses_two_sides_claiming_one_board(left, right):
    """resolve-by-serial ignores the taken-port list, so duplicated serials would have both sides read one stream."""
    from handumi_collector.config import HardwareCfg, ImuCfg
    from handumi_collector.devices.manager import DeviceManager
    hw = HardwareCfg(cameras=[], grippers=[],
                     imus=[ImuCfg("left", backend="teensy", serial_number=left), ImuCfg("right", backend="teensy", serial_number=right)])
    dm = DeviceManager(hw); dm.build()
    with pytest.raises(RuntimeError, match="LEFT/RIGHT identity would be"):
        dm.connect_all()


def test_manager_accepts_two_distinct_boards(monkeypatch):
    from handumi_collector.config import HardwareCfg, ImuCfg
    from handumi_collector.devices.manager import DeviceManager
    fake_bus(monkeypatch, [FakePort("/dev/a", "20540160"), FakePort("/dev/b", "20540199")])
    hw = HardwareCfg(cameras=[], grippers=[],
                     imus=[ImuCfg("left", backend="teensy", serial_number="20540199"), ImuCfg("right", backend="teensy", serial_number="20540160")])
    dm = DeviceManager(hw); dm.build(); dm.connect_all()
    assert dm.errors == {}
    assert (dm.imus["left"].port, dm.imus["right"].port) == ("/dev/b", "/dev/a")   # by serial, not by enumeration order


@pytest.mark.parametrize("quant_ms,limit_ms", [(50.0, 20.0), (5.0, 2.0)])
def test_read_quantisation_sets_the_clock_fit_residual(quant_ms, limit_ms):
    """Why READ_TIMEOUT_S matters: one host stamp per read means the residual is the quantisation, ~q/sqrt(12).

    Reproduces the bench measurement (50 ms -> 14.5 ms, 5 ms -> 1.05 ms) from synthetic samples."""
    from handumi_collector.pose.timing import fit_device_to_host

    n, dt_us = 5000, 5052                                        # the board's measured period: not a divisor of either quantum
    device_us = np.arange(n, dtype=np.int64) * dt_us + 1_000_000
    true_host_ns = device_us.astype(np.int64) * 1000
    q_ns = int(quant_ms * 1e6)
    phase = 1_234_567
    host_ns = ((true_host_ns + phase) // q_ns + 1) * q_ns - phase  # every sample in a read gets that read's stamp
    fit = fit_device_to_host(device_us, host_ns)
    assert abs(fit.slope_ns_per_us - 1000.0) < 1.0               # the device clock is recovered either way
    assert fit.residual_std_ms < limit_ms
    assert fit.residual_std_ms > quant_ms / 12                   # and it really is set by the quantisation


# ---------------------------------------------------------------- dual-wrist smoke

def test_dual_smoke_runs_both_sides_through_the_device_manager(tmp_path, capsys):
    """End to end on mock devices: two sessions, each labelled with its own side, both gated."""
    from handumi_collector.tools import imu_dual_smoke

    rc = imu_dual_smoke.main(["--hardware", "mock", "--seconds", "5.5", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and out.rstrip().endswith("distinct boards")
    dirs = sorted(p.name.rsplit("_", 2)[0] for p in tmp_path.iterdir())
    assert dirs == ["imu_left", "imu_right"]
    for d in tmp_path.iterdir():
        side = d.name.split("_")[1]
        meta = json.loads((d / "metadata.json").read_text())
        assert meta["side"] == side and meta["synthetic"] is True
        rows = list(csv.DictReader(open(d / "imu.csv", newline="")))
        assert rows and {r["side"] for r in rows} == {side}      # no cross-contamination between the two streams


def test_dual_smoke_refuses_an_unconfigured_side(tmp_path, capsys, monkeypatch):
    """A side whose serial was never filled in must stop the run, not be resolved by whichever port enumerates first.

    The profile is forced here rather than read, so this keeps testing the invariant after the left board's real serial
    lands in handumi_v1.yaml."""
    from handumi_collector.tools import imu_dual_smoke

    real_load = imu_dual_smoke.load_config

    def unconfigured_left(*a, **k):
        cfg = real_load(*a, **k)
        for imu in cfg.hardware.imus:
            imu.backend = "teensy"
            # A serial no board can answer to: with a real one here the test opens the Teensy on this desk, starts a
            # reader thread on it, and leaves it holding the port when main() bails out.
            imu.serial_number = "REQUIRED_SET_ME" if imu.side == "left" else "NO-SUCH-BOARD"
        return cfg

    monkeypatch.setattr(imu_dual_smoke, "load_config", unconfigured_left)
    rc = imu_dual_smoke.main(["--hardware", "mock", "--seconds", "1", "--out-dir", str(tmp_path)])
    assert rc == 2
    assert "never from port enumeration order" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_finds_a_teensy_that_exposes_no_serial_port(monkeypatch):
    """A board built with USB Type: RawHID is powered, enumerated and completely invisible to pyserial — it looks the
    same as a dead board or a power-only cable. Real capture from ioreg, trimmed."""
    import plistlib
    from handumi_collector.tools import imu_logger

    tree = [{"idVendor": 0x16C0, "USB Serial Number": "20540160", "USB Product Name": "USB Serial", "idProduct": 0x0483,
             "IORegistryEntryChildren": [
                 {"idVendor": 0x1A86, "USB Serial Number": "5B3D048082", "USB Product Name": "USB Single Serial"},
                 {"IORegistryEntryChildren": [
                     {"idVendor": 0x16C0, "USB Serial Number": "20778070", "USB Product Name": "Teensyduino RawHID",
                      "idProduct": 0x0486}]}]}]

    class Done:
        stdout = plistlib.dumps(tree)

    monkeypatch.setattr(imu_logger, "list_teensy_ports",
                        lambda: [dict(device="/dev/cu.usbmodem205401601", serial_number="20540160", vid=0x16C0, pid=0x483, product="USB Serial")])
    monkeypatch.setattr("subprocess.run", lambda *a, **k: Done())
    got = imu_logger.teensies_without_a_serial_port()
    assert [(g["serial_number"], g["product"]) for g in got] == [("20778070", "Teensyduino RawHID")]   # nested, and the CDC one is not reported


def test_empty_session_reports_rather_than_crashing(tmp_path):
    """A side that recorded nothing is a result. arrays() used to hand back a 1-D array and raise IndexError inside the
    code that was trying to report the failure, taking the healthy side's report down with it."""
    from handumi_collector.tools.imu_logger import Session

    sess = Session(tmp_path / "imu_left_empty", 10)
    arrs = sess.arrays()
    assert arrs["seq"].shape == (0,) and arrs["accel"].shape == (0, 3) and arrs["gyro"].shape == (0, 3)
    st = stream_stats(**arrs, expected_rate_hz=200.0)
    assert st["n_samples"] == 0
    assert [n for n, ok, _ in vs.gates(st, {}, static=True) if not ok] == ["samples"]


def test_double_triggered_samples_are_caught_by_their_own_gate():
    """A glitching data-ready line adds samples instead of losing them, so every drop check stays clean and p99 -- which
    looks only at the slow tail -- stays exactly on nominal. Measured on a wrist harness whose INT1 jumper ran beside the
    8 MHz SPI clock: a partner sample 19 us behind roughly one in nine, mean rate pulled to 226 Hz."""
    _, got = parse_all(synth_stream(n=2000)[0])
    rows = [(s.seq, s.device_timestamp_us, s.ax, s.ay, s.az, s.gx, s.gy, s.gz) for s in got]
    glitched, seq = [], 0
    for i, (_, t_us, *v) in enumerate(rows):
        seq += 1; glitched.append((seq, t_us, *v))
        if i % 9 == 0:                                   # a spurious edge 19 us later, sequence still contiguous
            seq += 1; glitched.append((seq, t_us + 19, *v))
    a = np.array(glitched, float)
    st = stream_stats(seq=a[:, 0], device_us=a[:, 1], host_ns=None, accel=a[:, 2:5], gyro=a[:, 5:8], expected_rate_hz=RATE)

    assert st["drops"]["lost_samples"] == 0                       # nothing was lost, so no drop check fires
    assert abs(st["interval_ms"]["p99"] - 5.0) < 0.1              # and the slow tail is untouched
    assert st["rate_hz_mean"] > 215                               # only the mean rate notices
    assert st["short_intervals"]["count"] == 223
    rows_out = dict((n, ok) for n, ok, _ in vs.gates(st, {}, static=False))
    assert rows_out["double_trigger"] is False and rows_out["rate"] is False


# ---------------------------------------------------------------- ring occupancy telemetry

def test_ring_high_water_is_the_max_of_the_reported_peaks(monkeypatch):
    """The device reports the peak since the last status packet and then resets, because a maximum cannot be differenced
    the way a counter can — so a since-boot value would carry in whatever piled up before anyone was listening. The host
    takes the max across the recording."""
    import handumi_collector.devices.teensy_imu as ti
    from handumi_collector.config import ImuCfg

    imu = ti.TeensyImu(ImuCfg("right", backend="teensy", rate_hz=200, serial_number="20540160"))
    assert imu.ring_high_water == 0
    for peak in (3, 11, 2, 7):
        imu.handle(ti.TYPE_STATUS, struct.pack(ti.STATUS_FMT, 1_000_000, 200, 500, 0, 1, 0, peak), host_ns=0)
    assert imu.ring_high_water == 11                       # the max, not the latest and not the sum
    assert imu.last_status["ring_high_water"] == 7         # while last_status keeps the most recent period


def test_ring_high_water_field_does_not_change_the_status_layout():
    """fw 1.0 sent this field as zero, so an old board still parses and simply reads as 'never queued'."""
    import handumi_collector.devices.teensy_imu as ti

    old_board = ti.encode_status(1_000_000, 200, 500, 0, fw=(1, 0))
    p = ti.PacketParser()
    out = p.feed(old_board)
    assert len(out) == 1 and p.crc_errors == 0
    assert struct.unpack(ti.STATUS_FMT, out[0][1][:ti.STATUS_LEN])[6] == 0


def test_a_stream_that_dies_partway_through_fails(tmp_path):
    """Every other gate is computed over whatever arrived, so a side that goes silent halfway still satisfies all of
    them — a 60 s dual run where one wrist stopped at 30 s came back PASS on nine gates out of nine."""
    _, got = parse_all(synth_stream(n=4000)[0])          # 20 s, so the surviving half is still a long-enough record
    half = got[: len(got) // 2]
    d = write_session(tmp_path, host_stamped(half))
    meta = json.loads((d / "metadata.json").read_text())
    wall_ns = int(len(got) / RATE * 1e9)                      # the session ran for the full record's worth of time
    meta.update(t_start_monotonic_ns=0, t_stop_monotonic_ns=wall_ns)
    (d / "metadata.json").write_text(json.dumps(meta))

    stats, prov = vs.load_session(d, None)
    rows = dict((n, (ok, det)) for n, ok, det in vs.gates(stats, prov, static=True))
    assert rows["duration"][0] is True and rows["rate"][0] is True     # the truncated record still looks healthy
    assert rows["coverage"][0] is False
    pct = float(re.match(r"([\d.]+)%", rows["coverage"][1]).group(1))
    assert 45.0 < pct < 55.0, rows["coverage"][1]


def test_a_stall_inside_the_record_fails_max_gap(tmp_path):
    _, got = parse_all(synth_stream(n=2000)[0])
    shifted = [type(s)(s.side, s.seq, s.device_timestamp_us + (2_000_000 if i > 1000 else 0), s.host_receive_ns,
                       s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.temperature_c) for i, s in enumerate(got)]
    d = write_session(tmp_path, host_stamped(shifted))
    stats, prov = vs.load_session(d, None)
    rows = dict((n, ok) for n, ok, _ in vs.gates(stats, prov, static=True))
    assert rows["max_gap"] is False


# ---------------------------------------------------------------- table frame / head mount

def test_table_frame_recovers_a_known_camera_pose():
    """Project a board from a known camera pose, solve it back, and check the transform points the way it claims:
    T_table_camera must carry camera coordinates into the table frame, not the other way round."""
    import cv2
    from handumi_collector.pose.se3 import inv_T, make_T
    from handumi_collector.tools.calibrate_table_frame import board_points, solve_view, table_frame

    rows, cols, sq = 6, 9, 0.025
    obj = board_points(rows, cols, sq)
    K = np.array([[600.0, 0, 424.0], [0, 600.0, 240.0], [0, 0, 1]])
    # camera 0.82 m above the table, 0.35 m back, tilted 38 deg down — the mount this tool exists for
    pitch = np.radians(180 - 38)
    R_cb = cv2.Rodrigues(np.array([pitch, 0.0, 0.0]))[0]
    T_camera_board = make_T(R_cb, [-0.1, -0.25, 0.9])

    views = []
    for jitter in (0.0, 3e-4, -2e-4, 1e-4):
        T = T_camera_board.copy(); T[:3, 3] += jitter
        P = (T[:3, :3] @ obj.T).T + T[:3, 3]
        uv = (P[:, :2] / P[:, 2:3]) @ K[:2, :2].T + K[:2, 2]
        views.append(solve_view(uv.astype(np.float64), obj, K, np.zeros(5)))

    res = table_frame(views)
    assert res["rms_reproj_px"] < 0.01
    assert res["view_spread"]["translation_mm"] < 1.0
    assert np.allclose(res["T_camera_board"], T_camera_board, atol=2e-3)
    assert np.allclose(res["T_table_camera"], inv_T(T_camera_board), atol=2e-3)

    origin_cam = T_camera_board[:3, 3]                       # the board origin, seen by the camera
    in_table = (res["T_table_camera"] @ np.append(origin_cam, 1.0))[:3]
    assert np.allclose(in_table, 0, atol=2e-3), in_table     # ...must land on the table frame's origin


def test_depth_plane_check_sees_a_tilted_depth_map():
    """The board is flat on the table, so a plane fitted to the depth inside it must agree with the pose solved from
    colour. Disagreement is depth-to-colour misalignment or a wrong depth scale, and is worth catching before a dataset."""
    from handumi_collector.pose.se3 import make_T
    from handumi_collector.tools.calibrate_table_frame import depth_plane_check

    K = np.array([[600.0, 0, 200.0], [0, 600.0, 150.0], [0, 0, 1]])
    ys, xs = np.mgrid[0:300, 0:400]
    corners = np.array([[60.0, 60.0], [340.0, 240.0]])

    flat = np.full((300, 400), 0.9)                          # a fronto-parallel plane at 0.9 m
    T = make_T(np.eye(3), [0, 0, 0.9])                       # board normal along +Z, same plane
    r = depth_plane_check(flat, K, corners, T)
    assert r["plane_rms_mm"] < 0.5 and r["tilt_vs_board_deg"] < 0.5

    tilted = 0.9 + (xs - 200) * 0.0005                       # same board pose, depth map tipped about Y
    r2 = depth_plane_check(tilted.astype(np.float64), K, corners, T)
    assert r2["tilt_vs_board_deg"] > 5.0, r2


# ---------------------------------------------------------------- rgb/depth spatial alignment

def _scene(shift_px=0, size=(240, 320)):
    """A bright square on a dark table: colour shows its outline, depth shows it standing 10 cm proud. `shift_px`
    slides the depth copy sideways, which is exactly the failure this metric exists to find."""
    h, w = size
    rgb = np.full((h, w, 3), 30, np.uint8)
    depth = np.full((h, w), 1.00, np.float32)
    y0, y1, x0, x1 = 80, 160, 110, 210
    rgb[y0:y1, x0:x1] = 220
    depth[y0:y1, x0 + shift_px:x1 + shift_px] = 0.90
    return rgb, depth


def test_edge_alignment_separates_aligned_depth_from_shifted_depth():
    from handumi_collector.pose.rgbd_align import edge_alignment, summarise

    good = edge_alignment(*_scene(0))
    assert good["median_offset_px"] <= 1.0, good
    assert good["overlap"] > 0.9, good

    # Canny and a depth gradient do not place an edge on exactly the same pixel, so agreement means within a pixel,
    # not identical. Anything larger is the depth map genuinely sitting somewhere else.
    assert good["shift_magnitude_px"] <= 1.5, good

    bad = edge_alignment(*_scene(8))
    # The distance distribution barely moves: the square's horizontal edges outnumber the vertical ones and a sideways
    # shift just slides them along themselves. The estimated shift is the number that actually sees it.
    assert bad["median_offset_px"] < 2.0, bad
    assert bad["shift_px"] == [-8, 0], bad          # the correction: depth is 8 px right, so move it 8 px left
    assert bad["residual_median_px"] <= good["median_offset_px"] + 0.5

    assert summarise([good] * 3)["problems"] == []
    gross = summarise([edge_alignment(*_scene(20))] * 3)
    assert gross["problems"] and "gross misalignment" in gross["problems"][0], gross


def test_edge_alignment_says_so_when_there_is_nothing_to_compare():
    from handumi_collector.pose.rgbd_align import edge_alignment, summarise

    flat = edge_alignment(np.full((240, 320, 3), 30, np.uint8), np.full((240, 320), 1.0, np.float32))
    assert "error" in flat and "median_offset_px" not in flat
    s = summarise([flat, flat])
    assert s["usable"] == 0 and s["problems"] == [] and s["warnings"]


def test_depth_holes_do_not_masquerade_as_object_edges():
    """Invalid depth is 0, so a hole boundary is a huge gradient. Counting it as an object edge would make a dropout
    look like perfect structure that happens to sit nowhere near the colour."""
    from handumi_collector.pose.rgbd_align import _edges_depth

    d = np.full((100, 100), 1.0, np.float32)
    d[40:60, 40:60] = 0.0                                   # a dropout, not a step
    assert _edges_depth(d).sum() == 0
    d2 = np.full((100, 100), 1.0, np.float32)
    d2[40:60, 40:60] = 0.9                                  # a real 10 cm step
    assert _edges_depth(d2).sum() > 50


def test_mount_geometry_reads_the_holder_off_the_transform():
    """Height, pitch, yaw and roll are properties of the solved pose, not things to measure with a tape. Getting them
    from the calibration makes them true by construction rather than true to within how the tape was held."""
    import cv2
    from handumi_collector.pose.se3 import make_T
    from handumi_collector.tools.calibrate_table_frame import mount_geometry

    def cam(pitch_below_deg, height_m=0.82, xy=(0.0, 0.0)):
        R = cv2.Rodrigues(np.array([np.radians(90 + pitch_below_deg), 0.0, 0.0]))[0]
        return make_T(R, [xy[0], xy[1], height_m])

    assert mount_geometry(cam(90))["pitch_deg"] == 90.0            # straight down at the table
    assert mount_geometry(cam(0))["pitch_deg"] == 0.0              # looking along the table
    g = mount_geometry(cam(38, height_m=0.82, xy=(0.30, -0.40)))
    assert g["pitch_deg"] == 38.0
    assert g["height_cm"] == 82.0
    assert g["board_distance_cm"] == round(np.hypot(0.30, 0.40) * 100, 1)


def test_table_frame_z_points_out_of_the_table():
    """The first real calibration wrote height -48.3 cm and pitch -62.5 deg -- a camera under the table looking up --
    from a solve whose reprojection was 0.36 px. Which way solvePnP faces a board's normal is not something to leave to
    the target, because a holder above a table is the only case there is."""
    from handumi_collector.pose.se3 import make_T
    from handumi_collector.tools.calibrate_table_frame import mount_geometry, z_out_of_the_table

    upside_down = make_T(np.diag([1.0, -1.0, -1.0]), [0.08, 0.28, -0.48])
    assert mount_geometry(upside_down)["height_cm"] < 0
    fixed = z_out_of_the_table(upside_down)
    assert mount_geometry(fixed)["height_cm"] > 0
    assert np.isclose(np.linalg.det(fixed[:3, :3]), 1.0)                     # still right-handed
    assert np.allclose(fixed[:3, 0], upside_down[:3, 0])                     # the board's in-plane X is kept
    assert np.allclose(z_out_of_the_table(fixed), fixed)                     # and it is idempotent


def test_roll_is_the_image_tipping_out_of_the_table_plane():
    """Roll must be zero for a level camera whatever it is pointing at. Built from the camera's +Y -- which is image
    *down* in OpenCV -- it read 170 deg for a rig that was 4.5 deg off level."""
    import cv2
    from handumi_collector.pose.se3 import make_T
    from handumi_collector.tools.calibrate_table_frame import mount_geometry

    def level_camera(pitch_deg, yaw_deg):
        """A physically ordinary downward-looking camera: image-right level with the table, image-down towards the near
        side. Columns are [right, down, forward] because that is the OpenCV frame -- +Y is down in the image."""
        p = np.radians(pitch_deg)
        R = np.stack([[1.0, 0.0, 0.0], [0.0, -np.sin(p), -np.cos(p)], [0.0, np.cos(p), -np.sin(p)]], axis=1)
        return make_T(cv2.Rodrigues(np.array([0.0, 0.0, np.radians(yaw_deg)]))[0] @ R, [0.0, 0.0, 0.8])

    for pitch in (20, 45, 70):
        for yaw in (0, 37, -95):
            assert abs(mount_geometry(level_camera(pitch, yaw))["roll_deg"]) < 0.05, (pitch, yaw)

    # Rolling about the camera's own optical axis must report that angle, unchanged by how far it is tilted down --
    # an earlier version measured the image x axis tipping out of the table plane, which is a projection and read
    # 7.1 deg for this 10 deg roll at 45 deg of pitch.
    for pitch in (20, 45, 70):
        tipped = level_camera(pitch, 0) @ np.block(
            [[cv2.Rodrigues(np.array([0.0, 0.0, np.radians(10)]))[0], np.zeros((3, 1))],
             [np.zeros((1, 3)), np.ones((1, 1))]])
        assert abs(abs(mount_geometry(tipped)["roll_deg"]) - 10.0) < 0.2, pitch


def test_a_board_turned_between_views_is_refused():
    """The table frame is defined by where the board is — origin at a corner, X along it — so turning it between views
    defines a second frame rather than giving a second look at the first, and their mean matches neither. Nothing in the
    reprojection error shows it: every view still solves perfectly. Any angle counts, not just a conspicuous one."""
    import cv2
    from handumi_collector.pose.se3 import make_T
    from handumi_collector.tools.calibrate_table_frame import board_moved, table_frame

    T0 = make_T(cv2.Rodrigues(np.array([np.radians(142.0), 0, 0]))[0], [-0.10, -0.25, 0.58])
    assert board_moved(table_frame([(T0, 0.3)] * 4)) is None            # a board held still is fine

    for deg in (5, 15, 45, 90, 180):
        turned = T0 @ make_T(cv2.Rodrigues(np.array([0.0, 0.0, np.radians(deg)]))[0], [0, 0, 0])
        mixed = table_frame([(T0, 0.3)] * 4 + [(turned, 0.3)] * 4)
        assert mixed["rms_reproj_px"] == 0.3, "the per-view fit stays perfect, which is why this needs its own check"
        assert board_moved(mixed) is not None, f"{deg} deg went unnoticed"
        # The mean sits halfway for a modest turn; at 180 deg it is degenerate — two opposite rotations have no
        # meaningful average — and the spread comes out as the full turn rather than half of it.
        assert mixed["view_spread"]["rotation_deg"] >= deg / 2 - 0.5, deg


# ---------------------------------------------------------------- synchronized replay

def _fake_meta(stream, caps_ns):
    from handumi_collector.pose.episode_io import FrameMeta
    n = len(caps_ns)
    return FrameMeta(stream, np.arange(n), np.arange(n), np.asarray(caps_ns, np.int64))


def test_replay_reports_the_delta_it_had_to_accept():
    """Streams do not share a clock tick, so a viewer must pick a nearest frame for each. Printing the delta it settled
    for is the difference between showing synchronisation and assuming it: on a real 30 fps take the wrist cameras land
    10-15 ms from the head, which is up to half a frame."""
    from handumi_collector.tools.inspect_episode import _Reader

    class FakeCap:
        def __init__(self): self.seeks = []
        def set(self, _prop, v): self.seeks.append(int(v))
        def read(self): return True, f"frame{self.seeks[-1]}"

    r = _Reader.__new__(_Reader)                                  # no real video: the time arithmetic is what is tested
    r.fm = _fake_meta("head", [0, 33_000_000, 66_000_000, 99_000_000])
    r.has_depth = False
    r.cap = FakeCap()
    r._vf, r._img = 1, "cached"

    img, d_ms, vf = r.at(34_000_000)
    assert (vf, img) == (1, "cached") and d_ms == pytest.approx(-1.0)       # nearest is just behind
    assert r.at(49_000_000)[2] == 1                                          # 16 ms past frame 1, 17 short of frame 2
    assert r.at(50_000_000)[2] == 2                                          # and 50 ms is the other side of halfway
    assert r.at(-10_000_000)[2] == 0 and r.at(10 ** 12)[2] == 3              # clamped at both ends
    assert r.cap.seeks == [2, 0, 3], "one seek per new frame, none while the answer is already decoded"


def test_replay_marks_the_imu_sample_it_would_use():
    from handumi_collector.pose.episode_io import ImuArrays
    from handumi_collector.tools.inspect_episode import _trace

    t = np.arange(0, 400) * 5_000_000                              # 200 Hz, 2 s
    imu = ImuArrays("left", np.arange(400), t // 1000, t,
                    np.zeros((400, 3)), np.zeros((400, 3)), np.full(400, 30.0))
    imu.gyro[:, 0] = np.linspace(0, 2, 400)
    canvas, nearest_ms = _trace(imu, 1_002_000_000, 0.5, (90, 400), "imu left", (255, 190, 120))
    assert canvas.shape == (90, 400, 3)
    assert nearest_ms == pytest.approx(-2.0, abs=0.6)              # the 200 Hz grid cannot land exactly on any instant

    empty, nan_ms = _trace(None, 0, 0.5, (90, 400), "imu left", (255, 190, 120))
    assert empty.shape == (90, 400, 3) and np.isnan(nan_ms)        # an episode without IMU still draws


def test_alignment_refuses_mismatched_grids():
    """Depth aligned to colour is the same grid by definition. Comparing edge maps of different sizes indexes one by the
    other's coordinates and returns numbers rather than an error — the mock profile's 32x24 depth against 848x480 colour
    is exactly that case."""
    from handumi_collector.pose.rgbd_align import edge_alignment

    r = edge_alignment(np.zeros((480, 848, 3), np.uint8), np.zeros((24, 32), np.float32))
    assert "error" in r and "not the same grid" in r["error"]
    assert "shift_px" not in r and "median_offset_px" not in r


# ---------------------------------------------------------------- temporal sync

def test_impulses_ignores_a_flat_signal_with_a_tiny_spread():
    """A deviation test alone is not enough. Head-camera frame difference sitting at 0.59 with 0.70 peaks has a MAD so
    small that 6 robust deviations is also small: it produced 39 "transients" in 30 s, and matching a few real taps
    against one candidate every 0.8 s found a partner for everything by chance. A real tap in that signal reached 6.5."""
    from handumi_collector.tools.temporal_sync import impulses

    t = (np.arange(900) * 33_000_000).astype(np.int64)                 # 30 s at 30 fps
    flat = 0.59 + np.random.default_rng(0).normal(0, 0.03, len(t))     # the real head signal, near enough
    assert len(impulses(t, flat)) == 0

    with_tap = flat.copy(); with_tap[400] = 6.5                        # and a real one is unmistakable
    got = impulses(t, with_tap)
    assert len(got) == 1 and got[0] == t[400]


def test_match_impulses_withholds_a_verdict_when_the_pairs_disagree():
    """Chance matches scatter. One run produced deltas of 136, 198, -94 and -102 ms and a median of +21 ms that meant
    nothing at all; quoting it would have been worse than saying nothing."""
    from handumi_collector.tools.temporal_sync import match_impulses

    taps = (np.array([1.0, 3.5, 7.0, 11.0]) * 1e9).astype(np.int64)
    scattered = taps + (np.array([136, 198, -94, -102]) * 1e6).astype(np.int64)
    m = match_impulses(taps, scattered)
    assert m["pairs"] == 4 and not m["agrees"] and "chance matches" in m["reason"]

    consistent = taps + int(40e6)
    ok = match_impulses(taps, consistent)
    assert ok["agrees"] and ok["median_ms"] == pytest.approx(40.0, abs=0.1) and ok["spread_ms"] < 1.0


def test_impulses_finds_taps_and_thins_each_burst_to_one():
    """A tap is several samples wide at 200 Hz; counting them all would turn three taps into a dozen."""
    from handumi_collector.tools.temporal_sync import impulses

    t = (np.arange(0, 2000) * 5_000_000).astype(np.int64)          # 200 Hz, 10 s
    s = np.abs(np.random.default_rng(0).normal(0, 0.02, len(t)))
    for centre in (200, 700, 1400):                                 # 1 s, 3.5 s, 7 s
        s[centre:centre + 4] += [3.0, 5.0, 2.0, 1.0]

    got = impulses(t, s)
    assert len(got) == 3, got
    assert np.allclose(got / 1e9, [1.005, 3.505, 7.005], atol=0.01)  # the strongest sample of each burst


def test_match_impulses_measures_the_offset_between_two_event_trains():
    from handumi_collector.tools.temporal_sync import match_impulses

    taps = np.array([1.0, 3.5, 7.0]) * 1e9
    seen = taps + 40e6                                              # the other stream sees them 40 ms later
    m = match_impulses(taps.astype(np.int64), seen.astype(np.int64))
    assert m["pairs"] == 3 and m["median_ms"] == pytest.approx(40.0, abs=0.1)

    assert match_impulses(taps.astype(np.int64), (taps + 5e8).astype(np.int64))["pairs"] == 0
    assert match_impulses(np.empty(0, np.int64), taps.astype(np.int64))["pairs"] == 0


def test_lookup_deltas_shows_the_phase_circulating_rather_than_a_fixed_offset():
    """Two free-running 30 fps cameras at slightly different rates wind their relative phase through a whole frame
    period. Measured once it looks like a calibration; applied later it is wrong by up to a frame. Real rates from the
    rig: 30.0260 Hz for the head against 30.0032 Hz for the left wrist."""
    from handumi_collector.pose.episode_io import FrameMeta, RawEpisode
    from handumi_collector.tools.temporal_sync import lookup_deltas

    def stream(name, hz, n=900):
        caps = (np.arange(n) * (1e9 / hz)).astype(np.int64)
        return FrameMeta(name, np.arange(n), np.arange(n), caps)

    ep = RawEpisode.__new__(RawEpisode)
    ep.frames = {"head_depth": stream("head_depth", 30.0260), "left_wrist": stream("left_wrist", 30.0032)}

    lk = lookup_deltas(ep, "head_depth", "left_wrist")
    assert lk["bound_ms"] == pytest.approx(16.66, abs=0.05)          # half a frame, structural
    assert lk["abs_p95_ms"] <= lk["bound_ms"] + 0.01
    assert lk["circulates"], lk["windows"]                           # signs flip: no constant to store

    same = lookup_deltas(ep, "head_depth", "head_depth")
    assert same["abs_median_ms"] == 0.0 and not same["circulates"]


def test_imu_interpolates_to_the_frame_instant():
    """At 200 Hz the nearest sample is within 2.5 ms, which is small — and interpolating costs nothing and stops that
    quantisation being carried into everything downstream."""
    from handumi_collector.pose.episode_io import ImuArrays
    from handumi_collector.tools.temporal_sync import imu_at

    t = (np.arange(400) * 5_000_000).astype(np.int64)
    gyro = np.stack([np.linspace(0, 4, 400), np.zeros(400), np.zeros(400)], 1)
    imu = ImuArrays("left", np.arange(400), t // 1000, t, gyro, np.zeros((400, 3)), np.full(400, 30.0))

    exact = imu_at(imu, int(t[10]))
    assert exact["gyro"][0] == pytest.approx(gyro[10, 0]) and exact["nearest_ms"] == 0.0

    half = imu_at(imu, int(t[10] + 2_500_000))                       # midway between two samples
    assert half["gyro"][0] == pytest.approx((gyro[10, 0] + gyro[11, 0]) / 2)
    assert half["nearest_ms"] == pytest.approx(2.5)

    assert imu_at(imu, int(t[-1] + 100_000_000)) is None             # well past the record, not extrapolated


def test_loss_ratio_is_safe_to_read_while_samples_arrive():
    """The reader thread appends to the window while the UI asks for the loss ratio. Summing the deque per call made
    that a race — `RuntimeError: deque mutated during iteration` — which killed a 3 h unattended run 46 minutes in and
    would have killed any soak through imu_monitor, which asks twice a second."""
    import threading
    from handumi_collector.devices.teensy_imu import SeqTracker

    t = SeqTracker(window=256)
    stop, errors = threading.Event(), []

    def produce():
        seq = 0
        while not stop.is_set():
            seq += 1
            t.update(seq, seq * 5000)

    def consume():
        try:
            while not stop.is_set():
                t.loss_ratio()
        except Exception as exc:                      # noqa: BLE001 - the point is that nothing escapes
            errors.append(exc)

    threads = [threading.Thread(target=produce), threading.Thread(target=consume)]
    for th in threads: th.start()
    time.sleep(1.0)
    stop.set()
    for th in threads: th.join()
    assert not errors, errors


def test_loss_ratio_matches_the_window_it_summarises():
    """The running sums have to mean what summing the window meant, evictions included."""
    from handumi_collector.devices.teensy_imu import SeqTracker

    t = SeqTracker(window=8)
    for seq in (1, 2, 3, 5):                          # one lost before 5
        t.update(seq, seq * 5000)
    assert t.loss_ratio() == pytest.approx(1 / 5)

    t2 = SeqTracker(window=4)
    for seq in range(1, 4): t2.update(seq, seq * 5000)
    t2.update(10, 10 * 5000)                          # 6 lost, fills the window
    assert t2.loss_ratio() == pytest.approx(6 / 10)
    for seq in range(11, 21): t2.update(seq, seq * 5000)   # push the gap out of the window entirely
    assert t2.loss_ratio() == 0.0


def test_a_host_that_slept_mid_run_fails_clock_sanity(tmp_path):
    """macOS stops the monotonic clock while it sleeps; the Teensy keeps sampling. An overnight Allan run recorded
    436 minutes of device time inside 46 minutes of host time across 28 naps, and every other gate passed it: the
    samples that did arrive were contiguous, on rate, CRC-clean and gravity-correct."""
    _, got = parse_all(synth_stream(n=4000)[0])
    d = write_session(tmp_path, host_stamped(got))
    meta = json.loads((d / "metadata.json").read_text())
    span_s = (got[-1].device_timestamp_us - got[0].device_timestamp_us) / 1e6
    meta.update(t_start_monotonic_ns=0, t_stop_monotonic_ns=int(span_s * 1e9), wall_duration_s=span_s)
    (d / "metadata.json").write_text(json.dumps(meta))

    stats, prov = vs.load_session(d, None)
    awake = dict((n, ok) for n, ok, _ in vs.gates(stats, prov, static=True))
    assert awake["clock_sanity"] is True and awake["coverage"] is True

    meta["wall_duration_s"] = span_s / 9.5              # the device outran the host, as it does across a sleep
    (d / "metadata.json").write_text(json.dumps(meta))
    stats, prov = vs.load_session(d, None)
    rows = dict((n, (ok, det)) for n, ok, det in vs.gates(stats, prov, static=True))
    assert rows["clock_sanity"][0] is False and "slept mid-run" in rows["clock_sanity"][1]
    assert rows["rate"][0] is True and rows["drops"][0] is True, "nothing else notices, which is why this gate exists"


# ---------------------------------------------------------------- sudo-recorded data ownership

def test_recorded_data_is_handed_back_to_the_invoking_user(tmp_path, monkeypatch):
    """The Orbbec SDK cannot claim its UVC interface without root on macOS, so every head-camera session is recorded
    under sudo and everything it writes is owned by root. Nothing downstream runs as root: pose_process failed with
    PermissionError on its own episode. sudo leaves SUDO_UID precisely so this can be undone."""
    from handumi_collector.collector import ownership

    d = tmp_path / "episode_000001"
    (d / "head_depth_depth").mkdir(parents=True)
    (d / "head_depth.mp4").write_bytes(b"x")
    (d / "head_depth_depth" / "000000.png").write_bytes(b"y")

    chowned = []
    monkeypatch.setattr(ownership.os, "chown", lambda p, u, g: chowned.append((str(p), u, g)))

    monkeypatch.setattr(ownership.os, "geteuid", lambda: 501)          # an ordinary run touches nothing
    ownership.give_back(d)
    assert chowned == [] and ownership.invoking_user() is None

    monkeypatch.setattr(ownership.os, "geteuid", lambda: 0)
    monkeypatch.delenv("SUDO_UID", raising=False)
    ownership.give_back(d)
    assert chowned == [], "root without SUDO_UID is a genuine root run, not a sudo one"

    monkeypatch.setenv("SUDO_UID", "501"); monkeypatch.setenv("SUDO_GID", "20")
    ownership.give_back(d)
    assert ownership.invoking_user() == (501, 20)
    paths = {p for p, _, _ in chowned}
    assert str(d) in paths and str(d / "head_depth.mp4") in paths
    assert str(d / "head_depth_depth" / "000000.png") in paths, "the depth PNGs are files too"
    assert all((u, g) == (501, 20) for _, u, g in chowned)


def test_a_failed_chown_does_not_lose_the_recording(tmp_path, monkeypatch):
    """Losing a take because the ownership handover failed would be far worse than a file to chown by hand later."""
    from handumi_collector.collector import ownership

    d = tmp_path / "episode_000002"; d.mkdir(); (d / "a.mp4").write_bytes(b"x")
    monkeypatch.setattr(ownership.os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "501"); monkeypatch.setenv("SUDO_GID", "20")

    def boom(*_a): raise PermissionError("nope")
    monkeypatch.setattr(ownership.os, "chown", boom)
    ownership.give_back(d)                                             # must not raise
    assert (d / "a.mp4").exists()


# ---------------------------------------------------------------- a camera that opens and then says nothing

def test_a_stream_that_recorded_nothing_is_not_a_PASS():
    """`if f and f < 25` skipped zero, because zero is falsy — so a stream that produced no frames at all, the worst
    outcome there is, was the single case that could not be flagged. A take with two empty cameras came back PASS and
    was kept."""
    from handumi_collector.collector.integrity import preliminary_qa

    healthy = {n: dict(frames=900, duration_s=30.0, fps_measured=30.0) for n in ("left_wrist", "right_wrist", "head_depth")}
    assert preliminary_qa(dict(streams=healthy), []) == ("PASS", [])

    empty = dict(healthy)
    empty["right_wrist"] = dict(frames=0, duration_s=0.0, fps_measured=0.0)
    empty["head_depth"] = dict(frames=0, duration_s=0.0, fps_measured=0.0)
    verdict, notes = preliminary_qa(dict(streams=empty), [])
    assert verdict == "REVIEW"
    assert any("right_wrist RECORDED NO FRAMES" in n for n in notes)
    assert any("head_depth RECORDED NO FRAMES" in n for n in notes)

    slow = dict(healthy)
    slow["left_wrist"] = dict(frames=300, duration_s=30.0, fps_measured=10.0)
    assert "left_wrist fps 10.0" in preliminary_qa(dict(streams=slow), [])[1]


def test_readiness_means_delivering_not_merely_opened(monkeypatch):
    """Three cameras once opened cleanly and reported connected while two produced 0.8 and 0.0 frames a second, and
    thirty seconds were recorded and kept before anything noticed."""
    from handumi_collector.config import CameraCfg, HardwareCfg
    from handumi_collector.devices import base
    from handumi_collector.devices.manager import DeviceManager

    class FakeCam:
        required = True
        def __init__(self, name, hz, running_for_s):
            self.name, self.cfg = name, CameraCfg(name, "pose_estimation", backend="mock", fps=30)
            self._hz, self._for = hz, running_for_s
        def status(self):
            st = base.DeviceStatus(self.name, connected=True, running=True, rate_hz=self._hz)
            st.running_since_ns = base.now_ns() - int(self._for * 1e9)
            return st

    dm = DeviceManager(HardwareCfg(cameras=[], imus=[], grippers=[]))
    dm.cameras = {"good": FakeCam("good", 30.0, 5.0), "dead": FakeCam("dead", 0.0, 5.0)}
    ok, missing = dm.required_ok()
    assert not ok and any("dead" in m and "0.0 Hz" in m for m in missing)
    assert not any("good" in m for m in missing)

    # ...but a camera reporting 0 Hz a millisecond after start has told you nothing yet
    dm.cameras = {"fresh": FakeCam("fresh", 0.0, 0.01)}
    assert dm.required_ok()[0] is True
