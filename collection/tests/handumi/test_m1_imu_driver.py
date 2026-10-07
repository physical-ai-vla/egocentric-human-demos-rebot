"""M1 host-driver tests, hardware independent: seq wrap, timestamp wrap, gaps, garbage/CRC resync, reconnect, identity."""
import threading
import time
import pytest
from handumi_collector.config import ImuCfg
from handumi_collector.devices.teensy_imu import (SEQ_MOD, PacketParser, SeqTracker, TeensyImu, TYPE_IMU, TYPE_STATUS,
                                                  decode_imu, encode_imu)
from handumi_collector.tools.gen_test_stream import build_stream


def _drive(imu: TeensyImu, data: bytes, chunk: int = 97):
    for i in range(0, len(data), chunk):
        for ptype, payload in imu.parser.feed(data[i:i + chunk]):
            imu.handle(ptype, payload, host_ns=1_000_000 * (i + 1))


def test_seq_wrap_is_not_a_gap_and_loss_is_counted():
    st = SeqTracker()
    for s in range(SEQ_MOD - 3, SEQ_MOD): st.update(s, s)
    lost, wrapped = st.update(0, SEQ_MOD)          # 4294967295 -> 0
    assert (lost, wrapped) == (0, False) and st.seq_gaps == 0
    lost, _ = st.update(4, SEQ_MOD + 4)            # skipped 1,2,3
    assert lost == 3 and st.seq_gaps == 1 and abs(st.loss_ratio() - 3 / 8) < 1e-9
    lost, _ = st.update(3, SEQ_MOD + 3)            # reordered/duplicate: not a loss
    assert lost == 0


def test_timestamp_wrap_reported_not_dropped():
    st = SeqTracker()
    st.update(0, (1 << 64) - 10); lost, wrapped = st.update(1, 5)
    assert (lost, wrapped) == (0, True) and st.ts_wraps == 1


def test_stream_with_garbage_crc_and_drops_recovers():
    data, truth = build_stream(3000, garbage_every=250, corrupt_every=333, drop_every=400, seq_start=SEQ_MOD - 1500)
    imu = TeensyImu(ImuCfg("left", backend="teensy"))
    events = []; imu.on_event = lambda k, d, t: events.append((k, d))
    _drive(imu, data)
    q = imu.quality()
    assert imu.buffer.total == truth["emitted"], (imu.buffer.total, truth)                 # every intact packet decoded
    assert imu.parser.crc_errors == truth["corrupted"] and imu.parser.resyncs >= truth["garbage"]
    gaps = [d for k, d in events if k == "imu_seq_gap"]
    assert sum(d["lost"] for d in gaps) == truth["dropped"] + truth["corrupted"]      # corrupted packets count as lost samples
    assert not [k for k, _ in events if k == "imu_timestamp_wrap"]
    assert imu.last_status and imu.last_status["sample_rate_hz"] == 400 and q.seq_gaps == len(gaps)
    s = imu.buffer.latest(); assert s.seq == (SEQ_MOD - 1500 + 2999) % SEQ_MOD          # seq wrapped through 2^32 cleanly


def test_grade_thresholds_from_config():
    from handumi_collector.devices.teensy_imu import ImuQuality
    th = {"green": {"min_hz": 380, "max_loss": 0.001, "max_age_ms": 50}, "yellow": {"min_hz": 300, "max_loss": 0.01, "max_age_ms": 250}}
    assert ImuQuality(hz=399, loss_ratio=0.0, age_ms=3, connected=True).grade(th) == "GREEN"
    assert ImuQuality(hz=350, loss_ratio=0.005, age_ms=3, connected=True).grade(th) == "YELLOW"
    assert ImuQuality(hz=399, loss_ratio=0.0, age_ms=900, connected=True).grade(th) == "RED"
    assert ImuQuality(hz=399, connected=False).grade(th) == "RED"


class _FakeSerial:
    """pyserial stand-in: yields a scripted stream, then raises (unplug), then works again after reopen."""
    instances = 0

    def __init__(self, port, baud, timeout=0.05):
        _FakeSerial.instances += 1; self.n = _FakeSerial.instances
        # Longer than the startup drain consumes: open() now discards the device ring's stale block (bounded by
        # _drain's byte cap), so a fixture that only holds what the test wants to see would be emptied by it.
        self.data, _ = build_stream(3000, t0_us=1_000_000 * self.n); self.pos = 0; self.closed = False

    def reset_input_buffer(self): pass

    def read(self, n):
        if self.pos >= len(self.data):
            if self.n == 1: raise OSError("device disconnected")     # first instance: unplug after its data
            time.sleep(0.01); return b""
        chunk = self.data[self.pos:self.pos + n]; self.pos += n; return chunk

    def close(self): self.closed = True


def test_usb_reconnect_emits_events(monkeypatch):
    import serial, serial.tools.list_ports as lp
    monkeypatch.setattr(serial, "Serial", _FakeSerial)
    monkeypatch.setattr(lp, "comports", lambda: [type("P", (), dict(device="/dev/cu.fakeL", serial_number="TEENSY-L-001"))()])
    monkeypatch.setattr("handumi_collector.devices.base.resolve_serial_port", lambda *a, **k: "/dev/cu.fakeL")
    monkeypatch.setattr("handumi_collector.devices.teensy_imu.resolve_serial_port", lambda *a, **k: "/dev/cu.fakeL")
    _FakeSerial.instances = 0
    imu = TeensyImu(ImuCfg("left", backend="teensy", serial_number="TEENSY-L-001"))
    events = []; imu.on_event = lambda k, d, t: events.append(k)
    imu.open(); assert imu.serial_number == "TEENSY-L-001"
    imu.start(); time.sleep(0.6); imu.close()
    assert "device_error" in events and "device_reconnect" in events and imu.reconnects >= 1
    assert imu.buffer.total >= 300                               # data from both the first and the reconnected instance


def test_identity_mismatch_refuses(monkeypatch):
    import serial, serial.tools.list_ports as lp
    monkeypatch.setattr(serial, "Serial", _FakeSerial)
    monkeypatch.setattr(lp, "comports", lambda: [type("P", (), dict(device="/dev/cu.fakeR", serial_number="TEENSY-R-002"))()])
    monkeypatch.setattr("handumi_collector.devices.teensy_imu.resolve_serial_port", lambda *a, **k: "/dev/cu.fakeR")
    imu = TeensyImu(ImuCfg("left", backend="teensy", serial_number="TEENSY-L-001"))
    with pytest.raises(RuntimeError, match="serial"):
        imu.open()


class _StaleThenLive:
    """A board that hands over its transmit ring before it reaches the present, which is what a Teensy does.

    `reset_input_buffer()` clears the HOST buffer only, so opening the port yields a block that is internally
    contiguous and then jumps by however many samples were produced while nobody was listening. Measured on the real
    boards as a 512 Hz burst against a 200 Hz stream, arriving tens of milliseconds AFTER the open rather than sitting
    in the buffer -- which is why `in_waiting` cannot be used to detect it."""

    def __init__(self, port, baud, timeout=0.05, *, stale_bytes=20_000, jump=5_000, live_gap_at=None):
        from handumi_collector.tools.gen_test_stream import build_stream
        probe, _ = build_stream(10, seq_start=0, t0_us=1_000)
        pkt = len(probe) // 10                           # framed packet size, not the payload size
        n_stale = max(1, stale_bytes // pkt)
        stale, _ = build_stream(n_stale, seq_start=1_000, t0_us=1_000_000)
        live, _ = build_stream(3_000, seq_start=1_000 + n_stale + jump, t0_us=9_000_000)
        if live_gap_at is not None:                      # a genuine loss well after the drain window
            live = live[:live_gap_at * pkt] + live[(live_gap_at + 120) * pkt:]
        self.data = stale + live
        self.pos = 0
        self.closed = False

    @property
    def in_waiting(self):                                # the block has not arrived yet, exactly as on hardware
        return 0

    def reset_input_buffer(self): pass

    def read(self, n):
        if self.pos >= len(self.data):
            time.sleep(0.005); return b""
        chunk = self.data[self.pos:self.pos + n]; self.pos += n; return chunk

    def close(self): self.closed = True


def _imu_on(monkeypatch, factory):
    import serial, serial.tools.list_ports as lp
    from handumi_collector.config import ImuCfg
    from handumi_collector.devices.teensy_imu import TeensyImu
    monkeypatch.setattr(serial, "Serial", factory)
    monkeypatch.setattr(lp, "comports", lambda: [type("P", (), {"device": "/dev/cu.usbmodemTEST", "serial_number": "T1"})()])
    d = TeensyImu(ImuCfg(side="left", backend="teensy", port_glob="/dev/cu.usbmodemTEST",
                         serial_number="T1", baud=2_000_000, rate_hz=200))
    d.open()
    return d


def test_startup_stale_block_is_not_counted_as_loss(monkeypatch):
    """Both IMUs read 90-98 % loss for the first ten seconds of every session and had lost nothing: one ring boundary,
    counted once as several thousand samples, taking the whole window to age out. A health panel that says 95 % when
    nothing is wrong is one the operator stops reading, and the pilot has nothing else watching the IMUs."""
    d = _imu_on(monkeypatch, _StaleThenLive)
    d.start(); time.sleep(0.6); q = d.quality(); d.close()
    assert q.loss_ratio == 0.0, f"loss {q.loss_ratio:.4f}, gaps {q.seq_gaps}"
    assert q.seq_gaps == 0


def test_a_reconnect_gets_the_same_treatment(monkeypatch):
    """A reconnect re-opens the port, so the board hands over its ring again."""
    d = _imu_on(monkeypatch, _StaleThenLive)
    d.start(); time.sleep(0.3)
    assert d._reconnect()
    time.sleep(0.6); q = d.quality(); d.close()
    assert q.loss_ratio == 0.0 and q.seq_gaps == 0, f"loss {q.loss_ratio:.4f}, gaps {q.seq_gaps}"


def test_a_real_gap_after_the_drain_is_still_counted(monkeypatch):
    """The drain must not be a way of not counting losses. A gap that happens once the stream is live is a loss."""
    import functools
    # Past the drain's byte cap: a fake returns instantly, so the drain reaches much further into the live stream
    # than it would against a board that produces 8 kB/s. The hole has to be somewhere the drain cannot have eaten.
    d = _imu_on(monkeypatch, functools.partial(_StaleThenLive, live_gap_at=2_200))
    d.start(); time.sleep(0.8); q = d.quality(); d.close()
    assert q.seq_gaps >= 1, "a 120-sample hole in the live stream was not reported"
    assert q.loss_ratio > 0.0
