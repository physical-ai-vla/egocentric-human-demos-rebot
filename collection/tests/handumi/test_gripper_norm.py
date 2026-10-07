import math
from handumi_collector.devices.feetech_gripper import EncoderUnwrapper, normalize_ticks
from handumi_collector.devices.feetech_bus import parse_status_block


def test_unwrap_across_seam():
    u = EncoderUnwrapper()
    vals = [4000, 4090, 10, 50, 4080]        # crosses 4095->0 and back
    out = [u(v) for v in vals]
    assert out == [4000, 4090, 4106, 4146, 4080]


def test_normalize_monotonic_clipped_and_convention():
    xs = list(range(900, 2101, 100))
    g_closed1 = [normalize_ticks(x, 1000, 2000, "closed") for x in xs]
    g_open1 = [normalize_ticks(x, 1000, 2000, "open") for x in xs]
    assert all(0.0 <= g <= 1.0 for g in g_closed1)
    assert all(a >= b for a, b in zip(g_closed1, g_closed1[1:]))     # monotonic non-increasing (1 = closed)
    assert all(abs(a + b - 1.0) < 1e-12 for a, b in zip(g_closed1, g_open1))
    assert normalize_ticks(2500, 3000, 2000, "closed") == 0.5            # reversed encoder direction works
    assert math.isnan(normalize_ticks(1500, None, None))


def test_parse_status_block_legacy15():
    d = [0] * 15
    d[0], d[1] = 0x00, 0x08          # position 2048
    d[2], d[3] = 0x10, 0x80          # speed -16 (sign bit)
    d[6], d[7] = 74, 31              # 7.4 V, 31 C
    d[13], d[14] = 0x02, 0x00        # current 2 * 6.5 mA
    b = parse_status_block(d)
    assert b["position"] == 2048 and b["speed"] == -16 and abs(b["voltage_v"] - 7.4) < 1e-9 and b["temperature_c"] == 31
    assert abs(b["current_ma"] - 13.0) < 1e-9


def test_normalize_across_seam_and_restart_invariance():
    """Real HandUMI unit (2026-09-10): closed 4069, open 5389 (unwrapped) crosses the 4095->0 seam."""
    closed, open_ = 4069, 5389
    assert normalize_ticks(4069, closed, open_, "open") == 0.0 and normalize_ticks(5389, closed, open_, "open") == 1.0
    assert abs(normalize_ticks(4069 + 660, closed, open_, "open") - 0.5) < 1e-9
    # raw (mod 4096) after a restart must read identically to the unwrapped value
    for unwrapped in (4200, 4700, 5300):
        assert abs(normalize_ticks(unwrapped % 4096, closed, open_, "open") - normalize_ticks(unwrapped, closed, open_, "open")) < 1e-12
    assert normalize_ticks(1293, closed, open_, "open") > 0.97                  # 5389 % 4096 = 1293 -> open, not "closed"
    assert normalize_ticks(3900, closed, open_, "open") == 0.0                  # just beyond closed -> clipped closed
    assert normalize_ticks(1600, closed, open_, "open") == 1.0                  # just beyond open -> clipped open
