import struct
from handumi_collector.devices.teensy_imu import (IMU_LEN, PacketParser, TYPE_IMU, TYPE_STATUS, STATUS_FMT, crc16_ccitt,
                                                  decode_imu, encode_imu, encode_packet)


def test_crc_known_vector():
    assert crc16_ccitt(b"123456789") == 0x29B1     # CRC-16/CCITT-FALSE check value


def test_roundtrip_and_resync_and_crc_error():
    p = PacketParser()
    pk1 = encode_imu(7, 123456, (0.1, 0.2, 9.8), (0.01, -0.02, 0.03), 33.5)
    pk2 = encode_packet(TYPE_STATUS, struct.pack(STATUS_FMT, 999, 400, 1000, 0, 1, 0, 0))
    bad = bytearray(pk1); bad[10] ^= 0xFF          # corrupt payload -> CRC failure
    stream = b"\x00garbage" + pk1[:11] + pk1[11:] + bytes(bad) + pk2
    out = []
    for i in range(0, len(stream), 5):             # feed in odd-sized chunks
        out += p.feed(stream[i:i + 5])
    types = [t for t, _ in out]
    assert types == [TYPE_IMU, TYPE_STATUS] and p.crc_errors == 1 and p.resyncs >= 1
    s = decode_imu("left", out[0][1], host_ns=42)
    assert (s.seq, s.device_timestamp_us, s.host_receive_ns) == (7, 123456, 42)
    assert abs(s.az - 9.8) < 1e-6 and abs(s.gy + 0.02) < 1e-6 and len(out[0][1]) == IMU_LEN
