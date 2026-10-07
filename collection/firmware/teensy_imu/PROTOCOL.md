# HandUMI Teensy 4.1 ↔ host IMU protocol (v1, 2026-09-09)

Wiring: ICM42688P over SPI (≥ 8 MHz), INT1 = data-ready → Teensy interrupt. Teensy 4.1 → host over USB CDC
(`/dev/cu.usbmodem*`, baud value is ignored by CDC but the host opens at 2 000 000). One Teensy per hand.

Sampling: ODR 200 Hz on accel + gyro (configurable 100/200/500/1000 — the part has no 400 Hz rate). Every INT1 edge the ISR latches `micros()`; the main loop
reads the FIFO/registers and emits one IMU packet per sample. `seq` increments per emitted sample (u32 wrap).

## Framing (little-endian)

```
offset  size  field
0       2     magic 0xA5 0x5A
2       1     type      0x01 = IMU sample, 0x02 = status/heartbeat, 0x03 = text log (utf-8)
3       1     len       payload length in bytes
4       len   payload
4+len   2     crc16     CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over bytes [2 .. 4+len)
```

### type 0x01 IMU (len = 40)
```
u32 seq | u64 t_us (device micros at INT1) | f32 ax ay az (m/s^2) | f32 gx gy gz (rad/s) | f32 temp_c
```
### type 0x02 status (len = 24), 1 Hz
```
u64 t_us | u32 sample_rate_hz | u32 samples_emitted | u32 samples_dropped (FIFO overflow) | u8 fw_major | u8 fw_minor | u16 ring_high_water
```
`ring_high_water` is the deepest the tx ring has been since the previous status packet, in packets, out of
`TX_RING_PACKETS` (256). It is reported and then reset each period, so a host takes the max across a recording to get
that recording's high-water; a maximum cannot be differenced like a counter, so a since-boot value would carry in
whatever piled up before anyone was listening. It occupies the field that was `reserved` in fw 1.0 and was always sent
as zero, so the layout, length and CRC are unchanged and an older host simply ignores it.

Host identity: the Teensy USB serial number (`serial.tools.list_ports` → `serial_number`) is written to `hardware.yaml`
`imus[].serial_number`. Host stamps every packet with `host_receive_ns` (monotonic); device↔host offset is estimated
offline (robust linear fit of `t_us` vs `host_receive_ns`, see `processing/sync.py`), so USB latency jitter does not
enter the canonical timeline. Firmware sketch: M1 (`firmware/teensy_imu/teensy_imu.ino`).
