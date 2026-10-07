# Teensy IMU firmware (M1)

Build: Arduino IDE / `arduino-cli` with Teensyduino, board **Teensy 4.1**, USB type **Serial**, CPU 600 MHz.
`config.h` holds pins, ODR (200 Hz), ranges (gyro ±2000 dps, accel ±8 g) and ring size. Flash one Teensy per hand and
write each board's USB serial number (`python -m handumi_collector.tools.imu_monitor --list`) into
`configs/handumi/hardware_handumi_v1.yaml` → `imus[].serial_number` (LEFT/RIGHT identity is by serial, never by port order).

Wiring (ICM42688P breakout ↔ Teensy 4.1): VDD 3V3, GND, SCLK→13, SDI/MOSI→11, SDO/MISO→12, CS→10, INT1→2.
Verify: `imu_monitor` shows ~200 Hz per side, loss 0.00 %, status packets 1 Hz, temperature ~25–40 °C, |accel| ≈ 9.81 m/s² at rest.
Timing: sampling is INT1-driven; `t_us` is a 64-bit monotonic extension of `micros()`; USB writes never block sampling
(ring drops are counted in the status packet `dropped`, visible as `imu_seq_gap` events on the host).
