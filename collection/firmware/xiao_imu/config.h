// HandUMI XIAO nRF52840 Sense + LSM6DS3TR-C — build-time configuration (edit here, not in the sketch).
// Wire protocol is the SAME as the Teensy unit: firmware/teensy_imu/PROTOCOL.md v1. That is the point — the host
// (handumi_collector.devices.teensy_imu) is a protocol parser, not a board driver, and needs no change for this board.
#pragma once
#define FW_MAJOR 1
#define FW_MINOR 0
// The IMU is ON the XIAO Sense module: internal I2C (Wire1 = P0.07/P0.27), power enable P1.08, INT1 P0.11.
// Nothing is wired externally, so there are no pin options here — the variant's pins_arduino.h names them.
#define I2C_HZ         400000   // 100 kHz (the Arduino default) makes a 14-byte burst ~1.4 ms = 58% of the 416 Hz
                                // budget. 400 kHz brings it to ~0.35 ms. Raise ODR only after re-checking this.
// Sensor configuration — LSM6DS3TR-C ODRs: 12.5 26 52 104 208 416 833 1660 3330 6660 Hz
#define IMU_ODR_HZ        416   // the spec rate; 208 is the documented fallback
#define GYRO_FS_DPS      2000   // ±2000 dps
#define ACCEL_FS_G          8   // ±8 g
// Streaming
#define STATUS_PERIOD_MS 1000
#define TX_RING_PACKETS   256   // 256 * 46 B ≈ 11.8 KB between the data-ready ISR and USB writes
