// HandUMI Teensy 4.1 + ICM42688P — build-time configuration (edit here, not in the sketch)
#pragma once
#define FW_MAJOR 1
#define FW_MINOR 0
// SPI wiring (Teensy 4.1 SPI0: MOSI 11, MISO 12, SCK 13)
#define PIN_CS      10
#define PIN_MISO    12          // SPI0 MISO; probed with a pull-up when WHO_AM_I fails, to tell open from held-low
#define PIN_INT1     2          // ICM42688P INT1 (data-ready), push-pull active-high — QCIoT board wiring, verified on hardware
#ifndef SPI_HZ                  // overridable from the build, to bisect signal integrity without editing this file:
#define SPI_HZ  8000000         //   arduino-cli compile --build-property compiler.cpp.extra_flags=-DSPI_HZ=1000000
#endif
// Sensor configuration
#define IMU_ODR_HZ        200    // 100 | 200 | 500 | 1000 (accel + gyro same ODR); the part has no 400 Hz rate
#define GYRO_FS_DPS      2000    // ±2000 dps
#define ACCEL_FS_G          8    // ±8 g
// Streaming
#define STATUS_PERIOD_MS 1000
#define TX_RING_PACKETS   256    // 256 * 46 B ≈ 11.8 KB ring between ISR-driven sampling and USB writes
