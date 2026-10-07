// HandUMI XIAO nRF52840 Sense + LSM6DS3TR-C IMU streamer.
// Protocol: firmware/teensy_imu/PROTOCOL.md (authoritative, v1) — byte-for-byte the same frames the Teensy unit
// sends, so handumi_collector.devices.teensy_imu parses this board with nothing changed but `rate_hz: 416`.
//
// Timing comes from the sensor's INT1 data-ready interrupt, never from a polling loop: the ISR latches a 64-bit
// monotonic micros() and raises a flag, the main loop burst-reads the sample into a ring, and USB writes drain the
// ring WITHOUT BLOCKING so a slow host can never stall sampling. Raw sensor axes are streamed as-is (no L/R flips).
//
// Two things differ from the Teensy sketch and both are load-bearing:
//   * the IMU is internal, on Wire1 (P0.07/P0.27) with a power-enable on P1.08 — LSM6DS3Core::beginCore() does both;
//   * this core's `Serial.write()` BLOCKS (mbed USBCDC::send waits with no timeout) and `availableForWrite()` is the
//     Print default, 0. The Teensy's `if (availableForWrite() < len) return;` guard would therefore send nothing at
//     all here. `_SerialUSB.send_nb()` is the non-blocking primitive, and it may accept a PARTIAL packet, so the
//     drain carries an offset and resumes mid-packet instead of re-sending bytes the host already has.
#include <Arduino.h>
#include <Wire.h>
#include "USB/PluggableUSBSerial.h"
#include "LSM6DS3.h"
#include "config.h"

// ---- LSM6DS3TR-C registers
#define REG_WHO_AM_I    0x0F
#define REG_DRDY_PULSE  0x0B
#define REG_INT1_CTRL   0x0D
#define REG_CTRL1_XL    0x10
#define REG_CTRL2_G     0x11
#define REG_CTRL3_C     0x12
#define REG_STATUS      0x1E
#define REG_OUT_TEMP_L  0x20    // OUT_TEMP(2) GYRO xyz(6) ACCEL xyz(6) = 14 bytes, little-endian int16

static const uint8_t MAGIC0 = 0xA5, MAGIC1 = 0x5A;
static const uint8_t TYPE_IMU = 0x01, TYPE_STATUS = 0x02, TYPE_LOG = 0x03;
static const uint8_t IMU_LEN = 40, STATUS_LEN = 24;
static const uint8_t LOG_MAX = IMU_LEN;
static const uint16_t PKT_MAX = 6 + IMU_LEN;
static_assert(STATUS_LEN <= IMU_LEN && LOG_MAX <= IMU_LEN, "IMU_LEN must stay the largest payload, it sizes the ring slot");

struct __attribute__((packed)) ImuPayload { uint32_t seq; uint64_t t_us; float ax, ay, az, gx, gy, gz, temp; };
struct __attribute__((packed)) StatusPayload { uint64_t t_us; uint32_t rate_hz, emitted, dropped; uint8_t fw_major, fw_minor; uint16_t ring_hiwater; };
static_assert(sizeof(ImuPayload) == IMU_LEN && sizeof(StatusPayload) == STATUS_LEN, "payload layout must match PROTOCOL.md v1");

LSM6DS3 g_imu(I2C_MODE, 0x6A);

// ---- 64-bit monotonic micros (the 32-bit hardware counter wraps every ~71.6 min)
static volatile uint32_t g_last_us32 = 0, g_wraps = 0;
static inline uint64_t micros64() {
  noInterrupts(); uint32_t now = micros(); if (now < g_last_us32) g_wraps++; g_last_us32 = now;
  uint64_t v = ((uint64_t)g_wraps << 32) | now; interrupts(); return v;
}

// ---- ISR → sample flag/timestamp
static volatile bool g_drdy = false; static volatile uint64_t g_drdy_us = 0; static volatile uint32_t g_isr_overrun = 0;
static void drdy_isr() { if (g_drdy) g_isr_overrun++; g_drdy_us = micros64(); g_drdy = true; }

// ---- TX ring of encoded packets
static uint8_t g_ring[TX_RING_PACKETS][PKT_MAX]; static uint8_t g_ring_len[TX_RING_PACKETS];
static volatile uint16_t g_head = 0, g_tail = 0; static uint8_t g_tx_off = 0;
static uint32_t g_dropped = 0, g_emitted = 0, g_seq = 0;

static uint16_t crc16_ccitt(const uint8_t* d, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) { crc ^= (uint16_t)d[i] << 8; for (int b = 0; b < 8; b++) crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : (crc << 1); }
  return crc;
}
// Deepest the ring has been since the last status packet: reported and reset each period, so the host takes the max
// over a session. A since-boot maximum cannot be differenced and would carry in whatever piled up before anyone listened.
static volatile uint16_t g_ring_hiwater = 0;
static inline void note_occupancy(uint16_t head, uint16_t tail) {
  uint16_t used = (uint16_t)((head + TX_RING_PACKETS - tail) % TX_RING_PACKETS);
  if (used > g_ring_hiwater) g_ring_hiwater = used;
}

static bool ring_push(uint8_t type, const uint8_t* payload, uint8_t len) {
  uint16_t next = (g_head + 1) % TX_RING_PACKETS;
  if (next == g_tail) { g_dropped++; note_occupancy(g_head, g_tail); return false; }  // host too slow: drop, count, keep sampling
  uint8_t* p = g_ring[g_head]; p[0] = MAGIC0; p[1] = MAGIC1; p[2] = type; p[3] = len; memcpy(p + 4, payload, len);
  uint16_t crc = crc16_ccitt(p + 2, 2 + len); p[4 + len] = crc & 0xFF; p[5 + len] = crc >> 8;
  g_ring_len[g_head] = 6 + len; g_head = next; note_occupancy(g_head, g_tail); return true;
}

static void ring_drain_usb() {
  if (!_SerialUSB.connected()) return;                 // no terminal: writing would block forever; let the ring drop
  while (g_tail != g_head) {
    uint32_t actual = 0;
    _SerialUSB.send_nb(g_ring[g_tail] + g_tx_off, g_ring_len[g_tail] - g_tx_off, &actual, true);
    g_tx_off += (uint8_t)actual;
    if (g_tx_off < g_ring_len[g_tail]) return;         // CDC buffer full or a transfer in flight: resume here later
    g_tx_off = 0; g_tail = (g_tail + 1) % TX_RING_PACKETS;
  }
}

static void log_line(const char* s) { ring_push(TYPE_LOG, (const uint8_t*)s, (uint8_t)min((int)strlen(s), (int)LOG_MAX)); }

// ---- sensor configuration. Written register by register rather than through the library's `settings` struct, so
// what the sensor is actually configured to do is visible in this file and does not move with a library version.
static float g_accel_mg_lsb, g_gyro_mdps_lsb;
static_assert(IMU_ODR_HZ == 104 || IMU_ODR_HZ == 208 || IMU_ODR_HZ == 416 || IMU_ODR_HZ == 833,
              "IMU_ODR_HZ must be an LSM6DS3TR-C rate in the usable band (104/208/416/833)");
static uint8_t odr_code(int hz) { return hz >= 833 ? 0x7 : hz >= 416 ? 0x6 : hz >= 208 ? 0x5 : 0x4; }
// FS_XL: 00 = ±2 g, 01 = ±16 g, 10 = ±4 g, 11 = ±8 g  (NOT monotonic — the datasheet's order, not a sorted one)
static uint8_t accel_fs_code(int g) { return g >= 16 ? 0x1 : g >= 8 ? 0x3 : g >= 4 ? 0x2 : 0x0; }
static float accel_sens(int g) { return g >= 16 ? 0.488f : g >= 8 ? 0.244f : g >= 4 ? 0.122f : 0.061f; }   // mg/LSB
// FS_G: 00 = 245, 01 = 500, 10 = 1000, 11 = 2000 dps. Sensitivity comes from the datasheet table, NOT from
// FS/32768: at ±2000 dps that would give 61 mdps/LSB where the part is specified at 70.
static uint8_t gyro_fs_code(int dps) { return dps >= 2000 ? 0x3 : dps >= 1000 ? 0x2 : dps >= 500 ? 0x1 : 0x0; }
static float gyro_sens(int dps) { return dps >= 2000 ? 70.0f : dps >= 1000 ? 35.0f : dps >= 500 ? 17.50f : 8.75f; }  // mdps/LSB

static bool imu_init() {
  if (g_imu.beginCore() != 0) return false;            // powers P1.08, starts Wire1, checks WHO_AM_I
  Wire1.setClock(I2C_HZ);
  g_imu.writeRegister(REG_CTRL3_C, 0x44);              // BDU = 1 (a burst read cannot straddle two samples), IF_INC = 1
  g_imu.writeRegister(REG_CTRL1_XL, (odr_code(IMU_ODR_HZ) << 4) | (accel_fs_code(ACCEL_FS_G) << 2));
  g_imu.writeRegister(REG_CTRL2_G,  (odr_code(IMU_ODR_HZ) << 4) | (gyro_fs_code(GYRO_FS_DPS) << 2));
  // PULSED data-ready. This part's DRDY is LEVEL by default: INT1 goes high when a sample is ready and only falls
  // when the output registers are read. With a RISING-edge ISR that deadlocks on the first sample — the pin goes
  // high at power-on, nobody has read anything yet, so it never falls and there is never a second edge. Measured on
  // this board before the fix: INT1 high 81753/81753 polls, 0 edges, 0 interrupts, STATUS_REG stuck at 0x07.
  g_imu.writeRegister(REG_DRDY_PULSE, 0x80);           // DRDY_PULSED: ~75 us pulse per sample instead of a level
  // Data-ready on INT1 from the GYRO only. Both sensors run at the same ODR and are sampled together, so enabling
  // both would fire two interrupts per sample period and half of them would find nothing new.
  g_imu.writeRegister(REG_INT1_CTRL, 0x02);
  g_accel_mg_lsb = accel_sens(ACCEL_FS_G); g_gyro_mdps_lsb = gyro_sens(GYRO_FS_DPS);
  delay(50);                                           // let the first samples settle before anyone reads them
  return true;
}

void setup() {
  Serial.begin(2000000);                               // CDC ignores the baud
  pinMode(PIN_LSM6DS3TR_C_INT1, INPUT);
  while (!imu_init()) {
    static const char NYB[] = "0123456789ABCDEF";      // not HEX: the Arduino core defines that as 16
    char m[] = "no IMU: WAI=0x.. (internal I2C, nothing to rewire)";
    uint8_t who = 0; g_imu.readRegister(&who, REG_WHO_AM_I);
    m[14] = NYB[who >> 4]; m[15] = NYB[who & 0x0F];    // 0x6A = LSM6DS3TR-C, 0x69 = LSM6DS3, 0x00/0xFF = bus dead
    log_line(m); ring_drain_usb(); delay(500);
  }
  attachInterrupt(digitalPinToInterrupt(PIN_LSM6DS3TR_C_INT1), drdy_isr, RISING);
  log_line("handumi xiao imu ready");
}

// One-shot diagnosis for the way this board can be useless while looking healthy: I2C answers, the sensor is
// configured, and no interrupt ever arrives. STATUS_REG says whether the sensor is producing data at all, which
// separates "the sensor is asleep" from "INT1 is not reaching the pin".
static void diagnose_silent_int1() {
  static const char NYB[] = "0123456789ABCDEF";
  uint8_t st1 = 0, st2 = 0, i1 = 0;
  g_imu.readRegister(&st1, REG_STATUS); delay(8); g_imu.readRegister(&st2, REG_STATUS);
  g_imu.readRegister(&i1, REG_INT1_CTRL);
  uint8_t pulse = 0; g_imu.readRegister(&pulse, REG_DRDY_PULSE);
  char m[] = "no DRDY: STATUS=..:.. INT1_CTRL=.. PULSE=.. pin=.";
  m[16] = NYB[st1 >> 4]; m[17] = NYB[st1 & 0x0F];
  m[19] = NYB[st2 >> 4]; m[20] = NYB[st2 & 0x0F];
  m[33] = NYB[i1 >> 4];  m[34] = NYB[i1 & 0x0F];
  m[42] = NYB[pulse >> 4]; m[43] = NYB[pulse & 0x0F];   // 80 = pulsed; 00 = level, and a RISING ISR will deadlock
  m[49] = digitalRead(PIN_LSM6DS3TR_C_INT1) ? '1' : '0';
  log_line(m);
}

static uint32_t g_last_status_ms = 0;
void loop() {
  if (g_drdy) {
    uint64_t t_us; noInterrupts(); t_us = g_drdy_us; g_drdy = false; interrupts();
    uint8_t raw[14];
    if (g_imu.readRegisterRegion(raw, REG_OUT_TEMP_L, 14) == 0) {
      int16_t t  = (int16_t)(raw[0]  | (raw[1]  << 8));                                    // little-endian, unlike the ICM
      int16_t gx = (int16_t)(raw[2]  | (raw[3]  << 8)), gy = (int16_t)(raw[4]  | (raw[5]  << 8)), gz = (int16_t)(raw[6]  | (raw[7]  << 8));
      int16_t ax = (int16_t)(raw[8]  | (raw[9]  << 8)), ay = (int16_t)(raw[10] | (raw[11] << 8)), az = (int16_t)(raw[12] | (raw[13] << 8));
      const float G = 9.80665f, D2R = 0.017453292519943295f;
      const float a = g_accel_mg_lsb * 1e-3f * G, w = g_gyro_mdps_lsb * 1e-3f * D2R;
      ImuPayload p; p.seq = g_seq++; p.t_us = t_us;
      p.ax = ax * a; p.ay = ay * a; p.az = az * a;
      p.gx = gx * w; p.gy = gy * w; p.gz = gz * w;
      p.temp = (t / 256.0f) + 25.0f;
      if (ring_push(TYPE_IMU, (const uint8_t*)&p, sizeof p)) g_emitted++;
    }
  }
  uint32_t ms = millis();
  static bool g_diagnosed = false;
  if (!g_diagnosed && g_emitted == 0 && ms > 3000) { g_diagnosed = true; diagnose_silent_int1(); }
  if (ms - g_last_status_ms >= STATUS_PERIOD_MS) {
    g_last_status_ms = ms;
    StatusPayload s; s.t_us = micros64(); s.rate_hz = IMU_ODR_HZ; s.emitted = g_emitted; s.dropped = g_dropped + g_isr_overrun;
    s.fw_major = FW_MAJOR; s.fw_minor = FW_MINOR; s.ring_hiwater = g_ring_hiwater; g_ring_hiwater = 0;
    ring_push(TYPE_STATUS, (const uint8_t*)&s, sizeof s);
  }
  ring_drain_usb();
}
