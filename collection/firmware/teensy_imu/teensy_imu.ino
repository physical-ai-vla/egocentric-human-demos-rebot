// HandUMI Teensy 4.1 + ICM42688P IMU streamer — protocol: PROTOCOL.md (authoritative), config: config.h
// Timing comes from the sensor's INT1 data-ready interrupt (never from a polling loop): the ISR latches a 64-bit
// monotonic micros() and raises a flag; the main loop reads the sample over SPI into a ring buffer, and USB writes
// drain the ring so a slow host can never block sampling. Raw sensor axes are streamed as-is (no L/R flips here).
#include <Arduino.h>
#include <SPI.h>
#include "config.h"

// ---- ICM-42688-P registers (bank 0)
#define REG_DEVICE_CONFIG   0x11
#define REG_INT_CONFIG      0x14
#define REG_TEMP_DATA1      0x1D
#define REG_ACCEL_DATA_X1   0x1F
#define REG_INT_STATUS      0x2D
#define REG_PWR_MGMT0       0x4E
#define REG_GYRO_CONFIG0    0x4F
#define REG_ACCEL_CONFIG0   0x50
#define REG_INT_CONFIG1     0x64
#define REG_INT_SOURCE0     0x65
#define REG_INT_SOURCE3     0x68
#define REG_WHO_AM_I        0x75
#define WHO_AM_I_VALUE      0x47

static const uint8_t MAGIC0 = 0xA5, MAGIC1 = 0x5A;
static const uint8_t TYPE_IMU = 0x01, TYPE_STATUS = 0x02, TYPE_LOG = 0x03;
static const uint8_t IMU_LEN = 40, STATUS_LEN = 24;
static const uint8_t LOG_MAX = IMU_LEN;                 // text logs are clamped so every packet fits one ring slot
// One ring slot holds the largest packet we ever emit: framing (magic 2 + type 1 + len 1 + crc 2) + the biggest payload.
// This was `4 + 255 + 2` in a uint8_t, which truncated to 5 and made ring_push() overrun 41 bytes into the next slots.
static const uint16_t PKT_MAX = 6 + IMU_LEN;
static_assert(STATUS_LEN <= IMU_LEN && LOG_MAX <= IMU_LEN, "IMU_LEN must stay the largest payload, it sizes the ring slot");

struct __attribute__((packed)) ImuPayload { uint32_t seq; uint64_t t_us; float ax, ay, az, gx, gy, gz, temp; };
struct __attribute__((packed)) StatusPayload { uint64_t t_us; uint32_t rate_hz, emitted, dropped; uint8_t fw_major, fw_minor; uint16_t reserved; };

// ---- 64-bit monotonic micros (extends the 32-bit hardware counter; called from ISR and loop)
static volatile uint32_t g_last_us32 = 0; static volatile uint32_t g_wraps = 0;
static inline uint64_t micros64() {
  noInterrupts(); uint32_t now = micros(); if (now < g_last_us32) g_wraps++; g_last_us32 = now;
  uint64_t v = ((uint64_t)g_wraps << 32) | now; interrupts(); return v;
}

// ---- ISR → sample flag/timestamp
static volatile bool g_drdy = false; static volatile uint64_t g_drdy_us = 0; static volatile uint32_t g_isr_overrun = 0;
static void drdy_isr() { if (g_drdy) g_isr_overrun++; g_drdy_us = micros64(); g_drdy = true; }

// ---- TX ring of encoded packets
static uint8_t g_ring[TX_RING_PACKETS][PKT_MAX]; static uint8_t g_ring_len[TX_RING_PACKETS];
static volatile uint16_t g_head = 0, g_tail = 0; static uint32_t g_dropped = 0, g_emitted = 0, g_seq = 0;

static uint16_t crc16_ccitt(const uint8_t* d, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) { crc ^= (uint16_t)d[i] << 8; for (int b = 0; b < 8; b++) crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : (crc << 1); }
  return crc;
}
// Deepest the ring has been since the last status packet. Reported and reset every period, so the host can take the
// max over a session and get that session's true high-water -- a maximum cannot be differenced the way a counter can,
// so a since-boot value would carry in whatever piled up before anyone was listening.
static volatile uint16_t g_ring_hiwater = 0;
static inline void note_occupancy(uint16_t head, uint16_t tail) {
  uint16_t used = (uint16_t)((head + TX_RING_PACKETS - tail) % TX_RING_PACKETS);
  if (used > g_ring_hiwater) g_ring_hiwater = used;
}

static bool ring_push(uint8_t type, const uint8_t* payload, uint8_t len) {
  uint16_t next = (g_head + 1) % TX_RING_PACKETS;
  if (next == g_tail) { g_dropped++; note_occupancy(g_head, g_tail); return false; }   // host too slow: drop, count, keep sampling
  uint8_t* p = g_ring[g_head]; p[0] = MAGIC0; p[1] = MAGIC1; p[2] = type; p[3] = len; memcpy(p + 4, payload, len);
  uint16_t crc = crc16_ccitt(p + 2, 2 + len); p[4 + len] = crc & 0xFF; p[5 + len] = crc >> 8;
  g_ring_len[g_head] = 6 + len; g_head = next; note_occupancy(g_head, g_tail); return true;
}
static void ring_drain_usb() {
  while (g_tail != g_head) {
    if (Serial.availableForWrite() < g_ring_len[g_tail]) return;   // never block on USB
    Serial.write(g_ring[g_tail], g_ring_len[g_tail]); g_tail = (g_tail + 1) % TX_RING_PACKETS;
  }
}

// ---- SPI helpers
static SPISettings spi_cfg(SPI_HZ, MSBFIRST, SPI_MODE0);
static void wr(uint8_t reg, uint8_t v) { SPI.beginTransaction(spi_cfg); digitalWriteFast(PIN_CS, LOW); SPI.transfer(reg & 0x7F); SPI.transfer(v); digitalWriteFast(PIN_CS, HIGH); SPI.endTransaction(); }
static uint8_t rd(uint8_t reg) { SPI.beginTransaction(spi_cfg); digitalWriteFast(PIN_CS, LOW); SPI.transfer(reg | 0x80); uint8_t v = SPI.transfer(0); digitalWriteFast(PIN_CS, HIGH); SPI.endTransaction(); return v; }
static void rdn(uint8_t reg, uint8_t* buf, size_t n) { SPI.beginTransaction(spi_cfg); digitalWriteFast(PIN_CS, LOW); SPI.transfer(reg | 0x80); for (size_t i = 0; i < n; i++) buf[i] = SPI.transfer(0); digitalWriteFast(PIN_CS, HIGH); SPI.endTransaction(); }

static float g_gyro_lsb_dps, g_accel_lsb_g;
// ACCEL/GYRO_ODR[3:0] per the ICM-42688-P datasheet: 0x06 = 1 kHz, 0x0F = 500 Hz, 0x07 = 200 Hz, 0x08 = 100 Hz.
// There is no 400 Hz rate on this part — asking for one used to silently select 0x07 (200 Hz), so the code is exact now.
static_assert(IMU_ODR_HZ == 100 || IMU_ODR_HZ == 200 || IMU_ODR_HZ == 500 || IMU_ODR_HZ == 1000,
              "IMU_ODR_HZ must be 100, 200, 500 or 1000 - the ICM-42688-P has no other UI rate in this range");
static uint8_t odr_code(int hz) { return hz >= 1000 ? 0x06 : hz >= 500 ? 0x0F : hz >= 200 ? 0x07 : 0x08; }
static uint8_t gyro_fs_code(int dps) { return dps >= 2000 ? 0 : dps >= 1000 ? 1 : dps >= 500 ? 2 : 3; }
static uint8_t accel_fs_code(int g) { return g >= 16 ? 0 : g >= 8 ? 1 : g >= 4 ? 2 : 3; }

static void log_line(const char* s) { ring_push(TYPE_LOG, (const uint8_t*)s, (uint8_t)min((int)strlen(s), (int)LOG_MAX)); }

static bool imu_init() {
  wr(REG_DEVICE_CONFIG, 0x01); delay(2);                       // soft reset
  if (rd(REG_WHO_AM_I) != WHO_AM_I_VALUE) return false;
  wr(REG_GYRO_CONFIG0,  (gyro_fs_code(GYRO_FS_DPS) << 5) | odr_code(IMU_ODR_HZ));
  wr(REG_ACCEL_CONFIG0, (accel_fs_code(ACCEL_FS_G) << 5) | odr_code(IMU_ODR_HZ));
  g_gyro_lsb_dps = (float)GYRO_FS_DPS / 32768.0f; g_accel_lsb_g = (float)ACCEL_FS_G / 32768.0f;
  wr(REG_INT_CONFIG, 0x03);                                    // INT1 push-pull, active high, pulsed
  wr(REG_INT_CONFIG1, 0x00);                                   // clear INT_ASYNC_RESET (required for INT1 to fire)
  wr(REG_INT_SOURCE0, 0x08);                                   // UI data-ready -> INT1
  wr(REG_PWR_MGMT0, 0x0F); delay(45);                          // gyro + accel low-noise mode; >45 ms before reading
  return true;
}

void setup() {
  pinMode(PIN_CS, OUTPUT); digitalWriteFast(PIN_CS, HIGH); pinMode(PIN_INT1, INPUT);
  SPI.begin(); Serial.begin(2000000);                          // CDC ignores the baud
  // Report the byte we actually read back, not just "not found": 0x00 is MISO stuck low (unwired, or no sensor power),
  // 0xFF is MISO floating high (CS never asserted, or MISO unwired), anything else is the wrong device or bus contention.
  // Without it a dead harness and a mis-wired one look identical from the host, which cost a session on the left board.
  while (!imu_init()) {
    static const char NYB[] = "0123456789ABCDEF";     // not HEX: the Arduino core defines that as 16
    char m[] = "no IMU: WAI=0x.. MISO=? INT1=?";   // snprintf would drag the whole printf in for two nibbles
    uint8_t who = rd(REG_WHO_AM_I);
    m[14] = NYB[who >> 4]; m[15] = NYB[who & 0x0F];
    // WHO_AM_I = 0x00 has several causes that look identical from the host. These two pulled-up reads are a ONE-SIDED
    // test -- they can convict, not acquit -- and run only on the failure path where SPI is dead anyway:
    //   MISO=0  conclusive: something holds SDO down. A breakout still strapped for I2C uses that pin as the address
    //           select and ties it low, which is the usual cause; otherwise a short to ground.
    //   INT1=0  conclusive: something drives INT1, so the sensor has power.
    //   MISO=1 / INT1=1  proves nothing either way. SDO is tri-stated while CS is high, and INT_CONFIG resets to
    //           open-drain active-low, so a perfectly healthy powered sensor floats both lines exactly like a dead one.
    SPI.end();
    pinMode(PIN_MISO, INPUT_PULLUP); pinMode(PIN_INT1, INPUT_PULLUP); delayMicroseconds(200);
    m[22] = digitalReadFast(PIN_MISO) ? '1' : '0';
    m[29] = digitalReadFast(PIN_INT1) ? '1' : '0';
    pinMode(PIN_INT1, INPUT);
    SPI.begin();
    log_line(m); ring_drain_usb(); delay(500);
  }
  (void)rd(REG_INT_STATUS);                                    // clear any pending
  attachInterrupt(digitalPinToInterrupt(PIN_INT1), drdy_isr, RISING);
  log_line("handumi teensy imu ready");
}

// One-shot diagnosis for the other way this board can be useless: SPI answers, the sensor is configured, and no sample
// ever arrives because the data-ready line is not landing where we think. Reading INT_STATUS says whether the sensor is
// producing data at all; routing data-ready to INT2 and watching the same Teensy pin says whether the wire is simply on
// the other INT pad, which looks identical at rest because both carry the breakout's pull-up.
static void diagnose_silent_int1() {
  static const char NYB[] = "0123456789ABCDEF";
  uint8_t st1 = rd(REG_INT_STATUS); delay(8); uint8_t st2 = rd(REG_INT_STATUS);
  char m[] = "no DRDY: INT_STATUS=..:.. pin=. I2=.";
  m[20] = NYB[st1 >> 4]; m[21] = NYB[st1 & 0x0F];
  m[23] = NYB[st2 >> 4]; m[24] = NYB[st2 & 0x0F];
  m[30] = digitalReadFast(PIN_INT1) ? '1' : '0';
  wr(REG_INT_CONFIG, 0x1B);              // INT1 and INT2 both push-pull, active high, pulsed
  wr(REG_INT_SOURCE0, 0x00);             // data-ready off INT1 ...
  wr(REG_INT_SOURCE3, 0x08);             // ... and on to INT2
  (void)rd(REG_INT_STATUS);
  uint32_t edges = 0; int prev = digitalReadFast(PIN_INT1); uint32_t t0 = millis();
  while (millis() - t0 < 300) { int now = digitalReadFast(PIN_INT1); if (now && !prev) edges++; prev = now; }
  m[35] = edges > 20 ? 'Y' : 'n';        // 300 ms at 200 Hz is ~60 edges if the wire is on INT2
  wr(REG_INT_SOURCE3, 0x00); wr(REG_INT_SOURCE0, 0x08); wr(REG_INT_CONFIG, 0x03);   // put INT1 back
  (void)rd(REG_INT_STATUS);
  log_line(m);
}

static uint32_t g_last_status_ms = 0;
void loop() {
  if (g_drdy) {
    uint64_t t_us; noInterrupts(); t_us = g_drdy_us; g_drdy = false; interrupts();
    uint8_t raw[14]; rdn(REG_TEMP_DATA1, raw, 14);             // TEMP(2) ACCEL xyz(6) GYRO xyz(6), big-endian int16
    int16_t t  = (int16_t)((raw[0] << 8) | raw[1]);
    int16_t ax = (int16_t)((raw[2] << 8) | raw[3]),  ay = (int16_t)((raw[4] << 8) | raw[5]),   az = (int16_t)((raw[6] << 8) | raw[7]);
    int16_t gx = (int16_t)((raw[8] << 8) | raw[9]),  gy = (int16_t)((raw[10] << 8) | raw[11]), gz = (int16_t)((raw[12] << 8) | raw[13]);
    const float G = 9.80665f, D2R = 0.017453292519943295f;
    ImuPayload p; p.seq = g_seq++; p.t_us = t_us;
    p.ax = ax * g_accel_lsb_g * G; p.ay = ay * g_accel_lsb_g * G; p.az = az * g_accel_lsb_g * G;
    p.gx = gx * g_gyro_lsb_dps * D2R; p.gy = gy * g_gyro_lsb_dps * D2R; p.gz = gz * g_gyro_lsb_dps * D2R;
    p.temp = (t / 132.48f) + 25.0f;
    if (ring_push(TYPE_IMU, (const uint8_t*)&p, sizeof p)) g_emitted++;
  }
  uint32_t ms = millis();
  static bool g_diagnosed = false;
  if (!g_diagnosed && g_emitted == 0 && ms > 3000) { g_diagnosed = true; diagnose_silent_int1(); }
  if (ms - g_last_status_ms >= STATUS_PERIOD_MS) {
    g_last_status_ms = ms;
    StatusPayload s; s.t_us = micros64(); s.rate_hz = IMU_ODR_HZ; s.emitted = g_emitted; s.dropped = g_dropped + g_isr_overrun;
    s.fw_major = FW_MAJOR; s.fw_minor = FW_MINOR;
    s.reserved = g_ring_hiwater; g_ring_hiwater = 0;         // the spare field now carries ring occupancy; layout unchanged
    ring_push(TYPE_STATUS, (const uint8_t*)&s, sizeof s);
  }
  ring_drain_usb();
}
