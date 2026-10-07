// ICM-42688-P bring-up / diagnostic sketch — NOT the production firmware.
//
// Recovered 2026-09-14 from the Arduino IDE's unsaved-sketch buffer, verbatim below the header. This is the sketch that
// first proved the QCIoT wiring on this desk: WHO_AM_I = 0x47, INT1 data-ready on pin 2, ~198 Hz, sane accel and gyro.
// Keep it for exactly that job — when a board or a harness is suspect, this is the shortest path to "is the part alive
// and is INT1 firing". It prints human-readable CSV at 115200 that any terminal can show.
//
// Do NOT develop this into the recording firmware; that is firmware/teensy_imu/ (binary framing + CRC16 + u32 sequence
// numbers + a u64 device clock — see PROTOCOL.md). This sketch has no sequence number, so dropped samples are invisible,
// and `handumi_collector.tools.imu_logger` marks any session recorded from it `drops_detectable: false` and fails the
// drop gate rather than reporting a comforting 0 %.
//
// It also differs from production on one setting: ACCEL_CONFIG0 = 0x07 here is +/-16 g (scale /2048), while production
// runs +/-8 g for the better resolution. Both use 0x07 for the ODR field, which is 200 Hz — this sketch's own comments
// are independent confirmation that the old firmware's "400 = 7" mapping was wrong.
//
// Build: Arduino IDE / arduino-cli, board Teensy 4.1, USB type Serial, 600 MHz.
//   arduino-cli compile -b teensy:avr:teensy41:usb=serial,speed=600 firmware/bringup/icm42688p_bringup
//   arduino-cli upload  -b teensy:avr:teensy41:usb=serial,speed=600 -p usb:<id> firmware/bringup/icm42688p_bringup
// Read it back with:
//   python -m handumi_collector.tools.imu_logger --port /dev/cu.usbmodemXXXX --baud 115200 --side right --seconds 60
// which auto-detects the text format and converts g/dps to SI on the way into the standard session schema.

#include <SPI.h>

constexpr int CS_PIN  = 10;
constexpr int INT_PIN = 2;

constexpr uint8_t WHO_AM_I       = 0x75;
constexpr uint8_t PWR_MGMT0      = 0x4E;
constexpr uint8_t GYRO_CONFIG0   = 0x4F;
constexpr uint8_t ACCEL_CONFIG0  = 0x50;

constexpr uint8_t INT_CONFIG     = 0x14;
constexpr uint8_t INT_CONFIG1    = 0x64;
constexpr uint8_t INT_SOURCE0    = 0x65;

constexpr uint8_t ACCEL_DATA_X1  = 0x1F;

volatile bool dataReady = false;
volatile uint32_t irqTimeUs = 0;

void imuISR() {
  irqTimeUs = micros();
  dataReady = true;
}

uint8_t readReg(uint8_t reg) {
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));

  digitalWrite(CS_PIN, LOW);
  SPI.transfer(reg | 0x80);
  uint8_t v = SPI.transfer(0x00);
  digitalWrite(CS_PIN, HIGH);

  SPI.endTransaction();
  return v;
}

void writeReg(uint8_t reg, uint8_t value) {
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));

  digitalWrite(CS_PIN, LOW);
  SPI.transfer(reg & 0x7F);
  SPI.transfer(value);
  digitalWrite(CS_PIN, HIGH);

  SPI.endTransaction();
}

void readRegs(uint8_t reg, uint8_t *buf, size_t n) {
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));

  digitalWrite(CS_PIN, LOW);
  SPI.transfer(reg | 0x80);

  for (size_t i = 0; i < n; i++) {
    buf[i] = SPI.transfer(0x00);
  }

  digitalWrite(CS_PIN, HIGH);
  SPI.endTransaction();
}

int16_t toInt16(uint8_t hi, uint8_t lo) {
  return (int16_t)((hi << 8) | lo);
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  pinMode(INT_PIN, INPUT);

  SPI.begin();
  delay(100);

  uint8_t who = readReg(WHO_AM_I);

  Serial.print("# WHO_AM_I=0x");
  Serial.println(who, HEX);

  if (who != 0x47) {
    Serial.println("# ERROR: IMU not detected");
    while (1);
  }

  // accel + gyro Low Noise mode
  writeReg(PWR_MGMT0, 0x0F);
  delay(50);

  // ±2000 dps, 200 Hz
  writeReg(GYRO_CONFIG0, 0x07);

  // ±16 g, 200 Hz
  writeReg(ACCEL_CONFIG0, 0x07);

  /*
    INT_CONFIG:
    INT1_MODE          = 0 : pulsed
    INT1_DRIVE_CIRCUIT = 1 : push-pull
    INT1_POLARITY      = 1 : active high

    bits 2:0 = 0b011
  */
  writeReg(INT_CONFIG, 0x03);

  // INT_ASYNC_RESET bit4 must be 0 for proper INT operation
  uint8_t ic1 = readReg(INT_CONFIG1);
  ic1 &= ~(1 << 4);
  writeReg(INT_CONFIG1, ic1);

  // Route UI Data Ready to INT1: bit3 = 1
  writeReg(INT_SOURCE0, 0x08);

  delay(20);

  attachInterrupt(
    digitalPinToInterrupt(INT_PIN),
    imuISR,
    RISING
  );

  Serial.println(
    "t_us,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"
  );
}

void loop() {
  if (!dataReady)
    return;

  noInterrupts();
  dataReady = false;
  uint32_t t = irqTimeUs;
  interrupts();

  uint8_t b[12];
  readRegs(ACCEL_DATA_X1, b, 12);

  int16_t axr = toInt16(b[0],  b[1]);
  int16_t ayr = toInt16(b[2],  b[3]);
  int16_t azr = toInt16(b[4],  b[5]);

  int16_t gxr = toInt16(b[6],  b[7]);
  int16_t gyr = toInt16(b[8],  b[9]);
  int16_t gzr = toInt16(b[10], b[11]);

  float ax = axr / 2048.0f;
  float ay = ayr / 2048.0f;
  float az = azr / 2048.0f;

  float gx = gxr / 16.4f;
  float gy = gyr / 16.4f;
  float gz = gzr / 16.4f;

  Serial.print(t);
  Serial.print(",");
  Serial.print(ax, 5);
  Serial.print(",");
  Serial.print(ay, 5);
  Serial.print(",");
  Serial.print(az, 5);
  Serial.print(",");
  Serial.print(gx, 3);
  Serial.print(",");
  Serial.print(gy, 3);
  Serial.print(",");
  Serial.println(gz, 3);
}