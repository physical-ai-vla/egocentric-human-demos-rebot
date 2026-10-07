// SPI harness probe — diagnostic only, for a Teensy 4.1 whose ICM-42688-P reads WHO_AM_I = 0x00.
//
// The production firmware can only say "the sensor did not answer". This pokes the harness from several angles and
// prints plain text at 115200, so the failure can be placed on a wire rather than guessed at. Flash it, read it with
// any terminal, then put firmware/teensy_imu back.
//
//   arduino-cli compile -u -b teensy:avr:teensy41:usb=serial,speed=600 -p usb:<id> firmware/bringup/spi_harness_probe
//
// What each result means is printed inline. The short version: SDO is tri-stated while CS is high, so the useful read
// is with CS asserted; and a bit-banged transfer at 50 kHz removes both the SPI peripheral and signal integrity from
// the list of suspects.
#include <Arduino.h>
#include <SPI.h>

#define PIN_CS    10
#define PIN_MOSI  11
#define PIN_MISO  12
#define PIN_SCK   13
#define PIN_INT1   2
#define REG_WHO_AM_I 0x75
#define WHO_EXPECT   0x47

static void hdr(const char* s) { Serial.println(); Serial.print("== "); Serial.println(s); }

static int pin_state(int pin) {
  // 1 = floats high under a pull-up AND low under a pull-down -> nothing drives it
  // 2 = held high by something, 0 = held low by something
  pinMode(pin, INPUT_PULLUP);  delayMicroseconds(500); int up   = digitalRead(pin);
  pinMode(pin, INPUT_PULLDOWN); delayMicroseconds(500); int down = digitalRead(pin);
  pinMode(pin, INPUT);
  if (up && !down) return 1;          // floating: follows whichever resistor is applied
  if (up && down)  return 2;          // driven high
  return 0;                           // driven low
}

static const char* state_name(int s) { return s == 1 ? "floating (nothing drives it)" : s == 2 ? "DRIVEN HIGH" : "DRIVEN LOW"; }

static void wr(uint8_t reg, uint8_t v) {
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
  digitalWrite(PIN_CS, LOW); SPI.transfer(reg & 0x7F); SPI.transfer(v); digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
}

static uint8_t spi_read(uint8_t reg, uint32_t hz) {
  SPI.beginTransaction(SPISettings(hz, MSBFIRST, SPI_MODE0));
  digitalWrite(PIN_CS, LOW);
  SPI.transfer(reg | 0x80);
  uint8_t v = SPI.transfer(0x00);
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
  return v;
}

// Manual mode-0 transfer at roughly 50 kHz: sample MISO on the rising edge, shift MOSI on the falling one.
static uint8_t bb_transfer(uint8_t out) {
  uint8_t in = 0;
  for (int i = 7; i >= 0; i--) {
    digitalWrite(PIN_MOSI, (out >> i) & 1);
    delayMicroseconds(10);
    digitalWrite(PIN_SCK, HIGH);
    delayMicroseconds(10);
    in = (in << 1) | digitalRead(PIN_MISO);
    digitalWrite(PIN_SCK, LOW);
    delayMicroseconds(10);
  }
  return in;
}

static uint8_t bb_read(uint8_t reg) {
  SPI.end();
  pinMode(PIN_SCK, OUTPUT);  digitalWrite(PIN_SCK, LOW);
  pinMode(PIN_MOSI, OUTPUT); digitalWrite(PIN_MOSI, LOW);
  pinMode(PIN_MISO, INPUT);
  pinMode(PIN_CS, OUTPUT);   digitalWrite(PIN_CS, LOW);
  delayMicroseconds(10);
  bb_transfer(reg | 0x80);
  uint8_t v = bb_transfer(0x00);
  digitalWrite(PIN_CS, HIGH);
  SPI.begin();
  return v;
}

static void hex2(uint8_t v) { Serial.print("0x"); if (v < 16) Serial.print('0'); Serial.print(v, HEX); }

static void report() {
  Serial.println("\n\n######## SPI harness probe (Teensy 4.1 + ICM-42688-P) ########");
  Serial.println("pins: CS=10 MOSI=11 MISO=12 SCK=13 INT1=2");

  hdr("1. lines at rest, before anything is driven");
  pinMode(PIN_CS, INPUT);
  Serial.print("   MISO(12): "); Serial.println(state_name(pin_state(PIN_MISO)));
  Serial.print("   INT1(2) : "); Serial.println(state_name(pin_state(PIN_INT1)));
  Serial.println("   MISO DRIVEN LOW here means the breakout is strapped for I2C (SDO = address select) or shorted.");
  Serial.println("   Floating proves nothing on its own: SDO is tri-stated while CS is high.");

  hdr("2. MISO with CS asserted low -- the read that actually tells you something");
  pinMode(PIN_CS, OUTPUT); digitalWrite(PIN_CS, LOW);
  delayMicroseconds(500);
  int m = pin_state(PIN_MISO);
  digitalWrite(PIN_CS, HIGH);
  Serial.print("   MISO(12) with CS low: "); Serial.println(state_name(m));
  Serial.println(m == 1 ? "   -> still floating: the sensor is NOT driving SDO. Suspect power, ground, the SDO wire, or CS."
                        : "   -> the sensor drives SDO, so it has power and SDO reaches the Teensy. Suspect SCK or MOSI.");

  hdr("3. hardware SPI, WHO_AM_I at three clocks");
  // Sections 1-2 put MISO back to plain GPIO to probe it, and a second SPI.begin() does not necessarily re-apply the
  // pin mux -- which made hardware SPI read 0x00 here while the bit-banged section read 0x47, deterministically, and
  // sent this diagnosis off after the harness for a round. End and restart so the mux is definitely reapplied.
  SPI.end();
  SPI.begin();
  pinMode(PIN_CS, OUTPUT); digitalWrite(PIN_CS, HIGH);
  const uint32_t clocks[] = {8000000, 1000000, 100000};
  for (uint8_t i = 0; i < 3; i++) {
    uint8_t v = spi_read(REG_WHO_AM_I, clocks[i]);
    Serial.print("   "); Serial.print(clocks[i] / 1000); Serial.print(" kHz -> WHO_AM_I = "); hex2(v);
    Serial.println(v == WHO_EXPECT ? "  OK" : "");
  }

  hdr("4. bit-banged at ~50 kHz (no SPI peripheral, no speed excuse)");
  uint8_t bb = bb_read(REG_WHO_AM_I);
  Serial.print("   WHO_AM_I = "); hex2(bb); Serial.println(bb == WHO_EXPECT ? "  OK" : "");

  hdr("5. a spread of registers at 1 MHz");
  const uint8_t regs[] = {0x75, 0x4E, 0x4F, 0x50, 0x1D, 0x11};
  uint8_t all_same = spi_read(regs[0], 1000000); bool same = true;
  for (uint8_t i = 0; i < 6; i++) {
    uint8_t v = spi_read(regs[i], 1000000);
    if (v != all_same) same = false;
    Serial.print("   reg "); hex2(regs[i]); Serial.print(" = "); hex2(v); Serial.println();
  }
  if (same) {
    Serial.print("   every register returned the same byte (");
    hex2(all_same);
    Serial.println(") -- the bus is not carrying data at all.");
    Serial.println("   0x00 = MISO never goes high; 0xFF = MISO never goes low. Either way nothing is answering.");
  }

  hdr("what to check next");
  Serial.println("   Loopback: short MOSI(11) to MISO(12) AT THE SENSOR END of the harness and re-run.");
  Serial.println("   Section 3 should then echo the register byte (0xF5 for WHO_AM_I's 0x75|0x80 -> reads back 0x00,");
  Serial.println("   so watch section 4, which sends 0x00 second and must read back 0x00; send-and-compare is in");
  Serial.println("   section 6 below instead).");

  hdr("6. loopback self-test (meaningful only with MOSI shorted to MISO)");
  SPI.end();
  pinMode(PIN_MOSI, OUTPUT); pinMode(PIN_MISO, INPUT); pinMode(PIN_SCK, OUTPUT); digitalWrite(PIN_SCK, LOW);
  bool echo = true;
  const uint8_t pat[] = {0xA5, 0x5A, 0x0F, 0xF0};
  for (uint8_t i = 0; i < 4; i++) {
    digitalWrite(PIN_MOSI, 0); delayMicroseconds(200);
    int low = digitalRead(PIN_MISO);
    digitalWrite(PIN_MOSI, 1); delayMicroseconds(200);
    int high = digitalRead(PIN_MISO);
    if (!(low == 0 && high == 1)) echo = false;
    (void)pat[i];
  }
  Serial.println(echo ? "   MISO follows MOSI -> those two wires reach the same point. If WHO_AM_I is still 0x00 with"
                        "\n   the short removed, suspect sensor power, ground, CS or SCK."
                      : "   MISO does NOT follow MOSI. With the short fitted that means one of those two wires is open;"
                        "\n   with no short fitted this line is expected and means nothing.");
  hdr("7. which INT pin is the wire actually on?");
  // A wire landing on INT2 instead of INT1 looks identical at rest -- both carry the breakout's pull-up, so pin 2 reads
  // DRIVEN HIGH either way -- but data-ready is routed per pin, so only one of them ever pulses. Configure the sensor,
  // route DRDY to one INT at a time, and count edges on the Teensy pin.
  wr(0x4F, 0x07);            // gyro  +/-2000 dps, ODR 200 Hz
  wr(0x50, 0x07);            // accel, ODR 200 Hz
  wr(0x4E, 0x0F);            // both in low-noise mode
  delay(60);
  wr(0x64, spi_read(0x64, 1000000) & ~(1 << 4));    // INT_ASYNC_RESET must be 0
  wr(0x14, 0x1B);                    // INT1 and INT2 both push-pull, active high, pulsed
  for (uint8_t which = 0; which < 2; which++) {
    wr(0x65, which == 0 ? 0x08 : 0x00);      // INT_SOURCE0: UI data-ready -> INT1
    wr(0x68, which == 1 ? 0x08 : 0x00);      // INT_SOURCE3: UI data-ready -> INT2
    (void)spi_read(0x2D, 1000000);                          // clear anything pending
    pinMode(PIN_INT1, INPUT);
    uint32_t edges = 0; int prev = digitalRead(PIN_INT1);
    uint32_t t0 = millis();
    while (millis() - t0 < 500) { int now = digitalRead(PIN_INT1); if (now && !prev) edges++; prev = now; }
    Serial.print(which == 0 ? "   DRDY routed to INT1 -> " : "   DRDY routed to INT2 -> ");
    Serial.print(edges * 2); Serial.println(" edges/s on Teensy pin 2");
  }
  Serial.println("   ~200/s on one line and ~0 on the other tells you which pad the wire is on.");
  Serial.println("   ~0 on both, with SPI otherwise healthy, means the INT wire is not landing on either INT pad.");
  wr(0x4E, 0x00);            // leave the sensor idle again

  Serial.println("\n######## done -- reflash firmware/teensy_imu when finished ########");
}

// Repeat forever rather than printing once at boot: a one-shot report is gone by the time a terminal attaches, which
// reads exactly like a silent board.
void setup() {
  Serial.begin(115200); while (!Serial && millis() < 3000) {} delay(200);
  pinMode(PIN_CS, OUTPUT); digitalWrite(PIN_CS, HIGH);
  SPI.begin();            // once here, so report()'s end/begin pair always follows a begin
}
void loop()  { report(); delay(5000); }
