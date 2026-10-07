# Teensy + ICM42688P IMU bring-up (2026-09-14)

Runbook for taking one freshly wired Teensy 4.1 + ICM-42688-P from "the LED is on" to "this board is fit to record
with". The production path is unchanged — binary framing, CRC16, u64 device clock (`firmware/teensy_imu/PROTOCOL.md`),
`devices/teensy_imu.py` on the host — and everything here reuses it. Nothing in this document introduces a second
parser, a second schema or a second notion of time.

## Wiring (QCIoT-ICM42688P breakout ↔ Teensy 4.1)

| ICM42688P J1 | Teensy 4.1 | |
|---|---|---|
| VDD | 3V3 | |
| GND | GND | |
| SCLK | 13 | SPI0 SCK |
| SDI / MOSI | 11 | |
| SDO / MISO | 12 | |
| CS | 10 | `PIN_CS` |
| INT1 | **2** | `PIN_INT1`, data-ready, push-pull active-high |

J3 jumpers on the QCIoT board must be set for SPI. `WHO_AM_I` (0x75) must read **0x47**.

## The ODR trap

The ICM-42688-P `ACCEL_ODR`/`GYRO_ODR` field encodes **0x06 = 1 kHz, 0x0F = 500 Hz, 0x07 = 200 Hz, 0x08 = 100 Hz**.
There is **no 400 Hz rate on this part.** The firmware used to map "400" onto `0x07`, so it ran at 200 Hz while every
document said 400; asking for 200 would have landed on `0x08` and quietly halved it to 100 Hz. `odr_code()` is now
exact and a `static_assert` rejects any `IMU_ODR_HZ` that is not 100 / 200 / 500 / 1000. The 500 Hz step is the one to
try when 200 Hz is proven — not 400.

## Flash

Arduino IDE with Teensyduino: board **Teensy 4.1**, USB type **Serial**, CPU 600 MHz, sketch
`firmware/teensy_imu/teensy_imu.ino`. Everything tunable lives in `config.h` (pins, `IMU_ODR_HZ`, ranges, ring size);
edit there, not in the sketch. **Close the IDE's Serial Monitor before logging** — macOS CDC ports are exclusive, and a
monitor holding `/dev/cu.usbmodem*` makes every host tool fail with "device reports readiness to read but returned no
data". Check with `lsof /dev/cu.usbmodem*`.

## "The board is not connected"

Before suspecting the cable, the hub or the board, check what USB Type it was flashed with. A Teensy built as RawHID,
MIDI or HID is powered, enumerated and working, and exposes **no CDC port at all** — macOS creates no
`/dev/cu.usbmodem*` node and `pyserial` cannot see it, which is indistinguishable from a dead board. This is how the
left wrist board spent a session being reported as absent; it was `Teensyduino RawHID` (16c0:**0486**) the whole time,
against **0483** for a Serial build.

`imu_logger --list` now reads the IOKit registry and says so. By hand:

```bash
ioreg -p IOUSB -a -l -w0 | grep -A 4 16C0        # or: ioreg -p IOUSB -w0 -l | grep -i teensy
arduino-cli board list                            # a non-CDC Teensy still shows, as HID=16c0:0486...
arduino-cli upload -b teensy:avr:teensy41:usb=serial,speed=600 -p usb:<id> firmware/teensy_imu
```

Serial numbers: take them from `--list` or `ioreg`. `arduino-cli board list` prints Teensy serials a digit short
(`2054016` for `20540160`), and a short serial is exactly what the substring matcher can attach to the wrong board.

## LEFT/RIGHT identity

Fixed by USB serial, never by port order — two identical boards on `/dev/cu.usbmodem*` differ by nothing else, and a
swap silently mislabels every episode with nothing in the data to catch it later. `handumi_v1` assigns
**right = 20540160**, **left = 20778070**; which board is physically on which hand is a decision to confirm at mounting.
`TeensyImu.open()` refuses an unset, placeholder or ambiguous serial and refuses a port another device already took;
`DeviceManager` refuses two sides that name the same board before anything is opened.

## Record and validate

```bash
.venv/bin/python -m handumi_collector.tools.imu_logger --list                      # USB identity; the Teensy is vid 0x16c0
.venv/bin/python -m handumi_collector.tools.imu_logger --side right --seconds 60   # dead still, on a solid surface
.venv/bin/python -m handumi_collector.tools.validate_session data/imu_right_<stamp>
.venv/bin/python -m handumi_collector.tools.plot_imu data/imu_right_<stamp> --save imu.png
```

A session directory is `imu.csv` + `metadata.json` + `session.log`. The CSV columns are the episode schema
(`pose.episode_io.ImuArrays`) in SI units — `side,seq,device_timestamp_us,host_receive_ns,ax,ay,az,gx,gy,gz,temp_c` —
so `imu_allan`, `camera_imu_offset` and `validate_session` read a bring-up log and a recorded episode identically.
`validate_session` also accepts a raw episode directory.

### Gates (`tools/validate_session.py`)

| gate | limit |
|---|---|
| duration | ≥ 5 s |
| rate | expected ODR ±2.5 % (200 Hz → 195–205) |
| jitter_p99 | p99 interval ≤ 1.4 × nominal (200 Hz → 7.0 ms) |
| drops | sequence loss ≤ 0.1 % |
| crc | 0 errors |
| gravity | median &#124;a&#124; in 0.8–1.2 g (static recordings only; `--moving` skips it) |
| coverage | device-time span ≥ 95 % of the session's wall time |
| max_gap | longest interval ≤ 20 × the nominal period |
| double_trigger | no sample closer than half the nominal period |
| accel_headroom | peak &#124;a&#124; ≤ 90 % of full scale |
| clock_fit | device→host residual < 5 ms |

`coverage` exists because every other gate is computed over whatever arrived, so a side that goes silent partway still
satisfies all of them. A 60 s dual run where the left wrist stopped at 30 s was reported **PASS on nine gates out of
nine** — 201.58 Hz, 0.0000 % loss, 0 CRC, jitter and gravity textbook — on its surviving half. Only the `duration` line
showed 30.4 s against the right wrist's 60.0 s, and nothing compared the two. `max_gap` is the same idea for a stall in
the middle rather than at the end.

The drop gate is the one that earns its keep: losing 2 % of samples only moves the mean rate to ~196 Hz, still inside
the rate band, so timing alone cannot see it. That is what the device sequence number is for
(`tests/handumi/test_imu_session.py::test_drop_gate_catches_what_the_rate_gate_cannot`).

`accel_headroom` is the ±8 g → ±16 g decision, made from data: widen the range only once real wrist motion starts
approaching full scale. At rest the reading is ~12 %. Worth watching, though: one knock on the bench during a dual run
put a **single** sample at 7.80 g norm / 6.60 g on one axis — 82 % of full scale — with everything else below 1.06 g.
A transient, not a signal, but it says an impact during real handling can get close to clipping, so check this gate on
the first recordings with actual wrist motion rather than assuming the at-rest number generalises.

## The device ring holds stale samples

The Teensy keeps sampling whether or not a host is listening, and its ring drops the **newest** packet when full — so
the packets waiting in it are the *oldest*. `reset_input_buffer()` empties the host's buffer, not the device's, so a
session that starts reading immediately records a stale block first. Measured once: 233 stale samples (device time
69.7–70.9 s) followed by a single 6357-sample jump, reported as **34 % loss and 129 Hz** when the board had in fact
lost nothing — after the jump the same recording ran at 197.91 Hz with 0.0000 % loss. `imu_logger` therefore discards
`--drain` seconds (default 0.5 s) before recording. Any other consumer of the raw stream needs the same drain.

The drain covers session start, where we know a backlog exists. A stall *during* a recording is a separate,
unresolved question — see `IMU_RING_POLICY.md` for the newest-drop vs oldest-drop trade-off and what to measure
before changing it.

## Host-stamp quantisation

`host_receive_ns` is stamped once per `serial.read()`, so **every sample that arrives in one read shares a timestamp**
and the serial read timeout sets the quantisation of the host clock. Measured on the bench, 25 s each, same board:

| read timeout | `clock_fit` residual |
|---|---|
| 50 ms | 14.543 ms |
| 5 ms | 1.054 ms |

14.5 ms is exactly the 50 ms uniform quantisation (50/sqrt(12) = 14.4), not USB latency. `imu_logger` defaults to 5 ms
(`--read-timeout`). **`devices/teensy_imu.py::_run()` still reads with `timeout=0.05`**, so a recorded episode carries
the 14 ms version — which matters because `estimate_camera_imu_offset` searches the camera<->IMU offset in 0.5 ms
steps. The device clock is unaffected either way (the fitted drift stayed within a few ppm in both runs), so this
limits how precisely the offset can be *measured*, not the timeline itself.

## Bench result (2026-09-14, right unit, serial 20540160)

First production run after the ODR and ring fixes — 60 s, board still on the desk:

```
firmware 1.0  device rate 200 Hz  device-side dropped 0 this session
[PASS] duration        60.0 s
[PASS] rate            197.91 Hz
[PASS] jitter_p99      5.053 ms (limit 7.00, nominal 5.00)
[PASS] drops           0 lost in 0 gaps = 0.0000%
[PASS] crc             0 errors, 0 resyncs
[PASS] gravity         median |a| 1.0109 g
[PASS] accel_headroom  peak |a| 0.97 g = 12% of +/-8 g
[PASS] clock_fit       drift +0.6 ppm, residual 1.651 ms
PASS
```

The sampling interval is **median = p95 = p99 = max = 5.0530 ms** — the jitter is not merely small, it is absent,
which is what INT1-latched hardware timestamps should look like. 5.0530 ms is 197.90 Hz, not 200: the part's internal
oscillator is ~1 % off nominal. Nothing downstream should care, because everything times off `device_timestamp_us`
rather than the nominal rate — but it is why the rate gate is a ±2.5 % band and not an equality.

Gyro bias at rest `(+0.22, -0.26, +0.02) dps`, per-axis noise `(0.61, 0.17, 0.20) dps` — x is visibly noisier than
y/z and is worth a second look during the Allan run. Accel norm `1.0109 g`, std `0.0026`.

## Dual wrist

```bash
.venv/bin/python -m handumi_collector.tools.imu_dual_smoke --seconds 60
```

Runs both boards through `DeviceManager` — the same builder, driver threads and serial resolution the recorder uses —
and gates each side separately into standard session directories. One board passing its gates says the part works; it
says nothing about the state the collector actually runs in, which is two at once.

**Status 2026-09-14: dual PASS, 9/9 on both sides**, 60 s simultaneous:

| | rate | drops | CRC | double_trigger | gravity | clock |
|---|---|---|---|---|---|---|
| left `20778070` | 201.57 Hz | 0.0000 % | 0 | 0.00 % | 1.0107 g | +4.0 ppm / 1.655 ms |
| right `20540160` | 197.93 Hz | 0.0000 % | 0 | 0.00 % | 1.0086 g | −2.6 ppm / 1.623 ms |

The two boards run 3.6 Hz apart (201.6 vs 197.9) — each part's internal oscillator, not a configuration difference.
Nothing downstream should care, because everything times off `device_timestamp_us`; it is why the rate gate is a ±2.5 %
band rather than an equality, and why a shared timeline has to come from the clock fit rather than from sample counting.

### Two faults this cost, both physical

**A marginal USB cable.** It browned the Teensy out, taking the 3V3 it supplies to the sensor with it. One cause, four
symptoms that each looked like something else: `WHO_AM_I = 0x00`, a board that worked once and not after a reboot, the
board vanishing from USB entirely, and a failed upload. `firmware/bringup/spi_harness_probe` settled it — after the
swap, `WHO_AM_I = 0x47` at 8 MHz, 1 MHz, 100 kHz and bit-banged 50 kHz alike.

**An INT1 jumper routed alongside the 8 MHz SPI clock.** Crosstalk added a spurious data-ready edge behind roughly one
sample in nine: 1463 extra samples 19–20 µs behind a partner, 10.8 % of a 60 s record. It passed every check we had —
sequence numbers stayed perfectly contiguous so no drop check fired, and p99 sat exactly on nominal because it looks
only at the slow tail — while the mean rate read 226 Hz. Re-routing the jumper took it to 0.00 %. The `double_trigger`
gate exists because of this: "rate too high" does not tell you what to go and look at.

## 200 vs 500 Hz (2026-09-14): 200 Hz stays the default

Both wrists, 60 s simultaneous, twice at each rate, same harness and same firmware apart from `IMU_ODR_HZ`:

| | 200 Hz L | 200 Hz R | 500 Hz L | 500 Hz R |
|---|---|---|---|---|
| effective rate (Hz) | 201.58 | 197.93 | 503.95 | 494.82 |
| jitter p99 (ms) | 4.961 | 5.053 | 1.985 | 2.021 |
| drops | 0.000 % | 0.000 % | 0.000 % | 0.000 % |
| CRC errors | 0 | 0 | 0 | 0 |
| double_trigger | 0 | 0 | 0 | 0 |
| coverage | 100 % | 100 % | 100 % | 100 % |
| clock residual (ms) | 1.667 | 1.624 | 1.731 | 1.750 |
| ring high-water / 256 | 2 | 1 | 2 | 2 |
| device drops (session) | 0 | 0 | 0 | 0 |
| USB (kB/s) | 9.1 | 8.9 | 22.6 | 22.2 |
| host CPU (% of one core) | 6.0 | | 11.1 | |

**500 Hz works and buys nothing measurable here.** Every quality figure is identical; the one number that governs how
precisely the camera↔IMU offset can be measured — the clock-fit residual — did not improve, and could not, because it
is set by the host's read quantisation rather than by the sample rate. What does change is cost: ~2× host CPU, 2.5×
USB and 2.5× storage (3.9 → 9.9 MB/min for the pair). At a 30 fps camera, 200 Hz already puts ~6.6 IMU samples between
frames, which is the ordinary range for VIO pre-integration.

So: **production default 200 Hz; 500 Hz is a validated optional mode**, one line in `config.h`, to turn on if a VIO
backend ever demonstrates it needs the samples.

It also closes `IMU_RING_POLICY.md`. Raising the rate 2.5× shrinks the ring's depth in time from 1.28 s to 0.51 s, and
the high-water mark stayed at **2 of 256 packets**. While a host is reading, the ring does not queue — there is no
latency to ratchet and no case for changing newest-drop to oldest-drop on present evidence.

## Then

200 Hz now passes end to end (above). Next, raise `IMU_ODR_HZ` to **500** and re-run the same recording and gates. Compare drop
rate, p99 jitter and the `clock_fit` residual against the 200 Hz session; if they hold, 500 Hz becomes the default for
VIO. The config is the only thing that changes.
