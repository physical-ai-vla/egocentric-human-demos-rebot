# P0 — three-sensor bring-up, first real run

Ladder rung P0 of `docs/ego_teleop/MULTISENSOR_FUSION.md`. Baseline frozen before any of this: `HW_BASELINE.md`.

```
wrist IMU     XIAO nRF52840 Sense   PRESENT   P0-1 PASS   (measured 2026-09-22)
wrist camera  Arducam 1080P         ABSENT    P0-2 blocked
chest RGB-D   Orbbec                ABSENT    P0-3 blocked
three at once                       —         P0-4 blocked
```

## P0-1 — XIAO nRF52840 Sense, alone: PASS

`/dev/cu.usbmodem31401`, USB serial `D2FB7365FE48ED0F`. Run twice: once ad hoc, once through the production device
path (`TeensyImu` + `SeqTracker` + the clock fit), which is the one that counts because it is the code the fusion
runs.

| | measured | target |
|---|---|---|
| device rate | **417.5 Hz** | 416 |
| host rate | 422.1 Hz | — |
| CRC errors | **0** | 0 |
| sequence gaps | **0** | 0 |
| samples lost | **0.0000 %** | — |
| timestamps monotonic | **yes** | yes |
| inter-sample gap median / p99 / max | 2395 / 2418 / 2427 µs | 2404 |
| firmware FIFO drops, while a host is reading | **0 /s** | 0 |
| tx ring high-water | 1–2 of 256 | — |

The firmware sketch, the protocol parser and the 416 Hz configuration all work as written. `rate_hz: 416` in the
profile was the only host-side change the runbook predicted, and it was the only one needed.

### Two things that looked like faults and are not

* **`dropped(FIFO) = 1,078,324`** on the first status packet. Cumulative since boot, accumulated while nothing was
  draining the port — `xiao_imu.ino:82` lets the ring drop when `!_SerialUSB.connected()`. The board had been
  plugged in unread for ~40 min; 40 min × 416 Hz ≈ 1.0 M. Differenced against the next status packet it is 0/s.
  Read the status counters **differenced**, never absolute.
* **A 19.5 s device-time gap and "54 % lost"** on the first ad-hoc read. `pyserial.reset_input_buffer()` does not
  clear the macOS CDC buffer, so one stale packet arrives first and the gap to the live stream reads as loss. The
  production path already does the right thing (`TeensyImu._drain`, read-and-discard for `DRAIN_S`). Any new
  script must drain the same way.

### One real finding: the XIAO clock runs +1.099 % fast

Measured the simplest possible way, first sample to last, no fit involved:

```
device elapsed  40.4436 s
host   elapsed  39.9991 s          ->  device clock is +1.099 % FAST
```

Stable: the fitted slope is 988.99 ns/µs (−11,000 ppm) across every window from 800 to 25,354 samples and across
separate captures, varying by ~0.03 %. The sketch takes its timestamps from the Arduino core's `micros()` and never
selects a clock source, so the nRF52840 is timing off its internal RC oscillator (±1–2 % typical) rather than a
crystal. That is exactly the magnitude observed.

**This does not block P0-1** — `fit_device_to_host` estimates the slope and absorbs it, which is why the host-side
numbers above are clean. What it costs:

* The `device_us` field is **not real microseconds** and must never be treated as such. `ego_teleop/tools/
  imu_allan.py:76` builds its time base from `device_timestamp_us × 1000`, so an Allan run on this board has a τ
  axis 1.1 % short. Harmless for noise densities, wrong if anyone reads τ off it precisely.
* `LiveImuClock` runs on a nominal 1000 ns/µs slope for its first `min_for_slope = 800` samples (~1.9 s at
  416 Hz). Its lower-envelope offset tracks a monotonically decreasing residual, so it stays close — but the
  sliding-min behaviour under a 1.1 % slope error has not been measured against a camera, and 1.9 s of VI
  initialisation is exactly when the timeline matters most. **Measure it in P0-4, do not assume it.**

Fixing it in firmware (start HFXO / select the 32.768 kHz crystal) would remove the whole class. Not done — it is
a firmware change and the host side already compensates.

### The residual number is 2.5 ms, not 45 ms

An early capture showed a 45 ms clock-fit residual. That was the measurement's own fault: stamping one host time
per `read(8192)` gives every sample in the burst the same stamp, up to 178 samples of quantisation. Re-measured
with one packet per read:

```
residual around the fitted line   p50 1.75 / p95 2.52 / max 8.54 ms
```

The production reader is already built for this — `teensy_imu.py:22` says so and `READ_TIMEOUT_S = 0.005` bounds
the quantisation to 5 ms, which is what its own 20 s run reported (p95 4.73 ms). At 416 Hz that is ~2 samples per
read. Nothing to change; the 45 ms was never real.

## Blocker for P4 and beyond: the wrist calibration bundle is for a different IMU

`wrist_bundle_right_v005` — and therefore `camera_imu_right_v002` (`T_camera_imu`, `time_offset_ms: 32.77`) and the
`imu_noise` block — carries `imu_serial: '20778070'`. That is the **Teensy / ICM42688P** unit, not this XIAO
(`D2FB7365FE48ED0F`). A different board, a different mounting, a different clock.

So before any fused number means anything:

* `T_camera_imu` must be re-estimated for the XIAO (Kalibr, as the bundle was),
* `time_offset_ms` must be re-estimated (gyro correlation against the Arducam),
* `imu_noise` must be re-run for the LSM6DS3TR-C (`tools/imu_allan.py`; the current values are ICM42688P).

Reusing 32.77 ms across boards would put a fixed, invisible skew into every camera↔IMU pairing. This is a
calibration task, not a code task, and it needs the Arducam physically mounted with the XIAO.

Stationary readings on this board, for reference when those are redone: gyro bias **3.73 °/s**, gyro noise
1.93 °/s, |accel| **10.031 m/s²** (+2.29 % vs g).

## P0-2 / P0-3 / P0-4 — blocked on hardware

```
wrist Arducam   'Arducam 1080P Low Light' is not in the AVFoundation listing
                (present: 3x C922, iPhone, MacBook camera)
chest Orbbec    pyorbbecsdk Context().query_devices() -> 0 devices
```

Both need to be physically connected. Once they are, the profiles are already written and the run is one command
each:

```bash
# P0-2 / P0-3 / P0-4: all three through the COLLECTOR's device path, which is deliberately not the fusion rig's
.venv/bin/python -m ego_teleop.tools.f0_sensors --hardware configs/handumi/hardware_fusion_p0.yaml \
    --seconds 30 --imu-hz 416 --json p0.json

# then P5 side-by-side on real hardware, no robot: RGB-D vs VI vs fused, live
sudo .venv/bin/python -m ego_teleop.tools.f5_teleop_hud --live --stage compare --protocol \
    --hardware configs/handumi/hardware_fusion_live.yaml
```

`hardware_fusion_p0.yaml` includes the Orbbec; `hardware_fusion_live.yaml` deliberately does not, because
`FusedLiveRig` opens the chest camera itself and one process cannot open it twice.

What P0-4 has to show, and the reason it exists: that `Arducam ~30 Hz`, `RGB-D ~30 Hz` and `XIAO ~416 Hz` all hold
**at the same time**, on one bus. Each alone proves nothing about the other two.
