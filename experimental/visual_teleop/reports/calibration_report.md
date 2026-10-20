# Calibration report — 2026-10-14

**Nothing on the human side is calibrated yet.** `config.unmeasured()` lists the following placeholders:

- `cameras.T_W_C`
- `imu.R_I_H.{left,right}`
- `time_sync.camera_imu_offset_ms.{left,right}`
- `init.R_W_H_rotvec_deg`

Until each one is measured, `live_teleop` refuses Stage 2 and above (arm motion). Stage 1 (grippers only) does not need
any of them. Spec §44 forbids patching around a missing calibration with an offset.

| item | status | how to measure | note |
|---|---|---|---|
| IMU identity L/R | **unknown** | `imu_bringup --list`; tap one wrist and see which serial moves; write it to `imu.<side>.serial_number` | Board seen today: XIAO nRF52840 `4A65C2BC893657D2`. It is in no existing profile, and it disconnected during bring-up. Known serials: Teensy `20540160` (L), `20778070` (R), XIAO `D2FB7365FE48ED0F` (R, fusion profile). |
| `R_I_H` (IMU → human TCP axes) | placeholder (identity) | Hold the hand in three poses: H z pointing down, H y pointing left, then the reverse. `align_vectors` against gravity gives two axes; repeat to check. | H axes are defined to match the robot TCP: z = forearm → thumb/index midpoint, y = index → thumb. |
| `T_W_C` (fixed Orbbec → table W) | placeholder | Checkerboard flat on the table: PnP gives the table plane (+Z up) and the X/Y axes on the board. | The old `~/c8/hra_red/global_calib` no longer exists. |
| init orientation `R_W_H` at X_L / X_R | placeholder | Build a two-slot jig and photograph the hand in it. This fixes the IMU yaw ψ. | Init aligns yaw only. The residual tilt is checked against `fusion.init_tilt_warn_deg` = 10°; a larger tilt means a wrong mount or a wrong pose. |
| camera ↔ IMU offset | none | Record a 5 s wrist wave and run `imu/time_sync.estimate_side` (RGB-D palm rotation vs gyro). OK needs peak correlation ≥ 0.8 and sharpness ≥ 0.02 (provisional). PAUSE stops init. | Synthetic test: 23 ms recovered as 23.5 ms. Live `sync` in the hub is not wired yet (see validation report). |
| RGB-D handedness | **unverified** | Live: raise the left hand and check that the dashboard shows LEFT. | `cameras.rgbd_selfie_mirrored` = false. A wrong value drives the wrong arm. |
| table height in RB | unmeasured | Measure the jaw tip touching the table at three points. | The workspace z floor is 0.03 m until then. |
| reBot L/R ports | from memory | Do the one-jaw test (`/grip/right/45` should open only the right jaw) after every power cycle. | `LEFT_LOC=51642368`, `RIGHT_LOC=1257472` |
| gripper zero | per power cycle | Read the closed jaw value: raw ≈ 0, open ≈ −270. | Offset drifts between power cycles (memory). |

## Robot TCP convention — measured, fixes a wrong assumption

From the pyroki rebot_b601 FK at home (numerical Jacobian), consistent with the URDF:

- **j6 rotates about TCP z.** A 5° step gives body rotation (0, 0, 5)° with zero TCP translation.
- **The tool direction is +z.** The j5 axis lies 0.253 m behind the TCP along −z.
- **y is the jaw axis.** The URDF `{side}_tcp` is `gripper_link` + 0.07 x with rpy (0, 90°, 0), and the finger prismatic
  joints open along `gripper_link` ±y, which is TCP ±y.

The repo comment "RE: x = finger/tool direction" (`configs/ego_teleop/frames.yaml`) describes `gripper_link`, not the
IK TCP. My first draft of the inter-arm wrist proxy repeated that mistake and has been fixed (`safety/collision.py`).

## 2026-10-14 afternoon — XIAO Sense Plus bring-up

### Flash and runtime result

- Flashed `firmware/xiao_imu` to the XIAO nRF52840 **Sense Plus**.
- **Its USB serial changed after the flash:** `4A65C2BC893657D2` became `63ACF722D6DCE23F`. Use the new serial in configs.
- The board had been sending 0 bytes before the flash.

### Root cause of `WAI=0x00`

- The Seeed mbed core defines `TARGET_SEEED_XIAO_NRF52840_PLUS` for this board.
- The Seeed LSM6DS3 library only switches to `Wire1` (the internal IMU bus) for `..._SENSE` or `..._SENSE_PLUS`.
- So the library used the external `Wire` pins.

**Fix:** pass the missing macro as a build flag. The library is not edited.

```
arduino-cli compile -b Seeeduino:mbed:xiaonRF52840SensePlus \
  --build-property "build.extra_flags=-DTARGET_SEEED_XIAO_NRF52840_SENSE_PLUS" firmware/xiao_imu
```

### Firmware timestamp fix

`micros64()` counted any backward `micros()` step as a 32-bit wrap. This gave one +2^32 µs jump in 20 s. A wrap now
requires a drop of more than 2^31; a small backward step holds the previous value. Backup:
`xiao_imu.ino.bak_pre_wrapfix`.

### 60 s check after the fix

| check | result |
|---|---|
| device rate | 415.9 Hz |
| host rate | 422.8 Hz (XIAO clock runs +1.6 %; the device→host clock fit absorbs it) |
| timestamps | monotonic, dt p99 2.43 ms, max 2.45 ms |
| sequence loss | 0 |
| host parser | `TeensyImu` reads it unchanged (same protocol v1) |

### 20 s AHRS check

| quantity | value |
|---|---|
| \|a\| | 10.19 m/s² (+4 % scale; AHRS uses the measured g_rest) |
| gyro z bias | 0.057 rad/s (removed by the still-window bias) |
| yaw drift while still | 1.1 °/min |
| tilt | 6.2° |

### Temporary IMU layout (user, 10-14)

One wrist uses the XIAO `63ACF722D6DCE23F`; the other uses the Teensy `20540160`.

- **Which wrist each one goes on is NOT assigned yet.** Do not guess it.
- The Teensy and FisheyeCamLeft dropped off USB together. Both belong to the left wrist unit, so check its cable or hub.
