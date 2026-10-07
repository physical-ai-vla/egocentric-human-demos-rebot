# HandUMI collector (`handumi_collector`)

This is the raw recorder for the human (HandUMI) side of the reBot work. It does not depend on any model. It records:

- a head C922 RGB stream, which is the **shared human/robot policy view** at 640x480@30;
- one or two HandUMI wrist units, each with an Arducam 160° fisheye (1920x1080 MJPG), a Teensy 4.1 + ICM-42688-P IMU, and
  a Feetech STS3215 jaw encoder.

Output is mp4 per stream plus one `sensors.mcap` per episode. Pose estimation, retargeting and export are offline steps
and live elsewhere in this repo (`../pipeline`, `../hra`). A pose failure never loses a raw episode.

Source: `~/ego_collector` (branch `orbbec-rgbd`, HEAD `12db478`, plus uncommitted working tree as of 2026-10-07). See
`MANIFEST.md`.

## Install (uv)

```bash
cd collector
uv sync                      # python 3.11 (.python-version); pulls ../experimental in editable mode (see below)
uv run pytest tests/handumi -q
```

`handumi_collector` lazily imports `ego_collector.camera` (the UVC/Orbbec capture layer from the V0 collector). The
ChArUco scripts also import `ego_collector.camera.calibration`. That package lives in `../experimental/ego_collector`,
and `pyproject.toml` declares it as a path dependency (`handumi-rebot-experimental`). Without uv, put both folders on
the path: `PYTHONPATH=collector:experimental`.

Test baseline, run with the source venv on this copy (identical to the source tree):

- 12 known failures, all from before the copy: the mock gripper "open at rest" gate, the camera listing, and the m15
  production UI tests.
- `test_virtual_view.py::test_calibrate_pinhole_on_rendered_checkerboard` segfaults on macOS (Qt/OpenCV). Deselect it
  with `--deselect`.
- One test is skipped because it needs the recorded pilot data.

Firmware (`firmware/teensy_imu`) is built with arduino-cli and Teensyduino:
`arduino-cli compile|upload -b teensy:avr:teensy41:usb=serial,speed=600 -p <port> firmware/teensy_imu`.
The board must be flashed as USB type **Serial**. A RawHID build shows no `/dev/cu.usbmodem*` at all and looks like a
dead board. Wire protocol: `firmware/teensy_imu/PROTOCOL.md`. `firmware/xiao_imu` is a XIAO nRF52840 alternative
that speaks the same protocol. `firmware/bringup/` holds the SPI and wiring probes.

## Devices and hardware profiles

Profiles are `configs/handumi/hardware_<name>.yaml`, selected with `--hardware <name>`. Device identity is resolved at
start-up and never by port order:

| device | identity | notes |
|---|---|---|
| head C922 | `match_name: C922` | Autofocus is OFF and `focus-abs` is fixed, because autofocus changed fx by session (417/531/501) |
| wrist Arducams | product name (`FisheyeCamLeft` vs `Arducam 1080P Low Light`); `strict_identity: true` | Both units report the same USB serial `UC684`, so recording is refused unless exactly one device matches each name |
| Teensy IMUs | USB serial number (PJRC vid 0x16c0) in `imus[].serial_number` | Find serials with `python -m handumi_collector.tools.imu_monitor --list`. The ICM-42688-P has no 400 Hz ODR; the firmware runs it at 200 Hz |
| Feetech jaws | CH343 adapter USB serial + servo id (1 Mbaud) | Scan with `python -m handumi_collector.tools.gripper_monitor --scan`. Error bit 0x01 ("voltage", from a 12.2 V supply against the 12.0 V EEPROM limit) is telemetry, not a failure |

Main profiles:

- `handumi_v1`: production, bimanual. Head C922, L/R fisheye, L/R IMU, L/R jaw. RGB only.
- `handumi_v1_right`: right hand only. Used for HRA_red and HRA_A100.
- `handumi_preimu`: the default when `--hardware` is omitted; L/R cameras and jaws, no IMU.
- `handumi_rgbd`: adds an Orbbec head RGB-D, for the RGB-D teacher tools only.
- `dev_current`, `dev_3c922`: desk and dev setups.
- `mock`: no hardware.
- `temporal_sync_calib`: camera↔IMU offset takes.

The Orbbec cannot share a host with streaming UVC wrist cameras: it breaks or blocks them in either open order. Do not
add it to a production profile. Group USB devices by host controller (`ioreg -p IOUSB`), not by port: a USB-C dock
splits USB2 and USB3 devices across two root ports.

Calibration files are in `configs/calibration/`; see its README. All of them are versioned `_vNNN` files, and you must
never change one mid-session.

| file | what it holds |
|---|---|
| `fisheye_{left,right}_vNNN.yaml` | Wrist fisheye intrinsics, including `valid_radius_px` |
| `camera_imu_*`, `wrist_bundle_*_vNNN.yaml`, `kalibr_raw/` | Camera↔IMU extrinsic and time offset (Kalibr) |
| `gripper_vNNN.yaml` | Jaw ticks for closed/open, set with SET CLOSED/OPEN in the UI. Normalisation is circular because one jaw crosses the 4095→0 seam |
| `head_c922_intrinsics_*`, `head_mount_*` | Head camera |
| `camera_tcp/` | Camera→TCP |
| `robot_right_wrist_c922_v001.json` | The measured robot wrist C922, used for the C922-view rendering |
| `tcp_convention_v1.yaml` | TCP frame convention |

## Protocols / dataset modes

The dataset mode comes from `--dataset <mode>`, defined under `datasets` in `configs/handumi/tasks.yaml`. Each mode sets
the session prefix, the per-order target, and optionally its own orders, instruction, AUTO timing and QA switches.
Defaults are in `configs/handumi/collector.yaml`. A protocol overlay comes from `--protocol <name>`, which reads
`configs/handumi/<name>.yaml`.

| mode | hardware | what |
|---|---|---|
| `Hpilot` / `H120` | `handumi_v1` | Bimanual 3-cube stacking. Orders RBP…PBR; 50 / 20 per order. Instruction is "Stack the {bottom} cube on the bottom, …". AUTO: 20 s episode, 10 s reset |
| `HRL80` + `--protocol robot_like_v1` | `handumi_v1` | Robot-like stacking, 6 × 14. AUTO episode is 30 s. Shows a live panel of wrist angular velocity and acceleration against R150 p95 (0.996 rad/s, 4.413 rad/s²) and grasp counting with 0.35/0.65 hysteresis. A FAIL makes the preliminary QA REVIEW. Every episode gets `protocol: robot_like_v1`. See `docs/ROBOT_LIKE_PROTOCOL.md` |
| `HRA_red` | `handumi_v1_right` | Right hand only, "Approach to the red cube". 10 s episode, 5 s reset. Gripper QA is off and robot-like checks are advisory only |
| `HRA_A100` | `handumi_v1_right` | 100 approach takes from the robot-A-matched start view (see the cycle below) |
| `HRGBD_qa10`, `CALIB_bodytcp` | `handumi_rgbd` | RGB-D teacher-sensor pilot and body→TCP calibration takes. Not stacking. See `docs/RGBD_TEACHER_QA.md` |
| `Vpilot_vision_only` | any | IMU-less observability pilot. Never counted |

The `HRA_A100` cycle:

1. 2 s still at the taped origin pose.
2. REC starts, with a 2 s origin hold ("2, 1").
3. "준비" ("get ready"): 3 s to reach the robot-like start pose.
4. "시작" ("start"): 10 s approach.

`record_c922_view: true` also writes `right_wrist_c922.mp4`: the fisheye remapped to the measured robot C922, frame for
frame with `right_wrist.mp4`. The RECORD tab shows a ROBOT C922 VIEW tile, with a start reference image for matching
the start pose (see the note on `start_ref_*.png` in `MANIFEST.md`).

## Running `collect`

```bash
# production bimanual stacking
uv run python -m handumi_collector.collect --hardware handumi_v1 --dataset Hpilot
# robot-like protocol
uv run python -m handumi_collector.collect --hardware handumi_v1 --dataset HRL80 --protocol robot_like_v1
# right-hand approach (HRA)
uv run python -m handumi_collector.collect --hardware handumi_v1_right --dataset HRA_A100
# no hardware / headless smoke
uv run python -m handumi_collector.collector.main --mock --headless-smoke 3
```

Options: `--session DIR` resumes an existing session, and `--config-dir` points at another config set. Some device and
ChArUco tools need `sudo` for UVC control access on macOS.

UI flow:

1. HARDWARE CHECK.
2. Optional preflight checklist.
3. START enters WAITING_FOR_STILLNESS. Both IMUs must be still (gyro < 6 °/s and accel-norm std < 0.35 over 0.25 s)
   before REC.
4. REC, with a still hold inside the recording ("셋 둘 하나 고", "3, 2, 1, go"). The hold gives VIO its init window.
5. GO, then the task, then STOP.
6. REVIEW, then KEEP or DISCARD. DISCARD moves the episode to `_discarded/`; nothing is ever deleted.

**AUTO** (button or `A`) runs this loop hands-free with Korean voice cues (macOS `say`, voice Yuna). The loop
auto-KEEPs PASS and pauses on REJECT, and it balances orders by the least-collected one. Disk gate: YELLOW below
50 GB free, START blocked below 10 GB.

## Output layout

```
datasets/human_handumi_raw/<Dataset>/<Dataset>_<YYYYMMDD_HHMMSS>/
  session_meta.json  counters.json  _discarded/
  episode_NNNNNN/
    head.mp4  left_wrist.mp4  right_wrist.mp4  [right_wrist_c922.mp4]  [head_depth.mp4 + head_depth_depth/*.png]
    sensors.mcap      # /<stream>/frame_meta, /{left,right}/imu, /{left,right}/gripper, /system/sync, /system/events
    events.json       # rec_start, hold/go cues, stillness, moved_before_go, ...
    episode_meta.json # task/order/instruction, devices, calibration versions, quality verdict, protocol, robot_like_live
    .incomplete -> .complete
    derived/          # offline products only (pose, robot_like/live_summary.json, ...); raw is never modified
```

MCAP `log_time` is the host monotonic clock in ns. IMU samples also carry `device_timestamp_us`, and every rate gate
uses that device clock.

## QA

- **Live**: per-device colour lamps from `collector.yaml` `imu_quality` / `gripper_quality` (rate, loss, age),
  camera drop and timestamp-jump warnings, and a HEAD MOTION HIGH warning.
- **Per-episode verdict** (`collector/integrity.py`), shown in REVIEW, PASS / REVIEW / REJECT:
  - Raw stream integrity.
  - Jaw checks: a frozen reading, no jaw travel, or a reading outside the calibrated span is a REJECT. These checks are
    off for approach-only modes.
  - Robot-like verdict, where a protocol is active.
- **Offline**:

  ```bash
  uv run python -m handumi_collector.tools.inspect_episode --episode EP --validate   # integrity + timing/grip/event plots
  uv run python -m handumi_collector.tools.validate_session <imu_log_dir>           # IMU rate, jitter p99, seq drops, CRC, gravity, headroom
  uv run python -m handumi_collector.tools.motion_qa ...                             # motion statistics
  uv run python -m handumi_collector.tools.wrist_exposure_qa ...                     # wrist exposure
  bash scripts/robotlike_session_check.sh <session>                                  # robot_like_v1 offline reachability check
  ```

  `robotlike_session_check.sh` runs on the GPU cluster via Ray. It also needs the `~/c8` chain (`c8_phase3`,
  `c8_collision_v1`), which is **not** in this folder.
- **Operating rule**: collection gates only on raw integrity and task validity. VIO quality never blocks collection;
  pose and VIO problems are handled in post-processing.

## Other tools and scripts

- Calibration:
  - `tools.calibrate_fisheye`, `tools.calibrate_c922`, `tools.calibrate_table_frame`, `tools.temporal_sync` (camera↔IMU
    offset).
  - `scripts/charuco_calibrate.py`, `scripts/identify_board.py`, `scripts/generate_charuco.py`,
    `scripts/table_frame_ui.py`, `scripts/calibrate_table_frame_charuco.py`, `scripts/fit_table_frame.py`,
    `scripts/calib_take_qa.py`.
- Cameras:
  - `scripts/verify_physical_side.py`: cover a lens to confirm which wrist is which.
  - `scripts/exposure_sweep.py`, `scripts/wrist_brightness_sweep.py`, `scripts/wrist_image_quality.py`,
    `scripts/wrist_cam_diagnose.py`.
  - Runbook: `docs/CAMERA_FREEZE.md`.
- IMU: `tools.imu_monitor`, `tools.imu_logger`, `tools.imu_dual_smoke`, `tools.plot_imu`. Runbook:
  `docs/IMU_BRINGUP.md`, `docs/IMU_RING_POLICY.md`.
- Robot-like protocol: `scripts/robotlike_replay_live.py`, `scripts/robotlike_offline_check.py`,
  `scripts/robotlike_compare_groups.py`.
- Export: `scripts/umi_export_orbslam.py` writes an episode in UMI ORB-SLAM3 layout. The robot-like check uses it.
- Design docs: `docs/PLAN.md`, `docs/POSE_V2.md` (fiducial-free pose), `docs/TEMPORAL_SYNC.md`, `docs/RGBD_POSE.md`.

Run scripts from this folder (`uv run python scripts/<x>.py`), because some of them import `scripts.<module>`.
