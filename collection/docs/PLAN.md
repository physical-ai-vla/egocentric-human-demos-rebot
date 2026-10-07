# HandUMI Egocentric Data Collector — audit, gap analysis, plan (2026-09-09)

Goal: collect `Hpilot → H120 → H240 …` human egocentric episodes (C922 head RGB as the *policy observation*, two HandUMI
wrist units: Arducam 160° fisheye + ICM42688P/Teensy IMU + Feetech jaw encoder) as **raw, model-independent** data, then
process offline to canonical TCP trajectories (`C_state`-compatible relative EE) and export to LeRobot for X-VLA.

Research question: H↑ ⇒ R*(H)↓ (H0+R150 vs H120+R150 first; then H120+R90, H240+R90/R60).

## 0. Decisions taken (so the code below is not a surprise)

| Topic | Decision | Why |
|---|---|---|
| Repo | New package `handumi_collector/` **inside `ego_collector`** (same py3.11 `.venv`), `ego_collector/` untouched | reuse `CameraCapture`, SE3 utils, QA, LeRobot v3 exporter; venv already has cv2 / PyAV / pyorbbecsdk / pyarrow |
| Deps added (uv pip, never `uv sync`) | `mcap`, `pyserial`, `PySide6`, `feetech-servo-sdk` | MCAP writer, Teensy CDC, desktop UI, Feetech jaw encoder |
| Head policy cam | C922 **640×480 @30**, MJPG → H.264 (VideoToolbox) mp4 | matches robot R150 observation; view gap minimised |
| Aux head sensor | Orbbec RGB-D **recorded raw if present**, never a V1 policy input | RGB / RGB-D ablations later |
| Wrist cams | Arducam fisheye 1080p30 → mp4, **pose estimation only** (not policy obs in V1) | O_human ≈ O_robot via C922 only |
| Raw container | `head.mp4`, `left_wrist.mp4`, `right_wrist.mp4`, (`head_depth/` if RGB-D), `sensors.mcap` (JSON-schema messages), `events.json`, `episode_meta.json` | raw never overwritten by processing; MCAP is Foxglove-inspectable |
| Timestamps | every sample: `host_monotonic_ns` (+ session `wall_ns` offset), `device_timestamp`, `sequence`; camera frames: `capture_ns`, `frame_index`, `capture_index` | sync on the head-frame timeline offline (30 Hz canonical) |
| Device identity | cameras matched by AVFoundation **name + ordinal** (ffmpeg listing order == OpenCV index order on macOS), serials by `/dev/cu.*` glob + Teensy USB serial | indices shift on replug; config, not hard-code |
| Crash safety | episode dir carries `.incomplete` until `episode_meta.json` is written last; discard = move to `_discarded/` | completed episodes survive a collector crash |
| Teensy protocol | binary framed packets (spec in `firmware/teensy_imu/PROTOCOL.md`), host parser + mock now, firmware M1 | no IMU code exists anywhere yet |
| Vendored code | `handumi_collector/vendor/` copies of handumi-sw `feetech/bus.py, gripper.py, telemetry.py` pieces, fisheye intrinsics, `local_delta` | handumi-sw rebot-ego work is **uncommitted** working-tree state |

## 1. Audit — what exists vs the spec

| Spec item | ego_collector | handumi-sw (rebot-ego, uncommitted) | robot-cockpit | Verdict |
|---|---|---|---|---|
| Head C922 RGB capture | `camera/capture.py` `CameraCapture` (threaded/poll, MJPG, monotonic_ns), ChArUco calib, live monitor | `cameras/preview.py` | 3× C922 via LeRobot OpenCVCamera | **reuse** `CameraCapture` |
| Aux RGB-D | `camera/orbbec_capture.py` (848×480, depth PNG16) | — | — | reuse as optional device |
| Wrist Arducam fisheye | only fisheye *intrinsics math* | `calibration/spatial.py` `calibrate_fisheye`, `tracking/apriltag.py` `CameraGeometry` (rectify) | — | **new device**, vendor fisheye calib |
| Multi-camera simultaneous record | ✗ (single cam; `record_global_aux.py` separate process) | ✗ | cockpit records 3 cams via LeRobot | **new** (`collector/recorder.py`) |
| Teensy / ICM42688P IMU | ✗ | ✗ | ✗ | **new** (protocol + parser + mock + firmware) |
| Feetech jaw encoder | ✗ (only −270..0 mapping) | `feetech/bus.py` `read_status_block`, `gripper.py` `FeetechGripperSampler` (100 Hz, midpoint ts, unwrap), `telemetry.py` | leader = FashionStar (different) | **vendor** sampler |
| MCAP writer | ✗ (parquet) | ✗ | `mcap_recorder.py` via rosbags/CDR, synthetic time | **new** (`mcap` lib, JSON schema, real host time) |
| Sync / 30 Hz timeline | per-frame ts only | `synchronization.py` nearest-sample rows, `SustainedHealthGate` | — | new `processing/sync.py`, copy health gate idea |
| SLAM → TCP | HaWoR/DROID import only (`hands3d/adapter.py`) | anti-SLAM (AprilTag bundles); `control_tcp.py` pivot calib | — | **new** `slam/` (M2), TCP calib from pivot/bundle |
| SE3 / quaternion utils | `tracking/transforms.py` (pose7 xyzw, qw≥0) | `tracking/transforms.py` `Pose` (xyzw) | `convert/rebot_ee.py` (xyzw, TCP = gripper_link +0.07 m x) | reuse ego_collector; conventions already aligned |
| Relative EE (`C_state` twin) | `delta_pose` (world-frame dxyz) | `dataset/eef_actions.py` `local_delta` = inv(T_t)·T_{t+h} (**this is C_state**) | `rebot_ee.relative_ee_state` | vendor `local_delta`; unit-test equality with cockpit FK version |
| Grip normalisation | `hands/grasp.py` clip((w−closed)/(open−closed)) | `feetech/calibration.py` | grip01 rule | reuse formula, config direction |
| QA | `qa/episode.py`, `qa/tracking.py` (jumps, jitter, dwell) | `eval/tracker_eval.py` gates | hw_event_in_episode flag | reuse + add IMU/SLAM discrepancy (M3) |
| LeRobot export | `scripts/build_e120_lerobot.py` (v3.0, 16-D EEF) | `convert_eef.py` (16 state / 14 delta) | 4090 converter (EE fields) | adapt to canonical schema (M5) |
| Human→robot mapper / replay | `scripts/retarget_to_rebot.py`, `ik_replay_rebot.py` (needs handumi-sw env) | `robots/kinematics.py` (pyroki IK), `rebot_b601.yaml` (`{side}_tcp` 70 mm) | `eval/rebot_fk.py` | reuse (M4) |
| Desktop UI | OpenCV HighGUI only | rich TUI | web (http.server) | **new** PySide6 |
| Tests | 59 synthetic tests (`uv run pytest -q` → use `.venv/bin/pytest`) | — | — | add `tests/handumi/` |

## 2. Raw episode layout (M0)

```
datasets/human_handumi_raw/<SESSION>/
├── session_meta.json                 operator, date, hardware/calibration versions, device listing, notes
├── counters.json                     per-order valid counts (rebuilt from episode_meta on start)
├── episode_000001/
│   ├── .incomplete                   (removed after episode_meta.json is written)
│   ├── head.mp4                      C922 640x480@30, H.264 (h264_videotoolbox → libx264 fallback)
│   ├── left_wrist.mp4 / right_wrist.mp4
│   ├── head_depth/  (optional)       uint16 PNG per frame + intrinsics.yaml (Orbbec)
│   ├── sensors.mcap                  JSON-schema channels, log_time = host_monotonic_ns
│   ├── events.json                   warnings/hardware events with host time (camera drop, IMU timeout, …)
│   └── episode_meta.json             order, instruction, frame counts per stream, durations, drops, status KEEP/DISCARD, quality (later)
└── _discarded/episode_0000NN/        moved, never deleted
```

MCAP topics: `/head/frame_meta`, `/left_wrist/frame_meta`, `/right_wrist/frame_meta`, `/left/imu`, `/right/imu`,
`/left/gripper`, `/right/gripper`, `/system/sync` (session wall↔monotonic anchor, per-device clock offsets), `/system/events`.
SLAM poses are **not** written by the collector (offline, M2) → `processing/` writes `slam/*.parquet` next to raw.

## 3. Milestones

- **M0 (now)**: devices (UVC cams, Orbbec optional, Teensy IMU parser, Feetech sampler, mocks), 3-cam preview, connectivity
  gate, order selector + counters, START/STOP/DISCARD/KEEP, raw MP4 + MCAP + meta, crash-safe episodes, tests
  (config, timeline, MCAP round-trip, episode integrity, episode manager). Runs fully with `--mock` (no hardware attached today).
- **M1**: Teensy firmware (ICM42688P SPI + INT1, 400 Hz, framed packets), live IMU/grip rates in UI, timestamp QA.
- **M2 (software done 2026-09-10, see POSE_V2.md)**: fiducial-free pose subsystem `handumi_collector/pose/` — Arducam+IMU VIO/VO backends (opencv_vo baseline, orbslam3/dpvo adapters), versioned camera/IMU/TCP calibration, VIO timestamp pipeline, HOME-return + static + IMU-consistency QA, canonical head-timeline table, relative TCP (= robot C_state), inspector/benchmark/offset tools, synthetic raw episodes. Hardware steps (intrinsics, T_camera_imu, offset, T_camera_tcp, Hpilot benchmark) pending.
- **M3**: IMU–SLAM rotation discrepancy gate, jumps/drops/timestamp checks, PASS/REVIEW/REJECT, episode inspector.
- **M4**: human→robot mapper + open-loop reBot replay on Hpilot (freeze mapper).
- **M5**: H120 collection, LeRobot/X-VLA exporter (`ΔTCP_L + grip_L, ΔTCP_R + grip_R`; optional pseudo-Δq artifact).

## 4. Hardware notes / open items
- Nothing HandUMI is attached today (3 C922 belong to the robot). M0 is validated with mocks + one real C922.
- Teensy packet format is ours to define → `firmware/teensy_imu/PROTOCOL.md`; firmware sketch in M1.
- TCP calibration (camera→pinch centre) needs the physical rig; use handumi-sw pivot calibration (`control_tcp.py`) in M2.
- Orbbec on macOS needs sudo for USB access (existing caveat).

## 5. M1 (software done 2026-09-09; hardware soak pending)

Hardware profiles (`--hardware`): `hardware_handumi_v1.yaml` = **production** (head C922 + L/R Arducam + L/R Teensy + L/R Feetech,
Orbbec optional aux), `hardware_dev_3c922.yaml` = the 3 robot C922 standing in for the cameras (dev only), `hardware_mock.yaml`.
The 3 C922 on this Mac are never the final sensor set.

- Firmware `firmware/teensy_imu/{teensy_imu.ino,config.h,README.md}`: INT1-driven sampling (no polling timing), 64-bit monotonic
  `micros()`, TX ring so USB never blocks sampling, 400 Hz, gyro ±2000 dps, accel ±8 g, 1 Hz status packet. PROTOCOL.md is authoritative.
- Host driver `devices/teensy_imu.py`: resync on garbage, CRC counting, uint32 seq wrap-aware loss ratio, device-timestamp wrap events,
  USB auto-reconnect (events `device_error` / `device_reconnect`), identity by USB serial (mismatch refuses to open),
  `ImuQuality.grade()` GREEN/YELLOW/RED from `collector.yaml imu_quality`.
- Gripper: vendored handumi-sw status-block parser + 4096 unwrap; raw / unwrapped / normalized (0 = closed, 1 = open) all stored;
  versioned calibration `configs/calibration/gripper_vNNN.yaml` (UI SET CLOSED / SET OPEN / SAVE), session + episode meta carry
  `gripper_calibration_version`; exporters transform to robot grip01 explicitly.
- Timestamp policy: device time (`device_timestamp_us`), host receive time (`host_receive_ns`), canonical session time
  (`host_monotonic_ns − session_start_monotonic_ns`, anchor in `/system/sync`) are all kept; IMU raw axes stored as-is,
  `T_camera_imu_{left,right}` belongs to calibration (M2), never to acquisition.
- UI: identity banner (LEFT/RIGHT IMU serial + port) before START, per-side IMU/GRIP live panels with quality colour.
- Tools: `python -m handumi_collector.tools.imu_monitor [--list|--seconds N --log CSV]`, `tools.gripper_monitor`,
  `tools.gen_test_stream` (deterministic fault injection for host tests).
- `slam/PoseEstimator` interface only (no SLAM until Arducam footage exists; ORB-SLAM3 vs DPVO decided by benchmark).

**M1 completion gate (needs hardware):** 10-min soak per side ~400 Hz, 0 systematic seq loss, no disconnect; gripper monotonic +
wrap OK; one 5-min episode with all streams → video/MCAP readable, monotonic timestamps, no NaN, frame_meta consistent.


## 6. M1.5 — pre-IMU production UI & recording (2026-09-10)

Entry point: `.venv/bin/python -m handumi_collector.collect [--dataset Hpilot|H120]` (defaults to `hardware_handumi_preimu.yaml`).
- Profiles: `hardware_handumi_preimu.yaml` (production now: head C922 + L/R Arducam + L/R Feetech, Orbbec optional, `imus: []` =
  "not installed", grey, never a blocker), `hardware_dev_current.yaml` (what is on the desk: 1 Arducam + real Feetech unit, C922 stand-in).
- Gripper identity by USB serial (`port_serial`), `tools.gripper_monitor --scan` finds bus adapters. Servo status error bits (0x01 voltage
  on the 12.2 V unit) no longer count as comm failure; they are telemetry (`error_bits`) + a UI flag.
- Session state machine: IDLE/READY → RECORDING → REVIEW → IDLE, plus RECORDING → ERROR_REVIEW on a required-device failure (`poll()`),
  KEEP_WITH_WARNING / DISCARD. `.complete` marker on KEEP. Gates: required devices, disk (`collector.yaml disk`), order, optional preflight.
- UI (`ui/app.py`): 3 previews (+ Orbbec line), workspace-guide overlay on head (preview only), undistort toggle when fisheye intrinsics
  exist, HEAD MOTION HIGH heuristic, per-hand grip gauge + 6 s sparkline + IMU "waiting for HW", task/order + suggestion, dataset
  counters, disk, preflight checklist, HOME READY countdown, REC live stats, REVIEW summary + head replay, HARDWARE CHECK (5 s test
  episode → `collector/integrity.validate_episode`), POSE and QA tabs.
- Tools: `tools.inspect_episode --episode EP [--validate] [--backend …]`, `tools.calibrate_fisheye` (plain checkerboard).
- Real-hardware validation done: 8 s + 6 s episodes with C922 + Arducam 1080p30 + C922 stand-in + the real Feetech gripper (100 Hz,
  telemetry) → video frames == frame_meta, monotonic, ~30 fps, integrity PASS. Not yet done: the 5 manual smoke episodes (needs the
  second Arducam, second gripper unit, both units' jaws moved by hand), gripper calibration (SET CLOSED/OPEN), Orbbec.


## 7. Design freeze — wrist observation (2026-09-10)

```
HEAD   human C922            = robot C922                       (shared policy input, 640x480)
WRIST  human Arducam fisheye RAW ──► VIO / TCP / C_state        (pose, raw kept)
                              └──► calibrated C922-like virtual pinhole (K_target ≈ robot wrist C922, 640x480, optional
                                    rotation to match mount tilt) ──► shared policy input with robot wrist C922
```
Rationale: putting a C922 on the HandUMI changes human motion (weight) and would confound the human-demo effect; matching the
*virtual* camera keeps the morphology light and R150 (3-view C922) reusable. Intrinsics/FOV and pure rotation offsets are corrected
exactly; translation offsets are not — keep the Arducam mount pose close to the robot wrist C922. Intrinsics/FOV, the target's own
distortion (`D_target`, so a rendered frame looks like a RAW C922 frame) and a pure rotation are matched exactly.

Implementation: `pose/virtual_view.py` (`VirtualPinholeView`, `fit_rotation`, `compose_rpy`), per-side target in
`configs/calibration/virtual_wrist_vNNN.yaml` (v001 = head C922 1080p calibration scaled to 4:3 640x480 — an ESTIMATE),
`tools.calibrate_c922` (robot wrist C922 at its real 640x480 mode → per-side K_target/D_target), `tools.calibrate_fisheye`
(Arducam, plain checkerboard), `tools.view_match_qa` (same-scene robot | raw | virtual panel, manual `--pick` correspondences,
grouped metrics E_view_all/center/edge/near/far + scale ratios, PARALLAX / FOV_MODEL / SCALE / MOUNT_MISMATCH flags, residual
`--fit-rpy`, full provenance), `tools.render_virtual_wrist` (derived mp4 + provenance; raw fisheye never deleted).

**Runbook: `docs/handumi_collector/CAMERA_FREEZE.md`** — robot C922 calibration → Arducam calibration → physical mount matching →
same-scene QA → `virtual_wrist_v002` freeze. Physical mount first; `rpy` only absorbs a small residual rotation (a fit larger than
`thresholds.mount_mismatch_rpy_deg` is flagged MOUNT_MISMATCH and marked not recommended). `E_near ≫ E_far` = parallax = mount problem.
