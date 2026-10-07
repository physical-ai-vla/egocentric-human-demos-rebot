# ego_teleop — reBot Damiao + Aero Hand wearable teleoperation v1 (tag-free) — audit & plan (2026-09-10)

Spec: "reBot Damiao + Aero Hand Wearable Teleoperation System v1" + "Tag-Free Revision" + **architecture correction** (user, 2026-09-10).

## Agreed sensing layout (authoritative, 2026-09-10 correction)

| Device | Role | Control role |
|---|---|---|
| Head Logitech C922 | egocentric RGB observation for policy/dataset (`head_rgb`) | **none** — no IMU, no 6-DoF head tracking in V1, never drives the arm |
| Left wrist unit: Arducam fisheye + ICM42688P (rigid, own `camera_imu_left`) | wrist VIO → `left_wrist_pose` in its own VIO world | drives the left arm via relative SE(3) (when teleoperated) |
| Right wrist unit: same, own `camera_imu_right` | wrist VIO → `right_wrist_pose` | drives the right arm (v1 teleop: the Aero-carrying arm) |
| Orbbec RGB-D | optional auxiliary raw only | none |
| Human finger articulation → Aero 7D | **source not decided** (`finger_source: unverified`) | independent provider interface; never coupled to wrist tracking |
| AprilTag / ArUco / ChArUco / Quest / T265 / OptiTrack | not in the runtime path | (AprilTag tooling allowed only as an offline VIO ground-truth instrument) |

Each wrist unit is itself the pose sensor (unlike EgoMI/Aria, where a head device is the tracking origin). `WristPoseProvider` stays;
there is no head-pose provider and none must be added. Synchronized streams kept separate: `head_rgb`, `left_wrist_rgb`,
`right_wrist_rgb`, `left/right_imu`, `left/right_wrist_pose`, `finger_state`, reBot state/command, Aero state/command.

The wrist fisheye's first job is scene capture for VIO; whether it also sees the operator's fingers well enough for Aero retargeting
is unverified. `LiveHandTracker` refuses to start until `finger_source` is set after a physical FOV check (options: wrist fisheye,
a dedicated small camera, or the head C922 as a passive image consumer only).

## 0. Where it lives

`~/ego_collector/ego_teleop/` (same py3.11 `.venv`; registered in pyproject `packages`). New code is ONLY what the spec adds on
top of what already existed; everything else is imported:

| Spec section | Existing (reused as-is) | New in `ego_teleop/` |
|---|---|---|
| §1/§12 sensors (fisheye, IMU, timestamps) | `handumi_collector.devices` (`camera.py`, `teensy_imu.py`, `base.py` ring buffers), `firmware/teensy_imu/` (400 Hz, PROTOCOL.md) | — |
| §2/§4 tag-free VIO, tracking states | `handumi_collector.pose.estimator` (`PoseEstimator`, `PoseEstimate`, `TrackingState` INIT/TRACKING/DEGRADED/LOST), backends `mock / opencv_vo / orbslam3 / dpvo`, `pose.timing` (camera↔IMU clock fit + offset cross-correlation), `pose.imu` | `tracking/interfaces.py` (`WristPose`, `WristPoseProvider`, `TrackingHealth` OK/DEGRADED/LOST), `tracking/wrist_pose_provider.py` (`EstimatorWristPoseProvider` = any backend → T_W_H via T_H_C; `TrackingSupervisor` = age/confidence/jump/hysteresis policy — the only place health is decided) |
| §3 frames + calibration | `handumi_collector.pose.calibration` (versioned `<prefix>_vNNN.yaml`), `pose.se3` (T_A_B convention, pose7 xyzw) | `transforms/frames.py` (`HumanRobotFrameMapper`, **the only axis-swap location**; conjugation R'=M R Mᵀ so mirror mappings stay proper), `transforms/calibration.py` (`T_C_I`, `T_H_C`, `T_RE_AH`, frame map; explicit `missing`, `require()`) |
| §5–§9 relative SE(3), clutch, drift | `pose.se3.local_delta` (= robot `C_state` definition) | `retarget/arm_relative_se3.py` (`RelativeSE3Retargeter`: engage/clutch/release re-anchor, LOST ⇒ hold, DEGRADED ⇒ speed factor; VIO never reset) |
| §7 arm control | robot-cockpit `robot_service.py` (`/observe`, `/execute_step`, `/estop`; 14-dim, arm **rad** in / gripper raw), handumi-sw `robots/kinematics.py` (pyroki IK, `{side}_tcp`), `rebot_arm_cfg.py` | `robot/rebot_client.py` (`HttpRebotClient`: **the single deg↔rad boundary**, other arm held, grippers passthrough, IK residual + joint-step guards; `PyrokiIkSolver`; `MockRebotController`), `robot/safety.py` (workspace box clamp/hold, lin/ang velocity + acceleration limiter, robot-obs watchdog) |
| §8/§10 finger tracking + calibration (camera source **undecided**) | `ego_collector.hands.mediapipe_tracker.HandLandmarker`, `ego_collector.hands3d.aero` (`human_semantic7`, `semantic7_normalized`, compact→16→actuations with SDK tendon coefficients, verified against SDK limit table) | `retarget/hand_features.py` (wrist-relative palm-normalized landmarks → u7; `HandTrackingSupervisor` OK/HOLD/LOST; `HandRectifier` fisheye→pinhole for the landmark model; `LiveHandTracker`), `retarget/aero_retarget.py` (`AeroRetargeter`: OPEN/CLOSED/PINCH calibration, per-channel min/max/gain/offset/sign/deadband/LPF/rate/clamp, `hold()`/`relaxed()`, JSON persistence) |
| **Aero branch: head RGB-D hand pose (ActiveUMI+RGB-D spec, 2026-09-11)** | `pyorbbecsdk` (already in the venv), `ego_collector.hands.mediapipe_tracker.HandLandmarker` (+ new `detect_all` raw detections), `ego_collector.hands3d.aero.AeroHandFK`, upstream `~/aero-hand/aero-hand-open` conventions, `dex_retargeting==0.5.0` | `hand3d/aero_mocap.py` (official 21→25 + palm-local, transcribed and pinned), `hand3d/depth.py` (deprojection, robust per-landmark depth, hand-level outlier rejection), `hand3d/head_camera.py` (`head_rgbd_vNNN.yaml`, `OrbbecHeadCamera`, `RecordedRgbdSource`), `hand3d/identity.py`, `hand3d/providers.py` (B0 monocular / B1 RGB-D, same output type), `hand3d/supervisor.py` (INITIALIZING/OK/DEGRADED/LOST, hold-on-loss), `retarget/aero_backends.py` (`DexPilotAeroRetargeter` = official node minus ROS, `Semantic7DAeroRetargeter`, `AeroCommandLimiter`), `tools/a1_hand3d.py`, `tools/a2_virtual_aero.py`, `configs/ego_teleop/head_rgbd.yaml`. Docs: `AERO_A0_AUDIT.md`, `AERO_RGBD_RUNBOOK.md`. The head camera gains a HAND role only — never an arm one.
| §9/§17 Aero SDK | `~/aero-hand/aero-hand-open/sdk` (`aero_open_sdk.AeroHand`: `set_joint_positions(7|16)`, `get_actuations/currents/temperatures`, `send_homing`) | `robot/aero_client.py` (`SdkAeroClient` thin wrapper, telemetry cadence; `MockAeroClient`) |
| §11–§14, §19 clock, coordinator, state machine | — | `robot/coordinator.py` (`TeleopStateMachine` INIT→…→TELEOP/CLUTCHED/PAUSED/ESTOP with readiness guards; `TeleopCoordinator.tick()` = one 30 Hz `TeleopCommand` with both execution timestamps) |
| §15–§16 raw episode | `handumi_collector.collector` (MP4 + MCAP raw recorder, `.incomplete` crash safety), `pose.sync` (30 Hz resampling for `processed/`) | `recorder/episode_logger.py` (`raw/vio_pose, hand_landmarks, events`, `robot/rebot_state, rebot_command, aero_state, aero_command`, `calibration/*.json`, `metadata.json`; refuses to overwrite) |
| §18 configs | `configs/handumi/*.yaml` (hardware profiles) | `configs/ego_teleop/{teleop,frames,rebot,aero}.yaml`, `ego_teleop/config.py` |
| §21 quality gates | `pose.yaml` gates (home-return, jumps, IMU consistency, valid ratio), `handumi_collector` QA | (M5) teleop-specific gates on `rebot_command` / `aero_state` |

Tests: `tests/teleop/` (all synthetic, no hardware): frames, relative SE(3) + clutch, tracking dropout (MockBackend through the
real provider), hand features + Aero retarget filters/calibration, safety limiter, HTTP client unit boundary, state machine +
full coordinator loop (arm follows / clutch freezes arm not hand / LOST holds / hand LOST relaxes / estop), episode logger layout,
official Aero convention pinning (`test_aero_official_conventions.py`), head RGB-D hand branch (`test_hand3d.py`),
retargeting backends + command limiter (`test_aero_backends.py`).
Run: `.venv/bin/pytest -q tests/teleop`.

Integration point for live sensing: `handumi_collector.pose.live.LivePoseRunner` (threaded online backend over the device buffers,
being built in parallel) → drain `runner.buffer` → `EstimatorWristPoseProvider.ingest_estimate()`; the provider does not own threads.

## 1. Decisions

- **Not a new repo.** Sensors, recorder, VIO interface, SE(3), Aero math already exist here; the spec's `sensors/`, `recorder/`,
  `calibration/` directories map onto `handumi_collector` and are not duplicated. Spec's `ego_teleop/` layout is kept for the new parts.
- **Invariant A enforced structurally:** `WristPose` can only be produced from a `PoseEstimate` that has a camera pose; there is no
  IMU-only path to XYZ. Age-based LOST reports the *last* pose with zero velocity.
- **Human relative motion is logged raw** (`rebot_command.human_d*` = inv(T_H0)·T_H, robot-agnostic) next to the mapped, safety-
  limited robot command (`arm_d*`) — re-retargeting offline needs only the raw column + the frame/scale config recorded per episode.
- **Clutch default:** arm frozen, hand live (`coordinator.clutch_freezes_hand: false`). Release re-anchors both T_H0 and T_R0 (zero jump).
- **Canonical hand order** everywhere: `thumb_cmc_abduction, thumb_cmc_flexion, thumb_tendon, index, middle, ring, pinky`
  (`AERO_CHANNELS` == `hands3d.aero.COMPACT_NAMES`). The SDK's own joints→actuations model does the thumb coupling (we send compact-7).
- **Units:** the ONLY angle conversion is `HttpRebotClient.observe()` (deg→rad); `/execute_step` already takes radians for the arm.
- Safety defaults are deliberately low (0.10 m/s, 30 °/s, workspace box must be verified on the arm before `ARM`).

## 1b. Open design item: left/right VIO worlds → one bimanual frame

Two independent VIO worlds are fine for everything built so far: teleop anchors each arm separately (`T_H0`, `T_R0` per side) and the
canonical per-hand action inv(T_t)·T_{t+k} (`pose.relative`, robot `C_state`) needs no shared L/R frame. A common frame is only needed
when a model or QA wants the *relative pose between the two hands* (bimanual coordination, hand-over, object-centric tasks).
Options, none confirmed yet:
1. **HOME jig alignment (recommended for v1):** both wrist units start each episode in a fixed two-slot jig with a measured
   `T_leftHome_rightHome`; the per-episode still-window (already detected by `pose.imu.still_mask` for the HOME-return QA) fixes
   `T_Wl_Wr = T_Wl_Hl0 · T_Hl0_Hr0 · inv(T_Wr_Hr0)`. Drift between the two worlds accumulates within the episode and is *measured*
   by the HOME-return residual at the end (already a gate) — no tags, no extra hardware, one printed jig.
2. Cross-view co-visibility (each wrist camera sees the other unit / shared scene features) — a joint map, i.e. a bimanual SLAM
   backend; robust but heavy; defer.
3. Head C922 seeing both wrist units — would give the head a tracking role, which the agreed layout forbids; rejected.
Decision needed before the first bimanual dataset, not before single-arm teleop.

## 2. Risks / what must be true before the robot moves

1. **Online metric VIO is the critical path (M1).** Existing backends: `orbslam3` = visual-inertial + metric but **offline external
   process**; `dpvo`/`opencv_vo` = online but **non-metric**. Live teleop needs online + metric ⇒ candidates: ORB-SLAM3 mono-inertial
   via a pybind wrapper on a Linux host, OpenVINS (ROS-free build) with a small bridge, or the offline ORB-SLAM3 run used first for
   replay-only validation (M4/M5 do not need online VIO). `WristPoseProvider` isolates this choice from everything downstream.
   Ground truth for M1 validation: run the *existing* AprilTag tooling (`ego_collector.tracking`) purely as an offline measurement
   instrument on the same recordings — it is not in the runtime path and not a dependency of `ego_teleop`.
2. **Camera–IMU extrinsic + time offset** (`camera_imu_<side>_vNNN.yaml`, `pose.timing.estimate_camera_imu_offset`) — mandatory,
   `TeleopCalibration.require()` blocks `CALIBRATED` without them. The Arducam↔Teensy mount must be rigid (FastUMI lesson).
3. **One camera for both branches:** confirm the fisheye sees enough environment texture for VIO *and* the fingers; if fingers leave
   the frame, add a second small camera for the hand branch only (do not add cameras pre-emptively).
4. **IK lives in the handumi-sw venv** (pyroki/jax not installed here). Options: run the arm branch from that interpreter, or wrap
   `PyrokiIkSolver` in a tiny local HTTP shim. Decide at M3.
5. **Aero on macOS:** SDK port autodetect is Linux-only ⇒ `aero.yaml port:` must be set. Bring-up (home/open/close/pinch/cylinder via
   SDK alone, telemetry logged) is required before any hand tracking is connected (M2 gate).
6. `robot_service` is joint-space and single-step; Cartesian control is client-side (IK → joint-limit → `/execute_step`). Its E-STOP
   semantics are kept; `HttpRebotClient.estop()` calls `/estop`.

## 3. Milestones (spec §15 revised) and gates

| M | Scope | Gate | Status |
|---|---|---|---|
| M0 | camera + IMU raw recorder, timestamps, calibration loading | `handumi_collector` M0/M1 gates (10-min IMU soak, MCAP/MP4 integrity) | software done (handumi_collector); hardware soak pending arrival |
| M1 | tag-free VIO → metric wrist SE(3); `EstimatorWristPoseProvider` on the chosen backend (**software readiness DONE; sensor validation NOT STARTED**) | return-to-origin < 10 mm / 3° over a 30 s manipulation window; LOST recovery; offline tag ground truth agrees on short windows | provider + supervisor done + tested with `MockBackend`; **backend choice open (risk 1)** |
| M2 | Aero hand only: SDK bring-up → MediaPipe → u7 → `AeroRetargeter` | open / close / pinch / individual fingers reliable; telemetry logged | retargeter + client done + tested; needs hand + fisheye |
| M3 | reBot relative wrist teleop with clutch (no hand) | reach targets at ≤ 0.10 m/s, no workspace/IK refusals in a 5-min session | retargeter, safety, client, coordinator done + tested with mocks; needs IK venv decision |
| M4 | arm + hand under one clock | reach-without-grasp, reach+pinch, pick, place | coordinator supports it; mocks pass |
| M5 | recording + replay + export | live vs replay same qualitative sequence; existing relative-motion replay unchanged | logger done; replay/export not started |
| M6 | swap VIO backend | nothing after `WristPoseProvider` changes | by construction |

## 4. M1 — online metric wrist VIO (right wrist first) — plan of record (2026-09-10)

Scope fence (user): M1 only. No new teleop features, no reBot motion, no L/R common-frame work, `finger_source` stays unverified.

| Step | What | Where | Gate |
|---|---|---|---|
| M1-0 calibration | **offline-primary chain** (policy fixed 2026-09-10): fisheye intrinsics → IMU noise (Allan) → **offline camera–IMU spatial + temporal (Kalibr, checkerboard/circle-grid)** → OpenVINS `mode: calibration` refine/validate across ≥ 2 takes → versioned bundle | `ego_teleop/calibration/bundle.py` → `configs/calibration/wrist_bundle_<side>_vNNN.yaml` (+ `export_pose_files` for the pose pipeline) | RMS < 0.5 px; between-take spread ≤ 1° / 5 mm / 2 ms; `require_production()` refuses an `openvins_online`-only extrinsic |
| M1-1 recorded → OpenVINS | `python -m ego_teleop.tools.m1_vio EPISODE --side right --backend openvins` on recorded MCAP/MP4 episodes (same data, only VIO config changes between runs) | `ego_teleop/tools/m1_vio.py` → `derived/pose_openvins/{right_camera_pose, m1_report_right}.json` | M1 protocol below |
| M1-2 live OpenVINS | same `OpenVinsBackend` under `handumi_collector.pose.live.LivePoseRunner` (needs the full-rate IMU tap, not the latest-only tap) | live runner (other session) + `ingest_estimate()` | ≥ 25 Hz pose output, no queue growth over 10 min |
| M1-3 WristPoseProvider | `EstimatorWristPoseProvider(OpenVinsBackend, T_H_C=...)` → `TrackingSupervisor` → (later) `RelativeSE3Retargeter` — nothing downstream changes | `ego_teleop/tracking/` | dropout tests + live health transitions |

M1 protocol (recorded, one take per item, right wrist): static 30 s; ±10 cm X, Y, Z moves with return; ~90° wrist rotations
(pronation, flexion); return-to-start; deliberate fast motion / blur; recovery after LOST. Metrics (`ego_teleop.tracking.m1_eval`
+ the pose package QA): static drift (mm, deg), return-to-start (mm, deg), valid ratio, lost runs / longest / time-to-recover,
and against ground truth when available: ATE, 1 s relative pose error, **scale ratio** (must be 1.00 ± 0.10 → metric). Real
recordings have no ground truth: use tape-measured 10 cm moves (expect 100 ± 10 mm displacement) and the return-to-start error.

Backend decision: **M1-A OpenVINS** (MSCKF, mono+IMU, explicit `equidistant` fisheye model, online; GPL-3.0 → built OUTSIDE the
repo in `~/vio/open_vins`, reached via the tiny pybind11 bridge `~/vio/ov_bridge` — `ego_teleop` only imports `ov_bridge` at runtime,
no GPL code is vendored). **M1-B Basalt** (BSD-3, double-sphere fisheye, strong camera–IMU calibration incl. time offset, macOS arm64
supported) is the fallback and its calibration tooling may be used for M1-0 regardless. VINS-Fusion rejected for mono 160° fisheye.
Only one backend is integrated at a time.

Conventions fixed in `ego_teleop/tracking/backends/openvins.py`: OpenVINS `T_imu_cam` = inv(our `T_camera_imu`); `distortion_model:
equidistant` for Kannala-Brandt; IMU already on the camera clock (pipeline offset) so `timeshift_cam_imu` starts at 0 and
`calib_cam_timeoffset: true` refines it; state → `T_world_camera = T_G_I · T_I_C` (current extrinsic estimate); LOST/DEGRADED from
feature counts + finite check (OpenVINS has no explicit lost flag). ICM42688P noise values are datasheet×4 placeholders until Allan.

Status 2026-09-10 (evening): OpenVINS + `ov_bridge` **built and running on this Mac** (`~/vio/ov_bridge/build.sh`; recipe + gotchas in
`~/vio/ov_bridge/README.md`: cmake-4 policy flag, brew `opencv@4` forced over OpenCV 5, `-include cassert`, OpenCV-dialect YAML).
Adapter + config writer + M1 metrics + CLI unit-tested (55 tests). Real-filter smoke test on the synthetic episode
(`handumi_collector.tools.synth_episode`, pinhole/radtan path, IMU simulated in the camera frame):

| run | init | post-init valid | scale ratio | RPE 1 s | ATE | M1 |
|---|---|---|---|---|---|---|
| 6 s synthetic, full res | 1.67 s | 1.00 | **1.018** | 12.7 mm / 0.16° | 26 mm | PASS |
| 6 s synthetic, downscale 2 | 1.67 s | 1.00 | 0.909 | 20.1 mm / 0.40° | 28 mm | PASS (scale at the edge) |
| 12 s synthetic (either res) | 1.9 s | 0.05 | — | — | — | FAIL: the tracker finds ≤ 2 features in this render (sparse dot scene) — a synthetic-scene limitation, not tuned further |

Reading: metric scale and short-window accuracy are right on the data OpenVINS can track; the synthetic dot renderer is too poor a
texture source to say more. **Nothing has run on real Arducam + ICM42688P data yet** (hardware not attached); M1-0 calibration and
the recorded protocol takes are the next real evidence. Note the pose package's episode QA reports REJECT for these runs because it
counts the VIO initialization window as invalid and expects a valid HOME-start pose; the M1 report separates `init_frames`
(time_to_first_valid) from tracking loss. Whether the QA policy should do the same is a question for the pose pipeline owner.

### M1-0 calibration path (locked 2026-09-10, second review)
Offline calibration is primary; OpenVINS online calibration only refines and sanity-checks it (wrist motion is far more aggressive
than a head camera, so a camera–IMU timestamp error degrades VIO fast). Runtime stays completely tag-free and so does calibration:
checkerboard or asymmetric circle grid, never AprilGrid — no fiducial dependency exists anywhere in `ego_teleop`.

| piece | where | notes |
|---|---|---|
| Calibration bundle | `ego_teleop/calibration/bundle.py` | one versioned file per side: camera model + K/D/size, four IMU noise terms, `T_camera_imu`, `time_offset_ms`, per-field provenance, `hardware_hash` over unit_id/mount/serials/resolution/fps/exposure; `missing`, `require_production()`, `export_pose_files()` (keeps `SideCalibration` working) |
| IMU noise | `ego_teleop/tools/imu_allan.py` | overlapping Allan deviation from an `imu_monitor --log` CSV → 4 Kalibr terms → bundle. Validated on synthetic noise: density within 1 %, random walk ~25 % on a 1 h log |
| Offline camera–IMU | `ego_teleop/tools/kalibr_export.py` / `kalibr_import.py` | episode → ROS1 bag (one clock, full-rate IMU, gray images) + tag-free `target.yaml`/`imu.yaml`; importer pins the conventions (Kalibr `T_cam_imu` == our `T_camera_imu`, `time_offset_ms = -1000·timeshift_cam_imu`, `pinhole+equidistant` → `kannala_brandt`) and marks provenance `kalibr` |
| OpenVINS modes | `openvins.py MODE_OVERRIDES`, `m1_vio --mode` | `production` = all calibration flags off; `calibration` = extrinsic + time offset optimised and traced per frame → `derived/pose_openvins/calib_trace_<side>.parquet` |
| Convergence across takes | `ego_teleop/tools/calib_converge.py` | per-take converged tail + between-take spread, gates 1° / 5 mm / 2 ms; **a single take never passes**; `--write-bundle` records `openvins_online` provenance which is not production-ready on its own |

### M1 addenda (2026-09-10, after the user's review)
- The synthetic 1.018 scale / 12.7 mm RPE proves integration (calibration → convention → adapter → metric SE(3)), **not** real wrist
  VIO performance; nothing about Arducam+ICM42688P drift, scale or blur robustness is known yet.
- **Hardware status: no Arducam and no Teensy attached to this Mac** (only the C922 and built-in cameras) → M1-0/M1-1 cannot start
  until the right wrist unit arrives. Everything that does not need hardware is in place; see `M1_RUNBOOK.md` for the exact steps.
- VIO initialization is a lifecycle state, not a QA exception: `TeleopState` now has `VIO_INITIALIZING → VIO_STABLE`
  (`ego_teleop.tracking.vio_stable.VioStableGate`: ≥ 2 s TRACKING_OK while still) and HOMING/anchoring/robot readiness are gated on
  it. The HOME QA in the pose package is untouched — HOME simply starts after the gate.
- The sparse 12 s synthetic scene is kept as a **low-feature regression test** (`tests/teleop/test_lowfeature_regression.py`, marker
  `slow`, real OpenVINS): features collapse → DEGRADED/LOST → supervisor LOST → retargeter HOLD, no runaway. Passes (4.7 s).
- **Full-rate IMU tap** for live VIO: `ego_teleop.tracking.imu_tap` (`install_imu_tap` fan-out on the device buffer without touching
  handumi_collector, `LiveImuClock` online device→host mapping, `LiveVioFeeder`). The 30 Hz synchronized stream is never the VIO
  IMU source. This is the M1-2 prerequisite the earlier `LivePoseRunner` latest-only tap lacked.
- Protocol metrics: `configs/ego_teleop/m1_protocol.yaml` (S0–S9 with physical displacements/angles + gates) and per-segment scale
  from known physical displacement in `m1_eval.segment_metrics`; `m1_vio --segments`.

## 5. Next actions (in order)
1. **Blocked on hardware — no further code until the right wrist unit is attached.** Then follow `docs/ego_teleop/M1_RUNBOOK.md` in
   this order: **M1-0a device smoke test** (imu_monitor soak + inspect_episode --validate; fixed FPS + manual exposure set first) →
   M1-0b intrinsics → M1-0c Allan → M1-0d **offline Kalibr** camera–IMU → M1-0e two `--mode calibration` takes + `calib_converge` →
   `require_production()` PASS → M1-1 protocol take. The first hand-held OpenVINS run is a **smoke test only** ("does it ingest real
   sensor bytes?"), never an architecture verdict — that verdict comes from the M1-1 take on a production-ready bundle.
2. **M1-2 dry run before any robot motion:** human wrist → OpenVINS → ΔSE(3) → *virtual* reBot target in a 3-D view (clutch, LOST
   hold, workspace clamp, IK feasibility). Low-speed real reBot teleop only after that.
3. Status line to keep honest: **M1 software readiness = DONE** (78 tests, calibration provenance enforcement, full-rate IMU path,
   OpenVINS bridge, failure regression); **M1 sensor validation = NOT STARTED** (no Arducam/Teensy attached as of 2026-09-10).
3. Aero SDK bring-up on the built hand (M2 gate), then `calibrate_open/closed/pinch` and save `aero_retarget.json`.
4. `ui/teleop_monitor.py` (PySide6, reuse `handumi_collector.ui`) + the 30 Hz runner that wires sensors → provider → coordinator → logger.
