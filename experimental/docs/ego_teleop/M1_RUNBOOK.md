# M1 runbook — right wrist unit, real hardware (M1-0 calibration → M1-1 recorded OpenVINS)

Scope: **right wrist only**. No robot, no left wrist, no Basalt, no common frame, no Aero; `finger_source` stays `unverified`;
the head C922 stays observation-only. Hardware needed: right Arducam fisheye + Teensy/ICM42688P **rigidly mounted** (the extrinsic
must not change between calibration and recording), a **checkerboard or asymmetric circle grid** (tag-free by design — no AprilGrid),
and a two-stop fixture or ruler for the known 20 cm / 15 cm displacements.

## Order of operations (corrected 2026-09-10, second review)

```
hardware arrives
  → M1-0a device smoke test        camera frames? 400 Hz IMU? monotonic timestamps? dropped samples?
  → M1-0b fisheye intrinsics
  → M1-0c IMU noise (Allan)
  → M1-0d offline Kalibr camera–IMU spatial + temporal      <- PRIMARY
  → M1-0e OpenVINS calibration-mode refinement, >= 2 excitation takes
  → require_production() PASS
  → M1-1 real OpenVINS validation (protocol take)
  → architecture verdict: GO / TUNE / CHANGE BACKEND
```

The first hand-held OpenVINS run answers exactly one question — **does OpenVINS ingest real sensor bytes and emit
something?** It is a smoke test. It must NOT be read as an architecture verdict: without a real `T_camera_imu`, a real
temporal offset and real IMU noise, its trajectory says nothing about achievable accuracy. The first meaningful trajectory is
the M1-1 take that ran on a bundle which passed `require_production()`.

## Camera mode: fix it before calibrating

Calibrate and record with **fixed FPS and manual exposure**. Auto-exposure changes the effective shutter with the scene, which
changes motion blur and the effective camera–IMU timing — a calibration taken at 60 fps / short exposure does not transfer to
30 fps / auto-exposure. `handumi_collector` pins resolution/FPS/FOURCC via the camera config, but exposure is a UVC control it
does not touch; set it out-of-band, e.g. with the existing binary:

```
~/robot-cockpit/bin/uvc-util -I <index> -s auto-exposure-mode=1          # manual
~/robot-cockpit/bin/uvc-util -I <index> -s exposure-time-abs=<value>     # short enough to limit blur at speed
~/robot-cockpit/bin/uvc-util -I <index> -c                               # list controls / confirm
```

Record the values you used in the bundle's `hardware` block (`fps`, `exposure`, plus `unit_id`, `mount_revision`, serials): they
are part of `hardware_hash`, so changing the camera mode later visibly invalidates the calibration instead of silently degrading it.

## Calibration policy (fixed 2026-09-10)

```
fisheye intrinsics  →  IMU noise (Allan)  →  OFFLINE camera–IMU spatial + temporal (Kalibr)  →  OpenVINS online refine/validate  →  versioned bundle
```

Offline calibration is **primary**. OpenVINS `mode: calibration` exists to *refine and sanity-check* a reasonable calibration, never
to invent an unknown camera–IMU transform: `WristCalibrationBundle.require_production()` refuses a bundle whose `T_camera_imu`
provenance is `openvins_online` alone. Wrist motion is far more aggressive than a head camera, so a camera–IMU timestamp error
degrades VIO quickly — that is why the temporal offset is calibrated offline and only then refined.

Everything lands in one versioned file, `configs/calibration/wrist_bundle_right_vNNN.yaml`, tied to the physical unit
(`hardware.unit_id`, `mount_revision`, camera/IMU serials, resolution/fps/exposure → `hardware_hash`), with per-field provenance
(source, tool, date, runs). `bundle.export_pose_files()` writes the per-kind files the pose pipeline already reads
(`fisheye_right_vNNN`, `camera_imu_right_vNNN`), so nothing downstream changes.

## M1-0, step by step

**0. Device smoke test (M1-0a)** — no calibration, no VIO; existing tools cover it:
```
python -m handumi_collector.tools.imu_monitor --hardware handumi_v1 --list                 # identity: serial <-> side
python -m handumi_collector.tools.imu_monitor --hardware handumi_v1 --seconds 600           # 10 min soak; exits 0 only if GREEN
# record a short episode in the collector UI, then:
python -m handumi_collector.tools.inspect_episode --episode EP --validate                   # per-stream fps/gaps + integrity
```
Gates: wrist camera delivers frames at the configured fixed FPS with no long gaps; IMU sustains ~400 Hz with no systematic
sequence loss and no disconnect over 10 minutes; device and host timestamps are monotonic; dropped-sample and gap counts are
recorded. Fix anything here before touching calibration — every later number inherits these timestamps.

**1. Fisheye intrinsics** (Kannala-Brandt, full 1920×1080). Either the in-repo tool or Kalibr:
```
python -m handumi_collector.tools.calibrate_fisheye --side right --index <cam> --cols 9 --rows 6 --square-mm 25 --views 30
```
Gate: RMS < 0.5 px, views covering the rim of the 160° field. On a 160° lens an asymmetric circle grid often localises better at the
rim than a checkerboard — look at the images and pick; the choice affects no code.

**2. IMU noise (Allan variance, M1-0c)** — long, dead-still, thermally settled log:
```
python -m handumi_collector.tools.imu_monitor --hardware handumi_v1 --seconds 7200 --log imu_right.csv
python -m ego_teleop.tools.imu_allan imu_right.csv --side right --write-bundle
```
Writes the four Kalibr noise terms into the bundle (`provenance: allan_variance`). The estimator was validated on synthetic noise
(noise density within 1 %, random walk within ~25 % on a 1 h log). Shorter than 30 min → warns; random walk is then unreliable.

**3. Offline camera–IMU spatial + temporal (M1-0d, PRIMARY)**. Record a calibration take with rich rotation about all three axes plus
translation, target filling the frame, then export a Kalibr bag and calibrate on a machine with Kalibr (docker is fine):
```
python -m ego_teleop.tools.kalibr_export EPISODE --side right --out calib/right1.bag --target checkerboard
kalibr_calibrate_cameras    --bag right1.bag --topics /cam0/image_raw --models pinhole-equi --target target.yaml
kalibr_calibrate_imu_camera --bag right1.bag --cam camchain-right1.yaml --imu imu.yaml --target target.yaml
python -m ego_teleop.tools.kalibr_import camchain-imucam-right1.yaml --side right --write-bundle --export-pose-files \
       --unit-id wristR-01 --mount-revision v1 --camera-serial <arducam> --imu-serial <teensy>
```
The bag carries **one clock** (IMU device time mapped onto host time with the pipeline's robust fit) and the **full-rate** IMU, so
Kalibr's `timeshift_cam_imu` is the real residual offset. Conventions handled by the importer and covered by tests:
Kalibr `T_cam_imu` == our `T_camera_imu` (no inversion); `time_offset_ms` (camera − imu) = −1000 × `timeshift_cam_imu`;
`pinhole` + `equidistant` → `kannala_brandt`.

**4. OpenVINS refinement / convergence validation (M1-0e)** — at least **two** excitation takes, never one:
```
python -m ego_teleop.tools.m1_vio TAKE1 --side right --backend openvins --mode calibration
python -m ego_teleop.tools.m1_vio TAKE2 --side right --backend openvins --mode calibration
python -m ego_teleop.tools.calib_converge TAKE1 TAKE2 --side right
```
`--mode calibration` turns on `calib_cam_extrinsics` + `calib_cam_timeoffset` and traces the per-frame estimate to
`derived/pose_openvins/calib_trace_right.parquet`. `calib_converge` reports each take's converged tail and the spread **between**
takes; gates: ≤ 1° rotation, ≤ 5 mm translation, ≤ 2 ms time offset. If the online estimate drifts far from the Kalibr prior, the
prior is suspect — re-run step 3 rather than accepting the online number (`--write-bundle` there records `openvins_online`
provenance, which is deliberately not production-ready).

**Production runs** use `--mode production` (default): calibrated values loaded, nothing optimised online.

## M1-1 validation recording (collector UI, right wrist + head C922)

Lifecycle: start streams → hold still → **wait for VIO_STABLE** (≥ 2 s TRACKING_OK while still, `ego_teleop.tracking.vio_stable`)
→ mark HOME start → protocol. The HOME window therefore begins after initialization; the HOME QA is unchanged.
Mark segment boundaries with `segment` events (detail.name = S0…S9) or note offsets for `--segments`.

| seg | motion | physical |
|---|---|---|
| S0 | still 10 s (HOME start) | — |
| S1 | slow X out-and-back between the two stops | 20 cm |
| S2 | slow Y | 20 cm |
| S3 | slow Z | 15 cm |
| S4 | yaw ±90° | 90° |
| S5 | pitch ±60° | 60° |
| S6 | roll ±60° | 60° |
| S7 | translation + rotation | — |
| S8 | fast motion / deliberate blur (loss allowed, runaway not) | — |
| S9 | back at HOME, still 10 s (HOME end) | — |

Edit `configs/ego_teleop/m1_protocol.yaml` if the physical distances differ.

## Offline evaluation

```
python -m ego_teleop.tools.m1_vio EPISODE --side right --backend openvins --downscale 1 \
       --segments "S0:0-10,S1:10-25,S2:25-40,S3:40-52,S4:52-62,S5:62-72,S6:72-82,S7:82-100,S8:100-110,S9:110-120"
```
Writes `derived/pose_openvins/{right_camera_pose.parquet, pose_qa.json, m1_report_right.json}` and prints the M1 line plus one
line per segment (scale from the known displacement, rotation ratio, drift, runaway flag).

### First real pass gate (M1-1, prototype-level)

| metric | first target |
|---|---|
| post-init valid | ≥ 95 % |
| 20–30 cm translation scale | 0.90–1.10 |
| return-to-home error | < 2–3 cm |
| static drift over 10 s | < 1 cm preferred |
| rotation error | < 5° preferred |
| RPE @ 1 s | < 2–3 cm preferred |
| fast motion | tracking LOSS is acceptable |
| while LOST | must HOLD |
| recovery | returns without a pose jump |

Also record: initialization time, lost intervals and time-to-recover, feature counts, pose jumps.

**What counts as failure.** Losing tracking during the fast/blur segment is expected for a wrist-mounted camera and is not a
failure. The failure mode is a *confidently wrong* pose:

```
motion blur → wrong pose still reported TRACKING_OK → 20–30 cm pose jump → robot target runaway
```

That chain is what the supervisor's jump/feature/age checks and the low-feature regression test exist to prevent; the M1 report's
`jumps` counters and the per-segment `runaway` flag are the numbers to read for it.

## M1-2 dry run before any robot motion

If M1-1 passes, do **not** move the reBot next. First run the chain end-to-end against a virtual robot:

```
human wrist → OpenVINS → WristPoseProvider → TrackingSupervisor → RelativeSE3Retargeter → virtual reBot target (3-D view)
```

Watch clutch/recenter, TRACKING_LOST hold, workspace clamping and IK feasibility on the visualised target trajectory. Only after
that dry run looks right does low-speed real reBot teleoperation follow (safety limits start at 0.10 m/s, 30 °/s).

## Before M1-2 (live)

Full-rate IMU must reach the VIO: `ego_teleop.tracking.imu_tap.install_imu_tap(device)` + `LiveVioFeeder` (never the 30 Hz teleop
clock). Then `EstimatorWristPoseProvider(OpenVinsBackend)` → `TrackingSupervisor` → `RelativeSE3Retargeter` target-trajectory
visualisation without a robot.

---

## Pre-ICM status (2026-09-11) — what is done, what is waiting on hardware

The HandUMI **main** tracking path is Arducam + ICM42688P → this M1 stack. The RGB-D/FoundationPose branch is frozen
(`docs/handumi_collector/RGBD_POSE.md`). **No new VIO abstraction is to be written**: `WristPoseProvider` is the
canonical backend interface, `tools/m1_vio.py` is the offline replay harness, and the HandUMI collector calls those.

### Software: ready, nothing further to build before the ICM arrives

`tracking/interfaces.py` · `tracking/wrist_pose_provider.py` · `tracking/backends/openvins.py` ·
`calibration/bundle.py` · `tools/{m1_vio,imu_allan,kalibr_export,kalibr_import,calib_converge}.py` ·
`handumi_collector/tools/{camera_imu_offset,calibrate_fisheye,inspect_episode}.py` —
all import and wire (checked 2026-09-11).

### Recorder timing audit (P0-4): the recorder is sound, the evidence is thin

`inspect_episode --validate` now also reports `median_gap_ms` and `p95_jitter_ms` alongside `max_gap_ms`, because VIO
integrates IMU between frames and cares about interval *stability*, not just the worst gap. `capture_ns` monotonicity
and duplicate timestamps were already covered (a duplicate breaks the strict-increase check).

Only take that exists — `DIAG/DIAG_20260911_105710` (left wrist, 5.0 s, stationary "headless smoke"):

```
1920x1080, 151 frames == 151 frame_meta, skipped 0, capture-index drops 0
fps 30.03   median_gap 33.35 ms   max_gap 35.0 ms   p95_jitter 1.31 ms
ok=true, problems=[]
```

This validates the **recorder**, not the camera for VIO: 5 s and stationary says nothing about motion blur, exposure
under motion, or sustained recording. Thresholds stay unfrozen until a real motion take exists.

### Blocked on the operator (P0-1 … P0-3)

1. **L/R motion takes, 30–60 s each**, production resolution/fps, manual exposure: static 3 s → slow X → slow Y →
   slow Z → roll → pitch → yaw → combined SE(3) → normal manipulation speed → faster wrist motion → static 3 s.
   Not a VIO dataset (no IMU yet) — a camera-quality audit.
2. **Exposure lock** per side before any calibration take. The target is not a pretty image: it is *do features
   survive fast motion*. Record `exposure_us`, `gain`, white-balance mode, fps, resolution in episode metadata.
3. **L/R fisheye intrinsics** with `tools/calibrate_fisheye.py` (do NOT write new calibration code), in the **final
   physical mount**, at final resolution/fps/exposure. Board coverage must include centre, four corners, near/mid/far
   and roll/pitch/yaw tilts — not a stack of frontal views. Freeze as
   `configs/calibration/{left,right}_arducam_fisheye_v001.yaml` (K, D, resolution, RMS/reprojection error, timestamp,
   device identity). `configs/calibration/` currently holds only `gripper_v001/v002` and `virtual_wrist_v001`, and the
   DIAG episode records `calibration_version: uncalibrated` — this is the single biggest pre-ICM blocker.
   **There is no right-wrist recording of any kind yet.**

The frozen virtual-C922 pipeline is not to be modified by any of this.

### On ICM arrival — sequence is the existing M1-0a … M1-1 above

Stage 1 raw IMU (~400 Hz, native axes preserved, sensor + host receipt timestamps) → Stage 2 IMU QA (`imu_allan.py`:
gyro bias/noise, gravity magnitude, rate, jitter, dropped samples) → Stage 3 **temporal** calibration
(`camera_imu_offset.py`, fast rotation, visual angular motion vs gyro; a gate before VIO) → Stage 4 **spatial**
(`kalibr_export` → Kalibr → `kalibr_import` → `calib_converge`, per side; do not proceed if convergence is unstable)
→ Stage 5 single-hand (LEFT first) OpenVINS **offline** via `m1_vio.py`, not live → Stage 6 ΔTCP
(`T_TCP = T_B · T_B_TCP`, then `ΔT_TCP(t,k) = T_TCP(t)⁻¹ T_TCP(t+k)`, physically the same quantity as the robot's
`C_state`) → Stage 7 grip from the Feetech encoder, kept out of vision entirely.

Do **not** build before the hardware lands: fake/synthetic IMU, a custom VIO algorithm, EKF/factor graph,
RGB-D fusion, final body→TCP calibration, dual-hand common-world fusion, or H120 collection.
