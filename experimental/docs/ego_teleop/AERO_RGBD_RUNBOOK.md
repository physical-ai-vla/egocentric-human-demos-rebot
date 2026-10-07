# Head RGB-D → Aero Hand teleoperation — runbook (A0 → A5)

The Aero branch of the ActiveUMI+RGB-D spec. **Nothing here touches the wrist-VIO / reBot arm branch**: the head
camera reconstructs the operator's hand and drives the Aero hand only, and the arm keeps coming from wrist VIO.

```
HEAD RGB-D ──► MediaPipe 2D ──► robust depth ──► metric 3D ──► palm-local ──► DexPilot ──► Aero 16 joints ──► Aero
   (Orbbec Gemini 336, depth aligned to colour, 848x480 @30)                   (official dex_retargeting + Aero URDF)

WRIST FISHEYE + IMU ──► VIO ──► relative SE(3) ──► reBot          (untouched; M1 work)
```

## Where things live

| | |
|---|---|
| `ego_teleop/hand3d/aero_mocap.py` | official conventions: 21→25 keypoints, palm-local transform, Aero joint order/limits, handedness rule |
| `ego_teleop/hand3d/depth.py` | intrinsics + deprojection, robust per-landmark depth, hand-level outlier rejection |
| `ego_teleop/hand3d/head_camera.py` | versioned `head_rgbd_vNNN.yaml`, live `OrbbecHeadCamera`, `RecordedRgbdSource` |
| `ego_teleop/hand3d/identity.py` | temporal hand identity (continuity first, label vote second) |
| `ego_teleop/hand3d/providers.py` | `MediaPipeHandPoseProvider` (B0) / `RgbdHandPoseProvider` (B1) |
| `ego_teleop/hand3d/supervisor.py` | `HAND_INITIALIZING / OK / DEGRADED / LOST` + hold-on-loss + recovery gate |
| `ego_teleop/retarget/aero_backends.py` | `DexPilotAeroRetargeter`, `Semantic7DAeroRetargeter`, `AeroCommandLimiter` |
| `ego_teleop/tools/a1_hand3d.py` | A1: pose only, acceptance gate, overlay video |
| `ego_teleop/tools/a2_virtual_aero.py` | A2: B0 vs B1 vs B2 through the same retargeter, virtual Aero video |
| `configs/ego_teleop/head_rgbd.yaml` | everything tunable (nothing tunable lives in code) |
| `docs/ego_teleop/AERO_A0_AUDIT.md` | what was read out of the upstream repo, and what we kept vs replaced |

Dependencies added to the venv: `dex_retargeting==0.5.0` (+ `pin`, `nlopt`). `pyorbbecsdk` and `mediapipe` were
already there. **No ROS** — the official nodes' payloads travel as in-process dataclasses with identical field order.

## A0 — audit (done)

`docs/ego_teleop/AERO_A0_AUDIT.md`, pinned by `tests/teleop/test_aero_official_conventions.py` (13 tests, of which
6 read the upstream checkout at `~/aero-hand/aero-hand-open` and skip without it).

```
.venv/bin/pytest -q tests/teleop/test_aero_official_conventions.py tests/teleop/test_hand3d.py tests/teleop/test_aero_backends.py
```

## Camera calibration (before A1)

Depth must be registered to colour before any `depth[u,v]` read. The Orbbec `AlignFilter` does it; the fact is
recorded, not assumed (`HeadRgbdCalibration.aligned`, checked every frame by `require_aligned()`).

```bash
# import the intrinsics the Orbbec recorder already writes, as a versioned calibration
.venv/bin/python -c "from ego_teleop.hand3d.head_camera import HeadRgbdCalibration as C; \
  print(C.from_orbbec_recorder_yaml('datasets/HumanRGBD_v1/_calibration/intrinsics.yaml').save())"
# -> configs/calibration/head_rgbd_v001.yaml   (fx/fy/cx/cy, depth scale, aligned, fps, model)
```

Re-import whenever the stream resolution or the camera changes: the intrinsics are resolution-specific.

## A1 — metric 3D hand pose, no Aero motion

```bash
# on a recorded episode (repeatable, no camera needed)
.venv/bin/python -m ego_teleop.tools.a1_hand3d --episode datasets/HumanRGBD_v1/episode_000001 \
    --out outputs/a1 --viz
# live (macOS: the Orbbec SDK needs the UVC device unclaimed -> sudo)
sudo .venv/bin/python -m ego_teleop.tools.a1_hand3d --live --window --out outputs/a1_live \
    --static 4:9 --motion 12:20
```

Record one take per required motion (spec §27) before reading anything into the numbers: **open · fist ·
thumb-index pinch · thumb-middle pinch · individual finger flexion · hand translation · camera motion with the hand
held still**. `--static a:b` = hand and camera both still (fingertip jitter). `--motion a:b` = camera moves, hand
does not (palm-local invariance — the property the whole head-camera design rests on).

Gate thresholds live in `configs/ego_teleop/head_rgbd.yaml:gates`; the report prints **distributions**
(p50/p95/max), never averages alone, and says out loud that the thresholds are pre-hardware guesses.

Current status on the one usable recorded episode (`episode_000001`, a third-person desk view, NOT a head-mounted
teleop take — it validates the pipeline, not the gate): 50–73 Hz throughput, 94.6 % metric landmark coverage,
1 identity swap. `--static` / `--motion` need a purpose-recorded take.

## A2 — virtual Aero, no motors

```bash
.venv/bin/python -m ego_teleop.tools.a2_virtual_aero --episode <ep> --arms b0 b1 b2 --out outputs/a2 --viz
```

```
b0  official monocular MediaPipe world landmarks ──┐
b1  head RGB-D metric hand pose ────────────────────┼──► the SAME DexPilot config ──► virtual Aero
b2  head RGB-D metric hand pose ──► semantic-7D (existing baseline)
```

Only the pose source differs between b0 and b1 — same config, same scale factors, same clip. **Never change pose
estimation and retargeting together and then attribute the result to depth** (spec §30).

The scoring metric that means something: for every frame with metric depth we know the operator's real thumb-index
fingertip distance, and FK gives the virtual Aero's thumb-index gap. b0 cannot be scored that way at all — it has no
metric scale. That is the gap this branch exists to close.

## A3 — real Aero, standalone (hardware-gated, not yet run)

Nothing new to write: `robot/aero_client.SdkAeroClient` already is the SDK boundary. The sequence is

1. `configs/ego_teleop/aero.yaml` → set `port` (macOS needs it explicitly), home the hand.
2. Lower `head_rgbd.aero_limits.max_joint_rate_rad_s` well below the 6.0 default for the first runs.
3. Feed `AeroCommandLimiter` output to the client; log `aero_target`, `aero_command`, `aero_state` every tick.
4. Watch currents/temperatures from `ActuatorStates`; stop on any warning.

The 16-joint path is the one to use (`AeroTarget.joints_rad`); do **not** hand-roll a 16→7 tendon mapping — the hand
is underactuated and the SDK already owns that conversion.

## A4/A5 — arm + hand, then demonstrations

Only after A3 is stable. The arm branch is untouched until then (spec §31). `finger_source` in
`configs/ego_teleop/teleop.yaml` stays `unverified` and moves to `head_rgbd` **only** when the A1 gate passes on a
purpose-recorded take — not because the camera is plugged in (spec §26).

## Episode layout addition

```
episode_xxxxxx/
├── raw/human_hand.parquet      2D + camera-metric + palm-local landmarks, per-landmark validity/filled/confidence,
│                               handedness, provider health AND supervisor health   <- canonical raw human data
├── head/color/*.jpg  head/depth/*.png (16-bit aligned)  head/frames.parquet  head/calibration.json
└── robot/aero_target.parquet   retargeter output (16 joints rad) + optimiser diagnostics, before the SDK
```

Human pose is never replaced by the Aero target: if Aero is swapped for another dexterous hand, the episodes stay
usable. Depth is stored raw (16-bit, lossless round-trip through `HeadRgbdWriter`) so a later policy can build point
clouds / geometry-aware tokens from it — but no 3D backbone is part of V1 (spec §25).

## Known sharp edges

* **Handedness is swapped on a head camera.** MediaPipe labels assume a selfie-mirrored image; ours is not.
  `selfie_mirrored: false` in the config handles it. Getting this wrong silently drives the wrong hand.
* **Depth between the fingers is background, not nothing.** A window median there is confident and wrong, so
  `reject_depth_outliers` invalidates landmarks further than 15 cm from the hand's own median depth.
* **Filled ≠ valid.** Landmarks the sensor could not measure are reconstructed from the monocular shape prior
  (similarity fit to the measured ones) so the optimiser gets complete geometry, but they stay `valid == False`, so
  the gates keep measuring real depth coverage. More than `max_fill` missing ⇒ the frame is LOST, not invented.
* **Hand size drives DexPilot saturation.** Aero's straight-finger reach is ~0.20 m from its base link; a hand pose
  scaled much smaller makes the optimiser curl everything to the limit. `diag_saturated` is logged per frame, and
  `dexpilot.rescale_palm_to_m` exists if an operator's hand needs normalising (off by default, as upstream).
* **HAND_LOST holds; it never opens.** Auto-opening on vision loss drops whatever is being carried. Resuming needs
  `recover_stable_ms` of continuous good tracking (state `HAND_INITIALIZING` in between, also non-commandable).
