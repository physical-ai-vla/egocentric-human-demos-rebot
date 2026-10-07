# Vision-only RGB-D arm teleoperation — POC runbook (V0 → V1)

**How much of a reBot arm teleoperation can a single fixed RGB-D camera carry, with no IMU?**

This branch is a *pose source*, not an architecture. The production arm path is unchanged and is still the one that
ships:

```
wrist fisheye + IMU ──► OpenVINS ──► WristPoseProvider ──► relative SE(3) ──► reBot        (production, untouched)

FIXED Orbbec RGB-D ──► MediaPipe 2D ──► aligned depth ──► metric 3D hand ──► palm pose ──┘  (this POC, temporary)
                                                                                 ▲
                                            everything downstream of here is shared, unmodified code
```

The comparison this sets up — RGB-D-only vs. RGB-D + constraints vs. wrist camera + IMU + OpenVINS — is only honest
because the two sources meet at the same interface and nothing below it knows which one it got.

## The camera must not move

`camera_mode: fixed` is an invariant, enforced in config (`RgbdArmPocCfg.__post_init__` refuses anything else) and
recorded in every log. `T_world_camera` here *is* `T_camera_palm`: the camera frame is the world frame. On a
head-mounted camera every head rotation would read as hand motion and the robot would follow the operator's head.

## Where things live

| | |
|---|---|
| `ego_teleop/hand3d/palm_pose.py` | palm origin/basis read-out, `ArmPoseHealth`, geometry sanity, `RgbdPalmPoseEstimator` |
| `ego_teleop/tracking/rgbd_hand_pose.py` | the factory that plugs it into `EstimatorWristPoseProvider` |
| `ego_teleop/tools/p1_rgbd_arm.py` | V0 (palm pose + metrics) and V1 (3-DoF virtual TCP) |
| `configs/ego_teleop/rgbd_arm_poc.yaml` | everything tunable, including the POC's own frame mapping and safety |
| `ego_teleop/tools/rgbd_arm_hud.py` | live HUD: palm marker + the gate numbers as they happen (pre-flight) |
| `ego_teleop/tools/rgbd_arm_metrics.py` | the post-hoc V0 metrics and the provisional V2 gate |
| `ego_teleop/tools/rgbd_arm_dashboard.py` | N takes -> one self-contained gate page |
| `~/orbbec_recorder.py` (`poc`, `preview`) | fixed color exposure + the 60 s protocol guide |
| `tests/teleop/test_rgbd_arm_poc.py` | 41 tests |

Reused unchanged, and deliberately not re-implemented: `RgbdHandPoseProvider`, `DepthSampler`, `HandIdentityTracker`,
`HandPoseSupervisor`, `aero_mocap.palm_frame`, `TrackingSupervisor`, `VioStableGate`, `HumanRobotFrameMapper`,
`RelativeSE3Retargeter`, `ArmSafetyPipeline`, `TeleopCoordinator`, `HttpRebotClient`/`MockRebotController`.

## Three health enums, three questions

| enum | question | owner |
|---|---|---|
| `HandPoseHealth` | may this drive the **Aero fingers**? | `HandPoseSupervisor` (existing) |
| `ArmPoseHealth` | is there a usable **arm-root** pose? | `palm_pose.py` (new) |
| `TrackingHealth` | what may the **robot** do about it? | `TrackingSupervisor` (existing, shared with OpenVINS) |

The one that matters: **five missing fingertips is `HAND_LOST` and `ARM_POSE_OK`.** The hand supervisor's verdict may
only lower the arm to `ARM_POSE_DEGRADED`; it can never veto it. `ARM_POSE_LOST` is reserved for what the arm actually
depends on — no hand reconstructed at all, a non-metric source, no palm origin, or an origin outside the depth band.

## Palm origin

Component-wise **median** of the valid palm landmarks (wrist + 4 MCPs), real measured depth only — a shape-prior
`filled` landmark never counts. A median needs no threshold to guess and one bad landmark in five moves it by
millimetres. There is no per-landmark outlier threshold on purpose: the palm spans 5–9 cm by anatomy and all of that
lands on the depth axis when the hand points at the camera, so any fixed threshold either rejects healthy palms or
catches nothing. Gross flying pixels are already removed upstream by the depth sampler.

The median does shift when the set of valid landmarks changes, so `min_palm_landmarks_ok: 5` — a frame with an
incomplete palm is declared `ARM_POSE_DEGRADED` rather than quietly moving the arm.

Orientation is the **canonical** `aero_mocap.palm_frame` basis, not a second convention; V0/V1 measure it every frame
and command none of it.

## Instruments

Three, and they run at different moments — mixing them up is how a take gets spoiled.

**Before the take — `rgbd_arm_hud`.** The live instrument: hand skeleton, the palm-origin marker, and every gate
number computed as it happens (jitter, catastrophic jumps, lost-run length, duty, plus rolling XYZ / orientation /
speed traces). It runs the same pose source the analysis runs, so what it shows is what V0 will report.

```bash
cd ~/ego_collector
.venv/bin/python -m ego_teleop.tools.rgbd_arm_hud --episode datasets/HumanRGBD_v1/episode_000001   # no camera needed
sudo .venv/bin/python -m ego_teleop.tools.rgbd_arm_hud --live --serve 8712 \
     --color-auto-exposure false --color-exposure 156                        # pre-flight, the take's camera mode
```

**Memory budget — the macOS MediaPipe GPU graph leaks.** Measured 2026-09-11 (mediapipe 1.0.1): ~3.4 MB per frame
(two 848×480 RGBA buffers) in `ImageCloneCalculator` → `ConvertToGpu` → `CreateCVPixelBufferWithoutPool`. RSS hits
13.8 GB by frame 4000 and the process aborts with CoreVideo `kCVReturnAllocationFailed (-6662)` around frame 7500 —
about four minutes at 30 Hz, exactly the length of a camera-alignment session. The CPU delegate is not an escape
(its graph aborts outright on macOS with 1.0.x). So the HUD rebuilds the landmarker every `--recycle-every` frames
(default 900): the peak stays near 3 GB, at one ~0.4 s hitch every 30 s, and a 9000-frame run that used to abort now
sustains 60 fps. A rebuild resets MediaPipe's frame-to-frame tracking, so the frame after it is a fresh detection —
which is why **analysis leaves recycling off** (`HandLandmarker(recycle_every=0)`, the default): a 60 s take is 1800
frames ≈ 6 GB on a 26 GB machine, and no periodic re-detection lands in the measurements. `a1_hand3d` / `a2_virtual_aero`
(Aero branch) carry the same latent limit and have not been changed.

**Pass the take's exposure to the pre-flight.** Measured on the Gemini 336 (2026-09-11): reopening the pipeline
turns auto-exposure back **on** — the value it re-converges to is the same in unchanged lighting, but the AE flag is
not inherited from whatever ran before. A pre-flight that does not set the exposure itself therefore runs in a
different camera mode from the recording, and its jitter number predicts nothing. The HUD warns when it finds AE on.

It writes `/tmp/rgbd_arm_hud.jpg` rather than opening a window — the Orbbec needs sudo on macOS and a root process
cannot reach WindowServer, which is why the recorder writes JPEGs too. `--serve PORT` also streams MJPEG plus a
`/stats.json` on localhost, so the numbers are readable on a phone while both hands are in front of the camera.
**It must not share a process with a take**: it runs MediaPipe on every frame, and the take must not drop frames.

**During the take — `orbbec_recorder.py poc`.** Segment banner, a remaining-time bar, and a Korean voice cue at each
boundary. No MediaPipe, so nothing competes with capture. It records the segment boundaries it actually used into
`metadata.yaml` and prints the matching `--segment` arguments — the analysis uses the recorded boundaries, never
assumed ones. Screen text is ASCII because OpenCV's Hershey fonts have no Korean glyphs; only the voice is Korean.

**After the takes — `rgbd_arm_dashboard`.** One page: the three headline numbers, the six-metric gate matrix across
takes, the per-segment table with camera Δ beside mapped robot Δ (this is where the axis map is read off), the
wrist-rotation translation leak in mm per 10°, and per take an XYZ trace with an arm-pose health ribbon.

```bash
.venv/bin/python -m ego_teleop.tools.rgbd_arm_dashboard --take outputs/poc/t1 --take outputs/poc/t2 \
    --take outputs/poc/t3 --out outputs/poc/gate.html
```

### The provisional V2 gate

Engineering permission for the first low-speed real-arm test, not a paper benchmark. **The worst take decides** —
three takes exist so one lucky take cannot open it, and an unmeasured metric reads NOT MEASURED rather than passing.

| metric | green | yellow | red |
|---|---|---|---|
| arm-pose valid duty | ≥95% | 90–95% | <90% |
| static XYZ RMS jitter | ≤5 mm | 5–10 mm | >10 mm |
| static XYZ p95 deviation | ≤10 mm | 10–20 mm | >20 mm |
| return-to-start error | ≤15 mm | 15–30 mm | >30 mm |
| catastrophic jump >30 mm/frame | 0 | 1 | >1 |
| long tracking loss >300 ms | 0 | 1 | >1 |

A single 5–10 cm frame is more dangerous to a real arm than a few mm of steady noise, which is why the jump and
long-loss rows demand exactly zero. Definitions that are easy to get wrong, and are pinned in tests: a run is
measured **to the next good frame** (a 10-frame outage at 30 Hz is 333 ms of hold, not 300); a jump is **never**
measured across a gap; jitter is **withheld** where the window drifted more than 10 mm rather than reported as noise.

## V0 — palm pose only, no robot

```bash
.venv/bin/python -m ego_teleop.tools.p1_rgbd_arm --stage v0 --live --protocol --viz --out outputs/rgbd_arm_poc
.venv/bin/python -m ego_teleop.tools.p1_rgbd_arm --stage v0 --episode datasets/HumanRGBD_v1/episode_000001   # replay
```

The trial (plan section 15), one 60 s take, labelled by `--protocol`:

```
 0-10 s stationary open hand   10-20 s slow X/Y/Z   20-30 s palm rotation only
30-40 s translation+rotation   40-50 s fast natural 50-60 s return to the start pose, then still
```

`--segment NAME:T0:T1` labels any other window. With nothing labelled the still windows are found automatically, so
the headline numbers exist on an unlabelled take too.

**The three numbers**, printed first:

```
stationary XYZ jitter p95 (mm)      return-to-start error (mm)      tracking-loss rate (%)
```

Plus: per-axis jitter and depth σ, orientation jitter (p50/p95/max), dropout count and durations, `ARM_POSE_*` counts
with reasons, the per-segment displacement table in **camera axes and mapped robot axes** — which is how you read the
axis map off and fill in `frames:` — palm scale, palm depth spread, and the frame-to-frame span jumps that the
geometry thresholds should later be set from.

## V1 — 3-DoF virtual reBot TCP, still no robot

```bash
.venv/bin/python -m ego_teleop.tools.p1_rgbd_arm --stage v1 --live --engage-at 3 --clutch-at 25 --release-at 30 \
    --record datasets/rgbd_arm_poc/episode_000001
```

The real coordinator, retargeter, workspace clamp and velocity/acceleration limiter run exactly as they would with a
robot attached; `arm_enabled: false` is the only difference. `tcp_*` is the pose that *would* have been sent,
`unclamped_*` is the retargeter target before safety.

Reported: hold fraction and reasons, workspace clamp fraction and depth, commanded speed vs. the limit, TCP path and
span, human travel per axis, and the clutch check (the frozen quantity is the **target** — the limiter is still
allowed to finish travelling to it).

Robot orientation is fixed at ENGAGE by construction: in `orientation: fixed` the emitted pose has identity rotation,
so the relative SE(3) delta carries no rotation. There is no separate 3-DoF code path.

### Readiness, honestly

`TeleopCoordinator`'s readiness flags are named for the production source. This POC has no IMU and no VIO, so two of
them are satisfied by RGB-D equivalents: `imu_online` = the depth stream is up, `vio_stable` = the reused
`VioStableGate` proved the palm pose held `TRACKING_OK` and still for `stable.min_stable_s`. `tracking_valid` is
**not** substituted — it is the real health of the RGB-D pose. Every episode records this as `readiness_note`.

## Dropouts

A 1–2 frame dropout is held, not extrapolated, and the numbers are `tracking.lost_after_ms` (how long a stale pose is
still `DEGRADED` rather than `LOST`) and `tracking.recover_after_n_ok` (good frames needed to resume). A `LOST` pose
holds the last arm target; nothing is ever integrated forward. Both live in `rgbd_arm_poc.yaml`.

## Geometry thresholds ship as `null`

`max_palm_scale_deviation`, `max_palm_scale_jump`, `max_span_jump_m`, `max_palm_depth_spread_m` are all measure-only
by default. The plan says thresholds come from measured data; a guessed one silently eats real motion. V0 prints the
distributions — set them from that table. `s_op` (operator palm scale) is learned once as the median of the first
`scale_reference_frames` good frames and is only ever used to *judge* a frame, never to rescale a command.

Maximum frame-to-frame palm motion is **not** reimplemented here: it is `TrackingSupervisor.max_jump_m` /
`max_jump_deg`, the same jump policy OpenVINS runs through.

## Storage

```
raw/rgbd_hand_pose_live_{left,right}.parquet    pose + arm_pose_health + raw palm position/orientation columns
raw/rgbd_relative_pose_live_{left,right}.parquet  per-tick ΔT_human, mapped delta, resulting TCP target
raw/human_hand.parquet   robot/rebot_command.parquet  robot/rebot_state.parquet  raw/events.parquet
head/color/*.jpg  head/depth/*.png                  (live runs; a replay records `source_episode` instead)
```

Provenance per stream: `pose_source: fixed_rgbd_hand`, `camera_mode: fixed`, `imu_used: false`, `causal: true`.
`wrist_pose_live_*` keeps meaning wrist fisheye + IMU + OpenVINS and is never written by this branch — a POC episode
can never be read as a wrist-VIO episode.

## What comes next, and what does not

V2 (3-DoF on the real arm) needs **no new code**: `arm_enabled: true` plus an `HttpRebotClient` in place of
`MockRebotController`. It is deliberately not wired until V1's numbers have been read. V3/V4 (palm orientation, 6-DoF
virtual) are a config flip to `orientation: palm` plus the same tooling; V5 after that.

Not implemented, on purpose: IMU of any kind, OpenVINS changes, a new VIO backend, learned pose or residual models,
MANO fitting, a second camera or view, head odometry or ego-motion compensation, object tracking, tactile, force
control. Whether any of them is the right next move is what the V0 numbers are for.
