# Multi-sensor wrist tracking: chest RGB-D + wrist Arducam + wrist IMU → reBot

Status 2026-09-22: the fusion stack, its configuration, its logging, the evaluation tooling, the **live MASt3R
visual frontend** and the **live HUD** are implemented and tested on synthetic takes. **No part of it has run on
the three-sensor rig** — the wrist IMU unit is the missing hardware. P0 is the first thing to run.

Added after the first pass, in this order of importance:

1. **`mast3r_live`, the live wrist visual frontend** (§2, §10). The branch used to default to OpenVINS, which the
   spec rules out as the production backend, and the only MASt3R path was an OFFLINE replay of a `result.txt`.
   MASt3R-Fusion now runs live on the GPU box behind a socket adapter — `ego_teleop/tracking/backends/
   mast3r_live.py` here, `scripts/mast3r_live_server.py` deployed into ITS checkout, nothing vendored either way.
   Protocol and rationale: `docs/ego_teleop/MAST3R_LIVE_PROTOCOL.md`.
2. **`f5_teleop_hud`, the live page** — the three branches overlaid per axis while the operator is still standing
   in front of the camera, plus the alignment, the correction, the latency and the §20 gates, live. §7 below.
3. **Two defects found while wiring those up**, both in the live path only:
   * `FusedLiveRig` built the VI backend and **never called `PoseEstimator.initialize()`**, so any live backend
     would have run on default intrinsics with no idea what camera it was looking at. `FusedWristCfg.
     initialize_estimator` now hands it the versioned `fisheye_<side>` + `camera_imu_<side>`, and the live rig
     calls it. Replay stages pass a ready estimator and are unaffected.
   * The wrist Arducam is Kannala-Brandt and MASt3R-Fusion's `Intrinsics.from_calib` knows pinhole and omnidir
     only. The client rectifies, with the same `cv2.fisheye` call and the same `balance` the EuRoC export uses —
     `ego_teleop/transforms/fisheye.py` is now the one implementation both share, so the live tracker and the
     offline bake-off cannot end up looking at two different cameras.

The three sensors are not alternatives to one another:

| sensor | contributes | fails at |
|---|---|---|
| chest RGB-D (Orbbec) | metric **absolute XYZ** (all three axes), long-term drift anchor, recovery reference | occlusion, hand out of frame, palm orientation |
| wrist Arducam | visual motion, keyframes, map, relocalization | featureless scene, motion blur, scale alone |
| wrist IMU (XIAO Sense / Teensy unit) | fast rotation and dynamics, motion between camera frames, dropout bridging | translation over any length of time |

```
chest RGB-D ─── metric XYZ ───┐
                              │
Arducam ─┐                    ▼
         ├─ visual-inertial → FUSION → ΔSE(3) → reBot
IMU  ────┘
```

## 1. What was built

| file | what |
|---|---|
| `ego_teleop/tracking/local_pose.py` | `LocalPoseContinuity` — map pose → continuous local pose (§3, §17) |
| `ego_teleop/tracking/anchor_fusion.py` | `AnchorAlignment` (T_W_D + lever arm), `TranslationAnchorFusion` (C(t)), `fusion_health` (§6, §8, §14) |
| `ego_teleop/tracking/fused_wrist.py` | `FusedWristPoseProvider` — a `WristPoseProvider`, so downstream is unchanged (§11) |
| `ego_teleop/tracking/fused_live.py` | `FusedLiveRig` — the three live sensors on their own threads (§15) |
| `ego_teleop/config.py` | `FusedWristCfg` + `configs/ego_teleop/fused_wrist.yaml` |
| `ego_teleop/recorder/episode_logger.py` | `fused_wrist_pose_live_*`, `fused_relative_pose_live_*` streams (§18) |
| `ego_teleop/tools/f0_sensors.py` | P0: three sensors at rate, simultaneously, + the IMU clock fit |
| `ego_teleop/tools/f4_fusion.py` | P4–P7 + the A/B/C ablation + the 60 s protocol report (§20, §21) |
| `ego_teleop/tools/f5_teleop_hud.py` | the LIVE page: three branches per axis, alignment, C(t), latency, gates |
| `ego_teleop/tracking/backends/mast3r_live.py` | live MASt3R-Fusion over a socket (§2, §10) |
| `scripts/mast3r_live_server.py` | the GPU-box half; deployed into the MASt3R-Fusion checkout, not imported here |
| `ego_teleop/transforms/fisheye.py` | KB -> pinhole, shared by the live client and `export_euroc.py` |
| `tests/teleop/test_fusion.py`, `test_fusion_tool.py`, `test_mast3r_live.py`, `test_teleop_hud.py` | 69 tests |

Downstream is untouched: `HumanRobotFrameMapper` → `RelativeSE3Retargeter` → `ArmSafetyPipeline` →
`TeleopCoordinator` → reBot client are the same objects the wrist-VIO and RGB-D-POC paths already run through. The
production `teleop.yaml` path and `rgbd_arm_poc.yaml` are unchanged.

## 2. The estimation problem, honestly stated

The spec sketch is `e(t) = p_RGBD(t) − p_VI(t)`. Taken literally that is wrong twice over, and both corrections are
what most of `anchor_fusion.py` is:

**(a) Different frames.** The RGB-D palm position lives in the chest camera frame D; the VI pose lives in the VI
world W. `T_W_D` is unknown. It is estimated online from paired samples (Kabsch, scale fixed at 1 — both are
metric). Until that fit converges the anchor is inactive and the pose is declared VI-only, not silently anchored to
a guess.

**(b) Different points.** RGB-D measures the **palm centroid**; the control frame is the **wrist**. A pure wrist
rotation swings the centroid by 5–10 cm while the control point does not move at all. Without modelling that lever
arm `r`, the residual reads the swing as drift and the anchor drags the arm around every time the operator turns
their hand. `test_wrist_rotation_does_not_leak_into_the_fused_position` is that failure, measured: the same take
with `r` forced to zero moves the control point by centimetres.

So the model is

```
predict    p̂_W(t) = p_VI(t) + R_VI(t)·r
rotation   R_WD    = Kabsch( p_D(t) → p̂_W(t) )
offsets    t_WD, r  jointly and in closed form from   [ I  −R_VI(t) ]·(t_WD; r) = p_VI(t) − R_WD·p_D(t)
residual   e(t)    = T_W_D·p_D(t) − p̂_W(t)            ← VI drift, in metres, in W
correction C_target ← (1−α)·C_target + α·e            → C slewed at max_correction_rate_m_s
output     p_fused = p_VI + C ,  R_fused = R_VI
```

`t_WD` and `r` are solved **together**, not alternately: a constant part of the lever arm is indistinguishable from a
camera translation, and alternating two separate means crawls (measured: still 1.3 cm out after 15 iterations) where
the joint 6-unknown solve converges in three.

### What is observable, and what is not

* `r` needs wrist rotation that is **not locked to the translation**. If the hand always rolls the same way it
  moves, "the palm swung" and "the camera is rotated slightly differently" fit the same data. Both conditions are
  checked — `min_rotation_span_deg`, and the posterior sigma of `r` from the normal equations against
  `max_lever_arm_sigma_m`. When either fails, `r` keeps its last converged value instead of being fitted to a
  coincidence. **The `rotation_only` window of the 60 s protocol exists for exactly this.** Once measured, freeze it
  in `alignment.lever_arm_m`.
* **The alignment is frozen once it is complete** (`freeze_after_align`). A continuously re-fitted `T_W_D` quietly
  absorbs slow VI drift — the residual goes to zero, C stops correcting and the drift survives in the fused pose.
  The anchor can only see drift the alignment is not free to explain. The price is the V1 assumption of §1.1: the
  chest camera is static in the VI world for the length of a take. A real torso shift shows up as a persistent large
  residual and re-opens the fit (`realign_residual_m` / `realign_after_s`), which is logged.
  *Freezing waits for the lever arm too* — otherwise the first window with enough travel (the translation window,
  where `r` is invisible) would lock `r` at zero before the rotation window ever ran.

### dC/dt is a robot velocity

Under relative-SE(3) teleoperation the arm follows increments of the fused pose, so the correction rate is injected
straight into the arm: `max_correction_rate_m_s: 0.01` is 1 cm/s of motion nobody asked for. Keep it an order below
hand speed. A large pending correction is mostly deferred to the clutch (§13), where the arm is frozen and the whole
correction can be applied at once — `deferred_rate_fraction` is what still creeps while commanding, because an
operator who never clutches must not be left with a permanently useless anchor.

## 3. Local pose vs map pose

`T_map_wrist` is the backend's globally optimised pose and **is never commanded**. A loop closure is a correction of
the past, not a motion of the hand, so it is absorbed into an offset that keeps `T_local_wrist` exactly continuous,
and the size of what was absorbed is logged.

Default policy: **absorb only corrections the backend announces** (`extra["map_update"] / loop_closure /
global_correction`). A tracking failure and a loop closure look identical in the pose alone; swallowing both would
hide the failure, so an unflagged jump stays a jump and `TrackingSupervisor.max_jump_m` grades it LOST.

## 4. States (§14)

| VI | chest RGB-D | fused state | `TrackingHealth` (what the coordinator acts on) |
|---|---|---|---|
| — | — | INITIALIZING | LOST (never commands) |
| OK | OK, aligned | TRACKING_OK | OK |
| OK | dropped < `rgbd_grace_s` | TRACKING_OK | OK |
| OK | dropped longer | DEGRADED | DEGRADED (speed ×`degraded_speed_factor`) |
| OK | never aligned > `max_unanchored_s` | DEGRADED | DEGRADED |
| DEGRADED | any | DEGRADED | DEGRADED |
| LOST | OK | LOST (HOLD) | LOST — translation may be observable but orientation confidence is not; `vi_lost_translation_only: true` opts into following translation at degraded speed with the orientation frozen |
| LOST | LOST | LOST (HOLD) | LOST |

A zero pose is never emitted. Once a pose has existed, LOST means *the last pose, held, with zero velocity*.

## 5. Sensors and time

* **Rates** (§15): IMU 416 Hz, Arducam 30 Hz, chest RGB-D 30 Hz, fusion output and robot command 50 Hz. Nothing is
  forced onto a common rate; each sensor thread pushes, the control loop pulls the latest state, and no queue is
  allowed to accumulate frames.
* **The XIAO nRF52840 Sense speaks the existing protocol.** `firmware/teensy_imu/PROTOCOL.md` v1 already carries
  exactly what §10 asks for — `seq`, device `t_us` at the data-ready interrupt, accel m/s², gyro rad/s, CRC-16 —
  so the host side needs only `rate_hz: 416` in the hardware profile; `handumi_collector.devices.teensy_imu` is a
  protocol parser, not a board driver. **The XIAO firmware sketch itself is not written** (the only remaining
  hardware task). Whatever runs on it must stamp the measurement time on the device; host arrival time is never a
  measurement timestamp. `f0_sensors.py` reports the device→host fit (slope, ppm drift, residual, gaps).
* **T_H_C is mandatory.** `configs/calibration/wrist_camera_<side>_vNNN.yaml`. The Arducam and the IMU sit on a
  lever arm from the wrist rotation centre; an identity there fabricates translation from every rotation. The replay
  tool refuses to run without it (`--allow-identity-mount` exists for plumbing only, and says so loudly).

## 6. Logging (§18)

`raw/fused_wrist_pose_live_<side>.parquet` is the exact causal trajectory the robot control had. Per row, next to
the pose: the RGB-D anchor (raw in D **and** mapped into W — all three axes, §16), the VI pose as both map pose and
local pose, the applied correction, the residual before and after correction, the absorbed map correction, both
branch healths and the unified state. `raw/fused_relative_pose_live_<side>.parquet` carries the per-tick human delta,
the mapped robot delta, the commanded TCP and the latencies. `wrist_pose_live_*` keeps meaning "wrist VIO alone" and
is never reused for this. An offline refined trajectory is a different stream and never overwrites either.

## 6a. The live page (`f5_teleop_hud`)

```
.venv/bin/python -m ego_teleop.tools.f5_teleop_hud --synthetic --stage virtual --protocol --hold --open
.venv/bin/python -m ego_teleop.tools.f5_teleop_hud --episode EP --vi-poses POSES.parquet --stage fused --protocol
sudo .venv/bin/python -m ego_teleop.tools.f5_teleop_hud --live --stage virtual --protocol
```

It serves one page on `127.0.0.1:8713` (`--host 0.0.0.0` to read it from a phone while both hands are busy). The
top plot is the reason it exists: **p_rgbd mapped into W, p_vi, and p_fused, overlaid per axis.** §P5 says to
record the two branches side by side before fusing so you can see which sensor is good on which axis; a table in a
report says that afterwards, and this says it while the window can still be re-run. Everything else on the page is
there to explain a bad overlay — the alignment block (is `T_W_D` fitted, is the lever arm observable yet), the
branch lamps (which sensor dropped), the correction trace (|e| vs |e−C|), the latency strip (§25).

It is an OBSERVER and the tests hold it to that: the pose it draws comes out of `f4_fusion.run`'s own tick loop
through `tick_hook` (a replay with the hook is asserted byte-identical to one without), the gate table is
`f4_fusion.report` on the rows so far rather than a HUD-local re-implementation, and `--stage real` is not offered
— watch a real arm with your eyes and the e-stop, not through a browser tab.

Served over HTTP, not a cv2 window, for the same reason `rgbd_arm_hud.py` is: the Orbbec needs sudo on macOS and a
root process generally cannot talk to WindowServer.

## 7. The ladder

| rung | how |
|---|---|
| **P0** sensors independently, simultaneously | `python -m ego_teleop.tools.f0_sensors --hardware handumi_rgbd --seconds 30 --imu-hz 416` |
| **P1** RGB-D wrist XYZ | existing: `python -m ego_teleop.tools.p1_rgbd_arm --stage v0 --live` |
| **P2** Arducam + IMU recording | existing: the collector, with a profile carrying the wrist camera + IMU |
| **P3** wrist VI | LIVE: start `mast3r_live_server.py` on the GPU box, then `vi_backend: mast3r_live`. OFFLINE: `python -m ego_teleop.tools.m1_vio EPISODE --side right --backend mast3r` (replays a MASt3R-Fusion `result.txt`), or `--backend openvins` as a reference. Nothing of MASt3R is vendored either way. |
| **P4** RGB-D vs VI, side by side | `f4_fusion --stage compare --episode EP --vi-poses derived/pose_openvins/right_camera_pose.parquet --protocol --ablation` |
| **P5** fused virtual wrist | `f4_fusion --stage fused …` — adds the human→robot mapping table per protocol window |
| **P6** virtual reBot | `f4_fusion --stage virtual … --clutch-at 45 --release-at 47` — full coordinator + safety, mock arm |
| **P7** low-speed real reBot | `f4_fusion --stage real --live --i-am-at-the-robot` — refuses to start otherwise; start at `translation_scale: [0.3,0.3,0.3]`, one axis at a time |

`--live` runs P5–P7 on the three real sensors through `FusedLiveRig`. **That path has never been executed on
hardware.** Run P0 first: it uses the collector's own device path, deliberately not this one, so it can prove the
sensors before the fusion rig is trusted.

Plumbing check with no hardware at all (this is what CI runs):

```
python -m ego_teleop.tools.f4_fusion --stage virtual --synthetic --protocol --ablation
```

The synthetic take follows the 60 s protocol window for window, the chest camera sees the truth, and the VI drifts
2 mm/s after the alignment windows. It is a fixture, never evidence about the hardware.

## 8. Evaluation and the ablation (§20, §21)

The 60 s protocol (`--protocol`), ×3, measured independently for **A** `rgbd_only`, **B** `vi_only`, **C** `fused` —
one config value apart, one code path. Fusion is not assumed to win; if C is not better than the better of A and B,
the reason is in the report's `branches` and `alignment` blocks.

Reported: valid duty, stationary XYZ RMS/p95, stationary rotation RMS, return-to-start, anchor↔fused residual
(the number that says whether fusion works) beside the drift the anchor sees and the raw anchor↔VI gap (which
contains the lever arm by construction and is therefore *not* a gate), rotation-only translation leakage,
catastrophic jumps, tracking-loss duration and shape, map corrections absorbed, command discontinuity, latency.

Thresholds live in `fused_wrist.yaml` `gates:` and are **pre-measurement starting points**. Re-fit them from the
first real A/B/C table before any PASS is read as evidence — the same rule as every other gate block in this repo.

## 9. Things found while building this, that are not in the spec

1. **The chest camera's frame and the palm lever arm have to be estimated before any correction exists** (§2 above).
   The spec's `e = p_RGBD − p_VI` has no frame and no lever arm in it.
2. **A sliding alignment cancels the thing it is supposed to measure.** Freezing the fit is what makes slow drift
   observable at all, and it costs the static-chest-camera assumption. Written down rather than hidden.
3. **`anchor_workspace` could jump the arm on engage.** The box is built around the ENGAGE TCP *and intersected with
   the configured workspace*; if the TCP is outside that workspace the intersection is non-empty but does not
   contain the TCP, and the clamp moves the first command to the nearest face — on the arm, an unannounced jump of
   however far outside it was standing. `FusedWristCfg.anchor_workspace` now refuses. **The same latent defect is
   still in `RgbdArmPocCfg.anchor_workspace`** (`ego_teleop/config.py`), untouched here because it would change the
   behaviour of a tool that is in use.
4. **Ablation A may never reach VIO_STABLE.** An unfiltered 30 Hz palm position with ~2 mm noise implies ~86 mm/s of
   apparent speed, above `VioStableConfig.max_lin_vel_m_s = 0.05`, so the stability gate can never pass and nothing
   is ever commanded. The report says so explicitly instead of returning empty command metrics. That is a real
   property of the RGB-D-only source, not a tooling artefact.

## 10. Not done

* `scripts/mast3r_live_server.py` has **never been run against a GPU**. It is `main.py`'s loop with the dataset
  replaced by the socket; the three adaptations to watch on the first run are `SocketStream` (one pending frame),
  `AppendableImuPool` (`IMUPool`'s `get_records` contract over a growing buffer, gyro in deg/s) and the
  map-update detector. Nothing about this path's tracking quality is known yet.
* The live path as a whole is written, not run — the wrist IMU unit is the missing hardware. P0 first.
* Everything in §22 of the spec stays out: Aero Hand control, finger retargeting, DexPilot, bimanual, learned
  correctors.
