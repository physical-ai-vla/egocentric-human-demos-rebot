# Visual Teleoperation MVP: gap audit against the existing code (2026-10-12)

This is a read-only audit. No code, config or robot state was changed. It maps the "Visual Teleoperation MVP" spec onto
what already exists, so that implementation can start later with as little new code as possible.

**Hardware assumed:** left and right wrist fisheye RGB, with an optional wrist IMU used for rotation only; a fixed Orbbec
Gemini 336 RGB-D camera; a dual-arm reBot with grippers.

**Audited code:**
- `~/ego_collector/ego_teleop/` (about 12.5 kLOC).
- `tests/teleop/`: 297 tests pass offline.
- `~/ego_cart20/ego_cart20/umi_fusion/` and `geometry/rotation6d.py`: `tests/test_umi_relpose.py` and `test_umi_fusion.py` both pass, 12 tests.
- `~/robot-cockpit/robot_service_mit.py` and `robot_service.py`, read only.
- The memory notes listed in the task.

**Status labels:**

| Label | Meaning |
|---|---|
| DONE | Usable as is. |
| PARTIAL | The mechanism exists but needs extending. |
| MISSING | Nothing exists yet. |
| CONFLICTS | The existing convention contradicts the spec, or would silently break it. |
| UNVERIFIED | I could not confirm this from code or data. |

**Repository state.** Much of `ego_teleop` is uncommitted (`git status`: `config.py`, `rebot_client.py`,
`episode_logger.py`, `wrist_pose_provider.py` and others are modified; several docs are untracked). `~/ego_cart20` has no
commits at all (`main` has no commits yet). Line numbers below refer to the working tree on 2026-10-12.

---

## 1. Spec item → existing code → status → change

### 1a. Pose input, frames, mapping

**World frame = table (+Z normal, +X left→right, +Y forward)**
- Existing: no table frame anywhere in `ego_teleop`. `transforms/frames.py:4-6` defines W as the VIO world (arbitrary per session) and RB as the reBot base. The UMI fusion world is the Orbbec "world (table) frame" (`ego_cart20/umi_fusion/fusion.py:36`).
- Status: **MISSING**
- Change: add `T_RB_Wtable` (rigid, from config) and a frame name `TW`. Use it only for the workspace and table-collision checks and for logging. Relative actions are invariant to it (see `ego_cart20/geometry/canonical_frame.py:9-12`). The memory note `hra-shared-base-frame` mentions an earlier checkerboard `T_base_cam` (table z −54.3 mm, tilt 2.9°) in `~/c8/hra_red/global_calib`. **That directory no longer exists** (`~/c8` contains only `robotlike`), so this is UNVERIFIED. Whether the Orbbec "table" axes match the spec axes is also UNVERIFIED.

**Human/robot init on table markers X_L / X_R**
- Existing: none. Engagement anchors at the *current* poses (`retarget/arm_relative_se3.py:50-53`, `robot/coordinator.py:150-153`).
- Status: **MISSING** (P1)
- Change: the CALIBRATING step should move each robot TCP to X_L / X_R (a joint-space ramp through the bridge `/execute_step`) and engage only after the human wrist is still at the marker (`tracking/vio_stable.py` gate, reused). The anchor math is unchanged.

**Pose input interface**
- Existing: the `WristPoseProvider` protocol and `WristPose` (`tracking/interfaces.py:19-56`). `EstimatorWristPoseProvider.ingest_estimate` (`tracking/wrist_pose_provider.py:85-115`) turns any `PoseEstimate` into a graded `WristPose`. A replay path already exists: `_ReplayEstimator`, `vi_provider_from_parquet`, `vi_estimates` (`tools/f4_fusion.py:59-87`).
- Status: **DONE** for live and VIO; **PARTIAL** for UMI replay.
- Change: move the f4_fusion replay helpers into `tracking/replay_tcp.py` and accept the UMI fusion columns (`pose_source`, `confidence_position`, `confidence_rotation`). `fuse_side()` returns an in-memory `SideFused` and **has no on-disk format yet** (its only callers are tests), so a writer must define one (see §3).

**Relative teleop `T_target = T_R0 (T_H0⁻¹ T_H(t))`**
- Existing: `RelativeSE3Retargeter.update` (`retarget/arm_relative_se3.py:66-80`): `delta_H = inv(T_H0) @ T_H`, then `T_R0 @ mapper.map_delta(delta_H)`. Test: `tests/teleop/test_arm_relative.py:21`.
- Status: **DONE**
- Change: none to the math. Note that the delta is a **body-frame** delta (in the anchor TCP frame), exactly as the spec writes it.

**Translation scaling and R_align**
- Existing: `HumanRobotFrameMapper` (`transforms/frames.py:64-92`): signed-permutation conjugation, per-axis `translation_scale`, `rotation_scale`. Config: `configs/ego_teleop/frames.yaml`. Test: `test_frames.py:15-44`.
- Status: **PARTIAL**
- Change: R_align must be a **signed permutation** (`basis_matrix`, `frames.py:24-34`). That suffices for the dataset→FK fix, `Ry(90)` (see §2.1). An arbitrary measured R_align needs a general-rotation option, about 15 lines. Scaling is applied in the anchor TCP frame, which is equivalent to world scaling only when it is isotropic.

**Human reach normalisation**
- Existing: only `translation_scale` (global, fixed).
- Status: **MISSING** (P2)
- Change: a per-operator scale in config, from an arm-span calibration. Keep it separate from R_align, the same rule as the Aero "s_op vs retarget" split in memory.

**Jump rejection**
- Existing: `TrackingSupervisor.grade_estimate` (`tracking/wrist_pose_provider.py:40-56`): `max_jump_m` 0.15 m and `max_jump_deg` 25° mark the frame LOST, with recovery hysteresis `recover_after_n_ok`. `LocalPoseContinuity` absorbs announced map corrections (`tracking/local_pose.py:63-85`). Test: `test_tracking_dropout.py:59`.
- Status: **DONE**
- Change: tighten the thresholds for the table-scale task. 0.15 m per frame is large at 30 Hz.

**Tracking-confidence gate → HOLD**
- Existing: `TrackingHealth` OK / DEGRADED / LOST (`interfaces.py:12-16`). Low confidence maps to DEGRADED (`wrist_pose_provider.py:50`). LOST holds the last target with `speed_factor=0` (`arm_relative_se3.py:73-74`). DEGRADED gives speed ×0.3 (`:79`). Age goes to DEGRADED after 80 ms and LOST after 250 ms (`:58-61`). Tests: `test_arm_relative.py:47`, `test_coordinator.py:81`.
- Status: **DONE** (HOLD); PAUSE is **MISSING**.
- Change: add an automatic LOST-longer-than-`pause_after_ms` transition to PAUSED in the coordinator. The UMI fusion confidences (1.0 / 0.8 / 0.5) need explicit thresholds: with `min_confidence_ok=0.5` and a strict `<` test, IMU "WARN" frames (conf_r 0.5) currently grade **OK**.

**Filters (EMA pos 0.2, SLERP rot 0.15, gripper 0.3)**
- Existing: none on the arm path. The Aero channel LPF exists only for the hand (`retarget/aero_retarget.py`, `aero_backends.py` `AeroCommandLimiter`). `interp_pose` is re-exported in `transforms/se3.py:4-5`.
- Status: **MISSING**
- Change: new `robot/filters.py` with a pose EMA+SLERP filter and a gripper EMA, applied to `T_target` *before* `ArmSafetyPipeline.check`. On clutch release or re-anchor, reset the filter to the robot pose.

### 1b. Safety, IK, gating

**Velocity / acceleration / jerk limits (start 0.05–0.10 m/s, 15–30 °/s)**
- Existing: `CartesianLimiter.step` (`robot/safety.py:46-70`): linear velocity, angular velocity and linear acceleration. Config `teleop.yaml` safety: 0.10 m/s, 30 °/s, 0.8 m/s². Tests: `test_safety.py:12-32`.
- Status: **PARTIAL**
- Change: add angular acceleration and a jerk limit (about 30 lines). Set the defaults to 0.05 m/s and 15 °/s for Stage 1. The MIT bridge adds its own joint-space profile on top (45 °/s, 180 °/s², `robot_service_mit.py:32-34`).

**Workspace clamp**
- Existing: `WorkspaceBox` with `clamp | hold` (`safety.py:14-23, 85-90`), one box in RB (`teleop.yaml`). Test: `test_safety.py:32`.
- Status: **PARTIAL**
- Change: per-arm boxes, and a table-plane floor in TW with margin, computed on the **jaw tip**, not the FK TCP (see §2.5).

**Single-arm IK**
- Existing: `PyrokiIkSolver` (`robot/rebot_client.py:216-256`, handumi `rebot_b601`, continuity IK) runs only in `~/xvla-mac/bin/python`. `HttpRebotClient.command_ee_pose` (`:171-190`) seeds from the previous command (`:159-161`). Test: `test_rebot_client_units.py:81`.
- Status: **DONE**
- Change: none. Run from the xvla-mac interpreter.

**Feasibility gate (IK, joint limits, self-collision, table, margin) → HOLD**
- Existing: IK residual (`rebot_client.py:178-179`), NaN check (`:176-177`), `JointLimits` limits plus a per-command step check (`:111-122`, refuse not clamp). Test: `test_rebot_client_units.py:92`. Self-collision and table collision: none.
- Status: **PARTIAL**, with a **CONFLICT** on limits (§2.6).
- Change: add `robot/feasibility.py` for the table check now, and self-collision later via pyroki collision, whose availability is UNVERIFIED. **Bug-level gap:** when the IK or guard refuses, `ArmSafetyPipeline._prev_cmd` has already advanced (`safety.py:91`) and so has `coordinator._prev_cmd_T` (`coordinator.py:181-182`). HOLD then does not mean "last executed": the limiter keeps walking an unexecuted pose, and on recovery the joint-step guard either refuses forever or allows a catch-up jump. Fix: roll back to the last *executed* command on any `arm_exec.ok == False`.

**Deadman (released = HOLD)**
- Existing: none. `hud_robot.py` has operator HOLD and E-STOP buttons for the M0 jog only (`tools/hud_robot.py:1-16, 142-160`).
- Status: **MISSING**
- Change: add `robot/deadman.py` (an input protocol plus keyboard/HTTP/scripted sources). The coordinator gates sending on it. On release it holds (no POST) and resets the limiter to the executed pose. On re-press it re-anchors like a clutch release (zero jump by construction, `arm_relative_se3.py:58-60`).

**E-STOP (FAULT, manual reset)**
- Existing: `coordinator.fire("estop")` calls `robot.estop()` (POST `/estop`), then disengages and enters ESTOP. `clear_estop` returns to ROBOT_READY (`coordinator.py:58-59, 141-142`). The bridge `/estop` holds both arms and latches; `/clear_estop` is manual (`robot_service_mit.py:670-685`). Test: `test_coordinator.py:81`.
- Status: **DONE**, with a caveat.
- Change: the bridge no longer auto-latches on a guard trip; it re-holds in place and logs `auto_events` (memory `rebot-mit-bridge`). Teleop must poll `/status` `auto_events` and `parked`, and enter FAULT itself.

**Clutch (re-anchor without robot motion)**
- Existing: `clutch()`, then `release()` re-anchors both anchors (`arm_relative_se3.py:55-60`). Coordinator events at `coordinator.py:54-55, 144-146`. Tests: `test_arm_relative.py:34`, `test_coordinator.py:108`.
- Status: **DONE** (single arm).
- Change: clutch per arm in bimanual mode.

**State machine IDLE / CALIBRATING / READY / TELEOP / HOLD / FAULT**
- Existing: `TeleopStateMachine` with INIT … VIO_STABLE, ROBOT_READY, ARMED, TELEOP, CLUTCHED, PAUSED, ESTOP (`coordinator.py:25-80`). HOLD is a per-tick `hold_reason`, not a state. Test: `test_coordinator.py:44`.
- Status: **CONFLICTS** in naming only.
- Change: do not fork it. Provide a spec-name view (IDLE=INIT, CALIBRATING=SENSOR_READY…VIO_STABLE, READY=ROBOT_READY/ARMED, TELEOP=TELEOP, HOLD=TELEOP with `hold_reason`, or CLUTCHED, FAULT=ESTOP) for the UI and logs. The VIO readiness requirements (`imu_online`, `vio_stable`) must become optional for an RGB-only or replay source; today `arm` requires `imu_online` (`:52`).

### 1c. Modes, dual arm, gripper

**Per-arm enable; modes Preview / single / bimanual**
- Existing: `CoordinatorConfig.arm_enabled` and a single `side` (`coordinator.py:118-124`, `rebot.yaml side`). For preview: `MockRebotController` (`rebot_client.py:197-213`), `f4_fusion --stage virtual` (`tools/f4_fusion.py:6, 171`), and the dry-run `OutboundGuard` (`tools/m0_eef_jog.py:66-91`), which reads the real `/observe` but never POSTs.
- Status: Preview **PARTIAL** (mock robot has no IK); bimanual **MISSING**.
- Change: Preview = `OutboundGuard(live=False)` plus real IK, for kinematic feasibility. Bimanual needs one 14-vector writer (§2.4).

**Dual-arm IK**
- Existing: pyroki solves both arms in one call (`rebot_client.py:253`, `left_pose=…, right_pose=…`), but the wrapper solves one side.
- Status: **PARTIAL**
- Change: add `BimanualIk` that calls `solver.ik(q, left_pose, right_pose)` once.

**Gripper mapping 0 = closed, 1 = open**
- Existing: the jaw is only *held* (`grip_hold_command`, `rebot_client.py:43-45, 168`). Units: observed raw 0 = closed, −270 = open; command 0..45; observed = command × −6 (`:4-8, 29-30`).
- Status: **MISSING** as a command path; units **DONE**.
- Change: `cmd = 45·g`, so `raw ≈ −270·g` ("mapping A", memory `rebot-gripper-semantics-r150`). The MIT bridge's internal normalisation is the **opposite** polarity (0 = open; `robot_service_mit.py:20-21, 68-73`). Never use it in teleop.

**Bimanual relative preservation monitor**
- Existing: none. `inter_gripper_relative_pose` exists (`ego_cart20/umi_fusion/relpose.py:25-27`).
- Status: **MISSING** (P1)
- Change: log the human T_LR and the robot-target T_LR each tick and alarm on divergence. Valid only if both human wrists are in one world (true for UMI fusion, which uses the Orbbec shared world; false for per-wrist VIO).

**Collision-aware retargeting**
- Existing: none.
- Status: **MISSING** (P2)

### 1d. Timing, UI, recording, export

**Estimator at 10–30 Hz decoupled from a 100+ Hz controller with interpolation**
- Existing: the coordinator is a single 30 Hz tick (`coordinator.py:1-3, 163-205`). Sensor threads push and the control loop pulls (`tracking/fused_live.py:1-15`). The MIT bridge runs 100 Hz internally with `MIT_GOAL_INTERP=1` segment tracking (`robot_service_mit.py:36-43, 498-516`).
- Status: **PARTIAL**
- Change: for the MVP, keep 30 Hz Cartesian targets and let the bridge interpolate at 100 Hz (no new controller). A separate 100 Hz Cartesian loop is P2.

**End-to-end latency timestamps per stage**
- Existing: `WristPose.timestamp_ns` (capture), `TeleopCommand.timestamp_ns`, and `ExecutionReport.command/execution_timestamp_ns` (`rebot_client.py:83-95`). `RebotState.timestamp_ns` is the host time *when /observe returned* (`:153`), not the acquisition time. The f5 HUD has a latency strip.
- Status: **PARTIAL**
- Change: add stage timestamps (capture, pose ready, target, IK done, POST sent, bridge accepted) to `TeleopCommand.to_row`. `/observe` JPEG-encodes three cameras on every call (`robot_service_mit.py:437-453`), so robot-state age is inflated. Use the bridge `t_wall`, and consider a joints-only endpoint (bridge change: out of scope, flag only).

**UI: global view + both wrists + status (pose source, confidence, IK, deadman, clutch, latency)**
- Existing: `tools/f5_teleop_hud.py` (fusion overlay, latency strip, engaged/clutch), `tools/hud_robot.py` (robot safety panel), and `tools/rgbd_arm_hud.py`.
- Status: **PARTIAL** (P1)
- Change: one page reusing the f5 HUD served-page pattern. Add a deadman lamp, per-arm IK verdict and `pose_source`.

**Recording: global RGB/depth, wrist RGB, actual robot TCP/joints/gripper, human targets, robot targets, executed action**
- Existing: `TeleopEpisodeLogger` (`recorder/episode_logger.py:43-187`): pose streams, `rebot_state`, `rebot_command` (includes `human_d*` and the commanded pose), events, calibration. `HeadRgbdWriter` (`:190-217`) writes global RGB-D. Wrist RGB goes through the handumi_collector MCAP/MP4 recorder (`:7`). Tests: `test_coordinator.py:127`, `test_pose_storage_contract.py`.
- Status: **PARTIAL** (P1)
- Change:
  - `rebot_state` rows carry only the teleoperated side (`RebotState.to_row`, `rebot_client.py:75-80`). Log both arms (`joints_deg_all` is already present) and both FK TCPs.
  - Add a `human_tcp_target_{side}` stream (the raw T_W_TCP used).
  - Add `metadata.data_source`.

**Export with the UMI action contract `[dSE3_L, dSE3_R, g_L, g_R]` and observation T_LR**
- Existing: ego_teleop exports a per-tick **incremental** `arm_action6 = [dxyz, rotvec]` of `inv(T_prev_cmd)·T_cmd` (`arm_relative_se3.py:35-37`, `coordinator.py:89, 181`). The UMI contract is `temporal_action_chunk` (`relpose.py:35-37`, UMI-relative, not incremental) plus `pose9` (xyz + rot6d rows, `relpose.py:20-22`, `rotation6d.py:38-40`).
- Status: **CONFLICTS**
- Change: never export `arm_action6` as the training action. Add `recorder/umi_contract_export.py`, which *imports* `ego_cart20.umi_fusion.relpose` and `geometry.rotation6d` (§3.3).

**Episode `data_source` UMI / VISUAL_TELEOP / ROBOT_AUTONOMOUS**
- Existing: none in either repo (searched both).
- Status: **MISSING**
- Change: a `metadata.json` key written by the logger, and a required field in the export.

**Executed trajectory as label, human target kept**
- Existing: the commanded pose and the human delta are logged; the executed pose (FK of observed joints) is not computed in the log.
- Status: **PARTIAL**
- Change: the export computes FK on observed joints. The label is the executed trajectory; human targets stay a separate stream.

### 1e. Staged tests and acceptance

**Stage 0: offline replay, no robot**
- Existing: `f4_fusion --stage virtual` (replayed VI parquet → coordinator → `MockRebotController`) and `p1_rgbd_arm --stage v1`.
- Status: **PARTIAL**
- Change: add the `tools/v0_replay.py` runner (§3).

**Stages 1–7 (free space → laundry)**
- Existing: M0 jog through the real path (`tools/m0_eef_jog.py`, `--execute --i-am-at-the-robot`).
- Status: Stage 1 **PARTIAL**; Stages 2–7 **MISSING**.
- Change: none until Stage 0 passes.

**Safety acceptance (100+ free-space commands, 0 unsafe jumps or collisions, 100 % HOLD on loss / deadman / IK failure)**
- Existing: unit tests cover the LOST hold (`test_arm_relative.py:47`), the stale-observation hold (`test_safety.py:41`) and guard refusal (`test_rebot_client_units.py:92`). There is no aggregate acceptance harness.
- Status: **PARTIAL**
- Change: add the Stage-0 acceptance test (§3.2), and fix the IK-failure rollback first.

---

## 2. Conventions the MVP must respect (conflicts)

1. **reBot TCP frame differs by 90° (memory `rebot-tcp-frame-90deg`).**
   - The relation is `R_fk = R_dataset @ Ry(90°)`. Constant: `R_DS_TO_FK = Rot.from_rotvec([0, π/2, 0])`, `~/softfold_infer/rebot/infer_core_v4.py:274-276`.
   - The teleop IK and FK (`PyrokiIkSolver`, the same solver as `eef_kin`) are in the **FK** frame.
   - UMI human TCPs are exported in **reBot dataset-TCP axes**: `X_TOOL = F @ C @ CT4` (`ego_cart20/sources/hfold_export.py:46-49, 151, 166`; `handumi_export.py:7-8`). `fusion.py:11` leaves `X_cam_tcp` to the caller, so whether a given T_W_TCP stream already includes `X_TOOL` must be recorded per stream. UNVERIFIED for a fusion-output stream, because none is written yet.
   - Consequence for the relative mapping: `Δ_fk = Ry(90)ᵀ Δ_ds Ry(90)`. This is a signed permutation, so it fits `HumanRobotFrameMapper`. Derived: `translation_axis_map [z, y, x]`, `sign [-1, 1, 1]`. **Pin it with a unit test against `R_DS_TO_FK` before any robot motion.** This exact class of error produced the 577 mm upward runaway.
   - Export must convert executed FK poses back with `R @ R_FK_TO_DS` (`infer_core_v4.py:580`).
2. **TCP point is not the jaw tip.** The pink/pyroki `TIP_X=0.07` TCP sits about 82 mm ahead of the physical jaw tip (memory `hra-shared-base-frame`). The table-collision check and the workspace floor must use the tip (`tip_z = tcp_z + 0.082·sin(pitch)`, from memory, UNVERIFIED here). Relative rotations about different TCP points produce different translations, so the human camera_tcp_v2 TCP and the robot TCP must be the *same physical point* or the bimanual geometry is biased.
3. **Gripper polarity.**
   - Observed raw: 0 = closed, −270 = open (wraps to about 358; `unwrap_grip`, `rebot_client.py:37-40`).
   - `/execute_step` jaw value: a command in 0..45, with observed = command × −6.
   - Spec g (0 = closed, 1 = open) therefore gives `cmd = 45·g` and executed label `g = clip(unwrap(raw)/−270, 0, 1)`.
   - The MIT bridge's own `jaw_norm` is reversed (0 = open). The `humanik_delta` "1 = closed" comment is wrong (memory `rebot-gripper-semantics-r150`).
   - The open offset drifts per power cycle (memory `rebot-gripper-offset-drift`), so executed-gripper labels need a per-session zero check.
4. **One writer, 14-vector, leader frame, radians.**
   - `/execute_step` takes `[L j1..6 rad, L jaw cmd, R j1..6 rad, R jaw cmd]` in the LEADER frame (`FLIP_IDX = (0,1,5,7,8,12)`, `rebot_client.py:28`). `/observe` returns FOLLOWER degrees. `leader_action()` (`:48-55`) is the only exit, and you must never POST by hand (memory `rebot-arm-port-location-mapping`).
   - `HttpRebotClient.build_action` re-sends the **other arm at its observed pose** (`:163-169`). Two single-side clients would fight: each POST resets the other arm's goal to its lagging observation.
   - `OutboundGuard` refuses any change to the other arm or the jaws (`m0_eef_jog.py:84-86`).
   - Bimanual and gripper control need one `BimanualRebotClient` that builds one vector per tick, and a guard variant that permits jaw and other-arm changes within step bounds.
5. **Port / location mapping.** Left and right are set by `LEFT_LOC` / `RIGHT_LOC` USB location IDs (`51642368` / `1257472`). `LEFT_FOLLOWER` is ignored; the bridge refuses to guess (`robot_service_mit.py:92-97`). Re-verify after every power cycle, replug or restart with a one-jaw test before teleop. A swap looks like a "mirrored" retarget bug.
6. **MIT bridge API and limits.**
   - Port 8021, but `rebot.yaml` `base_url` and `test_rebot_client_units.py:125` pin 8020. Use a config override; do not edit the test's meaning.
   - The bridge **silently clips** joints to `SEEED_LIM` in its raw frame (`robot_service_mit.py:54, 82`). It applies `SEEED_DIR` (negating pan, lift, roll), so its raw frame equals the follower frame. That was derived from code, UNVERIFIED on hardware.
   - The teleop `JointLimits` defaults (`rebot_client.py:33-34`) are **wider** than the bridge clip on j1 (±160° vs ±145°), j2 (−180° vs −170°), j4 (−107° vs −80°) and j6 (±180° vs ±130°). An IK solution that passes the teleop guard can be clipped by the bridge, so executed ≠ commanded with no error. Set teleop limits to the bridge limits minus a margin so the teleop refuses first.
   - Other bridge constraints:
     - `/execute_step` returns 409 while a PLAN session is open.
     - A parked arm (`MIT_REST_TORQUE_OFF`) is unparked by the next `/execute_step`.
     - `/gripper_zero` is not implemented.
     - Never `pkill -f robot_service_mit.py`, because it kills the live bridge.
     - Integral-law sag: about 8.9° at the elbow until the integral winds up.
     - One arm's five drives can drop out together at heavy poses.
7. **Quaternion order differs between repos.** ego_teleop and handumi `pose7` use xyzw (`transforms/se3.py:1-3`); ego_cart20 uses wxyz scalar-first (`rotation6d.py:5`). Cross the boundary only as 4×4 matrices.
8. **rot6d = first two ROWS of R** (`rotation6d.py:3-4, 38-40`). Never use columns, and never use the ego_teleop rotvec (`action6`) in exported data.
9. **Storage contract (frozen 2026-09-11).** Live pose streams are `*_live_*` and are never overwritten by refined ones (`episode_logger.py:24-35`; `test_pose_storage_contract.py`). A replayed UMI pose is not "live": declare it with `pose_type`/`causal=false` provenance (`declare_pose_stream`, `:140-149`) under its own stream name, never `wrist_pose_live_*`.
10. **GPU rule.** A live MASt3R pose source runs only on the 5090 via Ray (memory `mast3r-slam-pose-source`, `ray-submit-for-5090`), and never on a 5080. This is relevant to P2 live fusion, not to Stage 0.

---

## 3. P0 implementation plan

Order: Stage 0 offline replay first. No robot motion until its acceptance test passes.

### 3.1 Files

**New: `ego_teleop/tracking/replay_tcp.py` (about 120 LOC)**
- Move `_ReplayEstimator`, `vi_estimates` and `vi_provider_from_parquet` here from `tools/f4_fusion.py:59-87`; f4 then imports them, with no behaviour change.
- Add `load_tcp_stream(path)` for the columns `t_ns, x, y, z, qx, qy, qz, qw, valid, pose_source, confidence_position, confidence_rotation, tool_frame`.
- Map `pose_source == INVALID` to `valid=False` and `confidence = min(conf_p, conf_r)`.
- Build the provider with `T_H_C = I`, because the stream is already a TCP.

**New: `ego_cart20/umi_fusion/io.py` (about 40 LOC)**
- `write_tcp_stream(side_fused, X_cam_tcp, t_ns, out)` writes the parquet above, with the per-row provenance that `fusion.py:13-18` defines and `tool_frame` metadata stating whether `X_TOOL` (dataset-TCP axes) was applied.
- Lives in ego_cart20 so that the UMI side owns its format.

**New: `ego_teleop/robot/filters.py` (about 70 LOC)**
- `PoseFilter(alpha_pos=0.2, alpha_rot=0.15)` (EMA + SLERP via `interp_pose`), `ScalarEMA(0.3)` for the gripper, and `reset(T)`.

**New: `ego_teleop/robot/deadman.py` (about 60 LOC)**
- A `Deadman` protocol with `pressed(now_ns) -> bool`.
- Implementations: `ScriptedDeadman` (replay and tests), `HttpDeadman` (UI button heartbeat; released if no heartbeat within 150 ms), `AlwaysPressed` (preview only, refused when `live=True`).

**New: `ego_teleop/robot/feasibility.py` (about 80 LOC)**
- Table-plane check on the jaw tip in TW, with margin.
- Hooks for self-collision, initially returning `unknown` and logged as such.
- Wraps the existing IK residual and joint checks into one `FeasibilityVerdict`.

**Modify: `ego_teleop/robot/safety.py` (+40 LOC)**
- Angular acceleration and jerk limits.
- Per-side workspace.
- `rollback(T_exec)` so the limiter state equals the last *executed* command.

**Modify: `ego_teleop/robot/coordinator.py` (+80 LOC)**
- Gate on the deadman: on release, hold and roll back; on re-press, re-anchor.
- On `arm_exec.ok == False`, roll back and set `hold_reason=ik_infeasible` or a guard reason.
- LOST for longer than `pause_after_ms` fires `pause`.
- Make readiness optional for an RGB-only or replay source (`imu_online`/`vio_stable` from config).
- Apply the filter before safety.
- Add stage timestamps to `TeleopCommand`.

**Modify: `ego_teleop/robot/rebot_client.py` (+30 LOC)**
- Add `gripper_cmd_from_g(g)` and `g_from_raw(raw)` helpers (mapping A).
- Joint limits from config set to the bridge `SEEED_LIM` minus a margin.
- No change to the deg/rad or leader boundary.

**Modify: `ego_teleop/config.py` and add `configs/ego_teleop/visual_teleop.yaml` (about 60 lines)**
- `base_url: http://127.0.0.1:8021`.
- Filters.
- Limits: 0.05 m/s, 15 °/s, jerk.
- `T_RB_Wtable`.
- `frames`: the Ry(90) conjugation (`[z, y, x]`, `[-1, 1, 1]`) when the input is in dataset-TCP axes.
- `pause_after_ms`.
- Deadman source.

**New: `ego_teleop/tools/v0_replay.py` (about 200 LOC)**
- Pipeline: UMI TCP parquet (one side) → replay provider → filter → retargeter → safety → **preview robot** → logger.
- The preview robot is either `MockRebotController`, or `HttpRebotClient` + `PyrokiIkSolver` behind `OutboundGuard(live=False)` (real `/observe`, real IK, no POST). Run the second from `~/xvla-mac/bin/python`.
- Fault injection: tracking-loss windows, deadman releases, unreachable targets.
- Report: reuse `f4_fusion.jumps`, `f4_fusion.dropouts` and `f4_fusion.return_to_start` (`tools/f4_fusion.py:385-442`).
- Write `data_source=VISUAL_TELEOP` and `stage=0` in metadata.

Total new and modified code is about 800 LOC plus tests. Nothing in tracking, the frame mapper or the retargeter math is rewritten.

### 3.2 Tests (about 300 LOC, all offline, in `tests/teleop/`)

**`test_replay_tcp.py`**
- INVALID rows produce LOST, then HOLD.
- A WARN confidence produces DEGRADED at the configured threshold.
- The f4 replay is byte-identical after the move.

**`test_frame_fix.py`**
- The mapper configured as `[z, y, x]`/`[-1, 1, 1]` equals `Ry(90)ᵀ Δ Ry(90)` for random Δ.
- Assert against the literal `R_DS_TO_FK` (copy the constant, citing `infer_core_v4.py:274`).

**`test_deadman.py`**
- Released means zero POSTs (the `OutboundGuard.sent` count is unchanged).
- Re-press produces no jump (first command equals the observed pose).

**`test_ik_fail_hold.py`**
- An unreachable target is held.
- The limiter rolls back.
- After the target becomes reachable again, the first command lies within `v_max·dt` of the last *executed* pose.

**`test_filters.py`**
- EMA and SLERP step response, rotation stays on SO(3), and reset.

**`test_stage0_acceptance.py`** runs a synthetic UMI-like trajectory through `v0_replay` and requires:
- At least 100 commands.
- Zero per-tick translation steps above `v_max·dt + ε`, and zero rotation steps above `ω_max·dt + ε`.
- Zero commands outside the workspace or below the table + margin.
- HOLD on 100 % of injected loss, deadman-release and IK-failure ticks.

**`test_umi_contract_export.py`**
- For a synthetic bimanual executed trajectory, the export's actions equal `relpose.temporal_action_chunk` + `pose9` exactly.
- T_LR equals `inter_gripper_relative_pose`.
- g lies in [0, 1] with 0 = closed.

### 3.3 Recorder and export reuse of ego_cart20 (P1, designed now so P0 logs carry what it needs)

The logger change: `rebot_state` records **both** arms (14 joints in degrees, as observed) plus `t_ns` and the bridge `t_wall`. `rebot_command` keeps `human_d*` and the commanded pose. A new `human_tcp_target_{side}` stream holds the input T_W_TCP. `metadata.data_source ∈ {UMI, VISUAL_TELEOP, ROBOT_AUTONOMOUS}`.

`recorder/umi_contract_export.py` (about 150 LOC) proceeds in six steps:
1. Run FK of the observed joints per side (pyroki, FK frame) to get T_RB_TCP_fk. Convert with `R @ R_FK_TO_DS` to dataset-TCP axes, matching the UMI `X_TOOL` convention.
2. Resample both sides onto the dataset timeline with `ego_cart20.umi_fusion.relpose.resample_poses` (linear + SLERP, gap-aware validity).
3. Action: `temporal_action_chunk(T_side, t, H)` → `pose9` per side, plus `g_side = clip(unwrap(raw)/−270, 0, 1)`, resampled with a nearest or zero-order hold. The contract layout is `[dSE3_L, dSE3_R, g_L, g_R]`.
4. Observation: `inter_gripper_relative_pose(T_L, T_R)` → `pose9`. Both arms share RB, so T_LR is well defined; that is not the case for per-wrist-VIO human data.
5. Keep human targets as a separate, non-label stream. The label is the executed trajectory.
6. Import ego_cart20 and never copy it (its docstring is "the ONE implementation"). It is numpy-only, and its tests pass under `ego_collector/.venv`, so a path dependency or editable install suffices. **Commit ego_cart20 first** so the contract has a version to pin.

---

## 4. Top risks

1. **Frame-convention runaway.** A human TCP in dataset axes against IK in the FK frame reproduces the Ry(90) runaway: a constant-direction drift that every offline relative check hides, because a common rotation cancels in relative labels. Mitigation: `test_frame_fix.py`, plus the Stage-0 check that a pure human +X motion produces robot +X in TW.
2. **UMI fusion is offline and non-causal.** It uses a whole-episode Sim3 / r_c fit (`fusion.py:78-109`) and a two-ended gyro bridge (`:121-139`). Stage-0 smoothness and confidence overstate what a live source delivers. There is no causal real-time equivalent yet. Live MASt3R needs the 5090 via Ray, which adds network latency; `tracking/backends/mast3r_live.py` exists but has not been validated for this rig (UNVERIFIED).
3. **HOLD is not truly "hold" after a refused command.** The limiter and `_prev_cmd_T` advance on IK or guard failure (`safety.py:91`, `coordinator.py:181-182`). This is a stuck-or-jump hazard on recovery; fix it before Stage 1.
4. **Silent divergence between commanded and executed motion.** Causes: bridge joint clipping narrower than the teleop limits, integral-law sag, the 82 mm TCP-vs-tip offset, and gripper zero drift. Labels must come from observed joints, and the teleop limits must be stricter than the bridge.
5. **Bimanual plumbing.** The single-side client re-sends the other arm at its lagging observation. A second client would fight the first; one 14-vector writer is mandatory before dual-arm work. The port-location swap risk remains for every session.
6. **Latency and robot-state staleness.** `/observe` encodes three JPEGs per call. `RebotState.timestamp_ns` is the response time. The 30 Hz targets plus the bridge segment interpolation are adequate for 0.05 m/s but unmeasured; per-stage timestamps are needed before raising limits.
7. **Missing collision models.** Self-collision and table collision are absent, and pyroki collision support is UNVERIFIED. Until they exist, Stages 3+ rely on workspace boxes and a jaw-tip table floor only.
8. **Unversioned code.** ego_cart20 has no commits and much of ego_teleop is uncommitted, so the shared contract can drift silently between the UMI and teleop sides.
9. **Unverified table/world frame.** Spec axes vs Orbbec "table" axes vs `T_RB_Wtable` are not established, and the earlier HRA calibration directory is gone. Marker-based init (X_L/X_R) needs a new, measured calibration.
