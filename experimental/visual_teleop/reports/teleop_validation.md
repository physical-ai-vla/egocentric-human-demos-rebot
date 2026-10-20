# Teleop validation — 2026-10-14

## Spec §41 progress

| # | step | status |
|---|---|---|
| 1–2 | audit / inventory | done |
| 3 | transactional limiter bug | done (fixed 2026-10-13 in `ego_teleop`; re-implemented for both arms) |
| 4 | coordinate adapter tests | done |
| 5 | both IMUs + AHRS | **partial** |
| 6 | camera↔IMU offset | **partial** |
| 7 | real-time orientation fusion | done (offline) |
| 8 | fingertip detector interface | done |
| 9 | pinch → gripper | done |
| 10 | gripper-only test | **not run** (needs operator + robot) |
| 11 | Stage 0 offline replay | done (synthetic) |
| 12–16 | robot stages, dataset | **not started** |

**Step 3.** `ego_teleop` was fixed on 2026-10-13. In the new controller, every per-arm state (limits, filter, gripper,
q_exec) is committed only after the bridge acknowledges. All of it rolls back on a robot reject (test
`test_robot_reject_rolls_back_every_state`).

**Step 4.** The adapter is tested for identity, ±X/Y/Z, ±90° about each axis, permutations, composition and NaN.

**Step 5.** The streaming Mahony AHRS is tested (bias, roll/pitch, yaw following the gyro). The live bring-up tool is
ready, but the only attached board, XIAO `4A65C2BC893657D2`, disconnected from USB before it could be read. It is unknown
which wrist it belongs to.

**Step 6.** The estimator and grading are tested: synthetic 23 ms is recovered within 0.5 ms. It has not been run on real
data. The hub `sync` command is not wired yet, so for now record a wrist wave and run `time_sync` offline.

**Step 7.** IMU orientation is propagated at the IMU rate. Init yaw alignment reports the residual tilt. Vision/tag yaw
correction uses a rejection gate. Position comes from RGB-D with hold/HOLD on dropout. TRACKING/WARN/HOLD are tested.

**Step 8.** `FingertipDetector` has student and scripted backends. MediaPipe is not used. The node runs live on the
right Arducam at 28.7 Hz.

**Step 9.** Pinch control has hysteresis, a dropout hold then PAUSE, rate limit + LPF, and polarity 0 = closed. It is
wired end to end; the control process was tested against the mock robot.

**Step 10.** Not run. It needs the operator wearing a wrist cam and the robot connected.

**Step 11.** Synthetic replay with toy and pyroki IK; see below. There is no recorded real sensor log yet.

**Steps 12–16.** Blocked on the calibrations in `calibration_report.md` and on operator presence.

## Stage 0 (offline replay) — synthetic, both arms, 57 s scripted program

The program covers ±X/Y/Z at 4 cm and ±roll/pitch/yaw at 25° per hand, two pinches, a 0.6 s visual dropout, a 2 s
clutch with 5 cm repositioning and a 1 s deadman release. The IMU path is real: the AHRS runs on simulated 200 Hz IMU
data with bias and noise, and the fusion recovers an arbitrary IMU heading.

| run | direction segments (cos > 0.9) | jumps > v_max·dt | IK feasible | gripper transitions | QA flags |
|---|---|---|---|---|---|
| toy IK | 36 / 36 | 0 | 100 % | 4 / 4 per side | none* |
| pyroki IK, rotation about the TCP | **32 / 36** | 0 | 100 % | 4 / 4 | none |
| pyroki IK, rotation about the wrist (8 cm lever) | 36 / 36 | 0 | 100 % | 4 / 4 | none |

\*After fixing the QA definition: "valid tracking" now counts WARN as valid. IMU-only orientation is no longer a WARN
when no vision yaw source is configured.

Init alignment residual tilt was 0.02–0.04°. The executed per-tick maximum was 1.07 mm and 0.42°, which is the
v_max = 0.08 m/s and ω_max = 25°/s limit at WARN speed.

### Finding 1 — pitch about TCP x/y is tracked weakly by the continuity IK (kinematics, not a frame flip)

A rotation about the TCP's own x or y needs large shoulder and base motion, because the wrist joints sit 0.25 m behind
the TCP. Directly solving for 20° about x from home takes j1 −34°, j2 −32° and j5 −44°, and only reaches 17°. Per tick,
the IK's rest cost (`rest_weight` 20 = `ori_weight` 20) and the 2.9° joint-step guard keep the achieved rotation small.

- Pure rotation about the TCP's x: the arm rotated about 2° of the commanded ~20°, on a skewed axis (cos 0.65–0.76).
- When the human rotates about the wrist, as real hands do: direction is correct (cos ≥ 0.97), but the magnitude is
  about 1/3–1/2 of roll (0.09–0.16 rad vs 0.27 rad in the same window).

Roll about z (j6) and all translations track fully.

**Options (not applied; this is a decision for you):**
1. Lower the IK `rest_weight` for teleop. The REL16-v3 note says rest_weight sets the IK ceiling.
2. Accept it and use `rotation_scale` < 1 on x/y only.
3. Pivot human rotations at a point nearer the robot wrist.

Option 1 is the cleanest. It needs a Stage 0 rerun with a branch-jump check, because a weaker rest term makes branch
flips more likely.

### Finding 2 — FK TCP axes

FK TCP axes are z = tool and y = jaw, not "x = tool". See `calibration_report.md`. This is now pinned in the config and in
the collision proxy.

## Stage 1 control chain (mock robot, no hardware)

Run against the mock robot with `live_teleop --stage 1 --mock-robot`. Messages came over UDP from a fake pinch
publisher, and the dashboard endpoints were driven over HTTP.

- Before INIT: NOT_INITIALISED.
- After INIT, with the deadman held:
  - a pinch closes both grippers (g 1 → 0) with the rate limit applied;
  - deadman release gives HOLD even with open intent;
  - re-press opens.
- E-STOP latches, and RESET requires INIT again.
- Control loop p99 was 3.5 ms. 77 commands were sent out of 735 ticks; it sends only when something changes.

## Not done / open

- **Hardware runs.** Only the right-wrist Arducam was usable today. That unit is currently mounted on the HandUMI gripper,
  not worn as a bare-hand wristband. The student was distilled from a wristband mount, so re-record and re-distill if the
  mount changes.
- **Missing hardware.** The left wrist cam (FisheyeCamLeft) and both IMUs were not connected. The Orbbec needs `sudo`.
- **Robot not connected.** The MIT bridge on :8021 was up but reported `connected: false`.
- **Live sync command in the hub.** It should buffer palm rotation from `aero_mocap.palm_frame` and call
  `time_sync.estimate_side`.
- **Hardware E-STOP.** Its state is read from the bridge `/status` estop flag. Whether a physical E-STOP button is wired
  into that flag is **unknown**; please confirm.
- **LeRobot v3 parquet writing.** It waits for Agent B's feature schema. `contract.npz` carries `action20`, which is the
  executed trajectory, rot6d as rows, g 0 = closed.
- **Elbow posture hint** (spec §21). Optional, not implemented.
- **Real self-collision.** Only the TCP and wrist-segment proxy plus the no-crossing rule are checked.

## Shared action contract (Agent A, `~/laundry_system/contracts`)

The export imports the shared contract module; it does not copy it. The contract fixes:

- 20-D action, chunk relative to T(t)
- rot6d = first two **rows**
- gripper 0 = closed
- dataset-TCP axes, x = approach (= reBot FK z)
- H = 32 at `ACTION_DT_S` = 0.05 s (20 Hz)

Collection still runs at the robot execution rate. Resampling to the 20 Hz contract clock happens only at export.
Every exported chunk passes `validate_action_chunk`. Unit test `test_export_equals_shared_contract_encoder` shows the
export equals `encode_action_chunk` on the same executed poses (max difference < 1e-6).

Agent B's "column-interleaved rot6d" refers to the Piper source data, not to this contract.
