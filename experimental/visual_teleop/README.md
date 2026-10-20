# visual_teleop — reBot bimanual visual teleoperation (Agent C)

Fixed RGB-D + wrist RGB + wrist IMU → relative bimanual SE(3) → reBot IK + safety → robot. The robot-executed trajectory
is recorded as `VISUAL_TELEOP` data in the common 20-D action contract.

This package is new (2026-10-14). It reuses the tested pieces of `ego_teleop/` and `handumi_collector/` rather than
copying them:

- Cartesian limiter
- joint limits and feasibility gate
- bimanual 14-vector and leader-frame conversion
- gripper polarity
- RGB-D hand front-end
- Teensy/XIAO IMU driver
- clock fit and camera↔IMU cross-correlation
- UMI contract exporter

## Processes

The required packages live in three interpreters, so the system runs as three processes. They share
`time.monotonic_ns()` as the host clock.

| process | interpreter | does |
|---|---|---|
| `tools/fingertip_node.py` | `~/finger_distill/.venv` (ultralytics 8.4) | wrist cams → student fingertips → pinch metric (UDP 47811) |
| `tools/sensor_hub.py` | `ego_collector/.venv` (`sudo`, Orbbec) | IMUs → AHRS; Orbbec → wrists in W; fusion → hand (UDP 47810) |
| `tools/live_teleop.py` | `~/xvla-mac` (pyroki) | pinch hysteresis, dual-arm controller, bridge, recorder, dashboard :8730 |

```bash
cd ~/ego_collector
~/finger_distill/.venv/bin/python -m visual_teleop.tools.fingertip_node
sudo .venv/bin/python -m visual_teleop.tools.sensor_hub
~/xvla-mac/bin/python -m visual_teleop.tools.live_teleop --stage 1            # PREVIEW: real /observe, no POST
~/xvla-mac/bin/python -m visual_teleop.tools.live_teleop --stage 1 --live     # grippers move (operator present)
```

**Offline, with no hardware:**
- `.venv/bin/python -m visual_teleop.tools.stage0_replay --synthetic`
- `--ik pyroki` runs the same replay with real IK; use `~/xvla-mac`.
- `live_teleop --mock-robot` runs the control loop without a robot.

## Stages (spec §30) and what gates them

| stage | moves | refuses unless |
|---|---|---|
| 0 | nothing | — |
| 1 | grippers only (`arms_enabled` false) | `--live`, deadman held, INIT, no E-STOP |
| 2/3 | one arm (`--arm`) + its gripper | every calibration in `config.unmeasured()` measured |
| 4 | both arms | same |

## Control law (one tick, `robot/dual_arm_controller.py`)

```
human T_W_H (fusion) -> relative retarget (adapter + scaling) -> filter.propose -> limits.propose (NaN, jump, workspace, v/a)
  -> IK seeded with the last EXECUTED q -> feasibility (residual, bridge limits, joint step) -> inter-arm check (both HOLD)
  -> gripper (hysteresis -> rate limit -> LPF) -> ONE /execute_step vector -> accepted? -> ONLY THEN commit every state
```

Mode precedence: ESTOP > NOT_INITIALISED > DEADMAN_RELEASED > CLUTCH > HOLD (tracking/IK/safety, both arms) > FOLLOW.

**Frames (measured 2026-10-14):** the pyroki IK TCP is defined as follows.
- `z` is the tool/approach axis: it points from the wrist to the fingertips.
- `y` is the jaw opening axis.
- j6 rolls about `z`.

The human virtual TCP `H` is defined with the same axes, so the adapter is the identity. See `config/teleop_rebot.yaml`.

**Code structure:** see `tests/vteleop/` for the unit tests (spec §37).

**Reports:** see `reports/` for the calibration, latency and validation reports.
