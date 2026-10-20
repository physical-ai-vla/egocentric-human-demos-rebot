# Right-arm single-arm MVP — runbook (2026-10-14)

**Goal:** right hand motion moves the reBot right arm pose; right thumb/index pinch drives the right gripper.

**Hardware:**
- right wrist unit: Arducam 1080P Low Light + XIAO Sense Plus IMU `63ACF722D6DCE23F`
- fixed Orbbec
- MIT bridge on :8021

The left wrist stays out of scope. It gets a second XIAO later, and the same pipeline is copied to it.

All commands below run from `~/ego_collector`.

## 0. Before every session

1. **MIT bridge.** Check that the bridge is connected: `curl :8021/status` should show `connected: true`.
2. **One-jaw test.** Command only the right jaw and confirm that only the right jaw moves (memory rebot-arm-port-location-mapping).
3. **Right XIAO serial.** `.venv/bin/python -m visual_teleop.tools.imu_bringup --list` should show `63ACF722D6DCE23F`.

## 1. One-time calibrations, right side

Each one writes `visual_teleop/config/calibration.yaml`.

| # | command | what you do |
|---|---|---|
| 1 | `.venv/bin/python -m visual_teleop.tools.calib_right imu-mount --side right` | Pose A: palm down, fingers forward, horizontal. Pose B: fingers straight down. Hold each still. |
| 2 | `.venv/bin/python -m visual_teleop.tools.calib_right sync --side right --seconds 12` | Twist and wave the right wrist for 12 s, with the Arducam looking at a textured scene. |
| 3 | `sudo .venv/bin/python -m visual_teleop.tools.calib_right rgbd-level` | Do not touch the fixed camera for 3 s. |

Once these three are done, `config.unmeasured(c, ('right',))` returns `[]`. Until then, Stage 2 and above refuse to start.

**Init yaw is not a calibration.** At every INIT, push the right hand about 10 cm forward along the fingers, palm down,
and back, all within 3 s. RGB-D supplies the push heading. The IMU checks that the push was really along the fingers, so a
sideways push is rejected instead of producing a yaw 90° off.

## 2. Run (three terminals)

```bash
~/finger_distill/.venv/bin/python -m visual_teleop.tools.fingertip_node --sides right
sudo .venv/bin/python -m visual_teleop.tools.sensor_hub
~/xvla-mac/bin/python -m visual_teleop.tools.live_teleop --stage 2 --arm right   # PREVIEW first, then add --live
```

Open http://127.0.0.1:8730.

| key | action |
|---|---|
| SPACE (held) | deadman |
| C | clutch |
| Esc | E-STOP |

The INIT button runs the push window, then anchors.

## 3. Order (spec §30, right only)

1. **Handedness check.** Raise the right hand and confirm the dashboard shows the RIGHT hand. If it shows LEFT, flip
   `cameras.rgbd_selfie_mirrored`. Do not proceed until this is right.
2. **Stage 2, preview** (no `--live`). Move the hand in +X/+Y/+Z and roll/pitch/yaw. The **target TCP** must move the same
   way. Then try clutch, deadman release and hand out of view; the robot must HOLD.
3. **Stage 1, live:** `--stage 1 --arm right --live`. Grippers only. Test open, close, fast pinch, slow pinch, and hand
   lost (the jaw must not toggle).
4. **Stage 2, live:** `--stage 2 --arm right --live`. Start with small free-space motions. Limits stay at 0.08 m/s and
   25°/s.

## Stop conditions

Stop on any of:

- the arm moves opposite to the hand
- an orientation jump
- the jaw toggles while the hand is lost
- the deadman does not hold

Fix the frame, calibration or logic, rerun Stage 0, and only then return to live. Never add an offset (spec §44).
