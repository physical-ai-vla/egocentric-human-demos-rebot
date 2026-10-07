# A0 — official TetherIA Aero Hand Open stack audit (2026-09-11)

Source: `~/aero-hand/aero-hand-open` @ `d17c688`. Everything below is **pinned by**
`tests/teleop/test_aero_official_conventions.py`; if TetherIA changes the upstream stack those tests fail instead of
the numbers silently drifting. Nothing in this file may be "remembered" — it was read out of the source.

## Official pipeline

```
webcam (cv2, mirrored)                         webcam_mocap/webcam_mocap.py       30 Hz timer
  → mediapipe Hands  .multi_hand_world_landmarks (21, metres-ish)
  → x negated (undo the mirror), palm-local transform
  → 21 → 25 keypoint expansion (wrist duplicated as each finger CMC)
  → EMA (alpha 0.7) on the 25×3 array
  → /webcam_mocap_data   aero_hand_open_msgs/HandMocap {header, side, geometry_msgs/Pose[25]}
  → dex_retargeting (DexPilot, aero URDF)      aero_hand_open_retargeting/dex_retargeting_node.py
  → per-actuator scale/offset, clip to AeroHandConstants
  → /{side}/joint_control  aero_hand_open_msgs/JointControl {header, float32[16] radians}
  → aero_hand_open/aero_hand_node.py → aero_open_sdk → hardware
```

## Landmark / keypoint orders (source of truth)

MediaPipe 21 (input): `0 wrist; 1-4 thumb CMC,MCP,IP,TIP; 5-8 index MCP,PIP,DIP,TIP; 9-12 middle; 13-16 ring; 17-20 pinky`.

`HandMocap.keypoints` is **25**, five per finger, because MediaPipe has no finger CMC landmarks — the official node
substitutes the **wrist** for every finger CMC:

```
idx : 0 wrist
      1  thumb_cmc   2 thumb_mcp   3 thumb_ip    4 thumb_tip      <- mediapipe 0,1,2,3,4
      5  index_cmc   6 index_mcp   7 index_pip   8 index_dip  9 index_tip   <- mediapipe 0,5,6,7,8
      10 middle_cmc  ...  14 middle_tip                            <- mediapipe 0,9,10,11,12
      15 ring_cmc    ...  19 ring_tip                              <- mediapipe 0,13,14,15,16
      20 pinky_cmc   ...  24 pinky_tip                             <- mediapipe 0,17,18,19,20
```

i.e. `MEDIAPIPE_TO_MOCAP25 = (0,1,2,3,4, 0,5,6,7,8, 0,9,10,11,12, 0,13,14,15,16, 0,17,18,19,20)`.
The DexPilot config indexes **this 25-point array** (tips are 4, 9, 14, 19, 24; wrist is 0).

## Palm-local convention (`WebcamMocap.process_landmarks`) — reproduced exactly

```
x_axis = normalize(lm[5] - lm[13])        # index MCP - ring MCP;  negated for the LEFT hand
z_axis = normalize(lm[9] - lm[0])         # middle MCP - wrist
y_axis = normalize(z_axis × x_axis)
x_axis = normalize(y_axis × z_axis)       # re-orthogonalise
R      = [x_axis, y_axis, z_axis]ᵀ        # columns are the palm axes
local  = (lm - lm[0]) @ R                 # wrist at the origin, expressed in the palm basis
```

Units stay whatever came in (MediaPipe world landmarks ≈ metres). `R` is built from **MediaPipe-21 indices**, before
the 25-point expansion. Palm-local is what makes the representation invariant to camera pose — the property A1 tests.

## Mirroring and handedness — the one convention we must change

MediaPipe assigns handedness **assuming a selfie-mirrored input image**. The official node flips the frame
(`cv2.flip(frame, 1)`), so the label is already correct for the real hand, and it then negates `x` on the world
landmarks to undo the mirror in geometry.

Our head-mounted RGB-D looks **outward**: the image is not mirrored, so
* geometry needs **no** x negation (deprojected camera XYZ is already the true hand), and
* the handedness label must be **swapped** (`right ↔ left`).

This is `HandednessConvention.selfie_mirrored` in `ego_teleop/hand3d/`; production RGB-D runs with
`selfie_mirrored: false`. Mirroring is a *visualisation-only* operation from here on (spec §19).

## DexPilot retargeting config (`make_config("dexpilot", side)`)

```
urdf_path                : aero_hand_open_{side}.urdf
wrist_link_name          : {side}_base_link
finger_tip_link_names    : {side}_{thumb,index,middle,ring,pinky}_tip_link
target_link_human_indices: [[9,14,19,24,14,19,24,19,24,24, 0,0,0,0,0],
                            [4, 4, 4, 4, 9, 9, 9,14,14,19, 4,9,14,19,24]]
scaling_factor           : 1.2
low_pass_alpha           : 0.9
```

Reference vectors fed to the optimiser are `pose25[task_idx] - pose25[origin_idx]` (15×3): the 10 inter-fingertip
vectors of DexPilot plus the 5 wrist→tip vectors. `position` and `vector` variants also exist and stay available.

Two undocumented **hacks** live in `pose_callback` and are reproduced (both switchable, both logged):
1. `data += [0.0, 0.01, 0.0]` — shifts the human hand 1 cm to align with the Aero base link.
2. `if data[24][2] > 0.12: data[24][2] += 0.02` — pushes a straight pinky tip further up.

`dex_retargeting` returns joints in **its own** order (`retargeter.joint_names`, alphabetical-ish:
index, middle, pinky, ring, thumb), so the node reindexes into the Aero order by name. Then:

```
joint[0]   = q[0]*1.5  + 0°      thumb_cmc_abd
joint[1]   = q[1]*2    + (-60°)  thumb_cmc_flex
joint[2:4] = q*3.5     + 0°      thumb_mcp, thumb_ip
joint[4:7] = q*1.15    + (-10°)  index  mcp_flex, pip, dip
joint[7:10]= q*1.15    + (-10°)  middle
joint[10:13]=q*1.15    + (-10°)  ring
joint[13:16]=q*1.2     + (-10°)  pinky
clip to deg2rad(AeroHandConstants.joint_{lower,upper}_limits)
```

## Robot-side orders and limits (`aero_open_sdk.AeroHandConstants`, `JointControl.msg`, `ActuatorStates.msg`)

```
joints (16, JointControl.target_positions, RADIANS):
  0 thumb_cmc_abd  1 thumb_cmc_flex  2 thumb_mcp  3 thumb_ip
  4 index_mcp_flex  5 index_pip  6 index_dip     7 middle_*  10 ring_*  13 pinky_*
  lower (deg) = 0 everywhere
  upper (deg) = 100, 55, 90, 90, then 90 for all twelve finger joints

actuations (7, ActuatorStates, DEGREES of motor rotation):
  0 thumb_cmc_abd_act  1 thumb_cmc_flex_act  2 thumb_tendon_act
  3 index_tendon_act   4 middle_tendon_act   5 ring_tendon_act  6 pinky_tendon_act
  lower = 0,0,-15.2789,0,0,0,0     upper = 100, 104.1250, 247.1500, 288.1603 ×4
```

`ActuatorStates` also carries `actuator_speeds` (RPM, signed), `actuator_currents` (mA, signed) and
`actuator_temperatures` (°C) — the telemetry spec §17/§23 requires.

Our pre-existing compact-7 order (`ego_teleop.tracking.interfaces.AERO_CHANNELS` =
`ego_collector.hands3d.aero.COMPACT_NAMES`) is the **actuation** order, not the joint order. The two never mix: the
DexPilot path is 16-joint, the semantic-7D path is compact-7 → 16 → actuations.

## What we reuse vs. replace

| Official piece | Decision |
|---|---|
| `webcam_mocap.py` MediaPipe → palm-local → 25 keypoints | **Reused**, ported out of ROS as `hand3d/aero_mocap.py`; also the B0 baseline provider |
| monocular `multi_hand_world_landmarks` as the 3D source | **Replaced** by RGB-D metric deprojection (B1) — the one change the experiment isolates |
| `dex_retargeting` DexPilot + Aero URDF + scale factors + clip | **Reused verbatim** (`retarget/aero_backends.DexPilotAeroRetargeter`), ROS stripped |
| `aero_hand_node.py` / `aero_open_sdk` | **Reused** through the existing `robot/aero_client.SdkAeroClient` |
| ROS 2 transport (`HandMocap`, `JointControl` topics) | **Not used** — macOS has no ROS; the same payloads travel as in-process dataclasses with identical field order |

`dex_retargeting==0.5.0` + `pin==4.1.0` install and build the Aero URDF on macOS arm64 (verified in this venv), so the
whole A1–A2 path runs without ROS and without Linux.

## Consequences for our implementation

1. The **only** thing that changes between B0 and B1 is `HandPoseProvider`; the retargeter, the scale factors and the
   clip are byte-identical, so any A2/A3 difference is attributable to the pose source (spec §30).
2. Palm-local normalisation must stay the official one, or the pinned scale factors/hacks stop meaning anything.
3. `landmarks_local_3d` (and therefore the 25-point array) is in **metres** on the RGB-D path and in
   MediaPipe's "average hand" units on the monocular path — the same numbers only if the operator's hand happens to be
   average. `palm_scale_m` is logged on every frame so the two can be compared/rescaled offline.
