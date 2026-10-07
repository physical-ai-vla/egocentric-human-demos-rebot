# RGB-D teacher sensor — QA record

Dataset `HRGBD_qa10`, session `HRGBD_qa10_20260918_172229`, 8 episodes × 20.1 s, all `KEEP`.
Depth is a **label source** here, never a policy input: `RGB + depth + grip -> metric TCP / ΔEEF label`, policy input
stays RGB only.

## Verdict — 2026-09-18

| stage | state |
|---|---|
| depth sensor quality | **PASS** |
| RGB-depth sync / alignment | **PASS** |
| HandUMI body visibility | **PASS** |
| `camera_to_tcp_v1` | **EXISTS / FROZEN, not yet E2E validated** (L33/R14 accepted frames; rotation component never measured) |
| HandUMI identity tracking | **IN PROGRESS** |
| metric translation | **NOT YET PASS** |

The bottleneck is no longer "can a metric z be recovered at all" — it is one step: **which pixel's depth is the TCP.**

**The target is the HandUMI wrist-camera rigid body, not the palm.** `camera_to_tcp_v1` is a wrist-camera → TCP
offset, so a palm frame was always one link short: palm → wrist camera has never been calibrated, and a revived
MediaPipe would have carried that uncalibrated constant into every label.

Corrected chain:

```
head RGB-D -> HandUMI body detection/tracking -> body 3D position -> wrist-camera frame pose
           -> camera_to_tcp_v1 -> grasp TCP -> continuity IK -> pseudo joints
```

Because the rotation component of `camera_to_tcp_v1` is unmeasured, the **translation** chain
(`HandUMI body xyz -> TCP xyz`) is closed first. Orientation comes afterwards from the sources that were already
stable. Intended fusion:

```
translation   head RGB-D -> HandUMI body 3D tracking
rotation      wrist IMU / MASt3R
grip          existing Feetech signal
```

**AprilTag is not part of the operating system.** It is allowed only as a short calibration or validation aid
(measure `T_tag_tcp` once, or verify teacher pose on 5–10 pilot episodes) and removed afterwards. It is never a
permanent fixture, because a permanent tag has to be masked out of the policy input and has not been reliable here.

### Next QA gate — HandUMI body tracking

```
body track valid rate           identity-switch rate
stationary unit xyz std         moving unit frame-to-frame xyz p95
track gap length                TCP xyz tail
IK success                      pseudo-joint smoothness
```

The **idle left unit resting on the table is the sanity check**: if its 3D position std comes out at a few mm, the
translation detector is alive. No change to the full dataset until this passes.

## Sensor QA, items 1–6 and 10 — 8/8 PASS

`scripts/rgbd_qa10_report.py --session <session>`

```
valid ratio            85.9 – 89.2 %   (p05 72.7 – 82.4 %)
dropout                0                depth maps 605/605 in every episode, no collapsed runs
RGB-depth alignment    edge offset median 1.9 – 2.3 px, p95 8.2 – 11.0 px, no warnings
timing                 head_depth 29.98 fps, wrist 30.0x
                       wrist -> depth offset p50 8.1 ms, p95 15.7 ms   (inside half a frame)
depth noise (std)      0.20–0.30 m  0.84 – 1.50 mm
                       0.30–0.40 m  0.71 – 1.83 mm
                       0.40–0.50 m  0.40 – 0.62 mm
                       0.50–0.80 m  0.50 – 1.09 mm
plane residual (bias)  0.50–0.80 m  rms 5.1 – 5.8 mm   (only band that passed the flatness check)
grip                   100 Hz, offset -> depth p50 2.6 ms; jaw travel L 0.86 / R 0.73
intrinsics             device-reported in all 8 (fx 461.17, fy 461.45, 848x480, aligned to RGB, 1 mm unit)
```

Noise improved ~5x versus the 2026-09-11 take (2.3–3.6 mm) purely from freezing exposure.

**Bias is measured on only one band.** The others failed the planarity check, so their residual describes scene
geometry, not the sensor, and the report discards them. A flat board held at 20/30/40/50/60 cm for 5 s each would
give a real per-band bias in one 20 s episode. Not yet recorded.

## Camera freeze — `calibration_version: exposure_v004`

`configs/handumi/hardware_handumi_rgbd.yaml`, every value verified by read-back at record time
(`session_meta.json` → `devices.*.uvc_controls`, all `applied: true`).

| control | left (FisheyeCamLeft) | right (Arducam 1080P Low Light) |
|---|---|---|
| auto-exposure-mode | 1 (manual) | 1 (manual) |
| exposure-time-abs | 1 | 1 |
| auto-exposure-priority | 0 | 0 |
| gain | 0 | 0 |
| saturation | 32 | 32 |
| backlight-compensation | 0 | 0 |
| auto-white-balance-temp | false | false |
| **white-balance-temp** | **2800** | **3500** |
| power-line-frequency | 2 | 2 |
| brightness | −48 | −64 |

Four findings behind those numbers, each measured, none assumed:

* **`exposure-time-abs` was never dead.** The left unit sits in `auto-exposure-mode 8` by device default, and a UVC
  camera in auto mode stores the exposure you write and ignores it. Driven to mode 1 and verified by read-back, both
  units sweep cleanly (exposure 1 → mean 184 / sharpness 381; exposure 330 → mean 243 / sharpness 5).
* **`auto-exposure-priority` and `backlight-compensation` were fighting the setting**, at their defaults 1 and 1.
* **`brightness` is a luma pedestal**: it darkens without touching chroma, so saturation rises with every step
  (0 → 72, −64 → 110) and a faint lighting tint becomes a strong cast. Paired with `saturation: 32` to compensate.
  It also *raises* detail (sharpness 431 → 533 at −48) by pulling the histogram off the clipping ceiling.
* **`white-balance-temp` is effective, and 4600 was wrong.** 4600 is the device default and had never been verified.
  Two-extreme test on a fixed scene, read-back exact at both ends:

  ```
  left    2800 -> midtone R/G 0.985 B/G 0.985      6500 -> midtone R/G 0.981 B/G 0.792   Δ B/G 0.193, blue 24.9 levels
  right   2800 -> midtone R/G 0.980 B/G 0.990      6500 -> midtone R/G 0.992 B/G 0.955   Δ B/G 0.034, blue 15.0 levels
  ```

  Highlights read neutral at every setting, which is why the white desk never showed the problem: the whole cast
  lived in the midtones. Optimum differs per unit, so the two do not share a value.

**The 8 recorded episodes carry the OLD white balance (4600 K on both).** Depth and every number above are
unaffected; the wrist RGB in those episodes has the yellow-green midtone cast. Recorded here as dataset provenance.

## Items 7–9 — blocked at hand localization, not at depth

`scripts/rgbd_tcp_qa.py` over all 8 episodes, both sides: **16/16 tracks 100 % LOST, reason `no_hand`, 605/605
frames.** No trajectory, so no displacement percentiles and no non-physical-chunk share.

`scripts/hand_detector_audit.py` on 20 representative frames rules out the input path:

```
integrated (VIDEO, wrapper, 0.5)     0/20
standalone (IMAGE, raw, 0.5)         0/20
standalone (IMAGE, raw, 0.3)         0/20
standalone + ROI crop (0.3)          0/20
standalone + ROI crop 2x (0.3)       0/20
```

Input verified clean: 848×480 uint8 contiguous, decoded `bgr24`, no rotation or flip, correct stream, timestamps
strictly increasing, glove region ≈ 250×200 px. Positive control: the **same** code, machine and model detected a hand
in 100 % of the 1667 frames of the bare-hand Stage-A episode `Hpilot_rgbd_stageA_20260911_111811/episode_000001`.

Cause: the operator's hand is inside a grey work glove, wrapped around the black HandUMI body, so no finger
articulation is presentable; and the idle hand is often out of frame entirely. **A hand-landmark model is the wrong
detector for this rig** — this is a detector problem, not a sensor problem.

### Rejected routes to human metric translation, for the record

```
MASt3R              rejected      reproducibility 62 mm
Depth Anything V2   rejected      scale off 2.8–6.4x
MetricAnything      rejected
cube PnP            insufficient  40–92 mm on a stationary object
IMU bridge          insufficient  139 mm drift over 0.5 s
MediaPipe palm      REJECTED      2026-09-18: 0/605 frames, gloved hand around the device. Depth itself passed.
```

## What the pipeline does when a track exists

Validated end to end on the bare-hand Stage-A episode, so the chain below is known to work once a detector supplies
poses:

```
head RGB-D -> metric palm -> camera_to_tcp_v1 -> grasp TCP -> ego16.npz -> continuity IK -> pseudo reBot joints

episode_000001 (Stage A, bare hand, 1667 frames)
  IK fail 0.000 | FK mm L p50/p95 0.9/12.3 | arm_valid L 0.94 | posture L 0.94 | jumps>25° 0.000, p95 2.1°
  trajectory: path 8984 mm, mean speed 162 mm/s, bbox diag 584 mm
  displacement: step p50 3.3 mm p99 24.6 mm; chunk p50 14.9 mm p95 206 mm p99 330 mm
  non-physical: 0/1632 chunks over 0.5 m, 0 over 1.0 m; 6 frame steps over 50 mm
```

Two caveats that travel with every number from this chain:

* Poses are in the **head-depth camera frame, and the head moves.** No head odometry (Stage C) exists, so items 8/9
  contain head motion as well as hand motion.
* `camera_to_tcp_v1` is a **wrist-camera → TCP** offset and the palm route supplies a **palm** frame; palm → wrist
  camera has never been calibrated. Absolute xyz therefore carries an unknown constant hand-frame offset.
  Displacements are unaffected except through hand rotation within the chunk.

## Not done, deliberately

No further collection, no H250 reprocessing, no Route B training, no change to the depth extraction,
`camera_to_tcp` or IK code. GPU0/GPU1 A1/A2 runs untouched.

---

# HandUMI body tracking — translation chain, 2026-09-18

`scripts/handumi_body_track.py` → `scripts/body_to_ego16.py` → `scripts/ego_ik_retarget.py`, plus
`scripts/body_jump_audit.py`. No AprilTag, no hand model.

## Verdict

| stage | state |
|---|---|
| metric translation tracking | **provisional PASS** |
| pseudo-joint algorithm (relative motion) | **PASS** — at or above the Stage-A reference on all 8 |
| absolute TCP calibration (`body → wrist/TCP`) | **NOT DONE** — required before Route B labels |
| orientation | **not estimated** — comes from wrist IMU (restored to the profile), not from the body |

## Tracker health, 16 tracks over 8 episodes

```
identity switches   0 / 16 tracks
coverage            92.2 – 100 %
max gap             0 – 16 frames
table plane         inliers 67 – 71 %, rms 2.12 – 2.49 mm
frame step          p50 0.4 – 4.4 mm
```

### Noise floor — the metric had to be corrected twice before it meant anything

1. Whole-episode xyz std read 68–72 mm. That is workspace extent, not noise: both units move in a bimanual task.
2. A fixed 1.5 s prefix was then assumed to be still, because the collector has a stillness gate. **It has one, but it
   is driven by the IMUs and this profile had `imus: []`, so the gate never ran** — several "still" windows contained
   real motion and read 20–94 mm.
3. Corrected: search the trajectory for its quietest gap-free 1.0 s window, ranked on p95 then median frame speed.

```
genuine rest (window path < 15 mm) -- the detector's own noise floor
  ep2 right 0.71 mm   ep3 right 0.04 mm   ep4 left 0.08 mm   ep8 left 0.04 mm
  (in-window speed p50 0.04 – 0.18 mm/frame: really stationary)

no genuine rest window -- upper bounds, the unit kept moving
  0.75 – 1.91 mm on 11 tracks;  ep3 left 5.55 mm, whose window still ran at p50 2.26 mm/frame = 68 mm/s
```

**16/16 tracks ≤ 5.55 mm, 15/16 ≤ 1.91 mm.** For comparison, MASt3R's reproducibility was 62 mm.

`DARK_MAX = 70` frozen. Above-table pixels are bimodal — a large peak at 10–40 (the black body) against a tail at
70–160 (grey glove, skin, cables) — and stability barely moves across 40..95, so this is not a knife edge.

## A. Relative-motion IK QA — all 8 episodes

```
                IK fail   FK p50/p95 mm  L      R        arm_valid L/R   posture L/R   >25° jumps   Δq p95
ep1              0.000     2.1/12.5    2.8/17.6    0.92 0.97     1.00 1.00     0.000       5.4°
ep2              0.000     3.2/17.6    1.8/21.0    1.00 0.90     1.00 1.00     0.000       8.2°
ep3              0.000     2.2/15.7    0.8/13.8    0.95 0.95     1.00 1.00     0.000       4.2°
ep4              0.000     1.5/ 8.1    2.7/12.1    0.97 0.94     1.00 1.00     0.000       3.4°
ep5              0.000     3.8/10.1    2.1/11.0    0.98 1.00     1.00 1.00     0.000       3.1°
ep6              0.000     2.9/21.7    4.6/34.8    0.94 0.94     1.00 0.99     0.000       4.5°
ep7              0.000     2.6/12.6    3.6/17.7    0.99 0.98     1.00 0.99     0.000       4.4°
ep8              0.000     3.7/14.5    3.5/16.6    0.91 0.99     1.00 1.00     0.000       5.1°
reference (Stage-A palm, bare hand)
ep1              0.000     0.9/12.3      --        0.94  --      0.94  --      0.000       2.1°
```

Joint-limit occupancy improved (0.94 → 0.99–1.00). Δq p95 is higher than the reference (3.1–8.2° vs 2.1°), and
section B explains why.

Three choices behind these numbers, each forced by what the pilot profile has:

* **Orientation is the robot home quaternion, held fixed** (`ego16.py --orientation fixed`, the documented phase-1
  arm-only mode). The body's PCA first axis deviates by 0.8–57° p50 across tracks — nowhere near an orientation.
* **`camera_to_tcp_v1` is NOT applied.** It has no rotation component, so it is not a full SE(3) calibration and is
  not treated as one; with no body orientation it could only be a constant translation, which cannot change any
  relative-motion statistic. Kept behind `--apply-camera-tcp` for when `body → wrist/TCP` is measured.
* **`R_align` is built from the fitted table plane**, since the IMU-gravity branch had no IMU to use.
  `z_world` = table normal (sign resolved so it points away from the table towards the camera), `x_world` = the
  existing anchor rule (camera optical axis projected onto the plane), `y = z × x`, x re-orthogonalised. Asserted on
  every episode: `det(R_align) = 1` and the table normal maps to `[0,0,1]`.

**This is relative-motion QA only.** A constant offset does not touch smoothness, but it does move reachability and
joint posture, so absolute retarget QA waits for a real `body → wrist/TCP` calibration.

## B. Large-jump audit — `scripts/body_jump_audit.py`

Every >50 mm single-frame step classified from the ±5 frames around it.

```
76 events over ~9600 track-frames (0.8 %)
  segmentation_change  72  (95 %)
  true_fast_motion      4  ( 5 %)
  depth_failure         0
  identity_or_gap       0
```

**Two of the four buckets are empty, and that is the finding.** Depth coverage inside the blob was 0.94–1.00 at every
event and identity never switched. The largest events carry blob-area ratios of 16.7×, 12.7×, 23.9× (or 0.12×, 0.39×
the other way): the mask is merging the body with the forearm and releasing it again. The remedy is the segmentation,
not the sensor, not the association rule — and it is what raises Δq p95 above the reference.

## Wrist IMUs restored — `calibration_version: exposure_v005`

`imus: []` in the pilot profile was a configuration accident, not a decision. Both wrist IMUs are back in
`hardware_handumi_rgbd.yaml` (teensy, 200 Hz, `required: true`, serials 20540160 / 20778070), so the final profile
records **left IMU, right IMU, depth, left/right wrist RGB, grip**, all required.

Intended fusion, and the division of labour behind it:

```
translation   Orbbec RGB-D -> HandUMI body tracking      (IMU double integration is NOT used: 139 mm drift over 0.5 s)
rotation      wrist IMU gyro + gravity  (+ MASt3R if needed)
grip          existing Feetech signal
```

With the IMUs back, `ego16.py`'s IMU-gravity `R_align` becomes usable again and the table plane becomes a
**cross-check and fallback**: the angle between the IMU gravity axis and the table normal should be reported per side
per episode, and a departure from near-parallel is a calibration fault worth catching early.

## Still open

* `body → wrist/TCP` transform — the one thing between this and absolute Route B labels.
* Segmentation refinement to remove the forearm merge (95 % of all large jumps).
* Body orientation: PCA is not stable enough; either CAD-template registration or the wrist IMU.
* Per-episode IMU-gravity vs table-normal angle check (needs a take with IMUs recorded).
* Verification that the restored profile really records all five streams, with timestamp/sync QA, before any take
  larger than 10 episodes.

---

# Forearm-merge suppression — body-centre estimator, 2026-09-18

`scripts/body_center_bakeoff.py`: identical cached point clouds, identical identity association, estimator swapped.

```
estimator                >50mm  >100mm  step p95  p99   still med  coverage  switches
full-blob centroid          78      15     21.5   46.3     1.77      92.2%       0
uniform trim r=0.06         52      20     14.1   35.5     1.45      92.2%       0
uniform trim r=0.07         55      17     14.7   36.9     1.50      92.2%       0
uniform trim r=0.09         65      11     15.8   37.9     1.79      92.2%       0
uniform trim r=0.12         61       9     18.9   43.4     1.79      92.2%       0   <- ADOPTED
uniform trim r=0.15         68      12     20.5   45.3     1.73      92.2%       0
mean-shift bw=0.05         285      49     29.9   93.1     2.73      92.2%       0
mean-shift bw=0.09         142      18     21.9   60.8     2.23      92.2%       0
```

**Mean-shift was the intuitive answer and it is wrong here.** The HandUMI is a hollow multi-part mechanism, not one
dense lump: a small bandwidth hops between spurious internal modes (285 jumps, 3.7x the plain centroid) and a large
one converges back to the plain centroid. Density does not separate the body from the arm — extent does.

**A tight trim is also wrong, and instructively so.** r=0.06 gives the best >50 mm count (52) and the best step
percentiles, yet its >100 mm count is *worse than the baseline* (20 vs 15): the trim cannot follow a genuine fast
move, falls behind, and then snaps. Choosing on >50 mm alone would have picked it.

Adopted `BODY_RADIUS_M = 0.12`: **>100 mm −40 %, >50 mm −22 %**, stationary std 1.77 → 1.79 mm, coverage and
identity switches unchanged.

## Pseudo-joint quality, per arm

`jump_deg` from the IK is the max over all 12 joints, so a bimanual episode cannot be compared with a single-arm
reference — two moving arms take the max over twice as many joints and the percentile rises for that reason alone.
Reported per arm instead:

```
              FK p50/p95/p99 mm   arm_valid   Δq p50/p95/p99 deg   >25°   vel p95/p99 deg/s   acc p95   branch
ep1 L          2.0/11.5/21.8        91.6%      0.24/3.02/5.60     0.00%     90.5/168.1        1069        0
ep1 R          2.5/13.8/22.0        97.4%      0.54/3.63/5.04     0.00%    108.8/151.3        1609        0
ep2 L          2.2/15.3/25.5        98.8%      0.34/3.13/5.71     0.00%     94.0/171.2        1479        0
ep2 R          1.6/ 8.4/22.1        89.9%      0.51/2.64/6.93     0.00%     79.1/207.9        2156        0
ep3 L          1.8/11.8/21.2        94.9%      0.35/2.59/4.09     0.00%     77.6/122.8        1160        1
ep3 R          0.7/10.9/19.6        95.4%      0.17/1.45/2.82     0.00%     43.6/ 84.7         386        0
ep4 L          1.4/ 7.7/10.1        96.7%      0.16/0.90/2.32     0.00%     27.0/ 69.6         286        0
ep4 R          3.3/11.4/16.8        94.2%      0.28/2.02/4.45     0.00%     60.5/133.5         473        0
ep5 L          3.8/ 9.6/14.6        98.0%      0.40/2.11/3.45     0.00%     63.2/103.6        1121        0
ep5 R          2.4/11.7/21.4        99.8%      0.29/2.30/4.84     0.00%     69.1/145.1         912        0
ep6 L          2.8/13.3/22.2        94.4%      0.48/2.70/4.52     0.00%     81.1/135.7        1302        0
ep6 R          4.2/14.5/24.8        91.2%      0.26/2.69/4.15     0.00%     80.6/124.5         721        0
ep7 L          2.5/11.0/18.5        98.5%      0.49/3.02/5.95     0.00%     90.5/178.6        1461        0
ep7 R          3.5/17.6/24.5        98.2%      0.37/2.88/4.77     0.00%     86.4/143.0        1141        0
ep8 L          3.5/11.0/17.6        90.9%      0.54/2.34/3.95     0.00%     70.3/118.6        1104        0
ep8 R          3.1/15.6/21.4        99.0%      0.48/2.98/4.70     0.00%     89.4/141.1        1077        0
REF L (Stage-A, bare hand, single arm)
               0.9/ 4.3/ 7.7        94.2%      0.43/1.70/2.73     0.00%     51.1/ 82.0         640        0
```

IK fail 0.000 and joint-limit posture 0.99–1.00 on every episode. **Branch-switch candidates: 1 across all 16
tracks** (joints moving >10° while the target moved <5 mm). Δq p95 0.90–3.63° against the reference's 1.70°, in the
same band. FK p95 is 2–4x the reference (7.7–17.6 mm vs 4.3 mm) — inside the 30 mm gate, but the largest remaining
gap to the reference and the thing to watch once orientation arrives.

## The suppression did not move Δq p95, and that was predictable

Δq p95 went 4.79° → 4.74° (−1 %) over the 8 episodes. Chasing the jumps was the wrong lever for that statistic:
61–78 events over ~9600 track-frames is 0.7 %, far outside the 95th percentile. The correlation confirms both halves:

```
13 of 14 remaining >100 mm body jumps land in that arm's top 1 % of Δq
40–100 % of the >50 mm events do as well
```

So the jumps are real and they do reach joint space — they live in the p99 tail. Suppressing them improves p99 and
max, not p95, and the >100 mm count falling 15 → 9 is exactly that improvement.

---

# Calibration take — orientation + `body → TCP`, protocol

Hardware verified 2026-09-18 before the take was asked for, so a failed take cannot be blamed on it:

```
/dev/cu.usbmodem205401601  serial 20540160  left  Teensy
/dev/cu.usbmodem207780701  serial 20778070  right Teensy
imu_dual_smoke 12 s, both wrists, distinct boards:
  left  197.92 Hz  drops 0  CRC 0  gravity 1.0087 g  jitter p99 5.05 ms  clock drift +11.6 ppm   ALL PASS
  right 201.58 Hz  drops 0  CRC 0  gravity 1.0066 g  jitter p99 4.96 ms  clock drift +11.7 ppm   ALL PASS
```

```bash
sudo .venv/bin/python -m handumi_collector.tools.check_rgbd_ready --live
sudo .venv/bin/python -m handumi_collector.collect --hardware handumi_rgbd --dataset CALIB_bodytcp
```

Two episodes, **one per hand**, 20–30 s each. Not a stacking take — the order label is unused.

**Why the motion protocol is what it is.** `T_body_tcp` has a translation and a rotation, and a trajectory that only
translates cannot separate them: a lever arm is invisible until the body rotates. Each segment below exists to make
one term observable, so a take that skips a segment leaves that term unconstrained rather than merely noisier.

```
 1  hold still, 3 s              gravity reference; IMU bias; the detector's noise floor
 2  small xyz translation        body translation with orientation held -- the lever arm contributes nothing here
 3  roll, then pitch, then yaw   one axis at a time: rotation with the body centre near-fixed isolates the LEVER ARM
 4  rotation while translating   the coupled case the fit has to reproduce
 5  near workspace, then far     reachability spread; this is what the FK margin is measured against
 6  grip open / close, twice     the grip label crossing its own range inside the take
 7  hold still, 3 s              return drift, measured against an operator mark rather than the tracker's own output
```

Keep both units inside the head camera's view throughout; a unit that leaves the frame costs coverage in exactly the
segment it was needed for.

---

# Route B pseudo-joint dataset — schema and provenance (FROZEN; no dataset generated yet)

Defined now so that tomorrow's take turns into labels without a format argument. **Nothing is generated until the
absolute QA passes** (`T_body_tcp` measured, orientation from the IMUs, FK margin re-checked).

## Per-episode arrays, on the head-depth master clock

```
t_ns            (T,)      int64    head_depth capture_ns; every other stream is resampled ONTO this, never the reverse
q_abs_rad       (T,12)    float32  [L pan lift elbow wflex wyaw wroll | R ...]  canonical rebot_b601 solver order
dq_rad          (T,12)    float32  q[t+LEAD] - q[t], LEAD = 5 frames; the last LEAD rows are invalid, not zero-filled
grip01          (T,2)     float32  [L, R]; 1 = open, 0 = closed  (human raw convention, `norm_one_is: open`)
arm_valid       (T,2)     bool     pose valid AND FK error < fk_max_mm AND posture margin met -- the IK's own verdict
pose_valid      (T,2)     bool     the translation source had this arm this frame (body track coverage)
grip_valid      (T,2)     bool     gripper stream healthy this frame
fkerr_mm        (T,2)     float32  kept, not thresholded away: the margin is a dataset property worth training against
tcp_xyz_m       (T,2,3)   float32  T_world_tcp translation, world = table-aligned (z = table normal)
tcp_quat_xyzw   (T,2,4)   float32  T_world_tcp rotation; xyzw, never Euler (RPY is UI/debug only)
```

`dq` uses the **t+5 lead** already used by the exporter and the training patch. Left precedes right everywhere, and
the 14-D state the exporter writes stays `[q_L6, grip_L, q_R6, grip_R]`.

## Provenance, per episode

Written beside the arrays; a label without it is not reproducible.

```
translation_source   "head_rgbd_body_track"  + BODY_RADIUS_M, DARK_MAX, BAND, ABOVE_PLANE
orientation_source   "wrist_imu"             + R_body_imu version, yaw observability, gravity-vs-table angle
body_to_tcp          version + residual + observability warnings + which take it came from
world_frame          "table-plane aligned: +z = table normal, +x = camera optical axis projected onto the table"
                     + R_align, det, plane rms, inlier share
ik                   retarget_shakedown.yaml version + max_joint_delta, ori_weight, fk_max_mm, posture_margin_deg
camera               calibration_version (exposure_v005) + the read-back uvc_controls actually applied
episode              session, episode, duration, order, operator, collector version, clock_anchor
gates                the pseudo_joint_qa.py numbers this episode passed with
```

**Known provenance fault to carry forward:** the 8 `HRGBD_qa10` episodes were recorded at `white-balance-temp 4600`
on both wrist cameras, which the later measurement showed is wrong (midtone B/G 0.871 on the left). They are a
sensor-QA dataset, not Route B material.

## Collection ladder — gates, not a target count

Not 250 episodes in one go. `H250` stays useful for RGB-only pretraining (vision / task / order / phase / rotation /
grip, **xyz masked**) and is not the source of pseudo joints.

```
10 ep   full SE(3) + pseudo-joint QA        <- gate
30 ep   Route B small-data training          <- gate
60-100  the real comparison
150-250 only if the curve says it is needed
```

The comparison this ladder buys — how much robot teleop a human demonstration replaces — is more informative than a
single large take:

```
B0  C30 robot-only          B1  H30 pseudo-joint       B2  H60 pseudo-joint
B3  H100 pseudo-joint       B4  H100 + C30
```

---

# Tomorrow's runbook — the only hardware work left is two takes

Everything below the take is built, dry-run and self-tested as of 2026-09-18 night.

```bash
cd ~/ego_collector
.venv/bin/python scripts/calib_take_qa.py --audit                     # profile, no hardware: expect PROFILE READY

sudo .venv/bin/python -m handumi_collector.tools.check_rgbd_ready --live
sudo .venv/bin/python -m handumi_collector.collect --hardware handumi_rgbd --dataset CALIB_bodytcp
#   2 episodes, one per hand, 20-30 s, motion protocol above. POLARITY TEST once at collector start.

S=datasets/human_handumi_raw/CALIB_bodytcp/<stamp>
.venv/bin/python scripts/calib_take_qa.py --session $S                # 6 streams present, rates, drops, offsets
.venv/bin/python scripts/body_tcp_fit.py --selftest                   # conventions + fitter, before any real value
#   then the fit itself, then:
.venv/bin/python scripts/body_to_ego16.py --session $S --apply-camera-tcp
~/handumi-sw/.venv/bin/python scripts/ego_ik_retarget.py $S/episode_*
.venv/bin/python scripts/pseudo_joint_qa.py --session $S
```

Gate order, and what each one is allowed to conclude:

```
sync QA          all six streams, monotone timestamps, offsets inside half a frame
gravity check    IMU gravity vs table normal within a few degrees -- else suspect mounting, not the tracker
observability    rotation >= 40 deg, cond <= 25, yaw minimum depth > 0.60  -- else the take is redone, not fitted harder
absolute QA      IK fail, FK p50/p95/p99, arm_valid, posture, dq p95/p99, branch switch, joint-limit proximity
```

The metric to watch is **FK margin**: the relative-motion run's worst frames sat at 29.7 / 29.9 mm against a 30 mm
gate, which is also why `arm_valid` is 0.90-0.99 rather than 1.00. If `T_body_tcp` and a real orientation do not open
that margin, the next thing to look at is the retarget/IK objective, not the tracker.

After this passes: **10 new RGB-D human episodes** for Route B, then the 30 / 60-100 ladder.

## A1/A2 50k comparison — deferred, and why

```
xvla-R150-EEF-A1-300k   RUNNING   checkpoints/050000 (9.9 GB, 2026-09-18 18:15:48)   save_freq 50000, steps 300000
xvla-R150-EEF-A2-300k   RUNNING   checkpoints/050000 (9.9 GB, 2026-09-18 18:28:06)   save_freq 50000, steps 300000
both alive (2 DDP processes each); 4090 #0 97 % 21650/24564 MiB, #1 100 % 21535/24564 MiB
```

Both 50k checkpoints exist, so the comparison is unblocked in principle. It was NOT run, for two reasons:

* **No GPU headroom.** ~2.9 GB free on each 4090. An inference job there would OOM or contend with the training
  that must not be disturbed. A safe target exists: **node 100.64.0.5 (bh-ai-5090), RTX 5090, ~24 GB free, 0 % util.**
* **The historical frozen evaluator does not exist anywhere searched.** `moving first-5 cosine`, `endpoint cosine`,
  `idle false-motion`, `idle >5 mm`, `rotation error`, `grip accuracy`, `sampling variance` return zero hits in
  `~/ego_collector`, `~/robot-cockpit/eval` and a full grep of `/home/bh-aiteam`. Two leads were followed and
  rejected: `dir_eval_*` is the Cosmos world-model controllability eval (PSNR / TOP5 % MAD between Lxneg/Lxpos
  rollouts), and `/srv/data/hugh/vla/xvla` is an **OpenArm patty-flip** project (15 fps, 16-D, different robot).
  `r90_eval_*` is a real-robot success-rate gate, not an offline action metric.

The conventions survived even though the code did not: `eef_stability_gate.py` states the representation as
**dEEF, lead 5, chunk 30**, which fixes what "first-5" and "endpoint" refer to. Writing a new evaluator from that is
possible, but it must be validated by reproducing the historical D0 numbers (~0.66 / ~0.81 / idle p50 ~9 mm /
>5 mm ~0.9) on `/home/bh-aiteam/xvla_D0_R150_EEF_300k/checkpoints/{050000,100000,150000}` before it is trusted as
the same gate. Deferred deliberately: doing it tonight would have delayed the calibration take, and the training
keeps running regardless.

---

## Calibration take 1 — CALIB_bodytcp_20260919_121002 (2026-09-19)

Two episodes, one per hand: ep1 the right unit moving with the left resting, ep2 the reverse. The resting unit is
not wasted — its track std is the detector's noise floor for that take.

### What passed

| gate | ep1 right | ep2 left | verdict |
|---|---|---|---|
| body track coverage | 96.9% | 98.9% | PASS |
| identity switches | 0 | 0 | PASS (0 on all four tracks) |
| IMU↔depth clock residual p95 | 3.12 ms | 3.74 ms | PASS vs a 33 ms frame |
| motion observability: max rotation | 78.0° | 64.4° | PASS (gate ≥ 40°) |
| motion observability: cond | 10.6 | 9.6 | PASS (gate ≤ 25) |

Stream QA: all seven streams present, head_depth 29.98 Hz, wrists 30.0 Hz, IMUs 197.9/201.6 Hz with zero seq
drops, grippers 100 Hz. Detector noise floor from the resting unit: **0.54 mm** still-window std.

### What the take cannot support, and why

**No cube was held in either episode.** The moving gripper's raw position wanders (std 135–153 counts) with its
median at the jaw's hard stop and no plateau longer than 0.6 s; a held object parks the jaw at the object's width
for as long as it is held. `t_body_tcp` is solved from `p_tcp(t) = p_body(t) + R_world_body(t) @ t_body_tcp`, and
the only source of `p_tcp` is that cube, so the fit has no left-hand side.

The motion, however, is *fine* — that is what the observability gate settles. A re-take needs **a cube in the
gripper and nothing else changed**. `calib_take_qa.py` now checks for the grasp plateau so this is caught at the
bench rather than by the fitter.

### Three bugs this take exposed

1. **`clock_map` lost all conditioning on an uncentred fit.** The right Teensy had been up ~20 h, so `device_us`
   was 7.16e10 while the episode spanned 3.8e7; against a constant column, `lstsq` returned 662 ns/µs where the
   two-point slope was 999.73, with a **3.2 s** median residual. The left unit (`device_us` 9e8) happened to
   survive. Fixed by mean-centring both columns; the selftest now fits the same synthetic signal at both offsets
   and asserts the slopes agree (observed difference 5.7e-12 ns/µs). Had this reached the fit, every right-hand
   orientation would have been garbage.
2. **The yaw-drift audit counted deliberate motion as drift.** One `quiet` mask served both the gravity
   correction and the "is it turning" question, at 0.35 rad/s (20°/s) — far above a real turn rate — so a
   synthetic 60° turn read as +180°/min of drift. Split into `GRAVITY_*` and a tight `STILL_GYRO_TOL` (0.05 rad/s).
3. **`calib_take_qa.py` rejected IMU streams for non-monotonic `host_ns`.** All 287–587 offending steps are exact
   ties from USB batching, with zero backward steps, while `device_us` and `seq` are strictly monotonic and
   gap-free. The check now allows ties on host clocks and audits the device clock separately.

### Measured error floor to carry into the fit

Yaw is not observable from a gyro+accel IMU, so it is pure integration. Measured on the genuinely-still resting
units: **−7.46 °/min (ep1 left) and −10.83 °/min (ep2 right)**, i.e. **4.7° and 6.3° over a 35–38 s take**. The
fitter's yaw search estimates a *constant* yaw, so this within-take drift is not absorbed by it and sets the
error floor of any `t_body_tcp` this pipeline produces.

### Clock rule

Integrate on `device_us` (the IMU's own elapsed time), map `device_us → host_ns` with a centred affine fit, then
SLERP onto the depth timeline. `host_ns` is the synchronisation anchor, never the integration clock — it carries
the batching ties. The two units' 197.9/201.6 Hz difference is a crystal property, not data drift.

### New tooling

| script | does |
|---|---|
| `scripts/body_track_export.py` | joins the frozen body tracker and the IMU branch into `derived/body_track/track_export.npz` |
| `scripts/imu_orientation.py` | device-clock integration, centred clock map, SLERP resample; `--selftest` |
| `scripts/wrist_image_quality.py` | separates over-exposure from defocus on a centre crop |
| `scripts/wrist_brightness_sweep.py` | picks the brightness pedestal against a held scene |

---

## Calibration take 2 — CALIB_bodytcp_20260919_123133, and why the lever-arm fit cannot close (2026-09-19)

The re-take was recorded with a cube held (grasp plateau 18.7 s) and passes every gate:

| gate | value |
|---|---|
| streams / rates / drops | all present, IMU 197.9 & 201.6 Hz, 0 seq drops, depth-align p95 2.5 ms |
| body track coverage | 98.8%, 0 identity switches |
| IMU↔depth clock residual | p95 3.13 ms, rate error +1 ppm, mapped monotonic |
| motion observability | max rotation 82.0°, cond 8.5 |
| cube detection | 497 of 559 hold frames (88.9%), hue spread 1 |

**And the fit still fails.** residual rms 26.5 mm, inliers within 10 mm 26.5%.

### The diagnosis, in the order the evidence arrived

1. **The scalar invariant exonerates nothing by itself.** `d(t) = ||p_cube − p_body||` is invariant to every
   rotation. Its spread is 22.2 mm std / 183.5 mm range, with correlation to attitude of only +0.04. Filtering
   to confident detections more than 5 frames from a track gap takes this to 11.3 mm std — frames near a gap
   deviate 48.2 mm from the median against 4.0 mm elsewhere, a 12x tracking artefact.
2. **No grasp slip, no depth drift.** First-half vs second-half median of `d` differ by 0.6 mm.
3. **A better reference point helps `d` and does nothing for the fit.** Bake-off over six candidates on the
   clean 245 frames: bbox centre 8.8 mm std vs today's trimmed centroid 12.9 mm; shrinking the trim radius makes
   it *worse* (17.2 at r=0.06), geometric median and ICP-propagated points are no better. Yet swapping the body
   reference to bbox moved the fit residual from 26.5 to 26.9 mm — **no improvement at all**. Whatever is wrong
   is in the direction, not the distance.
4. **The offset does not rotate with the body.** Expressed in the body frame via the IMU, the body→cube
   direction scatters 7.6° p50; expressed in the *world*, with no rotation applied at all, it scatters 7.1°.
   Applying the measured rotation makes the offset 0.93x *less* constant. For a rigidly held cube on a body that
   rotated 82°, the world-frame direction should swing by tens of degrees and the body-frame one should be
   nearly fixed.
5. **The IMU is not the fault — measured independently.** Frame-to-frame ICP on the body's own point cloud,
   against the IMU over the same intervals: r = **+0.969**, medians 0.44° vs 0.53° per frame, cumulative path
   210° vs 246°, ICP correspondence error 1.7 mm p50.

### Conclusion

The body really does rotate (ICP), the IMU really does measure it (r = 0.97), and the cube really is held
rigidly (0.6 mm). The vector between the two tracked *positions* nevertheless does not rotate — because both
positions are **centroids of the surface visible from the camera**, which slide across their objects as the view
changes, and that sliding cancels the rotation. The visible dark-blob point count swings 2.0x across the take
(17.7k to 35.6k), which is the mechanism in one number.

So `p_tcp = p_body + R_world_body @ t_body_tcp` is not a model these observations can support, and no choice of
centroid, trim radius or frame filter repairs it. **A centroid is not a point on a rigid body.**

### What this changes

`T_body_tcp` cannot be frozen from centroid tracking. The lever arm needs a genuinely rigid body position, and
the evidence points at ICP for that role: it is already demonstrated here at 1.7 mm correspondence error and
r = 0.97 against the IMU. Note this is **frame-to-frame ICP on the live cloud**, a different thing from the
model-based ICP against a frozen mesh that was closed as FAIL in the Stage-A work — that closure does not apply
to this use.

The naive version does not work either: propagating a point through accumulated frame-to-frame transforms
scored 13.2 mm in the bake-off, no better than the centroid, because 245 frames of drift accumulate. The
structure that follows from the evidence is to de-rotate each cloud by the (verified) IMU rotation, accumulate a
canonical body model, and register each frame against that model — drift-free by construction.

### Two measurement lessons recorded

- **The 0.54 mm noise floor is a STILL-unit number and does not bound a moving unit.** A stationary object has
  an unchanging visible surface; the same detector on a handled object is off by 20x.
- **Half-vs-half repeatability was the wrong headline metric and I used it as one.** The take's rotation is
  concentrated, so the first half alone has cond 48 (first third: 135) and fits |t| = 296 mm — physically
  impossible for a 10 cm device. A temporal split is only meaningful where each part is independently
  observable; `body_tcp_fit` now says "NOT COMPARABLE" instead of printing the number. Interleaved splits
  (cond 16.6/16.3) and a bootstrap (t std 3.5/9.7/2.4 mm) are the valid forms.
- **Next take:** spread the rotation through the whole take rather than in one burst, so temporal subsets are
  each observable.

### New tooling

| script | does |
|---|---|
| `scripts/cube_tcp_detect.py` | held-cube centroid; size not colour, with compactness/depth/continuity gates and a per-frame confidence |
| `scripts/tcp_distance_invariant.py` | the rotation-free `d(t)` test and its fast/slow/attitude/gap decomposition |
| `scripts/body_reference_bakeoff.py` | six candidate body reference points ranked by `d(t)` invariance alone |

---

## The lever-arm failure, resolved: a world-frame yaw that was never modelled (2026-09-19, later)

The fit on CALIB_bodytcp_20260919_123133 was chased through four hypotheses. Three were wrong, and two of those
were wrong because of how the evidence was measured rather than what it said. The record below keeps the
corrections, because the measurement mistakes are as reusable as the result.

### What was actually wrong

**Positions and rotations were in two different worlds.** `p_body` and `p_cube` come from `align_from_plane`,
whose world is defined by the TABLE NORMAL. `R_world_imu` comes from the IMU, whose world is defined by GRAVITY.
The two share +Z and differ by an unknown yaw. Writing `v_body = R_imu^T · v_table` then gives
`R_imu^T · R_yaw · R_imu · C · t`, which is not constant -- it varies with the rotation itself. That is exactly
the observed symptom: a body-frame offset whose z component walked -25.3 → -0.4 → +2.0 → +44.5 mm across
attitude bins that should all have reported one constant.

`body_tcp_fit` was designed with `R_world_body = rot_z(yaw) @ R_imu @ R_imu_body_rp` for precisely this reason,
and `fit_imu_yaw` exists to solve that yaw. It was never run -- every fit above was made at yaw = 0.

Searching the yaw:

| | yaw 0 | best yaw (99°) |
|---|---|---|
| residual p50 | 13.1 mm | **6.9 mm** |
| inliers < 10 mm | 39.6% | **72.2%** |
| body-frame bin drift (z) | 58.8 mm | 27.0 mm |
| rms | 26.5 mm | 24.0 mm |

### Why the take still cannot be frozen

The yaw-search curve is FLAT: depth 0.10-0.12 against the 0.60 gate the fitter requires. The take contains
**35° of yaw (-12.5 to +22.7) against 64° of tilt** -- the operator tilted the wrist and barely turned it about
the vertical. With the yaw unpinned, `|t|` lands at 104 mm, larger than the device, and means nothing.

This is the fitter's own observability test doing its job, and it is the one gate the take fails.

**Next take: a deliberate yaw segment.** Rotate about the world vertical through at least 90°, spread through
the take rather than in one burst, in addition to the tilt the take already has well.

### Corrections to what was reported earlier in this session

- **"IMU verified, ICP vs IMU r = 0.969" was inflated.** That correlation was taken over per-frame rotation
  MAGNITUDES including hundreds of near-zero frames, where both series sit at zero and correlate trivially.
  Restricted to rotations above 0.8°, the magnitude correlation is 0.253. Magnitudes also say nothing about the
  AXIS, which is what a wrong rotation would corrupt.
- **The honest IMU verdict, measured properly:** at a 1-frame baseline (rotation p50 4.8°, ICP correspondence
  error 1.9 mm), the ICP and IMU rotation axes agree to **5.4° p50 / 13.3° p90** after removing a constant
  mounting rotation of 127.4° (det +1.000, a proper rotation, so absorbable). Longer baselines are useless for
  this test -- cloud overlap collapses and ICP itself fails (axis error 95° at a 15-frame gap, correspondence
  error 6.6 mm). The IMU is sound.
- **"Half-vs-half repeatability" was the wrong headline metric** and produced |t| = 296 mm from an
  unobservable half. Use interleaved splits and bootstrap.
- **The 0.54 mm noise floor is a still-unit number** and does not bound a moving unit.

### What is now fixed and worth keeping

**Canonical rigid body tracker** (`scripts/body_canonical_tracker.py`): each frame registered to one canonical
model, translation-only, never integrated. The first version built its model from voxels seen in many FRAMES,
which meant the surface facing the camera at the take's dominant attitude -- 194 of 245 frames sat within 15° of
the first -- so registration failed wherever the body had turned, correlating +0.828 with attitude and 0% of
frames under 4 mm beyond 30°. Weighting each frame by the inverse population of its attitude bin fixes it:

| | before balancing | after, visibility 0.05 |
|---|---|---|
| registration residual p50 / p95 | 3.57 / 11.82 mm | **3.31 / 5.44 mm** |
| corr(residual, attitude) | **+0.828** | **-0.371** |
| residual p50 at attitude ≥ 30° | 9.15 mm | **3.00 mm** |

Frozen at visibility 0.05, chosen on flatness across attitude rather than median alone (0.20 keeps 236 voxels
and registers at 15.8 mm; 0.35 keeps none -- no surface is visible from every attitude, because a solid body
occludes itself, so the model must be a union and not an intersection).

**Cube geometric centre** (`scripts/cube_center_planes.py`): RANSAC faces plus the known 20 mm half-edge, using
no orientation estimate so it cannot alias with `R(t) @ t`. Recovers a centre on all 497 hold frames (2 faces on
444, 1 on 53) and shifts it 18.3 mm p50 from the visible centroid -- the right magnitude for a 40 mm cube. It
did NOT improve the fit (z bin drift 58.8 → 71.8 mm), so the cube's visible centroid was **not** the limiter.
Kept because it is correct, not because it helped here.

---

## Calibration is now gated, and yaw-first (2026-09-19)

`body_tcp_fit --track` changed from a fit that prints numbers to a fit that refuses to produce a calibration
unless the take can support one.

**Yaw fitting is the default path.** Positions live in the table-normal world and rotations in the gravity
world; using `R_imu` directly is a frame error, not a simplification. `--no-yaw-fit` remains, documented as a
diagnostic for reproducing that error deliberately, and is what the regression fixture drives.

**A separate gate on VERTICAL-AXIS yaw**, because a gate on total rotation passes a take that cannot calibrate:
the 2026-09-19 right take has 64° of tilt and 35° of yaw and sails through the rotation gate. `yaw_excitation()`
reports the yaw span alone and gate F requires ≥ 90°.

**Thirteen freeze gates (A–M)**, all printed, and `--out` is refused unless every one passes -- a calibration
that failed a gate is worse than none, because it looks deliberate. On the existing take:

```
[PASS] A coverage 98.8%   [PASS] B switches 0   [PASS] C sync 3.13 ms   [PASS] D registration p95 5.44 mm
[PASS] E rotation max 64 deg, cond 16.5        [FAIL] F yaw span 35.2 deg      [FAIL] G yaw depth 0.114
[PASS] H yaw bootstrap 3.97 deg                [PASS] I |t| 119.8 mm           [FAIL] J residual p50 11.7 mm
[PASS] K interleaved cond 16.6/16.3            [PASS] L halves agree 3.5 mm    [FAIL] M pose drift 50.3 mm
```

Half-vs-half is gone; K and L use interleaved splits and require each subset to be independently conditioned.

**`scripts/yaw_regression_test.py`** keeps the failing take as a fixture. Turning the yaw correction off must
make the fit measurably worse; if a future change to the world-frame handling erases the distinction, the
numbers converge and the test fails.

| body reference | yaw off → on (residual p50) | inliers < 10 mm |
|---|---|---|
| canonical rigid | 14.0 → 11.7 mm (yaw +88.1°) | 6.1% → 24.1% |
| centroid | 11.0 → 8.4 mm (yaw +107.1°) | 40.4% → 66.9% |

**`body_track_export.py` now warns before dropping downstream fields.** It rewrites `track_export.npz` from
scratch, so re-running it silently deleted `*_body_xyz_rigid` and the TCP observations, and the next fit then
failed for a reason unrelated to what changed.

### Recording protocol for the yaw-identifiable take

Cube held from start to finish. Rotation spread through the take, not in one burst, and slow enough that RGB-D
registration keeps up.

| t | motion |
|---|---|
| 0–2 s | static |
| 2–5 s | yaw +30–45° |
| 5–8 s | small translation |
| 8–11 s | pitch |
| 11–14 s | yaw −45–60° |
| 14–17 s | small translation |
| 17–20 s | roll |
| 20–23 s | yaw +45–60° |
| 23–26 s | translation + moderate rotation |
| 26–29 s | yaw the other way |
| 29–32 s | static |

Target: vertical-axis yaw span ≥ 90° (100–120° preferred), tilt kept, cube held throughout.
