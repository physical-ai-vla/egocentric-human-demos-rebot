# HandUMI RGB-D rigid-body 6DoF tracking — Stage A (M2A)

ActiveUMI tracks a rigid VR controller and applies a fixed controller→EE transform. We keep the *principle* and drop the
VR: the **HandUMI body itself is the tracked rigid object**, seen by the head-mounted Orbbec RGB-D.

```
Head Orbbec RGB-D ──► HandUMI rigid body 6DoF  ──► body→TCP (fixed) ──► TCP 6DoF ──► relative EE  ⟷  robot C_state
   (localization)         T_depthcam_body            Stage B              Stage C/D
```

Head C922 stays the policy observation, the Arducam virtual-wrist pipeline is untouched, and **grip never comes from
vision** — the Feetech encoder is authoritative.

## Status

| stage | what | state |
|---|---|---|
| **A** | RGB-D → HandUMI rigid body 6DoF, one hand, offline | **FROZEN as an optional branch (2026-09-11)** — see "Freeze" below |
| B | body → TCP | not built (needs a pinch-centre calibration once A passes on real data) |
| B' | dual-hand, inter-hand transform | not built (`interhand_identity_swaps()` exists in `depth_qa`) |
| C | head RGB-D odometry → world TCP | not built — deliberately, until A passes (§13) |
| D | wrist VIO / fusion, HOME dock, reBot replay benchmark | not built |

## Freeze (2026-09-11)

The **main** HandUMI tracking path is now Arducam + ICM42688P VIO through the existing M1 stack
(`ego_teleop/tracking/WristPoseProvider` → OpenVINS → ΔTCP). This RGB-D branch is kept, not deleted: it may serve as an
`RGB-D external tracking baseline` for the paper. **No further development without a working container runtime.**

What was established on real data before the freeze (episode `Hpilot_rgbd_stageA_20260911_111811/episode_000001`,
55.6 s, head-mounted, background NCC 1.00→−0.01 so the camera genuinely moves):

* **ICP: FAIL.** With a clean operator-picked ROI it registers frame 0 (residual 6.6 mm, conf 0.82) then loses every
  subsequent frame — 1/1667 valid, one 55.57 s loss event. Earlier 0/300 runs were an over-generous ROI (forearm and
  desk inside the box), not the method; but even a clean ROI does not track. A thin ROI under-constrains the 6-DoF fit.
* **Frozen registration input** (keep these together — the mesh md5 is the guard):
  `assets/handumi/left_tracking_body_fisheye_support.stl` md5 `6e5f09ce5027e8575716d2599c69621d`,
  bbox `(251,156,208,138)`, `init_mask_left.png` (4690 px). Mask extent 0.125×0.109×0.085 m vs part
  0.140×0.114×0.083 m → 0.89/0.96/1.03 on the three axes.
* **Never use the `~/Desktop/handumi-print` copy of that STL**: identical geometry translated 93.9 mm for slicing, its
  origin lies outside the part, and `tracking_mesh.validate()` rejects it. It would offset every later `T_B_TCP`.
* **FoundationPose was never run.** Docker could not be installed on gpu-4090 (no sudo) and the native route fails:
  conda-provided `cuda-nvcc 12.1` installs fine without root, but `nvdiffrast` still fails with
  `RuntimeError: Error compiling objects for extension`. The only cluster machines with `nvcc` are an ARM Jetson Orin
  and a Blackwell RTX 5080 on CUDA 13.2. A verified 300-frame packet is preserved at `/tmp/fp_pkt_left_300` (247 MB).
  The `foundationpose` adapter remains **UNTESTED**.

The original question, for the record — it can still only be answered on real data:

> Can head-mounted RGB-D track the HandUMI rigid body's metric 6DoF well enough for robot training?

## Two things must arrive before a real pilot can run

1. **An RGB-D episode containing a HandUMI.** The collector already records one (`head_depth` camera, role `aux_depth`,
   in `hardware_handumi_preimu.yaml` / `hardware_handumi_v1.yaml` — it needs `sudo` on macOS for the Orbbec UVC
   interface). No such episode exists yet: `datasets/human_handumi_raw/Hpilot/*` holds sessions with no episodes, and
   `datasets/HumanRGBD_v1` predates the HandUMI hardware (bare hands + cubes). The recorder-layout reader has been
   exercised end-to-end against a real collector episode (mp4 ↔ depth-PNG join, capture timestamps, depth scaling), so
   what is missing is the recording itself, not the plumbing. **This does not wait for the ICM-42688P**: Stage A uses no
   IMU at all.
2. **A tracking-only rigid mesh.** See below — this is the real blocker.

## The tracking mesh (§3) — and why it cannot be generated automatically

The mesh must contain **only parts fixed to the controller body**. Moving parts (thumb / index-middle finger links,
crank plate, connecting links) change pose with the jaw and would make the tracker fight the grip.

`handumi-hw/hardware/STL/{left,right}_handumi/` has the 12 per-part meshes, and the classifier splits them correctly:

```
rigid (7)   fisheye_camera_main_support, <side>_controller_support, main_support_cover_plate,
            camera_mount, hand_support_base, <side>_servo_controller_support, servo_controller_cover
moving (5)  <side>_thumb_link, <side>_index_middle_finger_link, crank_mechanism_plate, connecting_link_1/2
```

**But every part is exported at its own local origin** — each part's bounding box is centred near zero, so the assembly
relationship is simply not in the files and the parts cannot be unioned. Two ways out:

* **Preferred — export one assembled rigid body from CAD (Shapr3D)** in the body frame, and point
  `configs/handumi/depth_pose.yaml` `mesh.<side>` at that `.obj`/`.stl`.
* Otherwise fill the per-part poses in the generated assembly YAML:

```bash
.venv/bin/python -m handumi_collector.tools.make_tracking_mesh --template left     # -> assets/handumi/left_tracking_body.yaml
#  ... fill every `pose:` (translation_m + quaternion_xyzw) from CAD ...
.venv/bin/python -m handumi_collector.tools.make_tracking_mesh --build assets/handumi/left_tracking_body.yaml
.venv/bin/python -m handumi_collector.tools.make_tracking_mesh --validate assets/handumi/left_tracking_body.obj
```

A missing pose is an **error**, never a guess: a silently wrong assembly becomes a silently wrong TCP.

### Body frame B

Origin at a repeatable CAD datum on the rigid body; `+x` gripper forward (approach), `+y` gripper left, `+z` gripper up
— the same axes as the TCP frame, so body→TCP is a pure offset. The datum must be named in the mesh's `.meta.yaml`.
Poses are stored as `xyz + quaternion xyzw`; RPY exists only in UI/debug output.

## Running the real Stage-A pilot

Dataset: **`Hpilot_rgbd_stageA`** — not H120. One hand (left) first; the right hand and dual-hand come only after a
single-hand PASS, so tracker instability and identity confusion are never debugged at the same time.

### Before recording (ten seconds, saves the take)

```bash
sudo .venv/bin/python -m handumi_collector.tools.check_rgbd_ready --live
```

The recorder writes `<stream>_depth/intrinsics.json` **only if the camera reported intrinsics at open()**. If it did
not, the episode still records and still looks complete — and the tracker then refuses it, because a guessed focal
length is a silent metric-scale error in every pose downstream. This check catches that before the take, not after.

### The take: ~20–30 s, one hand

Hold still at both ends and press the HOME mark, so jitter and return drift are measured against operator marks rather
than against the tracker's own output:

```
START HOME, hold still 2 s      -> mark home_leave when you start moving
  X / Y / Z translation, slow
  roll / pitch / yaw
  translation + rotation together
  a few moves at real manipulation speed
  occlude part of the HandUMI with the other hand for 0.5-1 s, then reveal it
  cube grasp -> lift -> place
  put the HandUMI back in the SAME physical spot          -> mark home_return
END HOME, hold still 2 s
```

`home_leave` / `home_return` are the marks the collector already emits (`Session.mark()`). Without them the still
windows are auto-detected *from the tracked pose*, which a frozen tracker also satisfies — the QA says so in its flags,
but the number is worth much less.

### Right after recording, before taking the rig off

```bash
.venv/bin/python -m handumi_collector.tools.check_rgbd_ready --episode <episode> --mesh assets/handumi/left_tracking_body.obj
```

### Then track and look at it

```bash
.venv/bin/python -m handumi_collector.tools.track_handumi_rgbd \
    --episode <episode> --side left --mesh assets/handumi/left_tracking_body.obj --pick
.venv/bin/python -m handumi_collector.tools.inspect_depth_pose --episode <episode> --side left
```

Everything lands in `<episode>/derived/depth_pose_<backend>/`: `<side>_body_pose.parquet`, `<side>_depth_track.json`
(full provenance + QA), `<side>_report.txt` (the §37 report), `<side>_overlay.mp4`, `<side>_trajectory.png`.
**Raw is never written to** — a tracker crash can only cost the derived directory.

### The first verdict is the overlay, not the millimetres

Real data has no ground truth, so judge these first, on `<side>_overlay.mp4`:

1. the mesh stays on the HandUMI body through the whole take
2. XYZ directions match the real motion
3. roll/pitch/yaw signs match
4. the body is not lost at normal manipulation speed
5. after the occlusion it reacquires the *same* pose
6. the cube is never mistaken for the HandUMI during the grasp
7. no 5–10 cm jump anywhere

Only then read the numbers. And remember what `fitness = 1.00` bought us on synthetic data: nothing.

### HOME return drift

With a still window at both ends, QA reports `inv(mean pose of the first) @ (mean pose of the last)` as
`return_translation_mm` / `return_rotation_deg`. **There is no mechanical dock yet**: the operator puts the HandUMI back
by hand, so this bounds *tracker drift and placement repeatability together*. It is an upper bound on drift, never
drift itself, and the report and flags say so. Thresholds (`qa.home_return`, provisionally 15 mm / 5° pass) are loose
for that reason and should tighten once a dock exists.

### No CUDA on the Mac: the packet workflow

`foundationpose` needs torch+CUDA, nvdiffrast, the FoundationPose repo and its weights. Record here, track there:

```bash
mac$ ... track_handumi_rgbd --episode <ep> --side left --mesh <mesh> --pick --export-packet /tmp/pkt_left
mac$ rsync -a /tmp/pkt_left gpu:/data/
gpu$ ... track_handumi_rgbd --episode /data/pkt_left --side left --backend foundationpose
mac$ rsync -a gpu:/data/pkt_left/derived/ <ep>/derived/
```

A packet is a flat `color/ depth/ intrinsics.json init_mask.png mesh.obj packet_meta.json timestamps.csv` directory —
the same reader handles it, so the identical command runs on both machines.

## Backends

| backend | runs on | how it works | use |
|---|---|---|---|
| `icp` | CPU (Mac) | numpy/scipy point-to-plane ICP, scene→model; multi-start random-orientation registration on the first frame | the local baseline, and the fallback number to compare against |
| `foundationpose` | CUDA | NVLabs FoundationPose `register()` / `track_one()` | the intended backend; **adapter written, never executed — there is no GPU here** |
| `mock` | CPU | replays given poses | tests only; refuses to run without explicit poses |

## Verification so far (synthetic, with ground truth)

`tools.synth_rgbd_pilot` renders a rigid body along the §14 test sequence (still → X/Y/Z → roll/pitch/yaw → combined
SE(3) → fast → partial occlusion → still) into an Orbbec-shaped 848×480 RGB-D stream and writes the true poses. This
separates "the pipeline is wrong" from "the sensor/scene is hard"; it does **not** stand in for a real pilot (no sensor
noise model, no multipath, no materials, and a simple body).

| episode | valid | abs translation err (med/p95) | abs rotation err (med/p95) | verdict |
|---|---|---|---|---|
| noiseless, 8 s, 240 f | 240/240 | 0.84 / 0.95 mm | 0.13 / 0.26 ° | PASS |
| +2 mm depth noise | 235/240 | 0.91 / 1.10 mm | 0.17 / 0.31 ° | PASS (5 lost during occlusion, reacquired) |

~35–50 fps single-hand on the Mac CPU.

## Known limitations — measured, not hypothetical

* **The ICP baseline can be confidently wrong under large inter-frame motion.** Squeezing the same motions into 1.5 s
  makes the pose drift 1 → 4 → 11 → 28 → 89 mm while fitness stays 1.00, the depth residual stays ~1.7 mm and model
  coverage stays 0.94. Nothing inside the backend detects it. QA therefore flags, from the observed motion, every frame
  whose step exceeds the backend's correspondence radius (`steps over the trust radius` in the report) and refuses a
  PASS verdict. **This is why the overlay video is part of QA, and why FoundationPose is the intended backend.**
* **A constant-velocity ICP seed is unstable** and is off by default. Against ground truth it predicts far better
  (p95 1.2 mm vs 8.1 mm for the last pose), but the increment carries both frames' tracking error and re-applying it
  amplifies: 1 → 2 → 5 → 14 → 37 → 107 mm over six frames, 40/240 valid, versus 240/240 and 0.97 mm worst error with the
  plain last-pose seed. Re-enable only with damping or a filtered velocity.
* **`icp` has no global re-detection.** It recovers only near the last good pose; a body that leaves the frame and
  re-enters elsewhere stays LOST until the run is re-initialised.
* **Auto-detected "static" segments are derived from the tracked pose**, so a frozen tracker also reads as static. QA
  says so in its flags. Mark still windows during recording (`static_begin` / `static_end` events) and they win.
* **open3d is not used.** The 0.18 wheel available for this Mac segfaults on every numpy→Eigen conversion under
  numpy 2 (`translate`, `scale`, `transform`, `registration_icp`, and `RaycastingScene.cast_rays` by extension). The
  mesh layer is trimesh (what FoundationPose consumes anyway) and ICP is written out in `pose/icp_core.py`.

## Never hides a failure (§17)

An uncertain frame is `tracking_valid=false` **with no pose at all** — `x/y/z/q*` are NaN in the table. Nothing here
copies the last pose forward, interpolates across a gap or smooths. Whether short gaps may be bridged is a decision for
a later processing stage, on top of this raw tracker output.

## Thresholds

All of them live in `configs/handumi/depth_pose.yaml`, and they are **provisional engineering criteria, not scientific
bounds** — freeze them against the first real pilot distribution. The report prints every raw number, so re-freezing
never needs a re-run. Verdict vocabulary: `PASS` | `NEEDS_WORK` | `FAIL`.

## Files

```
handumi_collector/pose/
  depth_hand_tracker.py     the interface: RigidPoseEstimate, DepthHandPoseTracker, backend registry
  rgbd_io.py                RgbdEpisode (recorder + flat layouts), CameraIntrinsics, backproject/project
  tracking_mesh.py          rigid-mesh load/validate/assemble, part classification, unit handling
  icp_core.py               point-to-plane ICP + voxel downsampling (numpy/scipy)
  depth_backends/           icp | foundationpose | mock
  depth_qa.py               §15/§16 metrics + the §37 report
  depth_run.py              offline run driver, derived output, portable packet export
  depth_config.py           configs/handumi/depth_pose.yaml
handumi_collector/tools/
  track_handumi_rgbd.py     the pilot CLI
  inspect_depth_pose.py     overlay video + trajectory plots + ground-truth comparison
  make_tracking_mesh.py     assembly template / build / validate
  synth_rgbd_pilot.py       synthetic RGB-D with ground truth (the self-test)
  check_rgbd_ready.py       pre-flight (--live) and post-flight (--episode) check for a pilot
tests/handumi/test_depth_pose.py
```
