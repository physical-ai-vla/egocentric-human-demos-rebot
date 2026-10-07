# Camera-side freeze runbook (do this before the IMUs arrive)

Goal: freeze the **wrist observation** so that after the IMUs arrive only `Arducam RAW → VIO` is added.

```
HEAD   human C922 640x480            = robot C922 640x480                       (already shared)
WRIST  human Arducam fisheye RAW ──► VIO / TCP / C_state                        (raw kept forever)
                              └──► virtual camera (K_target + D_target + rpy) ──► shared policy input with robot wrist C922
```

Order matters: **physical mount first, rotation last.** A translation offset between the two optical centres is parallax and
cannot be removed by any 2-D reprojection; `--fit-rpy` must only clean up a small residual rotation.

## 1. Robot wrist C922, in its real recording mode
```bash
.venv/bin/python -m handumi_collector.tools.calibrate_c922 --side left  --index <robot_left_wrist_cam>  --width 640 --height 480
.venv/bin/python -m handumi_collector.tools.calibrate_c922 --side right --index <robot_right_wrist_cam> --width 640 --height 480
```
Writes per-side `K_target` + `D_target` into the next `configs/calibration/virtual_wrist_vNNN.yaml` (the other side and `rpy_deg`
are carried over). Do **not** keep using v001 (head-C922 intrinsics scaled to 640x480) for real data — 640x480 modes crop/scale
differently per camera. Aim for RMS < 0.5 px, board covering the corners, tilted views included.

## 2. Arducam fisheye, mounted in its final position
```bash
.venv/bin/python -m handumi_collector.tools.calibrate_fisheye --side left  --index <arducam_left>  --width 1920 --height 1080
.venv/bin/python -m handumi_collector.tools.calibrate_fisheye --side right --index <arducam_right> --width 1920 --height 1080
```
Writes `fisheye_<side>_vNNN.yaml` (Kannala-Brandt). **Any lens/mount change ⇒ a new version.** Record the mount revision name
(e.g. `handumi_left_mount_v2`) and pass it to the QA / render tools; it lands in the provenance block.

## 3. Physical mount matching (the actual bottleneck)
Match, as closely as practical, for each side:
```
distance from TCP to the optical centre   ≈ same as the robot wrist C922
forward (optical axis) direction          ≈ same
pitch / yaw of the mount                  ≈ same
```
Millimetre accuracy is not required; the lever arm and viewing direction are what matter. Iterate with step 4.

## 4. Same-scene QA on a fixed cube layout (5-10 layouts)
```bash
.venv/bin/python -m handumi_collector.tools.view_match_qa --side left --out qa/left_layout01 \
    --robot-index <robot_wrist_cam> --human-index <arducam_left> --pick --mount-revision handumi_left_mount_v2 --scene layout_01
```
`--pick`: click the same feature in the robot frame then in the virtual frame (7-10 points: 3 cube centres, 4 plate corners,
2 fingertips, pinch centre). `n`/`f` label the next pair near/far, `u` undoes, `q` finishes. Manual points are the trusted path;
the colour-blob detector in `configs/handumi/view_match.yaml` is only a convenience for repeated identical layouts.

Read the groups, not just the mean:

| signature | meaning | action |
|---|---|---|
| `E_view_near ≫ E_view_far` (`near_far_ratio` > 2) → `PARALLAX` | optical centres differ | **move the mount** (step 3) |
| `E_view_edge ≫ E_view_center` → `FOV_MODEL` | `K_target`/`D_target` wrong | redo step 1 at the real resolution |
| `scale_ratio_mean` ≠ 1 → `SCALE` | focal length mismatch | redo step 1 |
| fit wants > `mount_mismatch_rpy_deg` (5°) → `MOUNT_MISMATCH` | mount orientation wrong | fix the mount, do not bake it into `rpy` |
| all small, uniform residual | pure residual rotation | accept `rotation_fit.rpy_proposed_deg` |

Then, only when the mount is right, add `--fit-rpy` and write the accepted rotation:
```python
from handumi_collector.pose.virtual_view import load_virtual_wrist, write_virtual_wrist
v, d = load_virtual_wrist()
d["rpy_deg"]["left"] = [...]            # rotation_fit.rpy_proposed_deg from view_match.json (only if `recommended: true`)
write_virtual_wrist(d["K_target"], d["size"], D_target=d["D_target"], rpy_deg=d["rpy_deg"], source="rpy from view_match qa/left_layout01")
```
Freeze target (an **engineering** acceptance criterion, not a fixed scientific bound): `E_view_all < 8 px`
(config `thresholds.pass_px`) on every layout, with no PARALLAX / FOV_MODEL / SCALE / MOUNT_MISMATCH flag, and a fitted residual
rotation ≤ 5°.

**The gate is the MEAN.** `pass_px` / `warn_px` are compared against `E_view_all_px` (mean over the scene's landmarks);
`E_view_median_px` and `E_view_p95_px` are **diagnostics** and never change the verdict — so `mean 6.9 / median 6.8 / p95 8.9`
is a correct PASS, not a bug. If p95 should ever gate too, add a separate `p95_pass_px` threshold and extend
`verdict_and_flags()`; never silently reinterpret `pass_px`. Each run writes `report.txt` with the distribution, so the criterion
can be revisited once 5-10 real layouts have been measured:

```text
side left   scene layout_01   mount handumi_left_mount_v2   fisheye fisheye_left_v003   target virtual_wrist_v002
E_view mean       4.3 px
E_view median     4.1 px
E_view p95        7.6 px
near              4.8 px
far               4.0 px
near/far          1.20
center            3.9 px
edge              5.1 px
scale ratio       1.02 ± 0.03
fit RPY           [0.8, -1.2, 0.4] deg
points            10 (7 near / 3 far)
verdict           PASS   (gate: mean < 8.0 px; median/p95 diagnostic only)
flags             NONE
```

Also record, per side, from step 1: `HFOV, VFOV, fx, fy, cx, cy, RMS, resolution, device identity, timestamp` — written to
`configs/calibration/robot_<side>_c922_<width>_vNNN.yaml` by `tools.calibrate_c922`, which prints the same block and the delta
against the provisional 49.1° HFOV.

## 5. Render the policy view (derived, regenerable)
```bash
.venv/bin/python -m handumi_collector.tools.render_virtual_wrist <session_or_episode> --mount-revision handumi_left_mount_v2
```
→ `<episode>/derived/virtual_wrist/{left,right}_wrist_c922like.mp4` + `virtual_wrist.json` (provenance: fisheye version,
virtual_wrist version, K/D/rpy/size, mount revision). **Never delete the raw fisheye**: if the virtual parameters turn out wrong,
all of H120 can be re-rendered from raw.
