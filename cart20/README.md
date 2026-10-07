# ego_cart20: ego-only CART20 pretraining data for X-VLA v4

Turns human egocentric demonstrations (HandUMI) into X-VLA v4 training samples. The samples have the same meaning and channel layout as the reBot R312c RELCART20 REL-only data used for fine-tuning.
Status, numbers and open questions: `SPEC_STATUS_2026-10-01.md`. Training recipe: `~/umi_bridge/RELCART20_RELONLY_D6_RECIPE.md`.

## Conventions (frozen, `ego_cart20/config.py`)

| | |
|---|---|
| rows | stored at 15 Hz on a timestamp grid `t_s + n/15`; `t_s` = first instant both arms are valid (task start) |
| time offsets | **UMI_DT = 3/59.94 s = 50.05 ms**, interpolated on the RAW trajectory. The 15 Hz row spacing is never used. REL16 = 0.80 s |
| action | `A_k = inv(T(t)) @ T(t + k*UMI_DT)`, k = 1..16, both arms at the same instants. Current-anchor, never sequential |
| CART20 | `[L xyz rot6d g, R xyz rot6d g]`: grippers at 9 and 19, LEFT first |
| action32 | `[16, 32]`: 0:20 = CART20, 20:32 = AUX12 = **exactly 0** (hard error otherwise, NaN included) |
| state.npy (default) | RELCART20 task anchor: per arm `inv(T(t_s)) @ T(t)`, `[L pose9, R pose9, gL, gR]` |
| state_prevrel.npy | diagnostic only: `inv(T(t)) @ T(t - UMI_DT)`, `[L pose9 g, R pose9 g]`, plus `state_prev_valid` |
| gripper | continuous aperture, 0 = closed, 1 = open (caliper mm / 80 mm); future values interpolated, never binarized |
| rot6d | first two ROWS of R (umi `mat_to_rot6d`); quaternions are wxyz |
| validity | interpolation only between valid raw samples ≤ 50 ms apart. A training row needs: both arms valid at t, all 16 targets valid, a head frame within 20 ms. No padding |
| canonical frame | per-arm task start (`geometry/canonical_frame.py`). A left-multiplied frame cancels in every label, so only the tool frame (fixed by the exporter) matters |

## Pipeline

```bash
PY=~/xvla-mac/bin/python
# 1  source adapter (HandUMI MASt3R + IMU-VI + gcal1 gripper) -> raw episodes
$PY ego_cart20/sources/handumi_export.py --out ~/c8/ego_cart20_v2_raw
# 2  raw -> processed episodes + {train,val,test}_manifest.jsonl (frozen HandUMI split)
$PY ego_cart20/scripts/convert_dataset.py ~/c8/ego_cart20_v2_raw ~/c8/ego_cart20_v2
# 3  validation (hard checks, exit 1 on failure), report + 3D samples, gcal1 parity
$PY ego_cart20/scripts/validate_dataset.py ~/c8/ego_cart20_v2
$PY ego_cart20/scripts/report_dataset.py ~/c8/ego_cart20_v2 ~/c8/ego_cart20_v2_report
$PY ego_cart20/validation/parity_gcal1.py ~/c8/ego_cart20_v2_raw <episode_id> ...
# 4  LeRobot v3 export for the REL-only D6 recipe
$PY ego_cart20/scripts/export_lerobot.py ~/c8/ego_cart20_v2 ~/c8/ego_cart20_v2_lerobot
# 5  submit (4090, Ray, physical GPU index)
launch/submit_ego_cart20v2.sh 1
# tests
$PY tests/test_contract.py
```

Single episode: `scripts/convert_episode.py <raw_dir> <out_dir>`. Native reader: `dataset/xvla_ego_dataset.XVLAEgoDataset(root, split, load_images)`. Weights: `.weights(order_balanced, stage_weights)`.

## Raw episode format (`io/raw_episode_loader.py`)
`raw_episode.json` holds episode_id, instruction, stack_order, task, videos, tool_frame, gripper_calibration and provenance.
`raw_episode.npz` holds, per arm on its own clock: `{arm}_t_ns`, `{arm}_position`, `{arm}_quaternion` (wxyz), `{arm}_valid`, `{arm}_gripper` (calibrated 0..1). Per camera: `cam_{c}_t_ns`, `cam_{c}_frame`.
Any new source only has to write this format.

## Processed episode (`io/processed_writer.py`)
`timestamps, row_valid, train_rows, state [N,20], state_prevrel [N,20], state_prev_valid [N], cart20 [Nv,16,20], action [Nv,16,32], left_pose/right_pose [N,4,4] (canonical), left/right_gripper [N], stage_id [N], camera_valid [N,3]`, plus `<camera>/frame_index.npy` and `metadata.json`. Row j of cart20/action belongs to row `train_rows[j]`.

## Not done / limits
- `stage_id` is −1 everywhere. A gripper-only stage detector was rejected: on HandUMI, grasp and release differ by only about 5–8 mm of aperture. Pass annotations to `labels/stage.stage_ids` once they exist.
- No test split: the frozen HandUMI split has none.
- The left and right HandUMI poses live in separate SLAM maps, so no cross-arm relation exists. All labels are per-arm relative.
