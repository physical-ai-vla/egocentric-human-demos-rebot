# HRA: right-arm approach prior from HandUMI ego data

HRA ("Human Right-arm Approach") is the single-arm task "Approach to the red cube". The data is egocentric HandUMI takes
(right hand only, 10 s each). The goal is to learn a right-arm approach prior that moves the reBot right arm towards the cube.
The narrative, numbers and lessons are in Part II of the report
(`docs/reports/full/part2_approach_en_full.pdf` and `docs/reports/methods_en.pdf`).

Three dataset generations were built, in this order:

| generation | data | label frame | wrist image | anchor / state |
|---|---|---|---|---|
| HRA_red v1 | 200 takes on 10-03, 166 kept (151 train / 15 val) | HandUMI TCP | pinhole re-render with a *guessed* robot cam | start-anchored RELCART20 |
| HRA_red robotcam v2 | same 151/15 split | robot TCP via hand-eye `M` | measured robot C922 (52 deg hfov) | start-anchored |
| HRA_A100 | 107 takes (10-06) + 27 (10-07), taped-cross origin protocol | robot TCP, task frame G | measured C922 | A = start-anchored, B = origin frame F, C/CT* = robotized and table-aware, state in F or G |

All generations keep the CART20 tensor contract (32-D action, 20-D RELCART20 state). The left arm is a dummy (identity pose, gripper 0)
and is loss-masked together with the gripper, so 9 of the 20 dims are supervised.
Training uses the loss-mask REL-only trainer (`training/relonly/rel16_relonly_lossmask.py`), with domain 20, batch 8, AdamW8bit and bf16.
Training is documented in `training/README.md` and serving in `deploy/README.md`.

## External dependencies (not in this folder)

- `~/ego_cart20` package. In this repo it is `pipeline/cart20/` (the `ego_cart20` Python package). It provides `right_only.py` (export / convert / sanity), `cube_pnp.py`,
  `scripts/export_lerobot_right_only.py` and `scripts/export_lerobot_right_only_robotcam.py`.
- `~/ego_collector` (collector, `scripts/umi_export_orbslam.py`, `handumi_collector.robotlike.{start_match,c922_view}`, calibration yamls
  under `configs/calibration/`, raw sessions under `datasets/human_handumi_raw/HRA_red|HRA_A100`). This lives in the collector part of the repo,
  which another owner maintains.
- MASt3R-SLAM per-frame poses: `pipeline/mast3r_pose/`, run as Ray jobs on the 5090/5080, plus `robotlike_session_check.sh` from the collector.
- Scripts hard-code `~/c8/hra_red` and `~/c8/hra_a100` as the working roots, and `~/xvla-mac/bin/python` as the Mac interpreter.
  Run them from a checkout laid out the same way, or edit `A=`, `OUT=` and `H /` at the top of each script.

## 1. HRA_red v1 (`hra_red/`)

1. **MASt3R poses.** Run the right-wrist MASt3R per-frame jobs (5080 and 5090 split). Then run `finalize_mast3r.sh`.
   It waits for every Ray job, checks that all 200 episodes are complete, retries missing ones on the 5090 once, and then pulls the poses. Its last step writes the
   `SKIP_CHECK=1: poses only` line that releases the chain.
   Variants used on the day: `resubmit_5090b_then_finalize.sh` and `wait_direct_then_finalize.sh`.
2. **`chain_after_mast3r.sh`** (CPU only). It runs these steps in order:
   - `python -m ego_cart20.right_only export <session> raw`: metric scale from **cube PnP** (cube edge 3.8 cm). IMU-VI scale is unobservable on slow approach motion.
   - `right_only convert raw processed --val-frac 0.1`: writes the labels and the loss mask.
   - `scripts/export_lerobot_right_only.py processed lerobot`: writes the LeRobot dataset `ego_hra_red_rightonly_v1_{train,val}`.
   - Writes `scale_qc_summary.json`.

   The per-episode sanity gate is `right_only sanity`, with these limits:
   - 0.1 <= s <= 1.0
   - travel_ratio >= 0.8
   - travel <= 0.8 m

   It takes the 181 scale-valid episodes down to 166.
3. **Train.** `smoke_then_main_4090.sh` runs these steps in order:
   1. QC gate.
   2. Upload with sha256 check.
   3. Wait for a free 4090 GPU.
   4. 300-step smoke run, using the same launcher as the main run: `training/launchers/node4090/train_hra_rightonly_lossmask_d20_b8.sh`.
   5. Automatic verdict.
   6. Main run `HRA-RIGHTONLY-LOSSMASK-D20-B8-300K`, plus the SSD archiver `training/archivers/trackb_archive_v2_relonly.sh`.

   The 5090 variants (`smoke_then_main_5090.sh`, `smoke_5090_then_main_4090.sh`) are unused: the 5090 has a different torch version and launcher.
   `smoke_now_4090.sh` and `smoke_when_ready.sh` are earlier forms of the same chain.
4. **Held-out loss.** `val_eval/val_loss.py` computes the masked loss on Mac MPS (`val_eval/results.json`).
5. **Side tools.** `hrl80_pnp_validation/hrl_val.py` validates cube PnP against the IMU on HRL80; there the PnP/IMU ratio has p50 1.06.

## 2. Robot-side calibration (`hra_red/global_calib`, `base_frame`, `handeye`, `start_*`)

These steps are needed to interpret v1 and to build v2 and A100.

- **Head cam (global cam).** `global_calib/checker_capture.py` captures the checkerboard; `collect_and_solve.py` and `board_frame.py` solve the calibration.
  - Outputs: `global_cam_intrinsics*.json` (f 728.7 px) and `global_cam_calib.json`.
  - `board_state.json` gives T_base_cam from 4 corners touched with the TCP (residuals 2-6 mm).
- **Link human takes to the robot base.** `base_frame/hra_to_base.py` and `hra_to_base_diag.py` do the link.
  - Head cube PnP, then wrist cube PnP, then IMU gravity.
  - Linked 64 of 166 takes. Judged too noisy to build a dataset on.
  - Found an 18 deg error in the exported camera-IMU rotation. The correction is `R_imu_corr.npy`.
  - `X_cam_tcp_full.npy` = the full HandUMI camera->TCP transform. Use `inv(cp6X @ HX.X)`, not the camera_tcp yaml alone.
- **Robot right wrist hand-eye.** `handeye/plan_poses.py`, `snap.sh` and `sweep.sh` capture the board views; `solve_handeye.py` solves the hand-eye
  (13 views, 0.16 px RMS, f 651.4).
  - Output: **`handeye/robot_right_wrist_handeye.json`** (`X_tcp_cam`, several methods; `park` is the one used downstream).
  - `T_base_board_now.npy` is the board pose used in that solve.
- **Start pose / view matching.** These scripts compare the robot start with the human start:
  - `base_frame/match_view*.py`
  - `start_reach_check.py`
  - `start_view/compare_start_views.py`
  - `start_pose/start_pose_stats.py` (writes `human_start.json`)
  - `head_link/head_cube.py`

  `umi_start_pose.json` is the robot start pose that the loaders/UI read. Candidate start poses are in `start_poses/*.json`.
  `08_medoid_back6_right6_down4.json` is the one the operator chose for CT7.

## 3. HRA_red robotcam v2 (`hra_red/robotcam/`)

1. `robotcam/make_raw_robotcam.py`: `raw/` -> `raw_robotcam/`. It sets T_robot = T_umi @ inv(M), with M = `X_tcp_cam(park)` @ `X_cam_tcp_full`, so the labels are in
   **robot TCP** and need no inference-side convention: `V4_UMI_TCP_PITCH_DEG=0` and `V4_UMI_TCP_OFFSET_MM=0,0,0`. `robotcam/ROBOTCAM.json` records M.
2. `right_only convert raw_robotcam processed_robotcam`.
3. `robotcam/run_export_loop.sh`: re-runs `export_lerobot_right_only_robotcam.py` until both splits finish. This works around the SVT-AV1 dealloc crash;
   the export resumes via `.complete` shards. It renders the wrist view with the measured C922 intrinsics plus the hand-eye.
4. `robotcam/robotcam_smoke_then_main_4090.sh`: smoke run, then main `HRA-RIGHTONLY-ROBOTCAM-LOSSMASK-D20-B8-300K`.
5. `robotcam/ik_check.py`: replays the labels through Pink IK from the robot start.

## 4. HRA_A100: shared origin (`hra_a100/`)

Protocol: a taped cross on the head-camera base plate is the task origin. Each take runs this cycle:
1. 5 s rest.
2. 2 s still with the jaw tip on the cross.
3. 2 s origin hold inside the recording.
4. 3 s "ready".
5. 10 s approach after "go". Training rows start at "go".

Calibration files:
- **`G_frame_calib.json`** is the task frame G. It comes from three robot jaw-tip touches on the tape line: origin (59.0, -74.9, 6.2) mm in base, heading 47.09 deg, table z = -27.1 mm.
- **`G_anchor.json`** and **`B_anchor_F.json`** are the fixed-anchor files that the UI loads (`V4_FIXED_ANCHOR_JSON`) for the G-state models (CT5-CT7) and the F-state model (B).
- `urdf_gripper/` holds the reBot gripper STLs and `tip_box_tcp.npy`, the URDF fingertip box in TCP coordinates. The table-aware contact model uses the box.

Scripts in pipeline order (all under `pipe/` unless noted):

1. `export_one.sh` (root): fisheye export per episode (`umi_export_orbslam.py`), keeping the origin hold.
   `make_c922_export.py` (root) builds the C922-view SLAM inputs.
   `origin_check.py` (root) measures origin repeatability.
2. The MASt3R keyframe/pointmap runs (`runs_kf`, Ray). `origin_scale_kf.py` computes the origin-plane metric scale; it is the second scale source next to cube PnP.
   `origin_scale.py` and `origin_scale_v1.py` are earlier variants.
   `origin_orient.py` measures the per-episode origin orientation: IMU tilt with the 18 deg fix, plus image yaw.
3. `export_a100.py`: right_only export with the occlusion-aware cube blob (`cube_pnp_occl.py`) and the IMU-VI scale fallback.
   `pnp_census.py` checks the PnP coverage.
4. `right_only sanity`, then `trim_go.py` (rows from the AUTO go), then `make_raw_robotcam_args.py` (robot-TCP labels, same M as v2).
5. Variants:
   - **A (start-anchored)**: `right_only convert` as is.
   - **B (origin frame F)**: `make_origin_frame.py`, then `convert_origin.py` (anchor = identity).
   - **C (robotized)**: `make_robotized_C.py`, checked with `check_C.py`.
   - **CT (table-aware)**: `table_aware_C*.py`. The latest is **`table_aware_C7.py`**. It adds the start-offset decay, the URDF-fingertip clearance QP (floor table + 5 mm, preferred 12 mm, endpoint lift <= 30 mm) and a Pink IK replay.
     Reports are in `reports/table_aware_C*_report.json`.
6. LeRobot export (`export_lerobot_right_only_robotcam.py --name ...`). Then `chain_generic.sh` or `chain_generic_steps.sh` (env `DS LD SMR MRUN [L4 MSTEPS]`) runs smoke + main on one 4090 GPU,
   using the launcher `train_hra_a93_lossmask_d20_b8.sh` (size-agnostic preflight).
7. Top-level orchestrators, in the order they were used on 10-07:
   - `run_all.sh` / `run_all_v2.sh`: mix with HRA_red. This was rejected because the two collections have no common origin.
   - `run_a100_only.sh`
   - `run_ab.sh` / `run_ab_train.sh`: A93START vs A93ORIGIN.
   - `run_C.sh`
   - `run_CT.sh`
   - `run_CT4.sh`
   - `run_CT5.sh`
   - `build_CT6.sh` / `run_CT6.sh` / `run_CT6_all.sh`
   - `run_CT7.sh`
   - **`run_CT7_old.sh`**: the current run. 85 takes from 10-06, start pose 08, run `HRA-RIGHTONLY-ROBOTCAM-CT7-OLD85-LOSSMASK-D20-B8-100K`.
8. Start-pose studies:
   - `start_candidates.py` and `start_candidates_down.py`
   - `start_stats_113.py`, which writes `start_poses/04_*` and `05_*`
   - `start_compare_113.py`
   - `ik_check_a93.py`
   - visualizations `viz_ab.py` and `viz_ct.py`
9. Serving a fresh checkpoint: `ui_CT5_5k.sh` and `ui_CT7old_5k.sh`. They wait for step 5000 and an idle UI, then call `deploy/loaders/load_hra_rightonly_ckpt_to_ui.sh`
   with `V4_FIXED_ANCHOR_JSON=G_anchor.json UMI_TCP_PITCH_DEG=0 UMI_TCP_OFFSET_MM=0,0,0`.
10. `val_eval/val_loss_ab.py` / `run_ab_val.sh`: held-out loss for A and B (`resA.json`, `resB.json`).

## What feeds training

| LeRobot dataset (on the 4090 `holobrain-data/lerobot/`) | built by | run |
|---|---|---|
| `ego_hra_red_rightonly_v1_{train,val}` | hra_red chain (v1) | HRA-RIGHTONLY-LOSSMASK-D20-B8-300K |
| `ego_hra_red_rightonly_robotcam_v2_{train,val}` | robotcam v2 | HRA-RIGHTONLY-ROBOTCAM-LOSSMASK-D20-B8-300K (stopped ~64k) |
| `ego_hra_a93*` (A start / B origin, 82/11) | `run_ab.sh` | HRA-RIGHTONLY-ROBOTCAM-A93START / A93ORIGIN-... |
| `ego_hra_ct5_86_v1`, `ego_hra_ct6_86all_v1`, `ego_hra_ct7_*` | `run_CT*.sh` | CT5/CT6 stopped early; CT7-OLD85 100k is the current run |

These calibration outputs are consumed downstream:
- `handeye/robot_right_wrist_handeye.json` and `base_frame/X_cam_tcp_full.npy`: label conversion, used by robotcam and A100.
- `G_frame_calib.json`, `G_anchor.json` and `B_anchor_F.json`: both the conversion and the UI fixed anchor.
- `umi_start_pose.json` and `start_poses/*.json`: robot start, read by the conversion and the UI.

## Caveats

- No success rate exists for any HRA model. The trial logger was not used during these runs (report Part II, Sec. "Robot Observations").
- Small sets overfit immediately. Held-out loss is lowest at the first checkpoint (5k-15k), so test early checkpoints.
- The cube looks orange in the fisheye-rendered training images and red on the robot camera. This gap is not corrected.
- The IK TCP sits ~82 mm ahead of the physical jaw tip. Contact reasoning must use `urdf_gripper/tip_box_tcp.npy`.
