# hra/ MANIFEST

Copied 2026-10-07 with `cp -p`, so mtimes are preserved. The sources were not modified. No overlap with `~/egocentric-human-demos-rebot`: the public
copy has no HRA pipeline code, only `analysis/v3/hra_val_loss.py`, which is in `evaluation/analysis/v3/`. Every file below therefore comes from the working dirs.

## hra_red/  <-  ~/c8/hra_red/

| dest | source | taken |
|---|---|---|
| `*.sh` | `~/c8/hra_red/*.sh` | all 9 orchestrators (chain_after_mast3r, finalize_mast3r, smoke_*, resubmit_*, wait_direct_*) |
| `umi_start_pose.json`, `scale_qc_summary*.json` | same dir | current robot start pose (CT6 variant, 10-07 13:53); scale-QC summaries (small) |
| `base_frame/` | `base_frame/*.py, *.json, *.npy` | hra_to_base*, match_view*, start_reach_check; `X_cam_tcp_full.npy`, `R_imu_corr.npy`; small result jsons |
| `handeye/` | `handeye/*.py, *.sh` + `robot_right_wrist_handeye.json`, `feasible.json`, `T_base_board_now.npy` | the solved hand-eye |
| `global_calib/` | `*.py` + `board_state.json`, `global_cam_intrinsics*.json`, `global_cam_calib.json`, `samples.json`, `probe.json`, `pred35.json` | head-cam calibration |
| `robotcam/` | `make_raw_robotcam.py`, `ik_check.py`, `run_export_loop.sh`, `robotcam_smoke_then_main_4090.sh`; `raw_robotcam/ROBOTCAM.json` -> `robotcam/ROBOTCAM.json` | the M matrix record |
| `val_eval/` | `val_loss.py`, `results.json` | |
| `start_pose/`, `start_poses/`, `start_view/`, `head_link/` | `*.py` + small json (`human_start.json` 48 KB, `start_views.json`, `head_f.json`, 10 start-pose candidates) | |
| `hrl80_pnp_validation/` | `hrl_val.py` + 4 small result jsons | |

## hra_a100/  <-  ~/c8/hra_a100/

| dest | source | taken |
|---|---|---|
| root `*.py`, `*.sh` | `make_c922_export.py`, `origin_check.py`, `export_one.sh` | |
| root json | `G_frame_calib.json`, `G_anchor.json`, `B_anchor_F.json`, `human_start_median.json`, `start_pose_CT6.json` | calibration / anchors |
| `pipe/` | `pipe/*.py`, `pipe/*.sh` (47 files) | all code and orchestrators; logs and `.out` excluded |
| `urdf_gripper/` | `gripper_{left,right,link}.STL` (2.1-2.7 MB each), `tip_box_tcp.npy` | STLs kept: they are under 5 MB, they are not images, and they define the gripper geometry |
| `val_eval/` | `val_loss_ab.py`, `run_ab_val.sh`, `resA.json`, `resB.json` | |
| `reports/` | `table_aware_C{,4,5,6,7}_report.json` (66-101 KB), `export_c922_report.json`, `pnp_census.json`, `start_compare_113.json`, `train_end_cube_px.json` | small reports referenced by the README / scripts |

## Moved elsewhere

- `~/c8/hra_red/node/*` are the 4090/5090 training launchers and wrappers for the HRA loss-mask runs. They went to `training/launchers/hra_node/` and `training/launchers/node4090/`
  (identical duplicates removed).

## Excluded

- Data:
  - `raw*/` (raw_episode npz trees) and `processed*/`
  - `lerobot*/` (LeRobot datasets, parquet + AV1 videos)
  - `export/` and `export_c922/` (1.4 GB of video + IMU)
  - `runs/` and `runs_kf/` (MASt3R outputs, 4.4 GB)
  - `raw_spread15_candidates/`
  - `wrist_snaps/`
- Every `*.jpg` and `*.png`:
  - handeye board snaps `hc*_{left,middle,right}.jpg`
  - global_calib `checker_*.jpg`, `img_*.jpg`, `probe_*.jpg`
  - `start_view/*.jpg`, `head_pair.jpg`, `compare_A.jpg`, `tcp_projection.jpg`
- Handeye observation captures `hc01..16.json`, `obs*.json`, `run*.json`, `v2/v3.json`, `viewA.json`, `far.json`. These are 60-95 KB per-view detection dumps (raw data, not calibration results).
- `base_frame/fit_imu.pkl` (pickled fit object), `head_link/head_dets.npy` (detections), `urdf_gripper/finger_points_tcp.npz` (2 MB, derived point cloud, not referenced by `pipe/`).
- Duplicate copies of data:
  - `table_aware_*_report.csv`
  - `origin_orient*.json` and `origin_scale_kf*.json` (per-episode outputs, 24-75 KB)
  - `check_C.json`, `ik_check_a93.json`, `human_starts_all.json`, `list_all.txt`, `lists*`, `x_touch_*.json`
- All `*.log` and `*.out`, `chain.log`, `export*.log` (up to 12 MB), `mast3r_jobs.txt`, `sid_5090`, `export.pid`.
- `*.bak*`: `umi_start_pose.json.bak_*`, `umi_start_pose.json.CT6_1007` (a snapshot; the live `umi_start_pose.json` is copied).
- `__pycache__`, `.DS_Store`.
- `start_match/` (empty).

## Notes and uncertainties

- `umi_start_pose.json` is the 10-07 13:53 version (CT6 start). The CT7 runs use `start_poses/08_medoid_back6_right6_down4.json` instead.
- No sanitization was needed: the secret scan found no hits in this folder. The scripts contain internal paths: `/home/bh-aiteam`, `/srv/data/johann`, Ray `100.64.0.1:8265`, node `100.64.0.2/.5`,
  and ssh aliases `gpu-4090`, `gpu-5090` and `head-lp`. They are kept because the scripts need them.
