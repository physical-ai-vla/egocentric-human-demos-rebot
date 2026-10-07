| section | claim | report | recomputed | evidence | status |
|---|---|---|---|---|---|
| data | recorded episodes | 349 | 349.0 | `results/episode_manifest.csv` | PASS |
| data | sessions | 15 | 15.0 | `results/episode_manifest.csv` | PASS |
| data | operators | 1 | 1.0 | `results/episode_manifest.csv` | PASS |
| data | first day | 2026-09-15 | 2026-09-15 | `results/episode_manifest.csv` | PASS |
| data | last day | 2026-09-17 | 2026-09-17 | `results/episode_manifest.csv` | PASS |
| data | total hours | 2.2 | 2.1964 | `results/episode_manifest.csv` | PASS |
| data | median episode s | 23.05 | 23.05 | `results/episode_manifest.csv` | PASS |
| data | order RBP | 64 | 64.0 | `results/episode_manifest.csv` | PASS |
| data | order RPB | 62 | 62.0 | `results/episode_manifest.csv` | PASS |
| data | order BRP | 61 | 61.0 | `results/episode_manifest.csv` | PASS |
| data | order BPR | 56 | 56.0 | `results/episode_manifest.csv` | PASS |
| data | order PRB | 55 | 55.0 | `results/episode_manifest.csv` | PASS |
| data | order PBR | 51 | 51.0 | `results/episode_manifest.csv` | PASS |
| data | early-domain episodes | 55 | 55.0 | `results/episode_manifest.csv` | PASS |
| data | early-domain sessions | 10 | 10.0 | `results/episode_manifest.csv` | PASS |
| data | main-domain episodes | 294 | 294.0 | `results/episode_manifest.csv` | PASS |
| funnel | sync failures | 65 | 65.0 | `results/phase3_census.json` | PASS |
| funnel | sync failure share % | 19 | 18.6246 | `results/phase3_census.json` | PASS |
| funnel | sync-passing episodes | 284 | 284.0 | `results/phase3_census.json` | PASS |
| funnel | candidate segments | 1306 | 1306.0 | `results/phase3_census.json` | PASS |
| funnel | workspace removals | 677 | 677.0 | `results/phase3_census.json` | PASS |
| funnel | metric-scale removals | 50 | 50.0 | `results/phase3_census.json` | PASS |
| funnel | IK removals | 18 | 18.0 | `results/phase3_census.json` | PASS |
| funnel | bimanual segments | 561 | 561.0 | `results/phase3_census.json` | PASS |
| funnel | kept fraction % | 43.0 | 42.9556 | `results/phase3_census.json` | PASS |
| funnel | workspace share of removals % | 91 | 90.8725 | `results/phase3_census.json` | PASS |
| funnel | removed fraction % | 57 | 57.0444 | `results/phase3_census.json` | PASS |
| funnel | workspace fails with one arm only | 531 | 531.0 | `results/phase3_census.json` | PASS |
| funnel | one-arm share % | 78 | 78.4343 | `results/phase3_census.json` | PASS |
| funnel | episodes with >=1 bimanual segment | 259 | 259.0 | `results/phase3_census.json` | PASS |
| funnel | segments starting in first 5 s % | 47 | 46.8806 | `results/phase3_census.json` | PASS |
| funnel | quarantined sessions | 2 | 2.0 | `results/c8_quarantine.json` | PASS |
| funnel | quarantined episodes | 6 | 6.0 | `results/c8_quarantine.json` | PASS |
| funnel | segments from quarantined sessions | 0 | 0.0 | `results/phase3_census.json` | PASS |
| split | val episodes | 26 | 26.0 | `results/split_index.json` | PASS |
| split | train episodes | 233 | 233.0 | `results/split_index.json` | PASS |
| split | train segments | 504 | 504.0 | `results/provenance.json` | PASS |
| split | val segments | 57 | 57.0 | `results/provenance.json` | PASS |
| split | train frames | 32760 | 32760.0 | `results/provenance.json` | PASS |
| split | val frames | 3705 | 3705.0 | `results/provenance.json` | PASS |
| split | val episodes per order (min) | 3 | 3.0 | `results/episode_manifest.csv` | PASS |
| split | val episodes per order (max) | 6 | 6.0 | `results/episode_manifest.csv` | PASS |
| split | val main-domain episodes | 22 | 22.0 | `results/episode_manifest.csv` | PASS |
| split | val sessions | 9 | 9.0 | `results/episode_manifest.csv` | PASS |
| split | non-quarantined sessions | 13 | 13.0 | `results/episode_manifest.csv` | PASS |
| calib | left fx px | 789 | 788.9385 | `calibration/fisheye_left_v002.yaml` | PASS |
| calib | right fx px | 791 | 791.2815 | `calibration/fisheye_right_v003.yaml` | PASS |
| calib | left cam-IMU offset ms | 30.2 | 30.17 | `calibration/camera_imu_left_v001.yaml` | PASS |
| calib | right cam-IMU offset ms | 32.8 | 32.77 | `calibration/camera_imu_right_v002.yaml` | PASS |
| calib | left reprojection mean px | 6.0 | 6.0454 | `calibration/kalibr/calib_left-results-imucam.txt` | PASS |
| calib | right reprojection mean px | 5.6 | 5.641 | `calibration/kalibr/calib_right-results-imucam.txt` | PASS |
| calib | cam->TCP ty mm | 55.3 | 55.3 | `calibration/camera_tcp/handumi_camera_tcp_v2.yaml` | PASS |
| calib | cam->TCP tz mm | 122.5 | 122.5 | `calibration/camera_tcp/handumi_camera_tcp_v2.yaml` | PASS |
| calib | jaw aperture mm | 67.5 | 67.5 | `pipeline/grip/correct_grip.py` | PASS |
| calib | jaw aperture +/- mm | 5 | 5.0 | `pipeline/grip/correct_grip.py` | PASS |
| pose | held-out tracks | 10 | 10.0 | `results/pose_benchmark/cp6_ss1_5080.log` | PASS |
| pose | IMU-VI scale err p50 % | 4.6 | 4.6 | `results/pose_benchmark/cp6_ss1_5080.log` | PASS |
| pose | IMU-VI scale err p90 % | 17.0 | 17.0 | `results/pose_benchmark/cp6_ss1_5080.log` | PASS |
| pose | global-constant scale err p50 % | 18.5 | 18.5 | `results/pose_benchmark/cp6_ss1_5080.log` | PASS |
| pose | global-constant scale err p90 % | 23.8 | 23.8 | `results/pose_benchmark/cp6_ss1_5080.log` | PASS |
| pose | k16 rel. position p90 mm | 11.3 | 11.3 | `results/pose_benchmark/cp6_ss1_5080.log` | PASS |
| pose | k32 rel. position p90 mm | 16.1 | 16.1 | `results/pose_benchmark/cp6_ss1_5080.log` | PASS |
| pose | rotation err p50 deg | 0.84 | 0.8369 | `results/pose_benchmark/cp7_eval.json` | PASS |
| pose | rotation err p90 deg | 2.13 | 2.1274 | `results/pose_benchmark/cp7_eval.json` | PASS |
| pose | benchmark tracks | 28 | 28.0 | `results/pose_benchmark/qa_all.json` | PASS |
| pose | MASt3R coverage (median) | 1.0 | 1.0 | `results/pose_benchmark/qa_all.json` | PASS |
| pose | ORB-SLAM3 stored-map coverage (median) | 0.03 | 0.0299 | `results/pose_benchmark/qa_all.json` | PASS |
| pose | ORB-SLAM3 reversal rate (median) | 0.24 | 0.2389 | `results/pose_benchmark/qa_all.json` | PASS |
| pose | MASt3R reversal rate (median) | 0.07 | 0.0719 | `results/pose_benchmark/qa_all.json` | PASS |
| model | depth | 24 | 24.0 | `results/model_config/{config,train_config}.json` | PASS |
| model | hidden_size | 1024 | 1024.0 | `results/model_config/{config,train_config}.json` | PASS |
| model | num_heads | 16 | 16.0 | `results/model_config/{config,train_config}.json` | PASS |
| model | len_soft_prompts | 32 | 32.0 | `results/model_config/{config,train_config}.json` | PASS |
| model | num_denoising_steps | 10 | 10.0 | `results/model_config/{config,train_config}.json` | PASS |
| model | chunk_size | 30 | 30.0 | `results/model_config/{config,train_config}.json` | PASS |
| model | num_image_views | 3 | 3.0 | `results/model_config/{config,train_config}.json` | PASS |
| model | image size px | 224 | 224.0 | `results/model_config/{config,train_config}.json` | PASS |
| model | vision encoder trainable | False | False | `results/model_config/{config,train_config}.json` | PASS |
| train | steps | 300000 | 300000.0 | `results/model_config/{config,train_config}.json` | PASS |
| train | batch size | 4 | 4.0 | `results/model_config/{config,train_config}.json` | PASS |
| train | lr | 0.0001 | 0.0001 | `results/model_config/{config,train_config}.json` | PASS |
| train | weight decay | 0.0001 | 0.0001 | `results/model_config/{config,train_config}.json` | PASS |
| train | grad clip | 10 | 10.0 | `results/model_config/{config,train_config}.json` | PASS |
| train | warm-up steps | 1000 | 1000.0 | `results/model_config/{config,train_config}.json` | PASS |
| train | final lr | 2.5e-06 | 0.0 | `results/model_config/{config,train_config}.json` | PASS |
| train | save every | 10000 | 10000.0 | `results/model_config/{config,train_config}.json` | PASS |
| train | aux C weight | 2.0 | 2.0 | `training/c8old_chain.sh` | PASS |
| train | FK D weight | 20 | 20.0 | `training/c8old_chain.sh` | PASS |
| eval | samples | 399 | 399.0 | `results/eval/300000.json` | PASS |
| eval | arm-samples | 798 | 798.0 | `results/eval/300000.json` | PASS |
| eval | moving arm-samples | 187 | 187.0 | `results/eval/300000.json` | PASS |
| eval | moving fraction % | 23.4 | 23.4336 | `results/eval/300000.json` | PASS |
| eval | static fraction % | 77 | 76.5664 | `results/eval/300000.json` | PASS |
| eval | pre-registered selection | 160000 | 160000 | `results/selection.json` | PASS |
| eval | geo score 160k mm | 2.78 | 2.7805 | `results/eval/160000.json` | PASS |
| eval | geo score 300k mm | 2.88 | 2.8789 | `results/eval/300000.json` | PASS |
| eval | moving geo score 300k mm | 23.0 | 22.9676 | `results/eval/300000.json` | PASS |
| eval | all-sample geo score 300k (rounded) mm | 2.9 | 2.8789 | `results/eval/300000.json` | PASS |
| table2 | all fk k1 @160k | 1.3 | 1.3377 | `results/eval/160000.json` | PASS |
| table2 | all fk k4 @160k | 1.7 | 1.6529 | `results/eval/160000.json` | PASS |
| table2 | all fk k8 @160k | 2.2 | 2.1623 | `results/eval/160000.json` | PASS |
| table2 | all fk k16 @160k | 3.1 | 3.067 | `results/eval/160000.json` | PASS |
| table2 | all fk k30 @160k | 5.7 | 5.6828 | `results/eval/160000.json` | PASS |
| table2 | all fk k1 @300k | 1.5 | 1.4818 | `results/eval/300000.json` | PASS |
| table2 | all fk k4 @300k | 1.8 | 1.7838 | `results/eval/300000.json` | PASS |
| table2 | all fk k8 @300k | 2.2 | 2.2363 | `results/eval/300000.json` | PASS |
| table2 | all fk k16 @300k | 3.2 | 3.1687 | `results/eval/300000.json` | PASS |
| table2 | all fk k30 @300k | 5.7 | 5.7242 | `results/eval/300000.json` | PASS |
| table2 | all mae k1 @300k | 0.3 | 0.3019 | `results/eval/300000.json` | PASS |
| table2 | all mae k4 @300k | 0.34 | 0.3381 | `results/eval/300000.json` | PASS |
| table2 | all mae k8 @300k | 0.42 | 0.4216 | `results/eval/300000.json` | PASS |
| table2 | all mae k16 @300k | 0.57 | 0.5742 | `results/eval/300000.json` | PASS |
| table2 | all mae k30 @300k | 0.97 | 0.9686 | `results/eval/300000.json` | PASS |
| table2 | all mae0 k1 @300k | 0.15 | 0.147 | `results/eval/300000.json` | PASS |
| table2 | all mae0 k4 @300k | 0.19 | 0.1911 | `results/eval/300000.json` | PASS |
| table2 | all mae0 k8 @300k | 0.25 | 0.2474 | `results/eval/300000.json` | PASS |
| table2 | all mae0 k16 @300k | 0.36 | 0.3565 | `results/eval/300000.json` | PASS |
| table2 | all mae0 k30 @300k | 0.51 | 0.5057 | `results/eval/300000.json` | PASS |
| table2 | moving fk k1 @160k | 11.2 | 11.18 | `results/eval/160000.json` | PASS |
| table2 | moving fk k4 @160k | 15.0 | 15.0439 | `results/eval/160000.json` | PASS |
| table2 | moving fk k8 @160k | 19.9 | 19.8873 | `results/eval/160000.json` | PASS |
| table2 | moving fk k16 @160k | 29.2 | 29.2175 | `results/eval/160000.json` | PASS |
| table2 | moving fk k30 @160k | 39.1 | 39.0725 | `results/eval/160000.json` | PASS |
| table2 | moving fk k1 @300k | 11.2 | 11.1699 | `results/eval/300000.json` | PASS |
| table2 | moving fk k4 @300k | 15.0 | 15.0004 | `results/eval/300000.json` | PASS |
| table2 | moving fk k8 @300k | 19.4 | 19.4339 | `results/eval/300000.json` | PASS |
| table2 | moving fk k16 @300k | 29.3 | 29.3426 | `results/eval/300000.json` | PASS |
| table2 | moving fk k30 @300k | 39.9 | 39.8912 | `results/eval/300000.json` | PASS |
| table2 | moving mae k1 @300k | 2.02 | 2.0199 | `results/eval/300000.json` | PASS |
| table2 | moving mae k4 @300k | 2.68 | 2.6788 | `results/eval/300000.json` | PASS |
| table2 | moving mae k8 @300k | 3.29 | 3.2921 | `results/eval/300000.json` | PASS |
| table2 | moving mae k16 @300k | 4.3 | 4.3028 | `results/eval/300000.json` | PASS |
| table2 | moving mae k30 @300k | 6.59 | 6.5903 | `results/eval/300000.json` | PASS |
| table2 | moving mae0 k1 @300k | 2.32 | 2.3232 | `results/eval/300000.json` | PASS |
| table2 | moving mae0 k4 @300k | 3.36 | 3.36 | `results/eval/300000.json` | PASS |
| table2 | moving mae0 k8 @300k | 4.38 | 4.3778 | `results/eval/300000.json` | PASS |
| table2 | moving mae0 k16 @300k | 6.34 | 6.3432 | `results/eval/300000.json` | PASS |
| table2 | moving mae0 k30 @300k | 8.68 | 8.6754 | `results/eval/300000.json` | PASS |
| table2 | moving cos k1 @300k | 0.72 | 0.7217 | `results/eval/300000.json` | PASS |
| table2 | moving cos k4 @300k | 0.8 | 0.8029 | `results/eval/300000.json` | PASS |
| table2 | moving cos k8 @300k | 0.84 | 0.8411 | `results/eval/300000.json` | PASS |
| table2 | moving cos k16 @300k | 0.86 | 0.855 | `results/eval/300000.json` | PASS |
| table2 | moving cos k30 @300k | 0.9 | 0.8946 | `results/eval/300000.json` | PASS |
| eval | MAE reduction vs zero, k30 % | 24 | 24.0345 | `results/eval/300000.json` | PASS |
| eval | MAE reduction range min % | 13 | 13.0529 | `results/eval/300000.json` | PASS |
| eval | MAE reduction range max % | 32 | 32.1675 | `results/eval/300000.json` | PASS |
| eval | moving k30 FK p90 mm | 90 | 89.5475 | `results/eval/300000.json` | PASS |
| eval | norm ratio k1 | 1.41 | 1.4075 | `results/eval/300000.json` | PASS |
| eval | collapse ratio k30 | 0.77 | 0.7695 | `results/eval/300000.json` | PASS |
| eval | geo flat after (k): 120k within 0.2 mm of best | True | True | `results/selection.json` | PASS |
| prompt | prompt spread k30 moving mm | 4.8 | 4.7518 | `analysis/out/prompt_swap_300k_summary.json` | PASS |
| prompt | noise spread k30 moving mm | 0.09 | 0.0938 | `analysis/out/prompt_swap_300k_summary.json` | PASS |
| prompt | prompt/noise ratio (about 50x) | 50 | 50.4218 | `analysis/out/prompt_swap_300k_summary.json` | PASS |
| prompt | predicted displacement k30 mm | 62 | 62.0444 | `analysis/out/prompt_swap_300k_summary.json` | PASS |
| prompt | spread / displacement % | 8 | 7.6588 | `analysis/out/prompt_swap_300k_summary.json` | PASS |
| prompt | err true k30 mm | 39.9 | 39.8912 | `analysis/out/prompt_swap_300k_summary.json` | PASS |
| prompt | err wrong k30 mm | 39.5 | 39.5435 | `analysis/out/prompt_swap_300k_summary.json` | PASS |
| prompt | mean rank true | 3.55 | 3.5455 | `analysis/out/prompt_swap_300k_summary.json` | PASS |
| prompt | rank-1 frequency % | 16 | 15.508 | `analysis/out/prompt_swap_300k_summary.json` | PASS |
| prompt | rank near chance at every k/subset (|rank-3.5|<0.25) | True | True | `analysis/out/prompt_swap_300k_summary.json` | PASS |
| prompt | reproduces reference eval (< 1e-5 mm) | True | True | `analysis/out/prompt_swap_300k_summary.json` | PASS |
| retarget | v1 usable % | 36.5 | 36.5 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v1 NN p50 | 1.49 | 1.494 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v1 NN p90 | 1.55 | 1.551 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v1 W1 dq | 0.00197 | 0.002 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v2-K usable % | 66.6 | 66.6 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v2-K NN p50 | 1.21 | 1.206 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v2-K NN p90 | 1.6 | 1.599 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v2-K W1 dq | 0.00275 | 0.0027 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v2-Kr usable % | 67.0 | 67.0 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v2-Kr NN p50 | 0.44 | 0.441 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v2-Kr NN p90 | 0.74 | 0.738 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v2-Kr W1 dq | 0.0031 | 0.0031 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v2-TR usable % | 56.1 | 56.1 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v2-TR NN p50 | 0.27 | 0.272 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v2-TR NN p90 | 0.45 | 0.454 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v2-TR W1 dq | 0.00208 | 0.0021 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | v1 start clusters | 1 | 1.0 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | TR start clusters | 100 | 100.0 | `results/retarget_v2/compare_R30_heldout.json` | PASS |
| retarget | candidate segments | 1229 | 1229.0 | `results/retarget_v2/hybrid_master_MASTER.json` | PASS |
| retarget | unsolved segments | 402 | 402.0 | `results/retarget_v2/hybrid_master_MASTER.json` | PASS |
| retarget | unsolved: no workspace window | 399 | 399.0 | `results/retarget_v2/hybrid_master_MASTER.json` | PASS |
| retarget | matched-segment finding mentions 686 / 0.00207 / 0.00226 | True | True | `results/retarget_v2/hybrid_master_MASTER.json` | PASS |
| hrl80 | episodes | 60 | 60.0 | `results/hrl80/hrl80_raw_census.json` | PASS |
| hrl80 | per order min | 10 | 10.0 | `results/hrl80/hrl80_raw_census.json` | PASS |
| hrl80 | per order max | 10 | 10.0 | `results/hrl80/hrl80_raw_census.json` | PASS |
| hrl80 | main session id | HRL80_20260928_101010 | HRL80_20260928_101010 | `results/hrl80/hrl80_raw_census.json` | PASS |
| hrl80 | live rot median old | 0.144 | 0.144 | `results/hrl80/hrl80_prereg_report.json` | PASS |
| hrl80 | live rot median HRL80 | 0.067 | 0.067 | `results/hrl80/hrl80_prereg_report.json` | PASS |
| hrl80 | U_e mean old | 0.59 | 0.592 | `results/hrl80/hrl80_prereg_report.json` | PASS |
| hrl80 | U_e mean HRL80 | 0.59 | 0.59 | `results/hrl80/hrl80_prereg_report.json` | PASS |
| hrl80 | pooled rho | -0.11 | -0.11 | `results/hrl80/hrl80_prereg_report.json` | PASS |
| hrl80 | pooled p | 0.05 | 0.051 | `results/hrl80/hrl80_prereg_report.json` | PASS |
| hrl80 | pooled n | 316 | 316.0 | `results/hrl80/hrl80_prereg_report.json` | PASS |
| hrl80 | TR usable % | 58.9 | 58.9 | `results/hrl80/compare_hrl80_heldout.json` | PASS |
| final | TR segments | 996 | 996.0 | `results/dataset_v2/final/MANIFEST_final.json` | PASS |
| final | old segments | 689 | 689.0 | `results/dataset_v2/final/MANIFEST_final.json` | PASS |
| final | train segments | 896 | 896.0 | `results/dataset_v2/final/MANIFEST_final.json` | PASS |
| final | val segments | 100 | 100.0 | `results/dataset_v2/final/MANIFEST_final.json` | PASS |
| final | source episodes | 316 | 316.0 | `results/dataset_v2/final/MANIFEST_final.json` | PASS |
| final | append invariants pass | True | True | `results/dataset_v2/final/APPEND_INVARIANT.json` | PASS |
| finetune | geo delta % | -10.0 | -9.9837 | `analysis/out/paired_bootstrap_r150_250k.json` | PASS |
| finetune | geo CI low % | -14.7 | -14.7324 | `analysis/out/paired_bootstrap_r150_250k.json` | PASS |
| finetune | geo CI high % | -5.6 | -5.5999 | `analysis/out/paired_bootstrap_r150_250k.json` | PASS |
| finetune | motion geo delta % | -7.2 | -7.1544 | `analysis/out/paired_bootstrap_r150_250k.json` | PASS |
| finetune | motion geo CI low % | -13.4 | -13.4431 | `analysis/out/paired_bootstrap_r150_250k.json` | PASS |
| finetune | motion geo CI high % | -1.8 | -1.8283 | `analysis/out/paired_bootstrap_r150_250k.json` | PASS |
| finetune | motion MAE delta % | -6.8 | -6.8197 | `analysis/out/paired_bootstrap_r150_250k.json` | PASS |
| finetune | motion MAE CI low % | -12.7 | -12.7024 | `analysis/out/paired_bootstrap_r150_250k.json` | PASS |
| finetune | motion MAE CI high % | 0.5 | 0.5288 | `analysis/out/paired_bootstrap_r150_250k.json` | PASS |
| finetune | episodes favouring ego init | 8 | 8.0 | `analysis/out/paired_bootstrap_r150_250k.json` | PASS |
| finetune | eval episodes | 10 | 10.0 | `analysis/out/paired_bootstrap_r150_250k.json` | PASS |
| finetune | bootstrap point == eval JSONs (geo) | True | True | `results/finetune/*.json` | PASS |
| finetune | R90 matched steps | 10k,20k | 10k,20k | `results/finetune/R90_MATCHED.md` | PASS |
| finetune | R90 10k all-sample geo delta | -33% | -33% | `results/finetune/R90_MATCHED.md` | PASS |
| finetune | R90 10k motion geo delta | +2% | +2% | `results/finetune/R90_MATCHED.md` | PASS |
| finetune | R90 20k all-sample geo delta | -11% | -11% | `results/finetune/R90_MATCHED.md` | PASS |
| finetune | R90 20k motion geo delta | +13% | +13% | `results/finetune/R90_MATCHED.md` | PASS |
| pretrain_v2 | stopped at 211.7k | True | True | `results/finetune/C_OLD_TR_HANDOFF.json` | PASS |
| pretrain_v2 | primary init 100k | True | True | `results/finetune/C_OLD_TR_HANDOFF.json` | PASS |
| pretrain_v2 | secondary 200k | True | True | `results/finetune/C_OLD_TR_HANDOFF.json` | PASS |
| pretrain_v2 | no fine-tuning result from v2 ckpts | none | none | `results/finetune/RUN_STATUS.json` | PASS |
| pretrain_v2 | planned steps | 300000 | 300000.0 | `results/finetune/RUN_STATUS.json` | PASS |
| pretrain_v2 | launch schedule steps | 300000 | 300000.0 | `results/finetune/FINAL_TR300K_LAUNCH.json` | PASS |
| v3_data | episodes converted | 330 | 330.0 | `results/v3/ego_cart20/metadata_v2.json` | PASS |
| v3_data | original episodes | 273 | 273.0 | `results/v3/ego_cart20/metadata_v2.json` | PASS |
| v3_data | HRL80 episodes | 57 | 57.0 | `results/v3/ego_cart20/metadata_v2.json` | PASS |
| v3_data | refused at export | 3 | 3.0 | `results/v3/ego_cart20/export_log_v2.json` | PASS |
| v3_data | 15 Hz rows | 111533 | 111533.0 | `results/v3/ego_cart20/metadata_v2.json` | PASS |
| v3_data | trainable rows | 57283 | 57283.0 | `results/v3/ego_cart20/metadata_v2.json` | PASS |
| v3_data | trainable share % | 51.4 | 51.3597 | `results/v3/ego_cart20/metadata_v2.json` | PASS |
| v3_data | UMI_DT ms | 50.05 | 50.0501 | `results/v3/ego_cart20/metadata_v2.json` | PASS |
| v3_data | v2 train episodes | 298 | 298.0 | `results/v3/ego_cart20/metadata_v2.json` | PASS |
| v3_data | v2 train rows | 51612 | 51612.0 | `results/v3/ego_cart20/metadata_v2.json` | PASS |
| v3_data | val episodes | 32 | 32.0 | `results/v3/ego_cart20/metadata_v2.json` | PASS |
| v3_data | val rows | 5671 | 5671.0 | `results/v3/ego_cart20/metadata_v2.json` | PASS |
| v3_data | v2b train episodes | 297 | 297.0 | `results/v3/ego_cart20/metadata_v2b.json` | PASS |
| v3_data | v2b train rows | 48411 | 48411.0 | `results/v3/ego_cart20/metadata_v2b.json` | PASS |
| v3_data | v2b val rows unchanged | 5671 | 5671.0 | `results/v3/ego_cart20/metadata_v2b.json` | PASS |
| v3_data | v2b train row drop % | -6.2 | -6.202 | `results/v3/ego_cart20/metadata_v2b.json` | PASS |
| v3_jump | jump events (train) | 44 | 44.0 | `results/v3/ego_cart20/manifest_v2b_train.jsonl` | PASS |
| v3_jump | episodes with jump events | 42 | 42.0 | `results/v3/ego_cart20/manifest_v2b_train.jsonl` | PASS |
| v3_data | gripper max observed | 0.847 | 0.8471 | `results/v3/ego_cart20/integrity_v2.json` | PASS |
| v3_jump | max adjacent step v2 mm | 422.5 | 422.5039 | `results/v3/state_jump/v2_vs_v2b_continuity.json` | PASS |
| v3_jump | max adjacent step v2b mm | 90.5 | 90.5028 | `results/v3/state_jump/v2_vs_v2b_continuity.json` | PASS |
| v3_jump | steps >100 mm v2 | 18 | 18.0 | `results/v3/state_jump/v2_vs_v2b_continuity.json` | PASS |
| v3_jump | steps >100 mm v2b | 0 | 0.0 | `results/v3/state_jump/v2_vs_v2b_continuity.json` | PASS |
| v3_jump | raw >3 m/s steps left in valid rows | 7 | 7.0 | `results/v3/state_jump/v2_vs_v2b_continuity.json` | PASS |
| v3_jump | raw >3 m/s steps (train) | 55 | 55.0 | `results/v3/state_jump/raw_jump_events.json` | PASS |
| v3_jump | episodes with raw >3 m/s steps | 24 | 24.0 | `results/v3/state_jump/raw_jump_events.json` | PASS |
| v3_jump | raw jumps that return (spikes) | 5 | 5.0 | `results/v3/state_jump/raw_jump_events.json` | PASS |
| v3_jump | raw steps with a lost/missing flag in the previous 15 samples | 0 | 0.0 | `results/v3/state_jump/raw_jump_stage_dump.json` | PASS |
| v3_jump | apparent jumps that are time gaps | 1,482 out of 2,598 (57%) | 1,482 out of 2,598 (57%) | `results/v3/state_jump/STATE_JUMP_ROOT_CAUSE.md` | PASS |
| v3_robot | IK success ego_pos_L % | 99.7 | 99.7324 | `results/v3/ego_vs_robot/kinematic_summary.json` | PASS |
| v3_robot | IK success ego_pos_R % | 99.5 | 99.4648 | `results/v3/ego_vs_robot/kinematic_summary.json` | PASS |
| v3_robot | IK success ego_full_nolock_L % | 72.3 | 72.2864 | `results/v3/ego_vs_robot/kinematic_summary.json` | PASS |
| v3_robot | IK success ego_full_nolock_R % | 74.6 | 74.6414 | `results/v3/ego_vs_robot/kinematic_summary.json` | PASS |
| v3_robot | IK success robot_full_L % | 45.2 | 45.24 | `results/v3/ego_vs_robot/kinematic_summary.json` | PASS |
| v3_robot | IK success robot_full_R % | 52.2 | 52.1933 | `results/v3/ego_vs_robot/kinematic_summary.json` | PASS |
| v3_robot | state workspace p95 L ego/robot | 0.221/0.395 | 0.221/0.395 | `results/v3/ego_vs_robot/INTERPRETATION.md` | PASS |
| v3_robot | state workspace p95 R ego/robot | 0.214/0.369 | 0.214/0.369 | `results/v3/ego_vs_robot/INTERPRETATION.md` | PASS |
| v3_robot | k8 translation p95 ego / robot | 93.1 / 99.2|91.0 / 99.9 | 93.1 / 99.2|91.0 / 99.9 | `results/v3/ego_vs_robot/INTERPRETATION.md` | PASS |
| v3_robot | both arms stationary k8 ego/robot | 19.5%/2.4% | 19.5%/2.4% | `results/v3/ego_vs_robot/INTERPRETATION.md` | PASS |
| v3_robot | wrist sharpness p50 ego_raw_val/left_wrist | 1854 | 1853.5 | `results/v3/robotized/wrist_sharpness_val.json` | PASS |
| v3_robot | wrist sharpness p50 ego_raw_val/right_wrist | 1487 | 1487.4 | `results/v3/robotized/wrist_sharpness_val.json` | PASS |
| v3_robot | wrist sharpness p50 robot_R312c_sample/left_wrist | 59 | 59.2 | `results/v3/robotized/wrist_sharpness_val.json` | PASS |
| v3_robot | wrist sharpness p50 robot_R312c_sample/right_wrist | 64 | 64.0 | `results/v3/robotized/wrist_sharpness_val.json` | PASS |
| v3_robot | wrist sharpness p50 robot100_val/left_wrist | 181 | 181.1 | `results/v3/robotized/wrist_sharpness_val.json` | PASS |
| v3_robot | wrist sharpness p50 robot100_val/right_wrist | 132 | 132.3 | `results/v3/robotized/wrist_sharpness_val.json` | PASS |
| v3_robot | robot100 train frames | 48411 | 48411.0 | `results/v3/robotized/info_robot100_train.json` | PASS |
| v3_robot | robot100 val frames | 5671 | 5671.0 | `results/v3/robotized/info_robot100_val.json` | PASS |
| v3_robot | mix70 train frames | 48411 | 48411.0 | `results/v3/robotized/info_mix70_train.json` | PASS |
| v3_robot | mix70 val frames | 5671 | 5671.0 | `results/v3/robotized/info_mix70_val.json` | PASS |
| v3_slots | trained slots (non-zero enc.bias) | 10-17 | 10-17 | `results/v3/domain_slots/lineage_comparison.json` | PASS |
| v3_slots | number of slots | 30 | 30.0 | `results/v3/domain_slots/lineage_comparison.json` | PASS |
| v3_slots | init loss slot 0 | 1.11 | 1.1121 | `results/v3/domain_slots/domain_probe_log_excerpt.txt` | PASS |
| v3_slots | init loss slot 6 | 1.12 | 1.1186 | `results/v3/domain_slots/domain_probe_log_excerpt.txt` | PASS |
| v3_slots | init loss slot 15 | 1.61 | 1.6113 | `results/v3/domain_slots/domain_probe_log_excerpt.txt` | PASS |
| v3_slots | init loss slot 10 | 2.35 | 2.3536 | `results/v3/domain_slots/domain_probe_log_excerpt.txt` | PASS |
| v3_slots | init loss slot 16 | 2.65 | 2.6528 | `results/v3/domain_slots/domain_probe_log_excerpt.txt` | PASS |
| v3_slots | init loss slot 17 | 4.07 | 4.0692 | `results/v3/domain_slots/domain_probe_log_excerpt.txt` | PASS |
| v3_slots | init loss slot 11 | 22.8 | 22.8008 | `results/v3/domain_slots/domain_probe_log_excerpt.txt` | PASS |
| v3_transfer | robot 40k k8 cos L/R | +0.81/+0.77 | +0.81/+0.77 | `results/v3/diagnostics/direction_cosine_ego40k_vs_robot40k.txt` | PASS |
| v3_transfer | ego-only 40k k8 cos L/R | -0.05/-0.03 | -0.05/-0.03 | `results/v3/diagnostics/direction_cosine_ego40k_vs_robot40k.txt` | PASS |
| v3_transfer | flip/swap variants | 24 | 24.0 | `results/v3/diagnostics/direction_cosine_flip_swap.txt` | PASS |
| v3_transfer | flip/swap best cosine | 0.16 | 0.16 | `results/v3/diagnostics/direction_cosine_flip_swap.txt` | PASS |
| v3_transfer | ego img + ego state range | 0.84-0.94 | 0.84-0.94 | `results/v3/diagnostics/image_state_ablation_ego40k.txt` | PASS |
| v3_transfer | ego img + other state range | 0.66-0.90 | 0.66-0.90 | `results/v3/diagnostics/image_state_ablation_ego40k.txt` | PASS |
| v3_transfer | ego img + robot state range | 0.69-0.90 | 0.69-0.90 | `results/v3/diagnostics/image_state_ablation_ego40k.txt` | PASS |
| v3_transfer | robot img + any state range | -0.25-0.49 | -0.25-0.49 | `results/v3/diagnostics/image_state_ablation_ego40k.txt` | PASS |
| v3_transfer | step-200 loss scratch | 0.945 | 0.945 | `results/v3/loss_curves/{A60,B60}*.csv` | PASS |
| v3_transfer | step-200 loss ego init | 0.244 | 0.244 | `results/v3/loss_curves/{A60,B60}*.csv` | PASS |
| v3_transfer | step-200 ratio | 3.9 | 3.873 | `results/v3/loss_curves/{A60,B60}*.csv` | PASS |
| v3_transfer | loss ratio B/A 0k-5k | 0.49 | 0.4899 | `results/v3/loss_curves/{A60,B60}*.csv` | PASS |
| v3_transfer | loss ratio B/A 5k-20k | 0.875 | 0.8746 | `results/v3/loss_curves/{A60,B60}*.csv` | PASS |
| v3_transfer | loss ratio B/A 20k-40k | 0.95 | 0.9501 | `results/v3/loss_curves/{A60,B60}*.csv` | PASS |
| v3_transfer | loss ratio B/A 55k-60k | 0.984 | 0.9838 | `results/v3/loss_curves/{A60,B60}*.csv` | PASS |
| v3_transfer | 55-60k means A/B | 0.0745/0.0733 | 0.0745/0.0733 | `results/v3/loss_curves/{A60,B60}*.csv` | PASS |
| v3_transfer | 55-60k points B lower | 13 of 24 | 13 of 24 | `results/v3/loss_curves/{A60,B60}*.csv` | PASS |
| v3_transfer | B60 stop step | 59800 | 59800.0 | `results/v3/loss_curves/{A60,B60}*.csv` | PASS |
| v3_runs | co-train reached step | 88600 | 88600.0 | `results/v3/loss_curves/COTRAIN_R312c_ROBOT100_4p4.csv` | PASS |
| v3_runs | MIX70 reached step | 5000 | 5000.0 | `results/v3/loss_curves/MIX70_pretrain_stopped.csv` | PASS |
| v3_runs | ROBOT100 pretrain step (snapshot) | 74000 | 74000.0 | `results/v3/loss_curves/ROBOT100_pretrain_300k_running.csv` | PASS |
| v3_runs | HRA step (snapshot) | 100000 | 100000.0 | `results/v3/loss_curves/HRA_rightonly_300k_running.csv` | PASS |
| v3_runs | HRA loss at snapshot | 0.019 | 0.019 | `results/v3/loss_curves/HRA_rightonly_300k_running.csv` | PASS |
| v3_runs | R312c episodes/frames | 312/174012 | 312/174012 | `results/v3/loss_curves/dataset_sizes_from_logs.txt` | PASS |
| v3_robot_runs | checkpoints executed | 71 | 71.0 | `results/v3/hardware_cycles_by_ckpt.csv` | PASS |
| v3_robot_runs | control cycles | 49431 | 49431.0 | `results/v3/hardware_cycles_by_ckpt.csv` | PASS |
| v3_robot_runs | run starts | 1681 | 1681.0 | `results/v3/hardware_cycles_by_ckpt.csv` | PASS |
| v3_robot_runs | HRA cycles | 8397 | 8397.0 | `results/v3/hardware_cycles_by_ckpt.csv` | PASS |
| v3_robot_runs | HRA starts | 658 | 658.0 | `results/v3/hardware_cycles_by_ckpt.csv` | PASS |
| v3_hra | recorded takes | 200 | 200.0 | `results/v3/hra_red/scale_qc_per_episode/` | PASS |
| v3_hra | IMU scale median | 0.012 | 0.0118 | `results/v3/hra_red/scale_qc_per_episode/` | PASS |
| v3_hra | IMU scale <= 0 | 95 | 95.0 | `results/v3/hra_red/scale_qc_per_episode/` | PASS |
| v3_hra | cube edge m | 0.038 | 0.038 | `results/v3/hra_red/scale_qc_per_episode/` | PASS |
| v3_hra | scale-valid | 181 | 181.0 | `results/v3/hra_red/scale_qc_per_episode/` | PASS |
| v3_hra | too few PnP frames | 14 | 14.0 | `results/v3/hra_red/scale_qc_summary_v0_181.json` | PASS |
| v3_hra | spread too large | 5 | 5.0 | `results/v3/hra_red/scale_qc_summary_v0_181.json` | PASS |
| v3_hra | sanity rejects | 15 | 15.0 | `results/v3/hra_red/sanity_rejected.jsonl` | PASS |
| v3_hra | travel-ratio-only rejects | 10 | 10.0 | `results/v3/hra_red/sanity_rejected.jsonl` | PASS |
| v3_hra | s>1 and travel>0.8 rejects | 3 | 3.0 | `results/v3/hra_red/sanity_rejected.jsonl` | PASS |
| v3_hra | accepted | 166 | 166.0 | `results/v3/hra_red/scale_qc_summary.json` | PASS |
| v3_hra | train episodes/rows | 151/20319 | 151/20319 | `results/v3/hra_red/scale_qc_summary.json` | PASS |
| v3_hra | val episodes/rows | 15/2011 | 15/2011 | `results/v3/hra_red/scale_qc_summary.json` | PASS |
| v3_hra | s_pnp p5/p50/p95 | 0.188/0.354/0.5 | 0.188/0.354/0.5 | `results/v3/hra_red/scale_qc_summary.json` | PASS |
| v3_hra | centre residual p50 cm | 0.17 | 0.17 | `results/v3/hra_red/scale_qc_summary.json` | PASS |
| v3_hra | HRL80 PnP-valid of episodes | 37/57 | 37/57 | `results/v3/hra_red/hrl_val_1face.json` | PASS |
| v3_hra | HRL80 PnP/IMU median | 1.062 | 1.0615 | `results/v3/hra_red/hrl_val_1face.json` | PASS |
| v3_hra | HRL80 PnP/IMU p16 | 0.959 | 0.9595 | `results/v3/hra_red/hrl_val_1face.json` | PASS |
| v3_hra | HRL80 PnP/IMU p84 | 1.274 | 1.2738 | `results/v3/hra_red/hrl_val_1face.json` | PASS |
| v3_hra | val loss 15k | 0.15 | 0.1502 | `results/v3/hra_red/val_loss_results.json` | PASS |
| v3_hra | train-subset loss 15k | 0.066 | 0.0662 | `results/v3/hra_red/val_loss_results.json` | PASS |
| v3_hra | partial val loss 30k seed 0 | 0.198 | 0.1982 | `results/v3/hra_red/val_loss_run_log_excerpt.txt` | PASS |
| v3_runs | run status entries | 11 | 11.0 | `results/v3/RUN_STATUS_v3.json` | PASS |
| v4_r30 | loss scratch @200 | 0.801 | 0.801 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss ego100k @200 | 0.239 | 0.239 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss ego300k @200 | 0.446 | 0.446 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss scratch @1000 | 0.312 | 0.312 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss ego100k @1000 | 0.103 | 0.103 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss ego300k @1000 | 0.117 | 0.117 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss scratch @10000 | 0.065 | 0.065 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss ego100k @10000 | 0.056 | 0.056 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss ego300k @10000 | 0.04 | 0.04 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss scratch @50000 | 0.019 | 0.019 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss ego100k @50000 | 0.017 | 0.017 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss ego300k @50000 | 0.014 | 0.014 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss scratch @100000 | 0.009 | 0.009 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss ego100k @100000 | 0.009 | 0.009 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | loss ego300k @100000 | 0.007 | 0.007 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | ratio ego100k 1k-5k | 0.544 | 0.544 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | ratio ego300k 1k-5k | 0.493 | 0.493 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | ratio ego100k 10k-20k | 0.823 | 0.823 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | ratio ego300k 10k-20k | 0.697 | 0.697 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | ratio ego100k 50k-100k | 0.97 | 0.97 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | ratio ego300k 50k-100k | 0.856 | 0.856 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | ratio ego100k 100k-150k | 1.021 | 1.021 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | ratio ego300k 100k-150k | 0.914 | 0.914 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | last steps scratch/100k/300k | 471400/305400/310200 | 471400/305400/310200 | `results/v4/r30_ablation_summary.json` | PASS |
| v4_r30 | ROBOT100 pretrain final loss | 0.007 | 0.007 | `results/v4/loss_curves/ROBOT100_pretrain_300k.csv` | PASS |
| v4_r30 | ROBOT100 pretrain epochs | 49.6 | 49.58 | `results/v4/loss_curves/ROBOT100_pretrain_300k.csv` | PASS |
| v4_r30 | R30_scratch epochs | 214 | 214.27 | `results/v4/loss_curves/R30_scratch.csv` | PASS |
| v4_r30 | R30_init_robot100pt100k epochs | 139 | 138.82 | `results/v4/loss_curves/R30_init_robot100pt100k.csv` | PASS |
| v4_r30 | R30_init_robot100pt300k epochs | 141 | 141.0 | `results/v4/loss_curves/R30_init_robot100pt300k.csv` | PASS |
| v4_hra | v1 val/train @15k | 0.15/0.066 | 0.15/0.066 | `results/v4/hra/summary.json` | PASS |
| v4_hra | v1 val/train @30k | 0.198/0.053 | 0.198/0.053 | `results/v4/hra/summary.json` | PASS |
| v4_hra | v1 val/train @105k | 0.299/0.018 | 0.299/0.018 | `results/v4/hra/summary.json` | PASS |
| v4_hra | A val/train @5k | 0.124 | 0.124 | `results/v4/hra/summary.json` | PASS |
| v4_hra | A train @5k | 0.11 | 0.11 | `results/v4/hra/summary.json` | PASS |
| v4_hra | A val/train @15k | 0.125 | 0.125 | `results/v4/hra/summary.json` | PASS |
| v4_hra | A train @15k | 0.058 | 0.058 | `results/v4/hra/summary.json` | PASS |
| v4_hra | A val/train @30k | 0.141 | 0.141 | `results/v4/hra/summary.json` | PASS |
| v4_hra | A train @30k | 0.031 | 0.031 | `results/v4/hra/summary.json` | PASS |
| v4_hra | A val/train @50k | 0.224 | 0.224 | `results/v4/hra/summary.json` | PASS |
| v4_hra | A train @50k | 0.02 | 0.02 | `results/v4/hra/summary.json` | PASS |
| v4_hra | B val @5k | 0.132 | 0.132 | `results/v4/hra/summary.json` | PASS |
| v4_hra | B train @5k | 0.097 | 0.097 | `results/v4/hra/summary.json` | PASS |
| v4_hra | B val @15k | 0.147 | 0.147 | `results/v4/hra/summary.json` | PASS |
| v4_hra | B train @15k | 0.057 | 0.057 | `results/v4/hra/summary.json` | PASS |
| v4_hra | B val @30k | 0.174 | 0.174 | `results/v4/hra/summary.json` | PASS |
| v4_hra | B train @30k | 0.031 | 0.031 | `results/v4/hra/summary.json` | PASS |
| v4_hra | B val @45k | 0.19 | 0.19 | `results/v4/hra/summary.json` | PASS |
| v4_hra | B train @45k | 0.022 | 0.022 | `results/v4/hra/summary.json` | PASS |
| v4_hra | C takes/accepted | 107/81 | 107/81 | `results/v4/hra/summary.json` | PASS |
| v4_hra | C clearance p50/min | 19.5/7.0 | 19.5/7.0 | `results/v4/hra/summary.json` | PASS |
| v4_hra | C4 takes/accepted | 107/87 | 107/87 | `results/v4/hra/summary.json` | PASS |
| v4_hra | C4 clearance p50/min | 20.6/7.9 | 20.6/7.9 | `results/v4/hra/summary.json` | PASS |
| v4_hra | C5 takes/accepted | 107/86 | 107/86 | `results/v4/hra/summary.json` | PASS |
| v4_hra | C5 clearance p50/min | 12.6/7.6 | 12.6/7.6 | `results/v4/hra/summary.json` | PASS |
| v4_hra | C6 takes/accepted | 107/86 | 107/86 | `results/v4/hra/summary.json` | PASS |
| v4_hra | C6 clearance p50/min | 12.6/7.6 | 12.6/7.6 | `results/v4/hra/summary.json` | PASS |
| v4_hra | C7 takes/accepted | 134/90 | 134/90 | `results/v4/hra/summary.json` | PASS |
| v4_hra | C7 clearance p50/min | 12.4/7.6 | 12.4/7.6 | `results/v4/hra/summary.json` | PASS |
| v4_hra | G origin mm | 59.0/-74.9/6.2 | 59.0/-74.9/6.2 | `results/v4/hra/summary.json` | PASS |
| v4_hra | G heading deg | 47.09 | 47.09 | `results/v4/hra/summary.json` | PASS |
| v4_hra | G collinearity max mm | 0.38 | 0.38 | `results/v4/hra/summary.json` | PASS |
| v4_hra | G baseline mm | 60.4 | 60.4 | `results/v4/hra/summary.json` | PASS |
| v4_hra | table z mm | -27.1 | -27.1 | `results/v4/hra/summary.json` | PASS |
| v4_hra | hand-eye RMS px | 0.16 | 0.1623 | `results/v4/hra/summary.json` | PASS |
| v4_hra | camera behind TCP mm | 169.5 | 169.4598 | `results/v4/hra/calib/robot_right_wrist_handeye.json` | PASS |
| v4_hra | camera off-axis mm | 76 | 75.9951 | `results/v4/hra/calib/robot_right_wrist_handeye.json` | PASS |
| v4_hra | optical axis below TCP x deg | 33 | 33.0544 | `results/v4/hra/calib/robot_right_wrist_handeye.json` | PASS |
| v4_hra | head cam f / rms / frames | 728.7/1.05/48 | 728.7/1.05/48 | `results/v4/hra/calib/global_cam_intrinsics.json` | PASS |
| v4_hra | origin-plane scale computed for | 107 | 107.0 | `results/v4/hra/calib/origin_scale_kf_107.json` | PASS |
| v4_hra | plan sessions total / 10-06 / 10-07 | 192/65/127 | 192/65/127 | `results/v4/hra/plan_sessions.csv` | PASS |
| v4_hra | 10-06 glitch holds | 1 | 1.0 | `results/v4/hra/plan_sessions.csv` | PASS |
| v4_hra | 10-07 cube-size stops / user stops | 69/58 | 69/58 | `results/v4/hra/plan_sessions.csv` | PASS |
| text | retired value absent: 16.5\% | True | True | `report/part*_en.tex` | PASS |
| text | retired value absent: 20.8/23.8 | True | True | `report/part*_en.tex` | PASS |
| text | retired value absent: versus 0.09 | True | True | `report/part*_en.tex` | PASS |
| text | retired value absent: five early | True | True | `report/part*_en.tex` | PASS |
| text | retired value absent: 6.1 and 5.6 | True | True | `report/part*_en.tex` | PASS |
| text | retired value absent: 128\textdegree | True | True | `report/part*_en.tex` | PASS |
| text | repository marked available on request (private since 2026-10-07) | True | True | `report/part*_en.tex` | PASS |
| text | value present: 211.7k | True | True | `report/part*_en.tex` | PASS |
| text | value present: $-33\%$ | True | True | `report/part*_en.tex` | PASS |
| text | value present: $+13\%$ | True | True | `report/part*_en.tex` | PASS |
| text | value present: 36.5\% | True | True | `report/part*_en.tex` | PASS |
| text | value present: 56.1\% | True | True | `report/part*_en.tex` | PASS |
| text | value present: 58.9\% | True | True | `report/part*_en.tex` | PASS |
| text | value present: 14.7 | True | True | `report/part*_en.tex` | PASS |
| text | value present: 996 segments | True | True | `report/part*_en.tex` | PASS |
| text | value present: 4.6/17.0 | True | True | `report/part*_en.tex` | PASS |
| text | value present: 18.5/23.8 | True | True | `report/part*_en.tex` | PASS |
| text | value present: versus 0.03 | True | True | `report/part*_en.tex` | PASS |
| text | value present: ten early | True | True | `report/part*_en.tex` | PASS |
| text | value present: 6.0 and 5.6 | True | True | `report/part*_en.tex` | PASS |
| text | value present: 0.24 versus 0.07 | True | True | `report/part*_en.tex` | PASS |
| text | value present: 39.9 | True | True | `report/part*_en.tex` | PASS |
| text | value present: 3.55 | True | True | `report/part*_en.tex` | PASS |
| text | value present: 233 train | True | True | `report/part*_en.tex` | PASS |
| text | value present: 561 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 48{,}411 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: $-6.20$\% | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 57{,}283 (51.4\%) | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 1{,}482 of the 2{,}598 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 422.5 to 90.5 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 99.7\% (left) and 99.5\% | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 72.3\%/74.6\% | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 45.2\%/52.2\% | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 1{,}854/1{,}487 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 59/64 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: slots 10--17 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 22.80 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: $+0.81$/$+0.77$ | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: $-0.05$/$-0.03$ | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: $+0.16$ | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 0.69--0.90 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: $-0.25$ and $+0.49$ | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 0.945 for A and 0.244 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 3.9$\times$ | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 0.49 over the first 5k | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: (0.0733 versus 0.0745 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 181/132 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 88.6k | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 71
checkpoints, 49{,}431 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: median of 0.012 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 95 of them | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 37 of 57 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 1.062 (p16--p84 0.959--1.274 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 151 train episodes (20{,}319 rows) | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 0.354 (p5 0.188, p95 0.500) | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 0.150 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 0.198 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: 8{,}397 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: version 3 | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: completed the full three-cube stack | True | True | `report/part*_en.tex` | PASS |
| text | v3 value present: with no success rate | True | True | `report/part*_en.tex` | PASS |
| readme | value present: 349 recorded | True | True | `README.md` | PASS |
| readme | value present: 284 pass sync | True | True | `README.md` | PASS |
| readme | value present: 259 source | True | True | `README.md` | PASS |
| readme | value present: 561 bimanual | True | True | `README.md` | PASS |
| readme | value present: 233 train | True | True | `README.md` | PASS |
| readme | value present: 26 held-out | True | True | `README.md` | PASS |
| readme | value present: 1.5 / 5.7 mm | True | True | `README.md` | PASS |
| readme | value present: 11.2 / 39.9 mm | True | True | `README.md` | PASS |
| readme | value present: 4.8 mm | True | True | `README.md` | PASS |
| readme | value present: 3.55 of 6 | True | True | `README.md` | PASS |
| readme | value present: 36.5 % | True | True | `README.md` | PASS |
| readme | value present: 56.1 % | True | True | `README.md` | PASS |
| readme | value present: 1.494 / 1.551 | True | True | `README.md` | PASS |
| readme | value present: 0.272 / 0.454 | True | True | `README.md` | PASS |
| readme | value present: 58.9 % | True | True | `README.md` | PASS |
| readme | value present: 0.067 vs 0.144 | True | True | `README.md` | PASS |
| readme | value present: 996 TR segments | True | True | `README.md` | PASS |
| readme | value present: 896 / val 100 | True | True | `README.md` | PASS |
| readme | value present: 27,776 | True | True | `README.md` | PASS |
| readme | value present: −10.0 % | True | True | `README.md` | PASS |
| readme | value present: −14.7 to −5.6 % | True | True | `README.md` | PASS |
| readme | value present: 8 of 10 | True | True | `README.md` | PASS |
| readme | value present: 211.7k | True | True | `README.md` | PASS |
| readme | value present: −33 % | True | True | `README.md` | PASS |
| readme | value present: +13 % | True | True | `README.md` | PASS |
| readme | value present: 4.6 / 17.0 % | True | True | `README.md` | PASS |
| readme | value present: 11.3 mm | True | True | `README.md` | PASS |
| readme | v3 value present: 48,411 rows (−6.20 %) | True | True | `README.md` | PASS |
| readme | v3 value present: 1,482 / 2,598 | True | True | `README.md` | PASS |
| readme | v3 value present: 99.7 / 99.5 % | True | True | `README.md` | PASS |
| readme | v3 value present: 45.2 / 52.2 % | True | True | `README.md` | PASS |
| readme | v3 value present: 1,854 / 1,487 | True | True | `README.md` | PASS |
| readme | v3 value present: slots 10–17 | True | True | `README.md` | PASS |
| readme | v3 value present: −0.05 / −0.03 | True | True | `README.md` | PASS |
| readme | v3 value present: +0.81 / +0.77 | True | True | `README.md` | PASS |
| readme | v3 value present: 0.244 vs 0.945 | True | True | `README.md` | PASS |
| readme | v3 value present: 0.984 (55–60k, tie) | True | True | `README.md` | PASS |
| readme | v3 value present: 1.062 (p16–p84 0.959–1.274) | True | True | `README.md` | PASS |
| readme | v3 value present: 151 episodes / 20,319 rows | True | True | `README.md` | PASS |
| readme | v3 value present: 49,431 | True | True | `README.md` | PASS |
| readme | stale text absent: is running; no results yet | True | True | `README.md` | PASS |
| readme | stale text absent: 234 checks | True | True | `README.md` | PASS |
| readme | stale text absent: This repository reports no closed-loop robot results | True | True | `README.md` | PASS |
| readme | stale text absent: outcomes were not logged). | True | True | `README.md` | PASS |

511 claims checked, 0 failed.
