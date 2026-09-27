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
| text | retired value absent: 16.5\% | True | True | `report/report.tex` | PASS |
| text | retired value absent: 20.8/23.8 | True | True | `report/report.tex` | PASS |
| text | retired value absent: versus 0.09 | True | True | `report/report.tex` | PASS |
| text | retired value absent: five early | True | True | `report/report.tex` | PASS |
| text | retired value absent: 6.1 and 5.6 | True | True | `report/report.tex` | PASS |
| text | retired value absent: 128\textdegree | True | True | `report/report.tex` | PASS |
| text | value present: 4.6/17.0 | True | True | `report/report.tex` | PASS |
| text | value present: 18.5/23.8 | True | True | `report/report.tex` | PASS |
| text | value present: versus 0.03 | True | True | `report/report.tex` | PASS |
| text | value present: ten early | True | True | `report/report.tex` | PASS |
| text | value present: 6.0 and 5.6 | True | True | `report/report.tex` | PASS |
| text | value present: 0.24 versus 0.07 | True | True | `report/report.tex` | PASS |
| text | value present: 39.9 | True | True | `report/report.tex` | PASS |
| text | value present: 3.55 | True | True | `report/report.tex` | PASS |
| text | value present: 233 train | True | True | `report/report.tex` | PASS |
| text | value present: 561 | True | True | `report/report.tex` | PASS |

177 claims checked, 0 failed.
