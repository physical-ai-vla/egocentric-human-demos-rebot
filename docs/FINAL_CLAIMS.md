# Headline claims and their evidence

Every row is also a row of `analysis/out/claim_audit.md`, which `analysis/verify_claims.py` regenerates from the
files below (it exits 1 on any mismatch). Status as of 2026-10-07 (version 4): all rows VERIFIED (`analysis/verify_claims.py`: 575 checks, 0 failed). From version 4 the report is split into Part I (stacking) and Part II (approach), each as a short main version and a full-length version in `report/archive/`; the checks run on the English sources of both, and the Korean versions are translations of them.

| Claim | Evidence file | Exact value | Status |
|---|---|---|---|
| Recorded bimanual egocentric demonstrations | `results/episode_manifest.csv` | 349 (15 sessions, 1 operator, 2026-09-15..17, 2.20 h) | VERIFIED |
| Six stacking orders, balanced at collection time (least-collected next) | `results/episode_manifest.csv` | 51–64 per order | VERIFIED |
| Episodes passing sync | `results/phase3_census.json` | 284 | VERIFIED |
| Usable source episodes / first-release segments | `results/phase3_census.json` | 259 / 561 (65-frame bimanual) | VERIFIED |
| First-release split (by episode, seed 0, not stratified) | `results/split_index.json`, `results/provenance.json` | 233 / 26 episodes, 504 / 57 segments | VERIFIED |
| Model | `results/model_config/` | X-VLA, 879M params, from `lerobot/xvla-base`, 300k steps | VERIFIED |
| Held-out median TCP error, all samples, k=1 / k=30 | `results/eval/300000.json` | 1.5 / 5.7 mm | VERIFIED |
| Share of static samples | `results/eval/300000.json` | 77 % (moving = 187 / 798 = 23.4 %) | VERIFIED |
| Moving samples, k=30 TCP error / direction cosine | `results/eval/300000.json` | 39.9 mm / 0.90 | VERIFIED |
| Moving samples, joint MAE vs zero-motion predictor, k=30 | `results/eval/300000.json` | 6.59° vs 8.68° (−24 %) | VERIFIED |
| Prompt swap: spread over 6 order prompts / predicted displacement | `analysis/out/prompt_swap_300k_summary.json` | 4.8 mm / 62 mm (≈ 8 %) | VERIFIED |
| Prompt swap: mean rank of the true prompt | `analysis/out/prompt_swap_300k_summary.json` | 3.55 of 6 (chance 3.5) | VERIFIED |
| Retarget v2 usable segments, v1 → TR | `results/retarget_v2/compare_R30_heldout.json` | 36.5 % → 56.1 % (of 1,229) | VERIFIED |
| Posture distance to held-out robot data (NN p50), v1 → TR | `results/retarget_v2/compare_R30_heldout.json` | 1.49 → 0.27 rad | VERIFIED |
| Start-posture clusters, v1 → TR | `results/retarget_v2/compare_R30_heldout.json` | 1 → 100 | VERIFIED |
| Reference robot data is disjoint from the calibration set | `pipeline/retarget_v2/contracts/r30_episodes.json` | calibrate on R30, evaluate on the other 120 | VERIFIED |
| Robot-like re-collection | `results/hrl80/hrl80_raw_census.json` | 60 episodes, 10 per order, one session | VERIFIED |
| Robot-like: live wrist-rotation over-threshold fraction (median) | `results/hrl80/hrl80_prereg_report.json` | 0.144 → 0.067 | VERIFIED |
| Robot-like: TR usable rate | `results/hrl80/compare_hrl80_heldout.json` | 56.1 % → 58.9 % | VERIFIED |
| Robot-like: live metric vs TR yield (pooled Spearman) | `results/hrl80/hrl80_prereg_report.json` | ρ = −0.11, p = 0.05, n = 316 | VERIFIED |
| Final dataset | `results/dataset_v2/final/MANIFEST_final.json` | 996 segments (689 + 307), 896 train / 100 val, 316 source eps | VERIFIED |
| Final dataset preserves old segments bit-identically | `results/dataset_v2/final/APPEND_INVARIANT.json` | I0–I5 PASS | VERIFIED |
| Final-dataset pretraining status | `results/finetune/C_OLD_TR_HANDOFF.json`, `RUN_STATUS.json` | stopped at 211.7k of 300k; ckpts 100k (primary), 200k; no FT result | VERIFIED |
| Ego init vs scratch, R150, 250k, geo score | `analysis/out/paired_bootstrap_r150_250k.json` | −10.0 %, 95 % CI −14.7 to −5.6 % | VERIFIED |
| Same, moving-sample geo / moving k30 MAE | `analysis/out/paired_bootstrap_r150_250k.json` | −7.2 % (CI −13.4 to −1.8) / −6.8 % (CI −12.7 to +0.5) | VERIFIED |
| Episodes favouring ego init | `analysis/out/paired_bootstrap_r150_250k.json` | 8 of 10 | VERIFIED |
| The 10 evaluation episodes are inside both fine-tuning sets | `evaluation/c8old_mac_eval.py` docstring, `results/finetune/SUMMARY.md` | yes | VERIFIED |
| R90 comparison | `results/finetune/R90_MATCHED.md` | 2 matched steps only; geo −33 % / −11 %, motion geo +2 % / +13 % → no conclusion | VERIFIED |
| **v3** Cartesian ego dataset: episodes / trainable rows | `results/v3/ego_cart20/metadata_v2.json` | 330 (273 + 57; 3 refused) / 57,283 of 111,533 rows (51.4 %) | VERIFIED |
| v3 split after jump filter (v2b) | `results/v3/ego_cart20/metadata_v2b.json` | train 297 eps / 48,411 rows (−6.20 %), val 32 / 5,671 | VERIFIED |
| Apparent state jumps that are time gaps | `results/v3/state_jump/STATE_JUMP_ROOT_CAUSE.md` | 1,482 of 2,598 (57 %) | VERIFIED |
| Silent MASt3R-SLAM jumps (> 3 m/s, no lost flag) | `results/v3/state_jump/raw_jump_events.json`, `raw_jump_stage_dump.json` | 55 steps in 24 train episodes, 0 with a lost flag | VERIFIED |
| Robot IK reach of ego poses (position / full pose, L / R) | `results/v3/ego_vs_robot/kinematic_summary.json` | 99.7 / 99.5 % ; 72.3 / 74.6 % | VERIFIED |
| Deployed IK (wrist joint locked) on the robot's own poses | same | 45.2 / 52.2 % | VERIFIED |
| Wrist sharpness p50 ego / robot / ROBOT100 | `results/v3/robotized/wrist_sharpness_val.json` | 1,854–1,487 / 59–64 / 181–132 | VERIFIED |
| Trained soft-prompt slots of `lerobot/xvla-base` | `results/v3/domain_slots/lineage_comparison.json` | 10–17 only (of 30) | VERIFIED |
| Ego-only 40k on robot frames, k8 direction cosine | `results/v3/diagnostics/direction_cosine_ego40k_vs_robot40k.txt` (session-log capture) | −0.05 / −0.03 (robot model +0.81 / +0.77, in-training frames) | VERIFIED (log capture, not re-run) |
| Image/state swap | `results/v3/diagnostics/image_state_ablation_ego40k.txt` (session-log capture) | ego images 0.66–0.94; robot images −0.25 to +0.49 | VERIFIED (log capture, not re-run) |
| Ego init vs scratch, R312c training loss, matched | `results/v3/loss_curves/`, `analysis/out/v3_ego_init_loss.json` | step 200: 0.244 vs 0.945; ratio 0.49 → 0.984 (tie) by 55–60k | VERIFIED (training loss, 1 seed) |
| Closed-loop three-cube stack, ego-init R312c FT (B300, 75k–205k) | operator observation only (no log, no video) | demonstrated; trials not counted | OPERATOR-REPORTED, not verifiable from repository files |
| v3 checkpoints executed on the robot | `results/v3/hardware_cycles_by_ckpt.csv` | 71 checkpoints, 49,431 cycles; no outcome field | VERIFIED |
| HRA_red IMU scale unobservable | `results/v3/hra_red/scale_qc_per_episode/` | median 0.012; ≤ 0 in 95 of 200 | VERIFIED |
| Cube-PnP scale vs IMU on HRL80 | `results/v3/hra_red/hrl_val_1face.json` | 37 / 57 valid; ratio 1.062 (p16–p84 0.959–1.274) | VERIFIED |
| HRA_red funnel and split | `results/v3/hra_red/scale_qc_summary*.json`, `sanity_rejected.jsonl` | 200 → 181 → 166; 151 / 20,319 train, 15 / 2,011 val | VERIFIED |
| HRA_red held-out loss | `results/v3/hra_red/val_loss_results.json`, `val_loss_run_log_excerpt.txt` | 15k 0.150 (train subset 0.066); 30k partial 0.198 | VERIFIED |
| R30 ablation: training-loss ratio to scratch, 1k–5k | `results/v4/r30_ablation_summary.json` | ego-PT 100k init 0.544, 300k init 0.493 | VERIFIED |
| R30 ablation: ratio over 50k–100k / 100k–150k | `results/v4/r30_ablation_summary.json` | 100k init 0.970 / 1.021; 300k init 0.856 / 0.914 | VERIFIED |
| R30 arms stopped (scratch / 100k / 300k) | `results/v4/loss_curves/R30_*.csv`, `results/v4/RUN_STATUS_v4.json` | 471.4k / 305.4k / 310.2k steps | VERIFIED |
| ROBOT100 pretrain finished | `results/v4/loss_curves/ROBOT100_pretrain_300k.csv` | 300k, loss 0.007, 49.6 epochs | VERIFIED |
| HRA_red v1 held-out loss rises (overfit) | `results/v4/hra/val_eval/v1_results.json` | 15k 0.150 → 105k 0.299 (train subset 0.066 → 0.018) | VERIFIED |
| HRA_A100 start vs origin anchoring | `results/v4/hra/val_eval/a93_*_results.json` | held-out 5k: 0.124 vs 0.132 | VERIFIED |
| Shared task frame G | `results/v4/hra/calib/G_frame_calib.json` | origin (59.0, −74.9, 6.2) mm, heading 47.09°, collinearity 0.38 mm, table z −27.1 mm | VERIFIED |
| Robot wrist hand-eye | `results/v4/hra/calib/robot_right_wrist_handeye.json` | 13 views, RMS 0.16 px, camera 169.5 mm behind TCP, axis 33° below TCP x | VERIFIED |
| Table-aware conversion CT5/CT6 | `results/v4/hra/table_aware/table_aware_C5_report.json` | 86 / 107 accepted, clearance p50 12.6 mm, min 7.6 mm | VERIFIED |
| Approach robot sessions | `results/v4/hra/plan_sessions.csv` | 192 (10-06: 65, 10-07: 127, of which 69 cube-size stops); no outcomes | VERIFIED |

## Not shown anywhere in this repository

- closed-loop robot success of an egocentric-pretrained policy;
- generalization to unseen cube layouts or unseen stacking orders;
- a benefit on robot test episodes disjoint from fine-tuning data;
- transfer to other operators or scenes;
- any robot fine-tuning result from the final 996-segment pretraining checkpoints;
- a closed-loop success rate for any v3 checkpoint (B300 stacking was observed, trials not counted);
- a held-out or closed-loop advantage of ego initialization (only matched training loss to 60k);
- transfer of the ego-only policy or the robotized wrist images to robot cameras;
- any robot or held-out evaluation of the three R30 arms;
- a success rate for the single-arm approach models (Part II).
