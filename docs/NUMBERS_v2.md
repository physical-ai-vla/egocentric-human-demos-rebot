# Key numbers of the v2 update, with their sources

Every number here is copied from a file in this repository. The **source** column gives the repo path plus the JSON key
path (dot-separated) or the line number. Percentages marked *computed* are `count / total` from the keys given; nothing
else is derived. Rates in the JSON files are fractions (0.561 = 56.1 %).

Terms:
- **old259** = the 259 C8 source episodes of the first release, 1,229 candidate 65-frame bimanual segments.
- **HRL80** = the new robot-like ego collection (60 episodes used, 521 candidate segments).
- **held-out R120** = R150 minus the 30 R30 calibration episodes (`pipeline/retarget_v2/contracts/r30_episodes.json`). Posture (NN) and `W1_dq_vs_heldout120` use this reference.
- **v1** = the single-anchor retarget of the first release. **v2-K** = kinematic multi-seed. **v2-Kr** = multi-seed with the frozen manifold penalty. **v2-TR** = R30 trajectory-window retrieval. **hybrid** = TR where feasible, otherwise Kr (`contracts/hybrid_frozen.json`).
- In `compare_*_heldout.json`, group `r30` holds `v1`, `v2` (= v2-K) and `v2r` (= v2-Kr); group `tr` holds `v2` (= v2-TR); group `hybrid` holds `v2` (= hybrid). See `pipeline/retarget_v2/v2k_compare.py`.

## 1. Retarget v2 (R30-only contract), evaluated against held-out R120

### old259 (1,229 segments)

| method | usable rate | NN p50 / p90 (rad) | W1 \|dq\| vs held-out | source |
|---|---|---|---|---|
| v1 | 0.365 (449) | 1.494 / 1.551 | 0.00197 | `results/retarget_v2/compare_R30_heldout.json` → `r30.v1.rate`, `r30.v1.usable`, `r30.v1.manifold.nn_dist_rad_p50`, `...nn_dist_rad_p90`, `...W1_dq_vs_heldout120` |
| v2-K | 0.666 (819) | 1.206 / 1.599 | 0.00275 | same file → `r30.v2.rate`, `r30.v2.usable`, `r30.v2.manifold.{nn_dist_rad_p50,nn_dist_rad_p90,W1_dq_vs_heldout120}` |
| v2-Kr | 0.67 (824) | 0.441 / 0.738 | 0.0031 | same file → `r30.v2r.rate`, `r30.v2r.usable`, `r30.v2r.manifold.{...}` |
| v2-TR | 0.561 (689) | 0.272 / 0.454 | 0.00208 | same file → `tr.v2.rate`, `tr.v2.usable`, `tr.v2.manifold.{...}` |
| hybrid (TR else Kr) | 0.673 (827) | 0.313 / 0.697 | 0.00296 | `results/retarget_v2/compare_hybrid_heldout.json` → `hybrid.v2.rate`, `hybrid.v2.usable`, `hybrid.v2.manifold.{...}` |

Other numbers from the same blocks:

| quantity | value | source |
|---|---|---|
| cluster coverage v1 / v2-K / v2-Kr / v2-TR | 0.14 / 0.735 / 0.93 / 0.9 | `compare_R30_heldout.json` → `r30.v1.manifold.cluster_coverage`, `r30.v2.manifold.cluster_coverage`, `r30.v2r.manifold.cluster_coverage`, `tr.v2.manifold.cluster_coverage` |
| distinct start clusters v1 / v2-TR | 1 / 100 | same file → `r30.v1.manifold.start_distinct_clusters`, `tr.v2.manifold.start_distinct_clusters` |
| matched segments (both TR and Kr solve), W1 velocity, NN p50/p90 | 686 segments; TR vs Kr W1 0.00207 vs 0.00226; NN 0.272/0.453 vs 0.439/0.568; the 138 Kr fallback segments: \|dq\| p95 0.052, NN p90 1.14 | `results/retarget_v2/hybrid_master_MASTER.json` → `matched_segment_finding` (string) |
| master old259 segment split TR / Kr_fallback / none | 689 / 138 / 402 of 1,229 (56.1 / 11.2 / 32.7 % *computed*) | `results/retarget_v2/hybrid_master_MASTER.json` → `segments.TR`, `segments.Kr_fallback`, `segments.none`, `total_segments`. The same percentages are written in `pipeline/retarget_v2/contracts/final_combined_contract.json` → `hrl80_report_preregistered.utility` |
| failure reasons (old259) | none:no_workspace_window 399, Kr_fallback:no_workspace_window 121, Kr_fallback:no_feasible_rollout 17, none:no_feasible_rollout 3 | `hybrid_master_MASTER.json` → `fallback_reasons` |

Legacy reference, **not** under the R30 contract: `results/retarget_v2/v2k_compare_old259.json` has the first v2-K run, which took its seed bank and workspace gate from the full R150: `old259.v1.rate` 0.451 and `old259.v2.rate` 0.728. `compare_R30full_heldout.json` repeats the `r30` group of `compare_R30_heldout.json` with the same values.

### HRL80 (521 segments), the same frozen pipeline

| method | usable rate | NN p50 / p90 (rad) | W1 \|dq\| vs held-out | source |
|---|---|---|---|---|
| v1 | 0.342 (178) | 1.495 / 1.558 | 0.00247 | `results/hrl80/compare_hrl80_heldout.json` → `r30.v1.*` |
| v2-K | 0.739 (385) | 1.156 / 1.523 | 0.00315 | same file → `r30.v2.*` |
| v2-Kr | 0.743 (387) | 0.441 / 0.77 | 0.00356 | same file → `r30.v2r.*` |
| v2-TR | 0.589 (307) | 0.283 / 0.455 | 0.00225 | same file → `tr.v2.*` |
| hybrid | 0.745 (388) | 0.329 / 0.731 | 0.00336 | same file → `hybrid.v2.*` |

## 2. HRL80 collection

| quantity | value | source |
|---|---|---|
| episodes used by the pipeline | 60 | `results/hrl80/hrl80_60_episodes.json` → `episodes` (length 60); `results/hrl80/hrl80_raw_census.json` → `summary.main_session_episodes` |
| session used | `HRL80_20260928_101010` (started 2026-09-28T10:10:10+0900) | `hrl80_raw_census.json` → `summary.main_session`, `sessions[3].started_wall_iso` |
| session directories on disk / episode directories | 4 / 63 (three short sessions at 09:59, 10:03 and 10:06 hold one RBP episode each and are not used) | `hrl80_raw_census.json` → `summary.session_dirs`, `summary.episode_dirs_total`, `sessions[0..2]` |
| date(s) | 2026-09-28 only | `hrl80_raw_census.json` → `summary.dates` |
| episodes per order (main session) | RBP, RPB, BRP, BPR, PRB, PBR = 10 each (target was 14 per order) | `hrl80_raw_census.json` → `summary.main_session_orders`, `sessions[3].target_per_order` |
| first / last episode start | 10:10:32 / 10:55:11 (+0900) | `hrl80_raw_census.json` → `summary.main_session_first_start`, `summary.main_session_last_start` |
| episode length | 33.01–33.18 s (AUTO episode_s 30) | `hrl80_raw_census.json` → `summary.duration_s_min`, `summary.duration_s_max`, `episodes_main_session[*].auto_episode_s` |
| live monitor verdict | FAIL on 60/60 (the live FAIL rule also counts rotation acceleration) | `hrl80_raw_census.json` → `summary.live_verdict` |
| live group split (rotation-speed cutoff 0.1, grasps ≥ 2, rot-acc diagnostic only) | 58 "robot-like" / 2 "NOT robot-like" | `results/hrl80/compare_groups_final.json` → `cutoff`, `groups["new robot-like"].episodes`, `groups["new NOT robot-like"].episodes` |
| frozen split | val = episodes 49–54 (one per order, round-block holdout), train 54 | `pipeline/retarget_v2/contracts/hrl80_split_frozen.json` → `val_source_episodes`, `counts` |

### HRL80 against old259 (pre-registered report)

| quantity | HRL80 | old259 | source |
|---|---|---|---|
| TR / Kr_fallback / none segments | 307 / 81 / 133 of 521 (58.9 / 15.5 / 25.5 % *computed*) | 689 / 138 / 402 of 1,229 (56.1 / 11.2 / 32.7 %) | `results/retarget_v2/master_combined_MASTER.json` → `parts.B.segments`, `parts.B.total`, `parts.A.segments`, `parts.A.total` |
| TR usable rate | 0.589 | 0.561 | `results/hrl80/compare_hrl80_heldout.json` → `tr.v2.rate`; `results/retarget_v2/compare_R30_heldout.json` → `tr.v2.rate` |
| hybrid usable rate | 0.745 | 0.673 | `compare_hrl80_heldout.json` → `hybrid.v2.rate`; `compare_hybrid_heldout.json` → `hybrid.v2.rate` |
| HRL80 failure reasons | none:no_workspace_window 133, Kr_fallback:no_workspace_window 68, Kr_fallback:no_feasible_rollout 13 | — | `master_combined_MASTER.json` → `parts.B.fallback_reasons` |
| TR NN p50 / p90 (size-matched old259 range) | 0.283 / 0.455, both "inside range" | median 0.274 / 0.453 | `results/hrl80/hrl80_prereg_report.json` → `size_matched.rows.nn_dist_rad_p50`, `...nn_dist_rad_p90` |
| cluster coverage | 0.72, "below min" | N-matched 0.73–0.82 | `hrl80_prereg_report.json` → `size_matched.rows.cluster_coverage` |
| start pairwise distance | 2.031, "above max" | N-matched 1.858–1.939 | `...rows.start_pairwise_rad_mean` |
| start distinct clusters | 78, "inside range" | N-matched 69–85 | `...rows.start_distinct_clusters` |
| W1 \|ddq\| vs held-out | 0.00454, "above max" | N-matched 0.00358–0.00396 | `...rows.W1_ddq_vs_heldout120` |
| W1 \|dq\| vs held-out | 0.00225, "inside range" | N-matched 0.00185–0.00235 | `...rows.W1_dq_vs_heldout120` |
| live rotation-speed over-p95 fraction, median | 0.067 | 0.144 | `hrl80_prereg_report.json` → `T1_TR_usable_fraction_U_e.HRL80.live_rot_speed_median`, `...old259.live_rot_speed_median` |
| mean per-episode TR fraction U_e | 0.59 (n 57) | 0.592 (n 259) | `...T1_TR_usable_fraction_U_e.{HRL80,old259}.U_mean`, `.n` |
| **T1 Spearman ρ** (live rot-speed over-p95 vs U_e, primary) | −0.172 (p 0.202) | −0.102 (p 0.103); pooled −0.11 (p 0.051, n 316) | `...T1_TR_usable_fraction_U_e.{HRL80,old259,pooled}["live_rot_speed_over_p95 (primary)"].{rho,p}` |
| T1 secondary: rot-acc / grasps_total | 0.121 (p 0.37) / −0.276 (p 0.038) | −0.05 (p 0.419) / −0.151 (p 0.015) | `...["live_rot_acc_over_p95"]`, `...grasps_total` |
| T0 (reproduces the old −0.009) argsort rank corr / Spearman with tie averaging | −0.101 / −0.123 (n 60) | −0.009 / −0.013 (n 259) | `hrl80_prereg_report.json` → `T0_v1_offline_fullpass_fraction.{HRL80,old259}.{argsort_rank_corr,spearman_avg_ties}` |

**Prereg reading.** `hrl80_prereg_report.json` does not store a reading label. The readings were written before the
results in `pipeline/retarget_v2/contracts/hrl80_interpretation_prereg.json` → `readings_stated_in_advance`. The numbers
above fit `live_up_offline_flat`: the live rotation-speed median fell from 0.144 to 0.067, but U_e stayed flat
(0.592 → 0.59), TR usable was 58.9 % rather than the 70–80 % the `strong` reading requires, and no_workspace_window is
still the main failure reason. The wording rule is `amendment_1.claim_wording`: "associated", not causal.

## 3. C-old v2 datasets (FINAL = TR-only old259 + HRL80)

| quantity | value | source |
|---|---|---|
| total TR segments (LeRobot episodes, 65 frames each) | 996 = old 689 + HRL80 307 | `results/dataset_v2/final/MANIFEST_final.json` → `composition.total.segments`, `composition.old259.segments`, `composition.HRL80.segments` |
| train / val segments | 896 / 100 (old 620/69, HRL80 276/31) | `MANIFEST_final.json` → `counts.train_segments`, `counts.val_segments`, `composition.old259.{train,val}`, `composition.HRL80.{train,val}`; `append_report.json` → `report.train`, `report.val` |
| source episodes train / val | 284 / 32 | `MANIFEST_final.json` → `composition.total.train_source_episodes`, `...val_source_episodes`; `results/dataset_v2/final/logs_final_validate.log` lines 3–5 (284 = 233 + 51; 32 = 26 + 6) |
| chunk starts train / val (LEAD 5, chunk 30) | 27,776 / 3,100 | `MANIFEST_final.json` → `counts.train_chunk_starts`, `counts.val_chunk_starts`; `logs_final_validate.log` line 17 |
| rows train / val | 58,240 / 6,500 | `append_report.json` → `report.train.rows`, `report.val.rows` |
| build gate (FK vs stored TCP) | pos max 10.963 mm, rot max 4.978 deg | `MANIFEST_final.json` → `gate.pos_mm_max`, `gate.rot_deg_max`; `logs_final_validate.log` line 7 |
| append invariants | I0–I5 PASS; 134,355 decoded old-video frames identical | `results/dataset_v2/final/APPEND_INVARIANT.json` → `checks`, `failed` |
| dataset tree sha256 | 50c6ed88ca6fee839a94c304f493c824a11feb0fad5a16c98b71bc64b794c594 | `MANIFEST_final.json` → `dataset_tree_sha256` |
| combined master | 1,750 segments: TR 996 / Kr_fallback 219 / none 535 | `results/retarget_v2/master_combined_MASTER.json` → `segments`, `total_segments` |
| old259 TR-only snapshot | 689 segments (620 / 69) | `results/dataset_v2/tr_old259/MANIFEST_tr.json`; `MANIFEST_final.json` → `composition.old259` |

## 4. Ego → robot fine-tuning

### R150, step 250k: scratch (B1-old) vs C-old v1 ego pretrain (300k)

All values come from `results/finetune/primary_R150_250k_comparison.txt` (n = 578 samples, line 2). They match the JSONs
`results/finetune/baseline/B1old_R150_250000.json` (scratch) and `results/finetune/r150ft600k/eval/250000.json` (C-old).
Lower is better.

| metric | scratch | C-old | Δ | txt line | JSON key (both files) |
|---|---|---|---|---|---|
| geo | 3.634 | 3.271 | −0.363 (−10.0 %) | 3 | `geo_score` |
| motion geo | 9.325 | 8.658 | −0.667 (−7.2 %) | 4 | `motion_tcp.geo_score` |
| k30 FK p50 (mm) | 4.543 | 4.358 | −0.185 (−4.1 %) | 6 | `k30.fk_mm_p50` |
| motion k30 MAE (deg) | 1.690 | 1.574 | −0.115 (−6.8 %) | 16 | `motion_tcp.k30.joint_mae_deg` |
| k30 MAE (deg) | 0.746 | 0.697 | −0.049 (−6.6 %) | 15 | `k30.joint_mae_deg` |
| motion k30 FK p50 (mm) | 11.787 | 11.010 | −0.778 (−6.6 %) | 8 | `motion_tcp.k30.fk_mm_p50` |
| episodes 66/77, motion geo | 9.687 | 10.092 | +0.405 (+4.2 %) | 22 | `r120_groups.heldout2_motion_tcp.geo_score` |

**Eval-set caveat.** The R150 "probe val-10" episodes are inside the R150 fine-tuning data for **both** arms: both arms
fine-tune on all 150 R150 episodes. See `evaluation/c8old_mac_eval.py`, docstring line 7 ("R150 probe val-10 episodes
(NOTE: included in the R150 fine-tuning data)"), and `results/finetune/SUMMARY.md` line 68. The episode ids are in
`baseline/B1old_R150_250000.json` → `r120_groups.seen8_episodes` and `r120_groups.heldout2_episodes`.
"heldout2" (episodes 66 and 77) means held out from R120 only, not from R150.

The C-old arm in this comparison is initialised from the **first-release (v1) ego pretrain**, 300k-final
(`results/finetune/SUMMARY.md` line 21, `pretrain300k_model_md5` 004974bb…). It is not the v2 dataset.

### R90 matched-step comparison

| quantity | value | source |
|---|---|---|
| steps evaluated | scratch 110k, C-old 20k, **2 matched steps** (10k, 20k) | `results/finetune/R90_MATCHED.md` line 6 |
| C-old init | legacy-v1 ego pretrain 300k-final | `R90_MATCHED.md` line 3 |
| 10k: motion geo / geo (all) / motion k30 MAE | +2 % / −33 % / +5 % (C-old vs scratch) | `R90_MATCHED.md` line 12 |
| 20k: motion geo / geo (all) / motion k30 MAE / heldout2 motion geo | +13 % / −11 % / −7 % / +15 % | `R90_MATCHED.md` line 13 |

Signs are mixed: C-old is better on all-sample geo at both steps, worse on motion geo at both steps, and better or worse
on the rest depending on the step. Two steps are too few to call a winner. `results/finetune/r90ft600k/SUMMARY.md` and
`r90scratch600k/SUMMARY.md` hold the full per-step tables. `r120scratch600k/SUMMARY.md` shows scratch R120 stopped at
200k. `r60scratch600k/SUMMARY.md` has no evaluated step yet.

### v2 final pretrain (5090)

`results/finetune/FINAL_TR300K_LAUNCH.json`: run `c8oldv2_finalTR_300k`, launched 2026-09-28 22:51 (`launched`), fresh
`lerobot/xvla-base` init (`init`), 300k steps with decay 300k and a save every 100k (`schedule`), on dataset tree sha
50c6ed88… (`dataset.tree_sha256`). The record has no status field. That the run is still going was reported by the
author and cannot be checked from repository files. No v2-pretrain fine-tuning result exists yet.
