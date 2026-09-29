# Sources for the v2 update

Every file below was **copied** (never moved or edited at the source) into this repository on 2026-09-29.
Text files were sanitized the same way as the first release: absolute paths became `${HOME}`, `${REMOTE_HOME}`, `${DATA_ROOT}`,
`${SHARED_ROOT}` and `${ARCHIVE_ROOT}`; host names and tailnet/LAN addresses became `<gpu-node>`, `<gpu-node-ip>` and `<ray-head>`;
user names became `<user>`; the few Korean comments/strings were translated to English. Binary files (`.npy`, `.parquet`) are byte copies.
`~` in the source column is the author's home directory on the collection Mac.

Files under `collection/handumi_collector/{config.py,ui/app.py,collector/main.py,collector/session.py}` and
`collection/configs/handumi/tasks.yaml` already existed from the first release; they were refreshed from the current
collector because the robot-like protocol (`--protocol robot_like_v1`) is wired into them. The earlier English translations
(voice name, spoken-cue docstring) were kept.

## pipeline/retarget_v2

| repo path | source |
|---|---|
| `pipeline/retarget_v2/v2k_seedbank.py` | `~/umi_bridge/track_c/v2k/v2k_seedbank.py` |
| `pipeline/retarget_v2/v2k_retarget.py` | `~/umi_bridge/track_c/v2k/v2k_retarget.py` |
| `pipeline/retarget_v2/v2k_compare.py` | `~/umi_bridge/track_c/v2k/v2k_compare.py` |
| `pipeline/retarget_v2/v2k_tail.py` | `~/umi_bridge/track_c/v2k/v2k_tail.py` |
| `pipeline/retarget_v2/hybrid_merge.py` | `~/umi_bridge/track_c/v2k/hybrid_merge.py` |
| `pipeline/retarget_v2/hrl80_prereg_report.py` | `~/umi_bridge/track_c/v2k/hrl80_prereg_report.py` |
| `pipeline/retarget_v2/contracts/v2kr_frozen.json` | `~/umi_bridge/track_c/v2k/v2kr_frozen.json` |
| `pipeline/retarget_v2/contracts/v2tr_frozen.json` | `~/umi_bridge/track_c/v2k/v2tr_frozen.json` |
| `pipeline/retarget_v2/contracts/hybrid_frozen.json` | `~/umi_bridge/track_c/v2k/hybrid_frozen.json` |
| `pipeline/retarget_v2/contracts/r30_episodes.json` | `~/umi_bridge/track_c/v2k/r30_episodes.json` |
| `pipeline/retarget_v2/contracts/R30_TR_EPISODES.md` | `~/umi_bridge/track_c/v2k/R30_TR_EPISODES.md` |
| `pipeline/retarget_v2/contracts/r150_nested_subset_v1.json` | `~/umi_bridge/track_c/v2k/r150_nested_subset_v1.json` |
| `pipeline/retarget_v2/contracts/final_combined_contract.json` | `~/umi_bridge/track_c/v2k/final_combined_contract.json` |
| `pipeline/retarget_v2/contracts/hrl80_split_frozen.json` | `~/umi_bridge/track_c/v2k/hrl80_split_frozen.json` |
| `pipeline/retarget_v2/contracts/hrl80_interpretation_prereg.json` | `~/umi_bridge/track_c/v2k/hrl80_interpretation_prereg.json` |
| `pipeline/retarget_v2/contracts/gripper_binary_contract.json` | `~/umi_bridge/track_c/v2k/gripper_binary_contract.json` |
| `pipeline/retarget_v2/contracts/rel16_ego_interface.json` | `~/umi_bridge/track_c/v2k/rel16_ego_interface.json` |
| `pipeline/retarget_v2/contracts/seedbank_R30.json` | `~/umi_bridge/track_c/v2k/seedbank_R30.json` |
| `pipeline/retarget_v2/contracts/seedbank_R30_q12.npy` | `~/umi_bridge/track_c/v2k/seedbank_R30_q12.npy` |
| `pipeline/retarget_v2/contracts/seedbank.json` | `~/umi_bridge/track_c/v2k/seedbank.json` |
| `pipeline/retarget_v2/contracts/seedbank_q12.npy` | `~/umi_bridge/track_c/v2k/seedbank_q12.npy` |
| `pipeline/retarget_v2/chains/v2kr_chain.sh` | `~/c8/robotlike/v2kr_chain.sh` |
| `pipeline/retarget_v2/chains/tr_chain.sh` | `~/c8/robotlike/tr_chain.sh` |
| `pipeline/retarget_v2/chains/hrl80_chain.sh` | `~/c8/robotlike/hrl80_chain.sh` |
| `pipeline/retarget_v2/chains/hrl80_retarget_chain.sh` | `~/c8/robotlike/hrl80_retarget_chain.sh` |

## pipeline/dataset_v2

| repo path | source |
|---|---|
| `pipeline/dataset_v2/c8oldv2_build_dataset.py` | `~/c8/c8oldv2_build_dataset.py` |
| `pipeline/dataset_v2/c8oldv2_build_hrl80.py` | `~/c8/c8oldv2_build_hrl80.py` |
| `pipeline/dataset_v2/c8oldv2_final_append.py` | `~/c8/c8oldv2_final_append.py` |
| `pipeline/dataset_v2/c8oldv2_validate.py` | `~/c8/c8oldv2_validate.py` |
| `pipeline/dataset_v2/c8oldv2_validate_hrl80.py` | `~/c8/c8oldv2_validate_hrl80.py` |
| `pipeline/dataset_v2/c8oldv2_validate_final.py` | `~/c8/c8oldv2_validate_final.py` |
| `pipeline/dataset_v2/c8oldv2_append_invariant.py` | `~/c8/c8oldv2_append_invariant.py` |
| `pipeline/dataset_v2/c8oldv2_hybrid_build.sh` | `~/c8/c8oldv2_hybrid_build.sh` |
| `pipeline/dataset_v2/c8oldv2_hybrid_resume.sh` | `~/c8/c8oldv2_hybrid_resume.sh` |
| `pipeline/dataset_v2/write_stage.py` | `~/c8/write_stage.py` |

## collection

| repo path | source |
|---|---|
| `collection/handumi_collector/robotlike/__init__.py` | `~/ego_collector/handumi_collector/robotlike/__init__.py` |
| `collection/handumi_collector/robotlike/monitor.py` | `~/ego_collector/handumi_collector/robotlike/monitor.py` |
| `collection/handumi_collector/config.py` | `~/ego_collector/handumi_collector/config.py` |
| `collection/handumi_collector/ui/app.py` | `~/ego_collector/handumi_collector/ui/app.py` |
| `collection/handumi_collector/collector/main.py` | `~/ego_collector/handumi_collector/collector/main.py` |
| `collection/handumi_collector/collector/session.py` | `~/ego_collector/handumi_collector/collector/session.py` |
| `collection/configs/handumi/robot_like_v1.yaml` | `~/ego_collector/configs/handumi/robot_like_v1.yaml` |
| `collection/configs/handumi/tasks.yaml` | `~/ego_collector/configs/handumi/tasks.yaml` |
| `collection/scripts/robotlike_offline_check.py` | `~/ego_collector/scripts/robotlike_offline_check.py` |
| `collection/scripts/robotlike_session_check.sh` | `~/ego_collector/scripts/robotlike_session_check.sh` |
| `collection/scripts/robotlike_compare_groups.py` | `~/ego_collector/scripts/robotlike_compare_groups.py` |
| `collection/scripts/robotlike_replay_live.py` | `~/ego_collector/scripts/robotlike_replay_live.py` |
| `collection/scripts/robotlike_resident_5080.sh` | `~/ego_collector/scripts/robotlike_resident_5080.sh` |
| `collection/docs/ROBOT_LIKE_PROTOCOL.md` | `~/ego_collector/docs/handumi_collector/ROBOT_LIKE_PROTOCOL.md` |
| `collection/tests/test_robotlike.py` | `~/ego_collector/tests/handumi/test_robotlike.py` |

## results/retarget_v2

| repo path | source |
|---|---|
| `results/retarget_v2/v2k_compare_old259.json` | `~/c8/robotlike/v2k_compare_old259.json` |
| `results/retarget_v2/compare_R30_heldout.json` | `~/c8/robotlike/compare_R30_heldout.json` |
| `results/retarget_v2/compare_R30full_heldout.json` | `~/c8/robotlike/compare_R30full_heldout.json` |
| `results/retarget_v2/compare_hybrid_heldout.json` | `~/c8/robotlike/compare_hybrid_heldout.json` |
| `results/retarget_v2/hybrid_master_MASTER.json` | `~/c8/robotlike/hybrid_master/MASTER.json` |
| `results/retarget_v2/master_combined_MASTER.json` | `~/c8/robotlike/master_combined/MASTER.json` |
| `results/retarget_v2/c8old_259_episodes.json` | `~/c8/robotlike/c8old_259_episodes.json` |

## results/hrl80

| repo path | source |
|---|---|
| `results/hrl80/hrl80_60_episodes.json` | `~/c8/robotlike/hrl80_60_episodes.json` |
| `results/hrl80/compare_hrl80_heldout.json` | `~/c8/robotlike/compare_hrl80_heldout.json` |
| `results/hrl80/hrl80_prereg_report.json` | `~/c8/robotlike/hrl80_prereg_report.json` |
| `results/hrl80/compare_groups_final.json` | `~/c8/robotlike/compare_groups_final.json` |
| `results/hrl80/compare_groups_oldonly.json` | `~/c8/robotlike/compare_groups_oldonly.json` |

## results/dataset_v2

| repo path | source |
|---|---|
| `results/dataset_v2/final/info_train.json` | `~/c8/c8oldv2_final_data/c8oldv2_final_train/meta/info.json` |
| `results/dataset_v2/final/info_val.json` | `~/c8/c8oldv2_final_data/c8oldv2_final_val/meta/info.json` |
| `results/dataset_v2/final/tasks_train.parquet` | `~/c8/c8oldv2_final_data/c8oldv2_final_train/meta/tasks.parquet` |
| `results/dataset_v2/final/tasks_val.parquet` | `~/c8/c8oldv2_final_data/c8oldv2_final_val/meta/tasks.parquet` |
| `results/dataset_v2/final/MANIFEST_final.json` | `~/c8/c8oldv2_final_data/MANIFEST_final.json` |
| `results/dataset_v2/final/APPEND_INVARIANT.json` | `~/c8/c8oldv2_final_data/APPEND_INVARIANT.json` |
| `results/dataset_v2/final/append_report.json` | `~/c8/c8oldv2_final_data/append_report.json` |
| `results/dataset_v2/final/c8oldv2_final_gate_distribution.json` | `~/c8/c8oldv2_final_gate_distribution.json` |
| `results/dataset_v2/final/c8oldv2_final_video_parity.json` | `~/c8/c8oldv2_final_video_parity.json` |
| `results/dataset_v2/final/logs_final_validate.log` | `~/c8/logs_final_validate.log` |
| `results/dataset_v2/tr_old259/MANIFEST_tr.json` | `~/c8/c8oldv2_data/MANIFEST_tr.json` |
| `results/dataset_v2/tr_old259/c8old_provenance.json` | `~/c8/c8oldv2_data/c8old_provenance.json` |
| `results/dataset_v2/tr_old259/c8oldv2_tr_gate.json` | `~/c8/c8oldv2_tr_gate.json` |
| `results/dataset_v2/tr_old259/c8oldv2_tr_gate_distribution.json` | `~/c8/c8oldv2_tr_gate_distribution.json` |
| `results/dataset_v2/tr_old259/c8oldv2_tr_video_parity.json` | `~/c8/c8oldv2_tr_video_parity.json` |
| `results/dataset_v2/hrl80/MANIFEST_hrl80.json` | `~/c8/c8oldv2_hrl80_data/MANIFEST_hrl80.json` |
| `results/dataset_v2/hrl80/c8old_provenance.json` | `~/c8/c8oldv2_hrl80_data/c8old_provenance.json` |
| `results/dataset_v2/hrl80/c8oldv2_tr_hrl80_gate.json` | `~/c8/c8oldv2_tr_hrl80_gate.json` |
| `results/dataset_v2/hrl80/c8oldv2_hrl80_gate_distribution.json` | `~/c8/c8oldv2_hrl80_gate_distribution.json` |
| `results/dataset_v2/hrl80/c8oldv2_hrl80_video_parity.json` | `~/c8/c8oldv2_hrl80_video_parity.json` |
| `results/dataset_v2/hybrid_ablation/MANIFEST_hybrid.json` | `~/c8/c8oldv2_hybrid_data/MANIFEST_hybrid.json` |
| `results/dataset_v2/hybrid_ablation/c8old_provenance.json` | `~/c8/c8oldv2_hybrid_data/c8old_provenance.json` |
| `results/dataset_v2/hybrid_ablation/c8oldv2_hybrid_gate.json` | `~/c8/c8oldv2_hybrid_gate.json` |
| `results/dataset_v2/hybrid_ablation/c8oldv2_hybrid_gate_distribution.json` | `~/c8/c8oldv2_hybrid_gate_distribution.json` |
| `results/dataset_v2/hybrid_ablation/c8oldv2_hybrid_video_parity.json` | `~/c8/c8oldv2_hybrid_video_parity.json` |

## results/finetune

| repo path | source |
|---|---|
| `results/finetune/primary_R150_250k_comparison.txt` | `~/c8/c8old_runs/primary_R150_250k_comparison.txt` |
| `results/finetune/baseline/B1old_R150_250000.json` | `~/c8/c8old_runs/baseline/B1old_R150_250000.json` |
| `results/finetune/r150ft600k/eval/250000.json` | `~/c8/c8old_runs/r150ft600k/eval/250000.json` |
| `results/finetune/r150ft600k/eval/300000.json` | `~/c8/c8old_runs/r150ft600k/eval/300000.json` |
| `results/finetune/R90_MATCHED.md` | `~/c8/c8old_runs/R90_MATCHED.md` |
| `results/finetune/r90scratch600k/SUMMARY.md` | `~/c8/c8old_runs/r90scratch600k/SUMMARY.md` |
| `results/finetune/r90ft600k/SUMMARY.md` | `~/c8/c8old_runs/r90ft600k/SUMMARY.md` |
| `results/finetune/r120scratch600k/SUMMARY.md` | `~/c8/c8old_runs/r120scratch600k/SUMMARY.md` |
| `results/finetune/r60scratch600k/SUMMARY.md` | `~/c8/c8old_runs/r60scratch600k/SUMMARY.md` |
| `results/finetune/SUMMARY.md` | `~/c8/c8old_runs/SUMMARY.md` |
| `results/finetune/FINAL_TR300K_LAUNCH.json` | `~/c8/c8old/FINAL_TR300K_LAUNCH.json` |

## Generated for this repository (no single source file)

| repo path | how |
|---|---|
| `results/hrl80/hrl80_raw_census.json` | counts and per-episode metadata read from `~/ego_collector/datasets/human_handumi_raw/HRL80/*/{counters.json,session_meta.json,episode_*/episode_meta.json}`; no raw video/IMU/mcap copied |
| `docs/NUMBERS_v2.md`, `docs/README_v2_section_draft.md`, `docs/SOURCES_v2.md` | written by hand from the files above |

## Not copied

- `*.bak*` files next to the sources (e.g. `v2k_retarget.py.bak_pre_tr`, `hybrid_merge.py.bak_old259`, `c8oldv2_validate.py.bak1`, `write_stage.py.bak_v1paths`, `robotlike_offline_check.py.bak_prefk`).
- Per-episode retarget outputs (`~/c8/robotlike/{hybrid_master,master_combined,master_old259_frozen,tr_old,r30_old,v2k_old,tr_hrl80,r30_hrl80,hybrid_hrl80}/*.json|*.npz`), live/offline per-episode checks, logs of the chains.
- LeRobot videos/parquet data and `meta/stats.json`/`meta/episodes` of every C-old v2 dataset; the raw HRL80 recordings.
- Evaluator `*.npz` dumps, checkpoints, and every other `r150ft600k/eval/*.json` except 250k and 300k.
- The 5090 launcher/archiver scripts (`~/c8/c8old/c8old5090_{train,archiver,setup}.sh`); only the launch record is included.
- Nothing larger than 5 MB was selected, so no size-based skip was needed.
