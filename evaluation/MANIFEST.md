# evaluation/ MANIFEST

Assembled 2026-10-07. Contains scripts only, no outputs. Sources were not modified.

| dest | source | choice |
|---|---|---|
| `c8old_eval_cache.py`, `c8old_selection.py` | `~/egocentric-human-demos-rebot/evaluation/` | repo copy (identical to `~/c8/`) |
| `c8old_mac_eval.py` | `~/c8/c8old_mac_eval.py` (09-29 10:28) | **newer working copy**. Adds the R120 / R60 / R30 seen-vs-held-out groups of the probe val-10, which the repo copy lacks. It reads `~/umi_bridge/track_c/v2k/r30_episodes.json` (= `pipeline/retarget_v2/contracts/r30_episodes.json`). |
| `analysis/*.py` (prompt_swap, summarize_prompt_swap, per_order_breakdown, paired_bootstrap_r150, verify_claims, v4_r30_ablation, v4_hra_summary) | `~/egocentric-human-demos-rebot/analysis/` | repo copy |
| `analysis/v3/*.py` (dir_cos, dir_cos_alt, domain_probe, domain_slot_audit, ego_val_cos, flow_axis_check, wrist_sharpness, hra_val_loss, hardware_cycles_by_ckpt, ego_init_loss_compare, loss_curves_from_logs, compare_ckpt_offline, viz_*, hrl80_cube_pnp_vs_imu) | repo `analysis/v3/` | repo copy, except the three below |
| `analysis/v3/domain_probe.py` | `~/ego_cart20/reports/preserved_diag_20261004/domain_probe.py` | original restored (repo copy = `${REMOTE_HOME}` placeholders in Python strings) |
| `analysis/v3/domain_slot_audit.py` | `~/ego_cart20/reports/domain_slot_audit/domain_slot_audit.py` | original restored (sanitized) |
| `analysis/v3/hrl80_cube_pnp_vs_imu.py` | repo copy (origin `~/c8/hra_red/hrl80_pnp_validation/hrl_val.py`) | the placeholder `"${HOME}/ego_cart20"` was replaced in the **destination** with `os.path.expanduser("~/ego_cart20")`. This is the only edit made to a copied file; the original hard-codes `/Users/jeonghwanlee/ego_cart20`. |
| `rel16_audit/` | `~/umi_bridge/rel16_audit/*.py, *.sh, *.md` and `r380/*.py, *.md, FIXTURES_FROZEN.txt` | REL16 / RELCART20 audit and diagnostic tooling. It covers: offline_diag, policy_decomp, step1/step2 replay, prompt swap, the cycle_logger, live_drift and latency bench, gate watchers, ssd_retention helpers, r380 runtime/image parity, and RESULTS / REAL_ROBOT_LOG / PROVENANCE notes. `relonly/` went to `training/relonly/`. `compare_ckpt_offline.py` was dropped as a duplicate of `analysis/v3/compare_ckpt_offline.py` (only the docstring wording differs). |

## Excluded

- `analysis/out/` (logs, npz, json results, claim_audit.md) and every `results/` file of the repo. These are outputs, not scripts.
- From `rel16_audit`:
  - `*.json`, `*.csv` and `v2eval_ckpt_*.txt` results
  - `*_keep_extra.txt` retention lists
  - `dryrun*.npz`, `*.png`
  - `r380/*.json`, `r380/*.csv`, `.mix50_eval_done`
  - every `*.bak*`
- `~/c8/c8old_mac_eval.py.bak-*`.

## Related evaluation code elsewhere in this repo

- `hra/hra_red/val_eval/val_loss.py` and `hra/hra_a100/val_eval/val_loss_ab.py`: HRA held-out masked loss.
- `deploy/hra_direction_probe.py`: offline, read-only direction probe on the robot.
- `deploy/mac_v4_smoke_ui.py`, routes `/trial` and `/trials`: the robot trial logger.

The secret scan is clean. `offline_diag.py` matches "token" only through `observation.language.tokens`.
