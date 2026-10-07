# training/ MANIFEST

Assembled 2026-10-07. Sources were not modified. Remote files were read with `ssh ... cat` (read-only) into a scratch dir and then copied here.

## Choice rule

The base is `~/egocentric-human-demos-rebot/training`. Every file was compared with its working origin:

- 2-line differences in "user" wording only: the working copy, which is the same code.
- Path-sanitization placeholders only (`${HOME}`, `${SHARED_ROOT}` inside Python strings and bash literals, which break execution): the **working original was restored**.
- No working copy had newer *code* than the repo for these files.

## Source -> dest

| dest | source | choice |
|---|---|---|
| `humanik_delta.py`, `eef_delta.py`, `lr_groups.py`, `aug_defaults.py`, `keep_indices.py`, `contrastive_sampler.py` | repo `training/` | repo copy (identical to `~/c8/c8old/` / node copies) |
| `train_bi.py` | `~/umi_bridge/track_c/v3/recipe/train_bi.py` | original restored (repo copy = path-sanitized) |
| `c8old_chain.sh` | `~/c8/c8old/c8old_chain.sh` | original restored (sanitized) |
| `c8old_launch.sh` | `~/c8/c8old_launch.sh` | original restored (sanitized) |
| `robot/reBot_B601_DM_dualarm.urdf` | repo `training/robot/` | identical to `~/holobrain-mac-model/reBot_B601_DM_dualarm.urdf` and every `*/urdf/` copy there (diff clean). Included on purpose: company-internal repo. Meshes (`package://rebotarm_bringup/...`) are NOT included. |
| `robot/rebot_fk.py` | repo | repo copy |
| `robot/rebot_fk_torch.py` | `~/c8/c8old/rebot_fk_torch.py` | original restored (sanitized `sys.path` / `REBOT_URDF`) |
| `relonly/train_rel16_relonly.py`, `train_rel16_relonly_lossmask.py`, `test_rel16_relonly_lossmask.py` | repo = `~/umi_bridge/rel16_audit/relonly/` | identical |
| `relonly/rel16_aux_relonly.py`, `rel16_relonly_lossmask.py` | `~/umi_bridge/rel16_audit/relonly/` | working copy (repo differs only by "user" wording) |
| `relonly/train_cotrain.py` | 5090 `/srv/data/johann/relonly/code_cotrain/train_cotrain.py` | original (repo sanitized `sys.argv[0]`) |
| `relonly/train_relonly_adamw8bit.py` | 5090 `/srv/data/johann/relonly/code_8bit/train_rel16_relonly.py` | original (repo sanitized); renamed as in the repo |
| `relonly/train_relonly_5090.sh` | `~/umi_bridge/rel16_audit/relonly/train_relonly_5090.sh` | working dir only |
| `umi76/*` | `~/umi_bridge/umi76/*` (non-.bak, non-log) | working dir only (derive_*, widen_*, gate_*, preflight_*, build_r*_umi76.sh, selections, contracts, pip freezes) |
| `umi76/run_derive_v2_ro_parent.py` | 4090 `~/umi_bridge/umi76/` (ssh cat) | node only |
| `docs/RELCART20_RELONLY_D6_RECIPE.md`, `UMI_XVLA_CONTRACT.md`, `TRACKB_HANDOFF.md` | `~/umi_bridge/` | |
| `docs/XVLA_SCRATCH_RECIPE.md` | `~/c8/XVLA_SCRATCH_RECIPE.md` (identical to `~/Downloads/` copy) | |
| `launchers/node4090/*.sh` (22) | 4090 `/home/bh-aiteam/{train_*,build_*,chain_*}.sh` (ssh cat, 10-07) | Gen-2 relcart20 v3/v3L/v4/relonly d6/d20/d20_b8/r384, ego cart20v2 d6/d20/d20_b8, ego cartonly v4, FT r312c from ego d6/d20_b8, HRA right-only + a93 loss-mask, cart20 v3, R312c build chains, `build_chain_r384.sh`. Older umi76/umi_v2/v3 launchers were not taken. |
| `launchers/node5090/` | 5090 `/srv/data/johann/relonly/code/` (ssh cat) | `train_{cotrain,ego_robot100,ft_r30,ft_r90,hra_rightonly,relonly_5090_main}*.sh`, `R30/R90_SELECTION.json`, `select_r90.py`, `build_r90.py`. `.bak1004` and `.disabled_0930` skipped. |
| `launchers/ego_cart20/` | `~/ego_cart20/launch/*.sh, *.py`, `LAST_SUBMISSION` | Mac waiters/chains + `train_xvla_softfold.py`. Logs skipped. Duplicates identical to node4090 copies removed (`train_ego_cart20v2_relonly_d6.sh`, `train_ft_r312c_from_ego_relonly_d6.sh`). `train_ego_cartonly_v4.sh` (from `~/c8/rel16ego/`) was identical to node4090 and removed. |
| `launchers/hra_node/` | `~/c8/hra_red/node/` | wrappers `wrap_lossmask*.py`, `orig_*`. Copies identical to node4090/node5090/relonly removed. |
| `launchers/mac_chains/` | `~/umi_bridge/{r384_train_when_built,v4_resume_when_gpu0_free,extend_to_300k,trainB_v4base}.sh`, `~/ego_cart20/reports/preserved_diag_20261004/ft_r30_chain.sh` | |
| `launchers/c8old/` | `~/c8/c8old/*.sh, *.py` + `FINAL_TR300K_LAUNCH.json`, `C_OLD_TR_HANDOFF.json` | `humanik_delta.py`, `rebot_fk_torch.py`, `c8old_chain.sh` duplicates removed (kept at training root) |
| `archivers/` | `~/umi_bridge/trackb_archive*.sh`, `state_to_ssd_then_rm.sh`, `~/c8/c8old/c8old5090_archiver.sh` | non-.bak only |

## Moved out

`~/umi_bridge/rel16_audit/*` (offline diag, policy decomposition, cycle logger, prompt swap, gate watchers, ssd retention, r380 audits)
is evaluation tooling. It is in `evaluation/rel16_audit/`, except `relonly/`, which is here.

## Excluded

- Every `*.bak*`, `*.orig`, `*.disabled_*`, `*.log`, `*.out`, `__pycache__`, `.DS_Store`.
- Checkpoints, bases (`xvla_base_*`), datasets, `runs/`, `ray_logs/`, the 5090 `venv/`, `lerobot-seeed/` and `pylib_bnb/`.
- `~/ego_cart20/reports/preserved_diag_20261004/{cotrain/l.sh, cotrain/orig.sh, train_ft_r30_5090_b8.sh, node_code/*}`: older snapshots of the node files that are now taken directly from the node.
- The remaining older 4090 launchers (`train_umi76.sh`, `train_umi_v2*.sh`, `train_umi_v3*.sh`, `train_r180h.sh`, `train_install*.sh`, `build_r30/r380/r663/r675/rbo*.sh`). They belong to superseded runs and are still on the node.

## Uncertainties

- The node launchers are snapshots from 10-07. A running job may have used an earlier revision, so check the sha logged in each run's startup.
- The training scripts import the node's patched LeRobot (`lerobot-seeed` / `bh_rebot_LeRobot`). That code is not here (upstream / colleague-owned).
- Secret scan: no hits. Only `HF_HOME` cache paths and the word "tokenizer" appear.
