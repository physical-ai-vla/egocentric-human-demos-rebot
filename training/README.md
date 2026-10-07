# training/: X-VLA on reBot (Gen-1 joint space, Gen-2 REL-only Cartesian)

The model is X-VLA 0.9B (`lerobot/xvla-base`) through LeRobot. The node-side LeRobot checkout is
`/home/bh-aiteam/workspace/bh_rebot_LeRobot` (4090) or `/srv/data/johann/relonly/code/lerobot-seeed` (5090). These are not copied here.

The full recipes, in Korean with exact shas, are in `docs/`:
- `docs/RELCART20_RELONLY_D6_RECIPE.md` (Gen-2)
- `docs/XVLA_SCRATCH_RECIPE.md` (Gen-1 / B1-old)
- `docs/UMI_XVLA_CONTRACT.md`, `umi76/UMI76_CONTRACT.md` and `umi76/RELCART20_CONTRACT.md` (data contracts)
- `docs/TRACKB_HANDOFF.md`

The methods summary is `report/methods_en.tex` Sec. "Model" and "Training" in `~/egocentric-human-demos-rebot`.

## Layout

| path | what |
|---|---|
| `train_bi.py`, `humanik_delta.py`, `eef_delta.py`, `lr_groups.py`, `aug_defaults.py`, `keep_indices.py`, `contrastive_sampler.py` | Gen-1 trainer and its patches (HumanIK delta target, C/D aux losses, LR groups) |
| `c8old_launch.sh`, `c8old_chain.sh` | Gen-1 C-old 300k pretrain -> R150 600k chain (4090) |
| `robot/` | `reBot_B601_DM_dualarm.urdf` (company-internal, kinematics; meshes are `package://rebotarm_bringup/...` and not included), `rebot_fk.py`, `rebot_fk_torch.py` |
| `relonly/` | Gen-2 REL-only plugin and entry points: `rel16_aux_relonly.py` + `train_rel16_relonly.py`, the 8-bit wrapper `train_relonly_adamw8bit.py` (= node `code_8bit/train_rel16_relonly.py`), the loss-mask variant `rel16_relonly_lossmask.py` + `train_rel16_relonly_lossmask.py` (+ test), the co-train wrapper `train_cotrain.py`, and `train_relonly_5090.sh` |
| `umi76/` | dataset derivation and base widening on the node: `umi76_to_lerobot.py`, `derive_v2.py`/`derive_v3d.py`/`derive_v3L.py`/`derive_cart20.py`/**`derive_relcart20.py`**, gates (`gate_v2.py`, `gate_r312c.py`, `gate_parity76.py`, `norm_parity.py`, `preflight_*.py`), **`widen_v3.py` -> `widen_cart20.py`** (+ `widen_proprio.py`, `widen_append94.py`), episode selections `r*_front*_v1.json`, `holobrain_4090_freeze*.txt` (pip freeze of the node env) |
| `launchers/node4090/` | 4090 launchers as they exist in `/home/bh-aiteam/` (read-only copies, 10-07) |
| `launchers/node5090/` | 5090 launchers from `/srv/data/johann/relonly/code/` + `R30_SELECTION.json`, `R90_SELECTION.json`, `select_r90.py`, `build_r90.py` |
| `launchers/ego_cart20/` | Mac-side submit / wait-then-submit chains for ego pretrain -> R312c FT, ROBOT100, soft-fold |
| `launchers/hra_node/` | HRA loss-mask wrapper variants (`wrap_lossmask*.py`, `orig_*`) |
| `launchers/mac_chains/` | other Mac waiters: `r384_train_when_built.sh`, `v4_resume_when_gpu0_free.sh`, `ft_r30_chain.sh`, `extend_to_300k.sh`, `trainB_v4base.sh` |
| `launchers/c8old/` | Gen-1 C-old ladder launchers (R30/R60/R90/R120 scratch + FT, 5090 setup/train), `cfg_parity.py`, `fb_parity.py`, `resume_bi_c8old.py` |
| `archivers/` | node -> Mac PortableSSD checkpoint archivers |

## Recipes

### Gen-1: joint space (C-old / B1-old)

- Trainer: `train_bi.py` + `humanik_delta.py`. The target is the pseudo-joint dq from IK retargeting (`pipeline/pseudo_joint`, `pipeline/retarget_v2`). The auxiliary losses are C (TCP delta) and D (FK consistency, `robot/rebot_fk_torch.py`).
- Optimizer: AdamW, lr 1e-4, VLM group 0.1x, wd 1e-4, clip 10.
- Schedule: 1k warm-up, cosine to 2.5e-6.
- Batch 4, fp32 + TF32, 300k steps, checkpoint every 10k.
- Smoke: 20 steps with `log_freq=1 EE_LOG_EVERY=1`; it must print finite losses and active aux terms.
- Long runs: `--log_freq` 100-200, `EE_LOG_EVERY=200`.
- Env: `HUMANIK_DELTA=1 HUMANIK_LEAD=5 HUMANIK_ROBOT=1 HUMANIK_TARGET=cmd`, `HF_HUB_OFFLINE=1`.
- Launch: `c8old_launch.sh` / `c8old_chain.sh` (300k pretrain on C-old ego, then R150 FT 600k). The R30-R120 ladder is in `launchers/c8old/`.
- Offline eval: `evaluation/c8old_mac_eval.py` (geo score, prompt swap).

### Gen-2: RELCART20 REL-only

- **Data.** Robot datasets go `umi76_to_lerobot` -> `derive_v2 --variant C` -> `derive_v3d` -> **`derive_relcart20`**.
  - Example: R312c = HEAD180 + FRONT132 rot180, `r312c_relcart20_rel16_v4`. The node chain is `launchers/node4090/chain_r312c_relcart20_v4.sh` (`build_r312c_umi76.sh`).
  - State = 20-D per-arm `inv(T_anchor) T_t` (pos + rot6d) + openness. Action = 32-D: REL16 dims 0:20, dq dims 20:32 zeroed.
  - Ego data comes from `pipeline/cart20` (`export_lerobot*.py`).
- **Base.** `lerobot/xvla-base`, then `widen_v3.py`, then `widen_cart20.py` (32-D action / 20-D state; existing rows copied exactly, gate = same output for every domain).
  Then set the processor domain id: `xvla_base_cart20v3_d6` (domain 6, Gen-2 v1) or `xvla_base_cart20v3_d20` (**domain 20 = our own slot**; only 10-17 are pretrained in the public weights).
- **Loss** (`relonly/rel16_aux_relonly.py`): MSE on action dims 0:20 only. Dims 20:32 are hard-zeroed at the transformer input and output at every denoising step, in training and inference.
  - Deploy must also install the zero mask: `V4_RELONLY=1`.
  - A preflight (`RELONLY_PREFLIGHT=N`) checks that the loss ignores padding targets, that padding predictions are 0 and that their decoder gradients are 0.
- **Optimizer, current recipe "D20-B8".**
  - bitsandbytes AdamW8bit wrapper (`train_relonly_adamw8bit.py`).
  - `ACCELERATE_MIXED_PRECISION=bf16`, batch 8, seed 1000, warm-up 1k.
  - Cosine with **STEPS == DECAY_STEPS**. The launcher refuses a mismatch, because LeRobot silently rescales the schedule.
  - SAVE_FREQ 5-10k.
  - Actions mean/std-normalized, states not.
  - Older runs: D6 = batch 4, fp32, torch AdamW.
- **Launchers (4090).**
  - `train_relcart20_v4_relonly_d20_b8.sh <twin> <gpu>` (R312c robot-only).
  - `train_relcart20_v4_relonly_d20_b8_r384.sh` (R384).
  - `train_ego_cart20v2_relonly_d20_b8.sh EGO <gpu>` (ego pretrain).
  - The env knobs are `STEPS DECAY_STEPS SAVE_FREQ MIN_START_GB GUARD_GB RUN_NAME RESUME EXPECT_DOMAIN RELONLY_PREFLIGHT`.
- **Loss mask (single arm, HRA).** `relonly/rel16_relonly_lossmask.py` + `train_rel16_relonly_lossmask.py`.
  - Each sample carries `aux.loss_mask` (20-D) and `aux.task_id`. Masked targets are replaced by the detached prediction before subtraction, which avoids NaN gradients. The loss is divided by the number of supervised elements.
  - With no mask present, it behaves as plain REL-only.
  - Launchers: `node4090/train_hra_rightonly_lossmask_d20_b8.sh HRA <gpu>` and the size-agnostic-preflight copy `node4090/train_hra_a93_lossmask_d20_b8.sh`, used by `hra/hra_a100/pipe/chain_generic*.sh`.
- **Fine-tune.**
  - The fine-tune starts from an ego pretrain checkpoint with a fresh optimizer and the **robot** dataset's normalization stats.
  - Episode subsets go through `--dataset.episodes` from an index file: `R30_SELECTION.json` = r312c eps 150..179, `R90_SELECTION.json`.
  - Do not use `dataset_tools.delete_episodes`: it fails on the (16, 32) action under HF arrow.
  - Launchers: `node4090/train_ft_r312c_from_ego_relonly_d20_b8.sh` (`FT_BASE=<dir>`) and `node5090/train_ft_r30_5090_b8.sh` / `train_ft_r90_5090_b8.sh`.
  - A base dir is built from a checkpoint with hard links + typed config + `BASE_SHA16`; see `launchers/mac_chains/ft_r30_chain.sh`.
- **Co-train.** `relonly/train_cotrain.py` patches `make_dataset` and the DataLoader so that every batch is exactly 4 ego + 4 robot samples.
  The robot (R312c) stats drive the normalizer, and items are cut to the shared keys. The launcher is `node5090/train_cotrain_5090_b8.sh`.
  Its retention guard skips the newest checkpoint; the training_state deletion race was fixed there.

## Ray submission

GPU work always goes through `ray job submit`, never ssh + nohup. Rules:
- 4090 or 5090 only, never the 5080s.
- Set `RAY_ADDRESS=http://100.64.0.1:8265`.
- Pin the node with a resource: 4090 = `node:100.64.0.2`, 5090 = `node:100.64.0.5`.
- Launchers pick the GPU themselves and refuse a GPU using more than 1 GB.

```bash
export RAY_ADDRESS=http://100.64.0.1:8265
ray job submit --no-wait --submission-id r312c-relonly-d20-b8-<HHMM> \
  --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' \
  -- bash -c 'ACCELERATE_MIXED_PRECISION=bf16 STEPS=300000 DECAY_STEPS=300000 SAVE_FREQ=10000 MIN_START_GB=60 RELONLY_PREFLIGHT=3 \
              bash /home/bh-aiteam/train_relcart20_v4_relonly_d20_b8.sh H 0'
```

- The worker's default python is the cluster one (3.12 / torch 2.11). The launchers call the holobrain conda python on the 4090 (`/home/bh-aiteam/miniforge3/envs/holobrain/bin/python`) or the `/srv/data/johann/relonly/venv` (py3.11, torch 2.7.1+cu128) on the 5090.
  The two nodes differ in torch version, so cross-node twins are not bit-identical.
- A job left PENDING for more than 900 s fails. To wait for a free GPU, run a Mac-side waiter that polls `nvidia-smi` and only then submits. Examples: `launchers/mac_chains/v4_resume_when_gpu0_free.sh`, `launchers/ego_cart20/*_when_*.sh`.
- Always run a smoke first (60-300 steps, same launcher, `log_freq=1`), including a save + resume. Then run the main job.
- Never edit a chain script while it runs: bash reads it incrementally.
- LeRobot prints `step:22K` after 1000 steps. Parse with `tr '\r' '\n'` and do not grep `step:[0-9]+` alone.

## Archiving

Checkpoints are 3.3 GB each. The node disk fills in about 11 checkpoints, and launchers have a disk guard (`GUARD_GB`).
Start an archiver from the Mac right after the step log appears:
`IDLE_EXIT=1000000 nohup bash archivers/trackb_archive_v2_relonly.sh <RUN> >> ~/archive_<RUN>.log 2>&1 &`.

| archiver | use |
|---|---|
| `trackb_archive_v2.sh` | 4090 runs using `ot_train.py` (Gen-1 / REL16 v3/v4) |
| `trackb_archive_v2_relonly.sh` | 4090 REL-only and loss-mask runs: the running-check also matches `train_rel16_relonly*.py`. With plain v2 the check misses these runs, so it exits "not running" and would delete the newest checkpoint. |
| `trackb_archive_v3_relonly.sh` | same, seeding the weights from the Mac UI copy when the sha matches (skips a 3.3 GB transfer) |
| `trackb_archive_v2_5090.sh` | 5090 runs (`/srv/data/johann/relonly/runs/$RUN`, SSD `5090_$RUN`) |
| `trackb_archive_v2_masked.sh`, `trackb_archive.sh` | older variants (v1 died on an ssh timeout; kept for reference) |
| `c8old5090_archiver.sh` | Gen-1 C-old 5090 pretrain |
| `state_to_ssd_then_rm.sh` | cleanup: move node `training_state` of weights-only archived runs, then delete the node copy |

Protocol: rsync to `<step>.partial`, then a per-file sha256 on both sides, then rename and write `.archived` / `SHA256SUMS`, and only then delete from the node.
The node keeps `training_state` only for the newest checkpoint and fixed multiples. The SSD is `/Volumes/PortableSSD/rebot_ckpts_archive/trackb_<RUN>/<step>`.
A Mac reboot kills the archivers; restart them by hand.

## Paths and hosts

This is a private company repo, so internal hosts are kept where scripts need them: Ray head `100.64.0.1`, 4090 `bh-aiteam@100.64.0.2` (via `-J head-lp`),
and 5090 `bh-ai-5090@gpu-5090` / `100.64.0.5`. No tokens or passwords are in this folder; the secret scan is clean.
