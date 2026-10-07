#!/bin/bash
# [2026-09-30 user] REL-only ablation of R312C-RELCART20-REL16V4-D600K on the 5090. ONE variable: Δq + FK supervision ON -> OFF
# (rel16_aux_relonly.py: rel_loss only; action channels 20:32 hard-zeroed at the transformer input/output, train + infer).
# Everything else copies train_relcart20_v4.sh: dataset r312c_relcart20_rel16_v4 (same bytes), base xvla_base_cart20v3,
# seed 1000, bs 4, fp32, chunk 16, max_state 20, max_action 32, lr/warmup/decay from the base config, decay 600k.
# Training stops at STEPS (300k, user) with the SAME 600k cosine decay, so the LR curve equals v4's up to 300k.
# MAIN ablation (user): R312c REL-only vs R312C-RELCART20-REL16V4-D600K. Retention: training_state kept only on 25k multiples + newest.
# Env note: the 5090 (sm_120) cannot run torch 2.6+cu124; this venv = torch 2.7.1+cu128 + every other package pinned to the
# 4090 holobrain freeze (flash_attn / pytorch3d dropped: not used by X-VLA). lerobot-seeed = the 4090 tree incl. its patches.
# usage: RUN_NAME=... STEPS=... SAVE_FREQ=... [RESUME=1] [RELONLY_PREFLIGHT=N] bash train_relonly_5090.sh [extra lerobot args]
set -e -o pipefail
R=/srv/data/johann/relonly
DS=r312c_relcart20_rel16_v4; DATA=$R/data/$DS; BASE=$R/base/xvla_base_cart20v3; SDIM=20
RUN=${RUN_NAME:-R312C-RELCART20-RELONLY-V4}
STEPS=${STEPS:-300000}; DECAY_STEPS=${DECAY_STEPS:-600000}; SAVE_FREQ=${SAVE_FREQ:-5000}
MIN_START_GB=${MIN_START_GB:-200}; GUARD_GB=${GUARD_GB:-100}; RESUME=${RESUME:-0}
PY=$R/venv/bin/python; SRC=$R/code/lerobot-seeed/src; ENTRY=$R/code/train_rel16_relonly.py
OUT=$R/runs/$RUN
RENAME='{"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}'
FACTORY_SHA=3998f9b70422548bc3ea7a622d027be91e97a1d403f2503219091bdf763bb3a9
PREFLIGHT=$R/code/preflight_rel16v2.py; PREFLIGHT_SHA=4c20d0f3819653556ff0611b601b0b4891cd1214078e4883a5c224bc07743b89
case "$RUN" in R312C-RELCART20-REL16V4-D600K*) echo "refusing: that is the v4 run name"; exit 1 ;; esac
[ "$(sha256sum $SRC/lerobot/datasets/factory.py | cut -c1-64)" = "$FACTORY_SHA" ] || { echo "factory.py sha mismatch"; exit 1; }
[ "$(sha256sum $PREFLIGHT | cut -c1-64)" = "$PREFLIGHT_SHA" ] || { echo "preflight sha mismatch"; exit 1; }
CUDA_VISIBLE_DEVICES= PYTHONPATH=$SRC $PY $PREFLIGHT "$DATA" "$BASE" "$SDIM" 32 || { echo "REL16 preflight FAILED -- refusing to start"; exit 1; }
FREE=$(df --output=avail -BG /srv/data | tail -1 | tr -dc 0-9)
[ "$FREE" -lt "$MIN_START_GB" ] && { echo "only ${FREE}G free, refusing"; exit 1; }
echo "[$RUN] 5090 dataset=$DS base=$BASE steps=$STEPS decay=$DECAY_STEPS save=$SAVE_FREQ free=${FREE}G cuda='${CUDA_VISIBLE_DEVICES:-}'"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
mkdir -p "$(dirname "$OUT")"; cd $R/code
if [ "$RESUME" = "1" ]; then
  CFG=$OUT/checkpoints/last/pretrained_model/train_config.json
  [ -f "$CFG" ] && [ -s "$OUT/checkpoints/last/training_state/optimizer_state.safetensors" ] || { echo "RESUME=1 but no resumable last"; exit 1; }
  TRAIN_ARGS=(--config_path="$CFG" --resume=true --save_freq=$SAVE_FREQ)
else
  TRAIN_ARGS=(
  --dataset.repo_id=rebot/$DS --dataset.root="$DATA" --policy.path=$BASE --policy.device=cuda --policy.push_to_hub=false
  --policy.dtype=float32 --policy.chunk_size=16 --policy.n_action_steps=16 --policy.max_state_dim=$SDIM --policy.max_action_dim=32
  --policy.action_mode=auto --policy.use_proprio=true --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false
  --policy.train_policy_transformer=true --policy.train_soft_prompts=true --policy.scheduler_decay_steps=$DECAY_STEPS
  --rename_map="$RENAME" --seed=1000 --steps=$STEPS --batch_size=4 --num_workers=4 --eval_freq=0 --save_freq=$SAVE_FREQ
  --wandb.enable=false "$@" --output_dir="$OUT"
  )
fi
( while sleep 30; do
    pid=$(pgrep -f "[t]rain_rel16_relonly.py.*$OUT(/| |\$)" | head -1); [ -z "$pid" ] && continue
    f=$(df --output=avail -BG /srv/data | tail -1 | tr -dc 0-9)
    [ "$f" -lt "$GUARD_GB" ] && { echo "[disk-guard $(date '+%F %T')] ${f}G < ${GUARD_GB}G -- stopping $RUN" | tee -a "$OUT.guard.log"; kill -TERM "$pid"; sleep 60; kill -KILL "$pid" 2>/dev/null; }
    last=$(readlink "$OUT/checkpoints/last" 2>/dev/null)
    for c in "$OUT"/checkpoints/[0-9]*; do s_=$(basename "$c"); [ "$s_" = "$last" ] && continue
      [ $((10#$s_ % 25000)) -ne 0 ] && [ -d "$c/training_state" ] && { rm -rf "$c/training_state"; echo "[retention $(date '+%F %T')] dropped training_state of $s_" >> "$OUT.retention.log"; }
    done
  done ) &
GUARD_PID=$!; trap 'kill $GUARD_PID 2>/dev/null' EXIT
env XVLA_STRICT_STATE_DIM=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$SRC HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} \
  $PY $ENTRY "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$OUT.log"
