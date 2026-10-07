#!/bin/bash
# [2026-10-07 user] Soft-Fold fine-tune "same as X-VLA" on the 5090: abs eef_6d, 30 actions / 2 s, binary gripper + BCE
# (action_mode ee6d), base lerobot/xvla-base (domain 5, ACTION IDENTITY), X-VLA optimiser recipe (train_xvla_softfold.py:
# 4 groups, AdamW 0.9/0.95, lr 1e-4, coef 0.1, freeze 1k, constant lr, clip 1.0), ColorJitter, bf16 AMP.
# User choice: bs 8 x 400k steps (bs 256 x 400k = ~34 days on one 5090).
set -e -o pipefail
R=/srv/data/johann/relonly; SF=/srv/data/johann/softfold
DS=${SF_DS:-softfold_abs30_v1_train}; DATA=$SF/lerobot/$DS; BASE=$SF/base/xvla_base_d5_identity
RUN=${RUN_NAME:-FT-SOFTFOLD-XVLA-ABS30-D5-B8-400K}
STEPS=${STEPS:-400000}; SAVE_FREQ=${SAVE_FREQ:-10000}; BS=${BS:-8}
MIN_START_GB=${MIN_START_GB:-150}; GUARD_GB=${GUARD_GB:-80}; RESUME=${RESUME:-0}
PY=$R/venv/bin/python; SRC=$R/code/lerobot-seeed/src; WRAP=$SF/code/launch/train_xvla_softfold.py
OUT=$SF/runs/$RUN
RENAME='{"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}'
FACTORY_SHA=3998f9b70422548bc3ea7a622d027be91e97a1d403f2503219091bdf763bb3a9
case "$RUN" in FT-SOFTFOLD-*) ;; *) echo "refusing: run name must start with FT-SOFTFOLD-"; exit 1 ;; esac
( cd $DATA && sha256sum -c --quiet SHA256SUMS ) || { echo "dataset SHA256SUMS mismatch"; exit 1; }; echo "[data] $DATA SHA256SUMS OK"
DOM=$($PY -c "import json,sys;print([s['config']['domain_id'] for s in json.load(open(sys.argv[1]))['steps'] if s.get('registry_name')=='xvla_add_domain_id'][0])" "$BASE/policy_preprocessor.json")
[ "$DOM" = 5 ] || { echo "domain gate: $DOM != 5"; exit 1; }; echo "[domain] $BASE processor domain_id $DOM"
NORM=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['normalization_mapping']['ACTION'])" "$BASE/config.json")
[ "$NORM" = IDENTITY ] || { echo "norm gate: ACTION $NORM != IDENTITY (BCE gripper needs raw 0/1)"; exit 1; }
[ "$(sha256sum $SRC/lerobot/datasets/factory.py | cut -c1-64)" = "$FACTORY_SHA" ] || { echo "factory.py sha mismatch"; exit 1; }
FREE=$(df --output=avail -BG /srv/data | tail -1 | tr -dc 0-9)
[ "$FREE" -lt "$MIN_START_GB" ] && { echo "only ${FREE}G free, refusing"; exit 1; }
echo "[$RUN] dataset=$DS base=$BASE steps=$STEPS bs=$BS save=$SAVE_FREQ free=${FREE}G precision=${ACCELERATE_MIXED_PRECISION:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
mkdir -p $SF/runs
if [ "$RESUME" = "1" ]; then
  CFG=$OUT/checkpoints/last/pretrained_model/train_config.json
  [ -f "$CFG" ] && [ -s "$OUT/checkpoints/last/training_state/optimizer_state.safetensors" ] || { echo "RESUME=1 but no resumable last"; exit 1; }
  TRAIN_ARGS=(--config_path="$CFG" --resume=true --save_freq=$SAVE_FREQ)
else
  TRAIN_ARGS=(
  --dataset.repo_id=softfold/$DS --dataset.root="$DATA" --dataset.image_transforms.enable=true
  --policy.path=$BASE --policy.device=cuda --policy.push_to_hub=false --policy.dtype=float32
  --policy.action_mode=ee6d --policy.chunk_size=30 --policy.n_action_steps=30 --policy.max_state_dim=20 --policy.max_action_dim=20
  --policy.use_proprio=true --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false
  --policy.train_policy_transformer=true --policy.train_soft_prompts=true --policy.scheduler_decay_steps=$STEPS
  --rename_map="$RENAME" --seed=1000 --steps=$STEPS --batch_size=$BS --num_workers=6 --eval_freq=0 --save_freq=$SAVE_FREQ
  --wandb.enable=false "$@" --output_dir="$OUT"
  )
fi
( while sleep 30; do
    pid=$(pgrep -f "[t]rain_xvla_softfold.py.*$OUT(/| |\$)" | head -1); [ -z "$pid" ] && continue
    f=$(df --output=avail -BG /srv/data | tail -1 | tr -dc 0-9)
    [ "$f" -lt "$GUARD_GB" ] && { echo "[disk-guard $(date '+%F %T')] ${f}G < ${GUARD_GB}G -- stopping $RUN" | tee -a "$OUT.guard.log"; kill -TERM "$pid"; sleep 60; kill -KILL "$pid" 2>/dev/null; }
    last=$(readlink "$OUT/checkpoints/last" 2>/dev/null); newest=$(ls "$OUT/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort | tail -1)
    for c in "$OUT"/checkpoints/[0-9]*; do s_=$(basename "$c"); [ "$s_" = "$last" ] || [ "$s_" = "$newest" ] && continue
      [ $((10#$s_ % 50000)) -ne 0 ] && [ -d "$c/training_state" ] && { rm -rf "$c/training_state"; echo "[retention $(date '+%F %T')] dropped training_state of $s_" >> "$OUT.retention.log"; }
    done
  done ) &
GUARD_PID=$!; trap 'kill $GUARD_PID 2>/dev/null' EXIT
env XVLA_STRICT_STATE_DIM=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$SRC HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} \
  $PY $WRAP "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$OUT.log"
