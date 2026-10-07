# [2026-10-03 user] CO-TRAINING on the 5090: R312c robot + EGO ROBOT100 (ego_cart20_v2b_robot100_train), every batch 4 ego + 4 robot
# (code_cotrain/train_cotrain.py, R312c stats/meta drive the normalizer), 300k = decay. Otherwise
# same B8 recipe as the 4090 runs: batch 8, bnb AdamW8bit (code_8bit wrapper, bnb 0.45.5 in pylib_bnb), AMP bf16 via
# ACCELERATE_MIXED_PRECISION, own domain 20 (base xvla_base_cart20v3_d20 = cart20v3 hard links + processor domain 20).
# Copy of train_ego_robot100_5090_b8.sh (untouched). NOTE: 5090 / torch 2.7.1+cu128 -> not a bit-identical twin of the 4090 runs.
set -e -o pipefail
R=/srv/data/johann/relonly
DS=r312c_relcart20_rel16_v4; DATA=$R/data/$DS; EGO=$R/data/${EGO_DS:-ego_cart20_v2b_robot100_train}; BASE=$R/base/xvla_base_cart20v3_d20; SDIM=20
RUN=${RUN_NAME:-COTRAIN-R312C-EGOROBOT100-RELONLY-D20-B8-300K}
STEPS=${STEPS:-300000}; DECAY_STEPS=${DECAY_STEPS:-$STEPS}; SAVE_FREQ=${SAVE_FREQ:-5000}
MIN_START_GB=${MIN_START_GB:-200}; GUARD_GB=${GUARD_GB:-100}; RESUME=${RESUME:-0}
PY=$R/venv/bin/python; SRC=$R/code/lerobot-seeed/src; ENTRY=$R/code/train_rel16_relonly.py
OUT=$R/runs/$RUN
RENAME='{"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}'
FACTORY_SHA=3998f9b70422548bc3ea7a622d027be91e97a1d403f2503219091bdf763bb3a9
PREFLIGHT=$R/code/preflight_rel16v2.py; PREFLIGHT_SHA=4c20d0f3819653556ff0611b601b0b4891cd1214078e4883a5c224bc07743b89
case "$RUN" in COTRAIN-*) ;; *) echo "refusing: run name must start with COTRAIN-"; exit 1 ;; esac
[ "$STEPS" = "$DECAY_STEPS" ] || { echo "need STEPS == DECAY_STEPS (LeRobot auto-scales decay otherwise)"; exit 1; }
( cd $EGO && sha256sum -c --quiet SHA256SUMS ) || { echo "ego SHA256SUMS mismatch"; exit 1; }; echo "[data] ego SHA256SUMS OK (robot R312c dir has no SHA256SUMS on the 5090: checked by preflight only)"
DOM=$($PY -c "import json,sys;print([s['config']['domain_id'] for s in json.load(open(sys.argv[1]))['steps'] if s.get('registry_name')=='xvla_add_domain_id'][0])" "$BASE/policy_preprocessor.json")
[ "$DOM" = 20 ] || { echo "domain gate: $DOM != 20"; exit 1; }; echo "[domain] $BASE processor domain_id $DOM"
WRAP=$R/code_cotrain/train_cotrain.py; export COTRAIN_EGO_ROOT=$EGO COTRAIN_EGO_PER_BATCH=${COTRAIN_EGO_PER_BATCH:-4}; echo "[precision] ACCELERATE_MIXED_PRECISION=${ACCELERATE_MIXED_PRECISION:-<unset>}"
[ "$(sha256sum $SRC/lerobot/datasets/factory.py | cut -c1-64)" = "$FACTORY_SHA" ] || { echo "factory.py sha mismatch"; exit 1; }
[ "$(sha256sum $PREFLIGHT | cut -c1-64)" = "$PREFLIGHT_SHA" ] || { echo "preflight sha mismatch"; exit 1; }
for d in $DATA $EGO; do CUDA_VISIBLE_DEVICES= PYTHONPATH=$SRC $PY $PREFLIGHT "$d" "$BASE" "$SDIM" 32 || { echo "REL16 preflight FAILED ($d) -- refusing to start"; exit 1; }; done
FREE=$(df --output=avail -BG /srv/data | tail -1 | tr -dc 0-9)
[ "$FREE" -lt "$MIN_START_GB" ] && { echo "only ${FREE}G free, refusing"; exit 1; }
echo "[$RUN] 5090 cotrain robot=$DS ego=$EGO ego_per_batch=$COTRAIN_EGO_PER_BATCH base=$BASE steps=$STEPS decay=$DECAY_STEPS save=$SAVE_FREQ free=${FREE}G cuda='${CUDA_VISIBLE_DEVICES:-}'"
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
  --rename_map="$RENAME" --seed=1000 --steps=$STEPS --batch_size=8 --num_workers=4 --eval_freq=0 --save_freq=$SAVE_FREQ
  --wandb.enable=false "$@" --output_dir="$OUT"
  )
fi
( while sleep 30; do
    pid=$(pgrep -f "[t]rain_cotrain.py.*$OUT(/| |\$)" | head -1); [ -z "$pid" ] && continue
    f=$(df --output=avail -BG /srv/data | tail -1 | tr -dc 0-9)
    [ "$f" -lt "$GUARD_GB" ] && { echo "[disk-guard $(date '+%F %T')] ${f}G < ${GUARD_GB}G -- stopping $RUN" | tee -a "$OUT.guard.log"; kill -TERM "$pid"; sleep 60; kill -KILL "$pid" 2>/dev/null; }
    last=$(readlink "$OUT/checkpoints/last" 2>/dev/null); newest=$(ls "$OUT/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort | tail -1)
    for c in "$OUT"/checkpoints/[0-9]*; do s_=$(basename "$c"); [ "$s_" = "$last" ] || [ "$s_" = "$newest" ] && continue
      [ $((10#$s_ % 25000)) -ne 0 ] && [ -d "$c/training_state" ] && { rm -rf "$c/training_state"; echo "[retention $(date '+%F %T')] dropped training_state of $s_" >> "$OUT.retention.log"; }
    done
  done ) &
GUARD_PID=$!; trap 'kill $GUARD_PID 2>/dev/null' EXIT
env XVLA_STRICT_STATE_DIM=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$SRC HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} \
  $PY $WRAP "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$OUT.log"
