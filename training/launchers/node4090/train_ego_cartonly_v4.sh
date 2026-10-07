#!/bin/bash
# [2026-09-30 user] EGO CART-ONLY PRETRAIN launcher: copy of train_relcart20_v4.sh (UNTOUCHED -- the real v4 run executes it).
# Recipe copied verbatim: base xvla_base_cart20v3, seed 1000, bs 4, fp32, save 5k, chunk 16, lambdas 1.0 / 20, LR / warmup / cosine,
# factory.py sha, launcher overrides, disk guard, physical GPU pin, XVLA_STRICT_STATE_DIM. ONLY diffs:
#   schedule  [2026-09-30 user] 200k steps / cosine decay 200k (NOT v4's 600k: 17,024 rows -> 600k = 141 epochs). 100k = PRIMARY
#             pretrain ckpt (23.5 ep, mid-cosine, as the C-old 100k-of-300k precedent), 200k = SECONDARY length ablation (47 ep) = STOP.
#             STEPS > 200k is refused for every run. SSD keeps 5/10/20/30/50/75/100/150/200k (ssd_retention_ego_cartonly.sh).
#   dataset   ego_relcart20task_rel16_v4_nopseudoq[*]/rel16ego_train (state20 = RelCart20 task-start, action 32 = REL20 + NaN dq12,
#             aux.has_dq_supervision = aux.has_fk_supervision = 0, no aux.q_t); its SHA256SUMS must verify at start
#   entry/aux train_rel16aux_masked.py / rel16_aux_masked.py (absent masks = ones -> bit-identical to rel16_aux on real v4)
#   preflight preflight_cartonly.py (preflight_rel16v2.py requires aux.q_t); runs on CPU before the GPU is touched
#   gate      dataset manifest: gripper calibration FINAL + calibrated QA-G PASS, else a full run is REFUSED
#             (replaces the real-robot gripper_contract_v2.json gate, which is about the reBot teleop label, not ego)
#   run       EGO-RELCART20-REL16V4-CARTONLY-PRETRAIN; never a real v4 run name / dataset; never writes into an existing run
#   guard     disk-guard pgrep pattern = train_rel16aux_masked.py
# Archive with trackb_archive_v2_masked.sh (Mac), NOT trackb_archive_v2.sh: the old pattern `train_rel16aux[.]py` does not match the
# masked entry, so it would read the run as finished and archive + delete the resume checkpoint.
# The dq head (action_decoder outputs 20:32) gets EXACTLY 0 gradient on every ego sample: it is NOT pretrained here; it is first learned
# in the R312c v4 real FT (only AdamW decoupled weight decay 1e-4 touches it, see the dataset manifest handoff note).
#
# usage: train_ego_cartonly_v4.sh <physical gpu> [extra lerobot_train args]
#   DRY_RUN=1   run every gate + the CPU preflight, print the lerobot_train command, exit 0 before touching the GPU
#   RESUME=1    resume $OUT/checkpoints/last (saved config is gated as in v4)
#   EGO_DS=ego_relcart20task_rel16_v4_nopseudoq<suffix>   the calibrated final re-derive (new path), default = the candidate
#   RUN_NAME=...SMOKE...   smoke: skips the gripper-FINAL and 600k gates (still never launch one without the user's go)
set -e
set -o pipefail
case "${1:-}" in -h|--help|"") sed -n '2,30p' "$0"; exit 0 ;; esac
GPU=$1
shift
EGO_DS=${EGO_DS:-ego_relcart20task_rel16_v4_nopseudoq}
case "$EGO_DS" in ego_relcart20task_rel16_v4_nopseudoq*) ;; *) echo "EGO_DS must be ego_relcart20task_rel16_v4_nopseudoq[*], got $EGO_DS"; exit 1 ;; esac
BASE=/home/bh-aiteam/xvla_base_cart20v3; SDIM=20
RUN=${RUN_NAME:-EGO-RELCART20-REL16V4-CARTONLY-PRETRAIN}
case "$RUN" in *R312C*|*REL16V4-D600K*|*REL16V3*|*HEAD180*) echo "run name $RUN collides with a real-robot run family, refusing"; exit 1 ;; esac
case "$RUN" in EGO-*) ;; *) echo "run name must start with EGO-, got $RUN"; exit 1 ;; esac
DRY_RUN=${DRY_RUN:-0}
PY=/home/bh-aiteam/miniforge3/envs/holobrain/bin/python
SRC=/home/bh-aiteam/lerobot-seeed/src
DATA_ROOT=/home/bh-aiteam/holobrain-data/lerobot/$EGO_DS
DATA=$DATA_ROOT/rel16ego_train
MAN=$DATA_ROOT/MANIFEST_ego_cartonly_v4.json
OUT=/home/bh-aiteam/holobrain-data/trainB/$RUN
RENAME='{"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}'

# dataset identity: the local copy must be byte-identical to its own SHA256SUMS (a stale candidate copy fails here)
[ -d "$DATA" ] && [ -f "$MAN" ] || { echo "dataset $DATA_ROOT missing (copy the frozen dataset from /mnt/shared/johann/rel16ego/)"; exit 1; }
(cd "$DATA_ROOT" && sha256sum -c --quiet SHA256SUMS) || { echo "dataset SHA256SUMS mismatch in $DATA_ROOT, refusing to start"; exit 1; }
echo "[$RUN] dataset $EGO_DS SHA256SUMS OK"

# gripper gate: a full run needs the FINAL calibrated dataset (manifest status + calibration fields + calibrated QA-G PASS)
GATE=$(python3 - "$MAN" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1])); st = m.get("status", {}); q = m.get("gripper_convention_qa", {}).get("calibrated")
bad = []
if st.get("gripper_calibration") != "FINAL": bad.append(f"status.gripper_calibration = {st.get('gripper_calibration')!r}")
if not str(st.get("overall", "")).startswith("FINAL"): bad.append(f"status.overall = {st.get('overall')!r}")
for k in ("gripper_calibration_version", "closed_aperture_mm", "open_aperture_mm", "raw_to_mm_mapping"):
    if m.get(k) is None or "PENDING" in str(m.get(k)): bad.append(f"{k} = {m.get(k)!r}")
if not q or not all(r["G1_range"]["pass_"] and r["G2_action_vs_future_state"]["pass_"] for r in q.values()): bad.append("calibrated QA-G missing or not PASS")
if m.get("PSEUDO_Q") != "NONE" or m.get("DQ_SUPERVISION") != "OFF" or m.get("FK_SUPERVISION") != "OFF": bad.append("manifest is not a cart-only (no pseudo-q) dataset")
print("OK" if not bad else "; ".join(bad))
EOF
)
case "$RUN" in
  *SMOKE*) [ "$GATE" = OK ] || echo "WARNING (smoke only): gripper gate would refuse a full run: $GATE" ;;
  *) [ "$GATE" = OK ] || { echo "gripper gate: dataset is not FINAL -- full pretrain REFUSED: $GATE"; exit 1; } ;;
esac

STEPS=${STEPS:-200000}
MAX_STEPS=200000   # [2026-09-30 user] ego cart-only contract: max 200k, 100k primary; 600k is never run
DECAY_STEPS=${DECAY_STEPS:-$STEPS}
SAVE_FREQ=${SAVE_FREQ:-5000}
MIN_START_GB=${MIN_START_GB:-150}
GUARD_GB=${GUARD_GB:-50}
case "$RUN" in *decay30k*) echo "refusing to write into a decay30k reference run"; exit 1 ;; esac
case "$RUN" in
  *SMOKE*) ;;
  *) [ "$STEPS" = 200000 ] && [ "$DECAY_STEPS" = 200000 ] && [ "$SAVE_FREQ" = 5000 ] || {
       echo "primary gate: need STEPS=200000 DECAY_STEPS=200000 SAVE_FREQ=5000 (ego cart-only contract), got $STEPS/$DECAY_STEPS/$SAVE_FREQ"; exit 1; } ;;
esac
[ "$STEPS" -le "$MAX_STEPS" ] && [ "$DECAY_STEPS" -le "$MAX_STEPS" ] || { echo "STEPS/DECAY_STEPS $STEPS/$DECAY_STEPS > ${MAX_STEPS} -- the ego cart-only pretrain is capped at 200k"; exit 1; }
RESUME=${RESUME:-0}
[ "$RESUME" != "1" ] && [ -e "$OUT" ] && [ "$DRY_RUN" != "1" ] && { echo "$OUT already exists and RESUME != 1 -- refusing to write into an existing run"; exit 1; }

FACTORY_SHA=3998f9b70422548bc3ea7a622d027be91e97a1d403f2503219091bdf763bb3a9
PREFLIGHT=/home/bh-aiteam/umi_bridge/umi76/preflight_cartonly.py; PREFLIGHT_SHA=777ccd15f42c959fe04a2b434512919c638aeb1ea9dc7480a8a2eb6607aee89d
AUX=/home/bh-aiteam/umi_bridge/umi76/rel16_aux_masked.py; AUX_SHA=5e13c15bb08c66b47c28f9484c4774ae1cc4629bc64dae3ab0b9731b31a7d14c
ENTRY=/home/bh-aiteam/umi_bridge/umi76/train_rel16aux_masked.py; ENTRY_SHA=a3e3b991e00910dad81fed901bad79fd3be87e98aaf2372d33da6b84ce7b67a2
export REL16_STATS=$DATA/meta/stats.json REL16_LAMBDA_Q=1.0 REL16_LAMBDA_FK=20 REL16_LOG_EVERY=200
[ "$(sha256sum $SRC/lerobot/datasets/factory.py | cut -c1-64)" = "$FACTORY_SHA" ] || { echo "factory.py sha mismatch -- REL16 chunk contract not guaranteed, refusing to start"; exit 1; }
[ "$(sha256sum $PREFLIGHT | cut -c1-64)" = "$PREFLIGHT_SHA" ] || { echo "preflight_cartonly.py sha mismatch, refusing to start"; exit 1; }
[ "$(sha256sum $AUX | cut -c1-64)" = "$AUX_SHA" ] && [ "$(sha256sum $ENTRY | cut -c1-64)" = "$ENTRY_SHA" ] || { echo "rel16_aux_masked / train_rel16aux_masked sha mismatch, refusing to start"; exit 1; }
CUDA_VISIBLE_DEVICES= PYTHONPATH=$SRC $PY $PREFLIGHT "$DATA" "$BASE" "$SDIM" 2>&1 | grep -E "^(PASS|FAIL)|PREFLIGHT" || { echo "CART-ONLY preflight FAILED -- refusing to start"; exit 1; }
free_gb () { df --output=avail -BG /home | tail -1 | tr -dc 0-9; }
FREE=$(free_gb)
if [ "$FREE" -lt "$MIN_START_GB" ]; then
  [ "$DRY_RUN" = "1" ] && echo "DRY_RUN note: only ${FREE}G free on /home (< ${MIN_START_GB}G) -- a real launch would refuse" || { echo "only ${FREE}G free on /home (< ${MIN_START_GB}G), refusing to start"; exit 1; }
fi
[ "$FREE" -lt 200 ] && echo "WARNING: only ${FREE}G free on /home, below the recommended 200G"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -dc 0-9)
if [ -n "$USED" ] && [ "$USED" -gt 1000 ]; then
  [ "$DRY_RUN" = "1" ] && echo "DRY_RUN note: physical GPU $GPU has ${USED} MiB in use (a real launch would refuse)" || { echo "physical GPU $GPU already has ${USED} MiB in use, refusing to start"; exit 1; }
fi
echo "[$RUN] physical GPU $GPU (${USED:-?} MiB used at launch; ray CUDA_VISIBLE_DEVICES was '${CUDA_VISIBLE_DEVICES:-}')"
echo "[$RUN] dataset=$EGO_DS base=$BASE(sha $(cat $BASE/BASE_SHA256)) gpu=$GPU free=${FREE}G steps=$STEPS decay=$DECAY_STEPS save=$SAVE_FREQ guard=${GUARD_GB}G extra=$*"

if [ "$RESUME" = "1" ]; then
  CFG=$OUT/checkpoints/last/pretrained_model/train_config.json
  TS=$OUT/checkpoints/last/training_state
  [ -f "$CFG" ] && [ -s "$TS/optimizer_state.safetensors" ] || { echo "RESUME=1 but $OUT/checkpoints/last has no config or optimizer state"; exit 1; }
  read -r c_steps c_decay c_save c_seed <<< "$($PY -c "import json;c=json.load(open('$CFG'));print(c['steps'],c['policy']['scheduler_decay_steps'],c['save_freq'],c['seed'])")"
  case "$RUN" in *SMOKE*) ;; *)
    [ "$c_steps" = 200000 ] && [ "$c_decay" = 200000 ] && [ "$c_save" = 5000 ] && [ "$c_seed" = 1000 ] || {
      echo "resume gate: saved config is $c_steps/$c_decay/$c_save/seed $c_seed, expected 200000/200000/5000/1000"; exit 1; } ;;
  esac
  echo "[$RUN] RESUME from $(readlink $OUT/checkpoints/last) (step $(cat $TS/training_step.json | tr -dc 0-9))"
  TRAIN_ARGS=(--config_path="$CFG" --resume=true --save_freq=$SAVE_FREQ)
else
  TRAIN_ARGS=(
  --dataset.repo_id=rebot/$EGO_DS
  --dataset.root="$DATA"
  --policy.path=$BASE
  --policy.device=cuda
  --policy.push_to_hub=false
  --policy.dtype=float32
  --policy.chunk_size=16
  --policy.n_action_steps=16
  --policy.max_state_dim=$SDIM
  --policy.max_action_dim=32
  --policy.action_mode=auto
  --policy.use_proprio=true
  --policy.freeze_vision_encoder=false
  --policy.freeze_language_encoder=false
  --policy.train_policy_transformer=true
  --policy.train_soft_prompts=true
  --policy.scheduler_decay_steps=$DECAY_STEPS
  --rename_map="$RENAME"
  --seed=1000
  --steps=$STEPS
  --batch_size=4
  --num_workers=4
  --eval_freq=0
  --save_freq=$SAVE_FREQ
  --wandb.enable=false
  "$@"
  --output_dir="$OUT"
  )
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "[$RUN] DRY_RUN: all gates + preflight passed; would run:"
  printf '  env CUDA_VISIBLE_DEVICES=%s XVLA_STRICT_STATE_DIM=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=%s %s %s' "$GPU" "$SRC" "$PY" "$ENTRY"
  printf ' %q' "${TRAIN_ARGS[@]}"; echo; echo "  | tee -a $OUT.log"
  exit 0
fi
mkdir -p "$(dirname "$OUT")"
cd /home/bh-aiteam/umi_bridge

# Disk guard (as v4). pgrep's [t] keeps it from matching this shell.
(
  while sleep 30; do
    pid=$(pgrep -f "[t]rain_rel16aux_masked.py.*$OUT(/| |\$)" | head -1)
    [ -z "$pid" ] && continue
    f=$(free_gb)
    if [ "$f" -lt "$GUARD_GB" ]; then
      echo "[disk-guard $(date '+%F %T')] ${f}G free < ${GUARD_GB}G -- stopping $RUN (pid $pid) before a save hits ENOSPC. Resume with RESUME=1." | tee -a "$OUT.guard.log"
      kill -TERM "$pid"; sleep 60; kill -KILL "$pid" 2>/dev/null
    fi
  done
) &
GUARD_PID=$!
trap 'kill $GUARD_PID 2>/dev/null' EXIT

env CUDA_VISIBLE_DEVICES=$GPU XVLA_STRICT_STATE_DIM=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$SRC $PY $ENTRY \
  "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$OUT.log"
