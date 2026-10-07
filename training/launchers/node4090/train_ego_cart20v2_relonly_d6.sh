#!/bin/bash
# [2026-10-01 user] EGO-ONLY CART20 v2 PRETRAIN with the REL-only domain-6 recipe (umi_bridge/RELCART20_RELONLY_D6_RECIPE.md).
# Copy of train_relcart20_v4_relonly_d6.sh (sha 6ecdd805..., untouched). Diffs ONLY: twin EGO = dataset ego_cart20_v2_train
# (~/ego_cart20 pipeline: human HandUMI CART20, state = RELCART20 task anchor [L9|R9|gL gR], action [16,32] = CART20 | AUX12 = 0,
# targets t + k*50.05 ms, gripper 0 closed / 1 open, aux.q_t = NaN placeholder), run name, schedule gate (STEPS == DECAY, recipe 6),
# the robot gripper-contract gate replaced by the ego EXPORT.json contract gate. Base xvla_base_cart20v3_d6 (domain 6, the R312c FT
# domain), same factory / preflight / REL-only plugin / entry shas, seed 1000, bs 4, fp32, lr / warmup / cosine unchanged.
# Downstream: R312C-RELCART20-RELONLY-D6 style FT with --policy.path=<ego ckpt>.
# usage: STEPS=200000 DECAY_STEPS=200000 SAVE_FREQ=10000 MIN_START_GB=60 RELONLY_PREFLIGHT=3 train_ego_cart20v2_relonly_d6.sh EGO <gpu>
# [2026-10-01 user] REL-only on R312c with X-VLA domain_id 6 (robotwin2 = AgileX dual-arm slot). Copy of train_relcart20_v3_relonly.sh
# (untouched) with: twin F = dataset r312c_relcart20_rel16_v4 (same bytes as v4 / the 5090 REL-only run) + base xvla_base_cart20v3_d6
# (= xvla_base_cart20v3, weights hard-linked, ONLY policy_preprocessor domain_id 0 -> 6). Same REL-only plugin/entry shas as the 5090
# run R312C-RELCART20-RELONLY-V4 (domain 0), seed 1000, bs 4, fp32, 300k steps / decay 600k, save 5k. Run R312C-RELCART20-RELONLY-D6.
# NOTE vs the 5090 domain-0 run: different GPU/env (4090 holobrain torch 2.6 vs 5090 torch 2.7.1+cu128) -- not a pure one-variable twin.
# usage: STEPS=300000 DECAY_STEPS=600000 SAVE_FREQ=5000 MIN_START_GB=60 train_relcart20_v4_relonly_d6.sh F <gpu>
# [2026-09-30 user] REL-only ablation of train_relcart20_v3.sh (untouched; the v3 run executes it). ONE variable: dq + FK supervision
# ON -> OFF (rel16_aux_relonly.py: rel_loss only, action channels 20:32 hard-zeroed at the transformer input/output, train + infer).
# Same dataset r180_relcart20_rel16_v3d, base xvla_base_cart20v3, seed 1000, bs 4, holobrain env as v3. Run HEAD180-RELCART20-RELONLY-V3.
# STEPS=300000 with DECAY_STEPS=600000 (user: stop at 300k, same LR curve as v3). usage: STEPS=300000 DECAY_STEPS=600000 train_relcart20_v3_relonly.sh E <gpu>
# [2026-09-29] CART20 state ablation of train_umi_v3.sh: IDENTICAL recipe (seed 1000, bs 4, fp32, 600k decay 600k, rel16_aux lambdas,
# factory/preflight/aux/entry shas); only dataset r180_cart20_rel16_v3d (state20 = L/R TCP pos+rot6d + openness, derive_cart20.py)
# and base xvla_base_cart20v3 (widen_cart20.py: v3 base with proprio rows 76 -> 20). usage: train_cart20_v3.sh D <gpu>
# [2026-09-29] RELCART20 copy of train_cart20_v3.sh: identical recipe and base; only dataset r180_relcart20_rel16_v3d (state20 = per-arm
# inv(T_anchor) T_t pos+rot6d, anchor = episode frame 0, + openness; derive_relcart20.py). usage: train_relcart20_v3.sh E <gpu>
# [2026-09-28] REL16-v2 twin on R312-CENTER (HEAD180 + FRONT132 pan=center, 22/order), from train_umi_v2.sh (R380, obsolete): A = state76 + binary leader gripper, B = A + base-frame TCP18 (state94, DIAGNOSTIC).
# Derived from train_umi76.sh (unchanged). Datasets: derive_v2.py (gates gate_v2.py 18/18). Bases:
#   A  /home/bh-aiteam/xvla_base_umi76   (the v1 REL16 base, untouched)
#   B  /home/bh-aiteam/xvla_base_umi94   (A + 18 fresh encoder columns, widen_append94.py; all else bit-identical)
# Same seed 1000 / bs 4 / fp32 / 600k decay 600k / save 5k / chunk 16 for both; only dataset state width + base differ.
# LAUNCH ONLY AFTER `gripper_smoke_test.py --cube` PASSES (user 2026-09-28): the gate below checks the contract file
# the Mac writes, copied here as ~/umi_bridge/gripper_contract_v2.json.
#
# [2026-09-29] REL16-v3 (from train_umi_v2c.sh): HEAD180, state76, action 32 = REL16 20 (continuous leader gripper) + dq12,
# trained with train_rel16aux.py (rel16_aux: rel + lambda_q*dq + lambda_fk*FK position consistency), base xvla_base_rel16v3
# (INPUT-major widen of lerobot/xvla-base, functional-equivalence verified). lambdas are FROZEN here, not tuned on results.
# usage: train_umi_v3.sh C <gpu> [extra lerobot_train args]      RESUME=1 as in train_umi76.sh
set -e
# [2026-09-28] pipefail: training runs as `... | tee`, and without this the pipeline's status is tee's (0), so an OOM
# came back as a SUCCEEDED Ray job on the first smoke. Now the job fails exactly when lerobot_train fails.
set -o pipefail
TWIN=${1:?E}
GPU=${2:?gpu index}
shift 2
case "$TWIN" in
  EGO) DS=${EGO_DS:-ego_cart20_v2_train}; BASE=/home/bh-aiteam/xvla_base_cart20v3_d6; SDIM=20 ;;   # [2026-10-01] ego CART20 v2, domain 6 base
  *) echo "twin must be EGO (ego CART20 v2, domain 6)"; exit 1 ;;
esac
RUN=${RUN_NAME:-EGO-CART20V2-RELONLY-D6-PRETRAIN}
case "$RUN" in EGO-CART20V2-*) ;; *) echo "refusing: run name must start with EGO-CART20V2- (got $RUN)"; exit 1 ;; esac
# ego contract gate (replaces the robot gripper_contract_v2 gate): the dataset's EXPORT.json must declare the frozen ego contract
EXP=/home/bh-aiteam/holobrain-data/lerobot/$DS/EXPORT.json
python3 -c "import json,sys;e=json.load(open('$EXP'));sys.exit(0 if (e.get('schema')=='ego_cart20_v2_lerobot/v1' and e.get('state')=='RELCART20 task anchor' and e.get('aux12')=='zero') else 1)" 2>/dev/null || {
  echo "ego contract gate: $EXP missing or not the ego_cart20_v2 contract"; exit 1; }
[ -f /home/bh-aiteam/holobrain-data/lerobot/$DS/SHA256SUMS ] && { ( cd /home/bh-aiteam/holobrain-data/lerobot/$DS && sha256sum -c --quiet SHA256SUMS ) || { echo "dataset SHA256SUMS mismatch"; exit 1; }; echo "[data] SHA256SUMS OK"; } || { echo "dataset has no SHA256SUMS, refusing"; exit 1; }
STEPS=${STEPS:-600000}
DECAY_STEPS=${DECAY_STEPS:-$STEPS}
SAVE_FREQ=${SAVE_FREQ:-10000}   # [2026-09-29] CART20: every 10k from the start (matches the v3 cadence after 50k)
MIN_START_GB=${MIN_START_GB:-150}
GUARD_GB=${GUARD_GB:-50}
case "$RUN" in *decay30k*) echo "refusing to write into a decay30k reference run"; exit 1 ;; esac
case "$RUN" in
  *SMOKE*) ;;
  *) [ "$STEPS" = "$DECAY_STEPS" ] && [ "$STEPS" -le 600000 ] && { [ "$SAVE_FREQ" = 10000 ] || [ "$SAVE_FREQ" = 5000 ]; } || {   # recipe 6: STEPS == DECAY (LeRobot auto-scales decay otherwise)
       echo "ego gate: need STEPS == DECAY_STEPS <= 600000 and SAVE_FREQ 10000|5000, got $STEPS/$DECAY_STEPS/$SAVE_FREQ"; exit 1; } ;;
esac
RESUME=${RESUME:-0}
PY=/home/bh-aiteam/miniforge3/envs/holobrain/bin/python
SRC=/home/bh-aiteam/lerobot-seeed/src
DATA=/home/bh-aiteam/holobrain-data/lerobot/$DS
OUT=/home/bh-aiteam/holobrain-data/trainB/$RUN
RENAME='{"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}'
# [2026-09-28] REL16 chunk contract (pre-training hardening, no semantics change): the patched datasets/factory.py keeps
# the stored (16, 20) action chunk from being re-expanded into a nested (16, 16, 20) horizon, which would train silently.
# Hard fail on any mismatch of the patch or the preflight script, and run the CPU preflight (same code path as
# lerobot_train) before touching the GPU. Hashes are in ~/umi_bridge/rel16v2b_r312c_freeze/INIT_FREEZE_ADDENDUM.json.
FACTORY_SHA=3998f9b70422548bc3ea7a622d027be91e97a1d403f2503219091bdf763bb3a9
PREFLIGHT=/home/bh-aiteam/umi_bridge/umi76/preflight_rel16v2.py; PREFLIGHT_SHA=4c20d0f3819653556ff0611b601b0b4891cd1214078e4883a5c224bc07743b89
AUX=/home/bh-aiteam/umi_bridge/umi76/rel16_aux_relonly.py; AUX_SHA=3511f81aec809103ddc97c7440c3f111c08c90d3d7083b577fc9f5fc004a4bf3
ENTRY=/home/bh-aiteam/umi_bridge/umi76/train_rel16_relonly.py; ENTRY_SHA=1f0b486133b4590a7685be667fd0c87d5c6459c323b450e4adc5a14d4a41df35
export REL16_STATS=/home/bh-aiteam/holobrain-data/lerobot/$DS/meta/stats.json REL16_LAMBDA_Q=1.0 REL16_LAMBDA_FK=20 REL16_LOG_EVERY=200
[ "$(sha256sum $SRC/lerobot/datasets/factory.py | cut -c1-64)" = "$FACTORY_SHA" ] || { echo "factory.py sha mismatch -- REL16 chunk contract not guaranteed, refusing to start"; exit 1; }
[ "$(sha256sum $PREFLIGHT | cut -c1-64)" = "$PREFLIGHT_SHA" ] || { echo "preflight_rel16v2.py sha mismatch, refusing to start"; exit 1; }
[ "$(sha256sum $AUX | cut -c1-64)" = "$AUX_SHA" ] && [ "$(sha256sum $ENTRY | cut -c1-64)" = "$ENTRY_SHA" ] || { echo "rel16_aux / train_rel16aux sha mismatch, refusing to start"; exit 1; }
# [2026-10-01] domain contract: the processor (loaded from BASE by lerobot_train) must carry domain_id 6
DOM=$($PY -c "import json,sys;print([s['config']['domain_id'] for s in json.load(open(sys.argv[1]))['steps'] if s.get('registry_name')=='xvla_add_domain_id'][0])" "$BASE/policy_preprocessor.json")
[ "$DOM" = "${EXPECT_DOMAIN:-6}" ] || { echo "domain gate: $BASE processor has domain_id $DOM, expected ${EXPECT_DOMAIN:-6}"; exit 1; }
echo "[domain] $BASE processor domain_id $DOM"
CUDA_VISIBLE_DEVICES= PYTHONPATH=$SRC $PY $PREFLIGHT "$DATA" "$BASE" "$SDIM" 32 || { echo "REL16 preflight FAILED -- refusing to start"; exit 1; }
free_gb () { df --output=avail -BG /home | tail -1 | tr -dc 0-9; }
FREE=$(free_gb)
[ "$FREE" -lt "$MIN_START_GB" ] && { echo "only ${FREE}G free on /home (< ${MIN_START_GB}G), refusing to start"; exit 1; }
[ "$FREE" -lt 200 ] && echo "WARNING: only ${FREE}G free on /home, below the recommended 200G"
# Physical GPU pin (2026-09-28): Ray's logical GPU state is NOT trusted here -- scratch R90 holds physical GPU0
# outside Ray's accounting, so Ray handed the first smoke "GPU 0" and it OOM'd. The caller passes the PHYSICAL index;
# refuse to start on a GPU that already has >1 GB in use, and log what nvidia-smi saw.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -dc 0-9)
[ -n "$USED" ] && [ "$USED" -gt 1000 ] && { echo "physical GPU $GPU already has ${USED} MiB in use, refusing to start"; exit 1; }
echo "[$RUN] physical GPU $GPU (${USED:-?} MiB used at launch; ray CUDA_VISIBLE_DEVICES was '${CUDA_VISIBLE_DEVICES:-}')"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "[$RUN] dataset=$DS base=$BASE(sha $(cat $BASE/BASE_SHA256)) gpu=$GPU free=${FREE}G steps=$STEPS decay=$DECAY_STEPS save=$SAVE_FREQ guard=${GUARD_GB}G extra=$*"
mkdir -p "$(dirname "$OUT")"
cd /home/bh-aiteam/umi_bridge

if [ "$RESUME" = "1" ]; then
  CFG=$OUT/checkpoints/last/pretrained_model/train_config.json
  TS=$OUT/checkpoints/last/training_state
  [ -f "$CFG" ] && [ -s "$TS/optimizer_state.safetensors" ] || { echo "RESUME=1 but $OUT/checkpoints/last has no config or optimizer state"; exit 1; }
  # the resumed run is governed by the SAVED config, so gate that, not the CLI defaults
  read -r c_steps c_decay c_save c_seed <<< "$($PY -c "import json;c=json.load(open('$CFG'));print(c['steps'],c['policy']['scheduler_decay_steps'],c['save_freq'],c['seed'])")"
  case "$RUN" in *SMOKE*) ;; *)
    [ "$c_steps" = "$c_decay" ] && [ "$c_seed" = 1000 ] || {
      echo "resume gate: saved config is $c_steps/$c_decay/$c_save/seed $c_seed, expected steps == decay, seed 1000"; exit 1; } ;;
  esac
  echo "[$RUN] RESUME from $(readlink $OUT/checkpoints/last) (step $(cat $TS/training_step.json | tr -dc 0-9)), saved config steps=$c_steps decay=$c_decay save=$c_save seed=$c_seed"
  # [2026-09-28] the only CLI override on a resume is the save cadence (50k); steps/decay/LR/seed/data come from the saved config
  TRAIN_ARGS=(--config_path="$CFG" --resume=true --save_freq=$SAVE_FREQ)
else
  TRAIN_ARGS=(
  --dataset.repo_id=rebot/$DS
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

# Disk guard. Runs on the node, so it keeps working when the Mac or the link does not.
# pgrep's [l] keeps it from matching this shell.
(
  while sleep 30; do
    pid=$(pgrep -f "[t]rain_rel16_relonly.py.*$OUT(/| |\$)" | head -1)   # [2026-09-28] also matches a RESUMED run (--config_path=$OUT/checkpoints/...)
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

# XVLA_STRICT_STATE_DIM: refuse to zero-pad a narrow state up to 76. Without it a slim20 dataset would
# train silently as UMI76-with-56-zeros (and B's 94 against a 76 base would be refused, not truncated).
env CUDA_VISIBLE_DEVICES=$GPU XVLA_STRICT_STATE_DIM=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$SRC $PY $ENTRY \
  "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$OUT.log"
