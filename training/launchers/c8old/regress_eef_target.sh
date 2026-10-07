#!/bin/bash
# [2026-09-25] C-old EEF_TARGET_SOURCE regression on ONE 4090 GPU: 20-step runs, per-batch C/D logging (EE_LOG_EVERY=1).
#   T0a / T0b : original probe code (humanik_delta 783de894), cmd           -> determinism baseline
#   T1        : C-old code, legacy (default), cmd                             -> must equal T0a (legacy invariance)
#   T2        : C-old code, legacy, HUMANIK_TARGET=state                      -> reference for T3
#   T3        : C-old code, measured + EEF_MEASURED_SELFTEST=1, state         -> must equal T2 (measured equivalence)
B=/home/bh-aiteam/c8old; W=/home/bh-aiteam/workspace/bh_rebot_LeRobot; PY=$W/.venv/bin/python
EP=$($PY -c "import json; print(json.load(open('/home/bh-aiteam/c8probe/split.json'))['train'])" | tr -d ' ')
run() {  # run <name> <code_dir> <target> <extra env...>
  local N=$1 X=$2 T=$3; shift 3; rm -rf $B/reg/$N
  env HF_HOME=/home/bh-aiteam/.cache/huggingface HF_HUB_OFFLINE=1 HUMANIK_DELTA=1 HUMANIK_LEAD=5 HUMANIK_ROBOT=1 HUMANIK_TARGET=$T \
      XVLA_EE_AUX=1 EE_AUX_SOURCE=state EE_AUX_SCALE=10 EE_AUX_LAMBDA=2.0 EE_FK_LAMBDA=20 EE_LOG_EVERY=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XVLA_TF32=1 XVLA_FUSED_ADAM=1 "$@" \
      $PY $X/train_bi.py --dataset.repo_id=rebot/rebot_3stack_R150_headview --dataset.root=/home/bh-aiteam/holobrain-data/lerobot/rebot_3stack_R150_headview \
      --dataset.episodes="$EP" --policy.path=lerobot/xvla-base --policy.dtype=float32 --output_dir=$B/reg/$N --job_name=$N \
      --steps=20 --save_freq=100000 --save_checkpoint=false --log_freq=1 --policy.scheduler_decay_steps=400000 --seed=1000 --batch_size=4 --num_workers=8 \
      --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
      --policy.normalization_mapping='{"STATE":"IDENTITY","ACTION":"IDENTITY","VISUAL":"IDENTITY"}' \
      --rename_map '{"observation.images.global":"observation.images.image","observation.images.left_wrist":"observation.images.image2","observation.images.right_wrist":"observation.images.image3"}' \
      > $B/reg/$N.log 2>&1; echo "$N rc $?"; grep -c "aux_loss" $B/reg/$N.log; }
mkdir -p $B/reg; cd $B
run T0a /home/bh-aiteam/c8probe/xvla cmd
run T0b /home/bh-aiteam/c8probe/xvla cmd
run T1 $B/xvla cmd
run T2 $B/xvla state
run T3 $B/xvla state EEF_TARGET_SOURCE=measured EEF_MEASURED_SELFTEST=1
ext() { grep -oE "batch [0-9]+: FK pos err.*fk_loss [0-9.e+-]+|step:[0-9]+ .*loss:[0-9.e+-]+" $B/reg/$1.log; }
for p in "T0a T0b" "T0a T1" "T2 T3"; do set -- $p; diff <(ext $1) <(ext $2) > /dev/null && echo "IDENTICAL $1 vs $2 ($(ext $1 | wc -l) lines)" || { echo "DIFFERENT $1 vs $2"; diff <(ext $1) <(ext $2) | head -6; }; done
grep -m2 "EEF_TARGET_SOURCE\|installed" $B/reg/T3.log | cut -c1-200
echo REGRESS_DONE
