#!/bin/bash
# [2026-09-25] Track C-old overnight chain on ONE 4090 GPU (Ray assigns it; GPU0 runs Track B). Contract §22b / §24.
# Stages (a FAIL stops everything after it; status in $B/stage_status.json):
#   smoke_real  : 20 steps on the real C-old dataset, EEF_TARGET_SOURCE=measured -> finite losses
#   pretrain    : ego C-old pretrain 300k (xvla-base init, B1 recipe, decay 300k, save 10k), measured C/D
#   integrity   : 300k checkpoint loadable + finite; weights copied to ft_init (kept, never archived away)
#   r150_ft     : R150 (150 ep) B1 recipe 600k from the pretrained WEIGHTS ONLY (fresh optimizer / scheduler), decay 400k
#                 (= the real-robot-verified B1 schedule; 250k is the primary comparison ckpt), EEF_TARGET_SOURCE=legacy
# Guards: NaN/Inf in the training log -> kill + FAIL; OOM / nonzero rc -> FAIL. The Mac supervisor archives checkpoints,
# evaluates them, enforces the disk guard and writes SUMMARY.md.
B=/home/bh-aiteam/c8old; W=/home/bh-aiteam/workspace/bh_rebot_LeRobot; PY=$W/.venv/bin/python; X=$B/xvla; S=$B/stage_status.json
R150=/home/bh-aiteam/holobrain-data/lerobot/rebot_3stack_R150_headview; DS=$B/data/c8old_train
mkdir -p $B/runs
st() { $PY - "$@" <<'P'
import json, sys, time, os
p = "/home/bh-aiteam/c8old/stage_status.json"; d = json.load(open(p)) if os.path.exists(p) else {}
for kv in sys.argv[1:]:
    k, v = kv.split("=", 1); d[k] = v
d["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S"); json.dump(d, open(p, "w"), indent=1)
P
}
common_env() {
  export HF_HOME=/home/bh-aiteam/.cache/huggingface HF_HUB_OFFLINE=1 HUMANIK_DELTA=1 HUMANIK_LEAD=5 HUMANIK_ROBOT=1 HUMANIK_TARGET=cmd
  export XVLA_EE_AUX=1 EE_AUX_SOURCE=state EE_AUX_SCALE=10 EE_AUX_LAMBDA=2.0 EE_FK_LAMBDA=20 EE_LOG_EVERY=1000
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XVLA_TF32=1 XVLA_FUSED_ADAM=1
}
train() {  # train <run> <dataset_repo> <dataset_root> <policy_path> <steps> <decay> <save> [extra args]
  local N=$1 REPO=$2 ROOT=$3 POL=$4 STEPS=$5 DEC=$6 SAVE=$7; shift 7
  $PY $X/train_bi.py --dataset.repo_id=$REPO --dataset.root=$ROOT --policy.path=$POL --policy.dtype=float32 \
    --output_dir=$B/runs/$N --job_name=$N --steps=$STEPS --save_freq=$SAVE --log_freq=${LOGF:-100} --policy.scheduler_decay_steps=$DEC --seed=1000 \
    --batch_size=4 --num_workers=8 --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false \
    --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
    --policy.normalization_mapping='{"STATE":"IDENTITY","ACTION":"IDENTITY","VISUAL":"IDENTITY"}' \
    --rename_map '{"observation.images.global":"observation.images.image","observation.images.left_wrist":"observation.images.image2","observation.images.right_wrist":"observation.images.image3"}' \
    "$@" > $B/runs/$N.log 2>&1 &
  local P=$!
  while kill -0 $P 2>/dev/null; do
    if tail -c 20000 $B/runs/$N.log | grep -E "step:[0-9]+ " | tail -3 | grep -qiE "loss:(nan|inf)|grdn:(nan|inf)"; then
      echo "NAN/INF in $N -> kill"; kill $P; sleep 5; kill -9 $P 2>/dev/null; st "$N=FAIL" "fail_reason=nan_inf_in_training_log"; return 3; fi
    sleep 60
  done
  wait $P; local rc=$?
  grep -qi "out of memory" $B/runs/$N.log && { st "$N=FAIL" "fail_reason=cuda_oom"; return 4; }
  [ $rc -eq 0 ] || { st "$N=FAIL" "fail_reason=train_rc_$rc"; return 5; }
  return 0; }
echo "GPU $CUDA_VISIBLE_DEVICES: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader -i $CUDA_VISIBLE_DEVICES)"; df -h /home/bh-aiteam | tail -1
st chain=RUNNING
# ---- smoke_real
common_env; export EEF_TARGET_SOURCE=measured EE_LOG_EVERY=1   # [2026-09-26] smoke logs EVERY step / batch (the 1000/100 defaults printed nothing in 20 steps -> false FAIL)
rm -rf $B/runs/smoke_real; st smoke_real=RUNNING
LOGF=1 train smoke_real local/c8old_train $DS lerobot/xvla-base 20 300000 100000 --save_checkpoint=false || { st chain=FAIL "failed_stage=smoke_real"; exit 1; }
L=$B/runs/smoke_real.log
nl=$(grep -aoE "aux_loss [0-9.e+-]+ fk_loss [0-9.e+-]+" $L | wc -l); ns=$(grep -aoE "step:[0-9]+ .*loss:[0-9.e+-]+ " $L | wc -l)
grep -m1 "sampler:" $L; grep -aoE "step:[0-9]+ .*loss:[0-9.e+-]+ grdn:[0-9.e+-]+" $L | tail -2; grep -aoE "aux_loss [0-9.e+-]+ fk_loss [0-9.e+-]+" $L | tail -2
echo "smoke: $ns step-loss lines, $nl aux/fk lines"
if [ "$nl" -lt 20 ] || [ "$ns" -lt 20 ] || grep -aqiE "loss:(nan|inf)|grdn:(nan|inf)|aux_loss (nan|inf)|fk_loss (nan|inf)" $L || ! grep -aq "EEF_TARGET_SOURCE=measured" $L; then
  st smoke_real=FAIL chain=FAIL "failed_stage=smoke_real" "fail_reason=measured path not finite / not active (step lines $ns, aux/fk lines $nl)"; exit 1; fi
unset EE_LOG_EVERY; common_env
st smoke_real=PASS
# ---- pretrain 300k
[ -f $B/runs/pretrain300k.done ] || {
  rm -rf $B/runs/pretrain300k; st pretrain=RUNNING
  train pretrain300k local/c8old_train $DS lerobot/xvla-base 300000 300000 10000 || { st chain=FAIL "failed_stage=pretrain"; exit 2; }
  touch $B/runs/pretrain300k.done; }
st pretrain=PASS pretrain_final_step=300000
# ---- integrity + weights-only handoff
CK=$B/runs/pretrain300k/checkpoints/300000/pretrained_model; FI=$B/runs/ft_init_pretrain300k
$PY - <<P || { st integrity=FAIL chain=FAIL "failed_stage=integrity" "fail_reason=ckpt_load_or_nonfinite"; exit 6; }
from safetensors import safe_open; import torch
n = bad = 0
with safe_open("$CK/model.safetensors", "pt") as f:
    for k in f.keys():
        t = f.get_tensor(k); n += 1; bad += int(not torch.isfinite(t.float()).all())
print("tensors", n, "nonfinite", bad); raise SystemExit(0 if (n > 0 and bad == 0) else 1)
P
rm -rf $FI && cp -r $CK $FI && H1=$(md5sum < $FI/model.safetensors | cut -c1-32) && H0=$(md5sum < $CK/model.safetensors | cut -c1-32)
[ "$H0" = "$H1" ] || { st integrity=FAIL chain=FAIL "failed_stage=integrity" "fail_reason=copy_md5_mismatch"; exit 6; }
st integrity=PASS ft_init_copied=true "pretrain300k_model_md5=$H1"
# ---- R150 fine-tune 600k (weights only: --policy.path = the copied pretrained_model -> new optimizer / scheduler)
common_env; unset EEF_TARGET_SOURCE; export EEF_TARGET_SOURCE=legacy
[ -f $B/runs/r150ft600k.done ] || {
  rm -rf $B/runs/r150ft600k; st r150_ft=RUNNING "r150_ft_parent_md5=$H1"
  train r150ft600k rebot/rebot_3stack_R150_headview $R150 $FI 600000 400000 10000 || { st chain=FAIL "failed_stage=r150_ft"; exit 7; }
  touch $B/runs/r150ft600k.done; }
st r150_ft=PASS r150_final_step=600000 chain=DONE
echo C8OLD_CHAIN_DONE
