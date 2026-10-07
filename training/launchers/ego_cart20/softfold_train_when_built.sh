#!/bin/bash
# [2026-10-07] Mac waiter: when the 5090 Soft-Fold build log says BUILD DONE (and LOAD OK), submit the main pretrain
# SOFTFOLD-RELCART20-RELONLY-D21-B8-150K (steps = decay = 150k, save 5k, bf16 + 8-bit AdamW, domain 21) via Ray on the 5090.
# Stops (no submit) if the build log shows FAIL / Traceback.  Log: ~/ego_cart20/launch/softfold_train_when_built.log
export RAY_ADDRESS=http://100.64.0.1:8265; RAY=~/ray-env/bin/ray; H=bh-ai-5090@100.64.0.5
L=/srv/data/johann/softfold/build_softfold_cart20_v1.log
while true; do
  s=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $H "grep -c 'BUILD DONE' $L; grep -c 'VALIDATION FAIL\|Traceback' $L" 2>/dev/null | tr '\n' ' ')
  set -- $s
  [ "${2:-0}" -gt 0 ] && { echo "[$(date '+%F %T')] build FAILED -- not submitting"; exit 1; }
  [ "${1:-0}" -gt 0 ] && break
  sleep 300
done
used=$(ssh -o BatchMode=yes $H "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | head -1)
[ "$used" -gt 1000 ] && { echo "[$(date '+%F %T')] 5090 busy (${used} MiB) -- not submitting"; exit 1; }
ID=softfold-d21-b8-150k-$(date +%m%d%H%M)
$RAY job submit --no-wait --submission-id $ID --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.5": 0.001}' \
  -- bash -c 'ACCELERATE_MIXED_PRECISION=bf16 RELONLY_PREFLIGHT=3 STEPS=150000 SAVE_FREQ=5000 bash /srv/data/johann/relonly/code/train_softfold_5090_b8.sh --log_freq=200'
echo "[$(date '+%F %T')] submitted $ID"
