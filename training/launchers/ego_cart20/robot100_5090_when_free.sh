#!/bin/bash
# [2026-10-02 user] "5090에 학습 하나 돌고 있는데 이거 끝나면 믹스70 말고 100으로 프리트레인 100케이": wait until the 5090 GPU is free
# (< 1 GB used on 3 checks in a row), run a 60-step smoke (bnb AdamW8bit on sm_120 + checkpoint save + memory), then the main (rev: user "프리트레인도 300케이로" -> 300k)
# EGO-CART20V2-RELONLY-D20-B8-300K-V2B-ROBOT100 (steps = decay = 300k, save 5k) via Ray on node 100.64.0.5. nohup.
export RAY_ADDRESS=http://100.64.0.1:8265; H5=bh-ai-5090@100.64.0.5; R=/srv/data/johann/relonly; L=$R/code/train_ego_robot100_5090_b8.sh
log() { echo "[$(date '+%F %T')] $*"; }
used() { ssh -o BatchMode=yes -o ConnectTimeout=20 $H5 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | tr -dc 0-9; }
sub() {  # <submission id> <env+cmd>
  ray job submit --no-wait --submission-id $1 --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.5": 0.001}' -- bash -c "$2" 2>&1 | grep -E "submitted|rror"
}
log "waiting for the 5090 GPU to be free"; ok=0
while [ $ok -lt 3 ]; do u=$(used); if [ -n "$u" ] && [ "$u" -lt 1000 ]; then ok=$((ok+1)); else ok=0; fi; sleep 120; done
log "5090 free -> smoke"
SM=ego-robot100-5090-smoke-$(date +%H%M)
sub $SM "ACCELERATE_MIXED_PRECISION=bf16 RUN_NAME=EGO-CART20V2-ROBOT100-5090-SMOKE STEPS=60 DECAY_STEPS=60 SAVE_FREQ=50 MIN_START_GB=200 RELONLY_PREFLIGHT=2 bash $L --log_freq=10"
peak=0; for i in $(seq 1 90); do u=$(used); [ -n "$u" ] && [ "$u" -gt "$peak" ] && peak=$u; st=$(ray job status $SM 2>&1 | grep -oiE "succeeded|failed|stopped" | tr a-z A-Z | head -1); [ -n "$st" ] && break; sleep 10; done
s=$(ray job logs $SM 2>&1 | tr '\r' '\n'); echo "$s" | grep -aE "\[domain\]|\[data\]|8bit\]|cfg.steps|step:|Checkpoint|End of training|Traceback|Error" | tail -12 | cut -c1-170
log "smoke status $st, peak GPU ${peak} MiB"
ckok=$(ssh $H5 "ls $R/runs/EGO-CART20V2-ROBOT100-5090-SMOKE/checkpoints/000050/training_state/optimizer_state.safetensors >/dev/null 2>&1 && echo yes")
[ "$st" = SUCCEEDED ] && [ "$ckok" = yes ] && echo "$s" | grep -q "End of training" || { log "SMOKE FAILED -> main run NOT started (check by hand)"; exit 1; }
ssh $H5 "rm -rf $R/runs/EGO-CART20V2-ROBOT100-5090-SMOKE"
MAIN=ego-robot100-d20-b8-300k-5090-$(date +%H%M)
sub $MAIN "ACCELERATE_MIXED_PRECISION=bf16 RUN_NAME=EGO-CART20V2-RELONLY-D20-B8-300K-V2B-ROBOT100 STEPS=300000 DECAY_STEPS=300000 SAVE_FREQ=5000 MIN_START_GB=200 RELONLY_PREFLIGHT=3 bash $L"
for i in $(seq 1 60); do s=$(ray job logs $MAIN 2>&1 | tr '\r' '\n'); echo "$s" | grep -qE "ot_train.py:4[0-9]+ step:|Traceback|refusing|FAILED|gate" && break; sleep 20; done
echo "$s" | grep -aE "\[domain\]|\[data\]|\[precision\]|8bit\] bits|cfg.steps|Auto-scal|step:|refusing|gate|Traceback" | head -8 | cut -c1-170
log "main run submitted: $MAIN"
