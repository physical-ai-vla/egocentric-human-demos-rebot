#!/bin/bash
# [2026-10-04 user] after the ROBOT100 ego pretrain 300k finishes: base dir from its 300000 ckpt -> 60-step smoke -> FT-R30 600k (5090)
export RAY_ADDRESS=http://100.64.0.1:8265; H5=bh-ai-5090@100.64.0.5; R=/srv/data/johann/relonly; L=$R/code/train_ft_r30_5090_b8.sh
PT=ego-robot100-pt300k-5090-10041616; PRUN=EGO-CART20V2-RELONLY-D20-B8-300K-V2B-ROBOT100; B=$R/base/xvla_base_egorobot100_pt300k_d20
log() { echo "[$(date '+%F %T')] $*"; }
while true; do st=$(curl -s -m 20 http://100.64.0.1:8265/api/jobs/$PT | python3 -c "import json,sys;print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
  case "$st" in SUCCEEDED) break ;; FAILED|STOPPED) log "pretrain $st -- FT not started"; exit 1 ;; esac; sleep 300; done
log "pretrain SUCCEEDED"
ssh -o BatchMode=yes $H5 "set -e; S=$R/runs/$PRUN/checkpoints/300000/pretrained_model; test -s \$S/model.safetensors; mkdir -p $B
  for f in \$S/*; do n=\$(basename \$f); [ \$n = config.json ] || ln -f \$f $B/\$n; done
  $R/venv/bin/python -c \"import json;c=json.load(open('\$S/config.json'));json.dump({'type':'xvla',**{k:v for k,v in c.items() if k!='type'}},open('$B/config.json','w'),indent=2)\"
  sha256sum $B/model.safetensors | cut -c1-16 > $B/BASE_SHA16; grep -o '\"domain_id\": [0-9]*' $B/policy_preprocessor.json; cat $B/BASE_SHA16
  echo 'ROBOT100 ego pretrain 300k (EGO-CART20V2-RELONLY-D20-B8-300K-V2B-ROBOT100 300000), hard links + typed config.json' > $B/BASE.md"
SM=ft-r30-smoke-5090-$(date +%m%d%H%M)
ray job submit --no-wait --submission-id $SM --entrypoint-resources '{"node:100.64.0.5": 0.001}' -- bash -c "CUDA_VISIBLE_DEVICES=0 ACCELERATE_MIXED_PRECISION=bf16 RUN_NAME=FT-R30-SMOKE STEPS=60 DECAY_STEPS=60 SAVE_FREQ=50 RELONLY_PREFLIGHT=2 bash $L --log_freq=10" 2>&1 | grep -E "submitted|rror"
for i in $(seq 1 120); do st=$(ray job status $SM 2>&1 | grep -oiE "succeeded|failed|stopped" | tr a-z A-Z | head -1); [ -n "$st" ] && break; sleep 10; done
ray job logs $SM 2>&1 | tr '\r' '\n' | grep -aE "\[r30\]|\[domain\]|8bit\]|num_frames|num_episodes|cfg.steps|step:|Checkpoint|End of training|Traceback|Error|refusing" | tail -12 | cut -c1-200
[ "$st" = SUCCEEDED ] || { log "smoke $st -- main NOT started"; exit 1; }
ssh -o BatchMode=yes $H5 "rm -rf $R/runs/FT-R30-SMOKE $R/runs/FT-R30-SMOKE.*"
MAIN=ft-r30-from-egorobot100pt300k-600k-5090-$(date +%m%d%H%M)
ray job submit --no-wait --submission-id $MAIN --entrypoint-resources '{"node:100.64.0.5": 0.001}' -- bash -c "CUDA_VISIBLE_DEVICES=0 ACCELERATE_MIXED_PRECISION=bf16 STEPS=600000 DECAY_STEPS=600000 SAVE_FREQ=5000 RELONLY_PREFLIGHT=3 bash $L" 2>&1 | grep -E "submitted|rror"
for i in $(seq 1 60); do s=$(ray job logs $MAIN 2>&1 | tr '\r' '\n'); echo "$s" | grep -qE "ot_train.py:4[0-9]+ step:|Traceback|refusing|FAILED" && break; sleep 20; done
echo "$s" | grep -aE "\[r30\]|num_frames|num_episodes|cfg.steps|step:|Traceback" | tail -5 | cut -c1-200
log "main FT submitted: $MAIN"
