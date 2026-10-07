#!/bin/bash
# [2026-09-30 user] "4는 펜딩": Ray fails a job left PENDING > 900 s (JobSupervisor start timeout), so the wait happens HERE on the
# Mac. Every 60 s: when 4090 physical GPU0 has < 1 GB in use (tobey's job finished) and the node has >= 70 G free, submit the v4
# resume through Ray (same command as before) and confirm a training step. Exits after one submit.
NODE=bh-aiteam@100.64.0.2
export RAY_ADDRESS=http://100.64.0.1:8265
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for 4090 GPU0 < 1000 MiB and >= 70G free"
while true; do
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $NODE 'echo $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -dc 0-9) $(df --output=avail -BG / | tail -1 | tr -dc 0-9)' 2>/dev/null)
  used=${r% *}; free=${r#* }
  if [ -n "$used" ] && [ "$used" -lt 1000 ] && [ "$free" -ge 70 ]; then
    sleep 60   # make sure GPU0 is really released (not a restart gap of the other job)
    used2=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $NODE 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -dc 0-9' 2>/dev/null)
    [ -n "$used2" ] && [ "$used2" -lt 1000 ] || { log "GPU0 busy again ($used2 MiB) -- keep waiting"; continue; }
    J=r312c-relcart20-rel16v4-resume65k-$(date +%H%M)
    log "GPU0 free ($used2 MiB), ${free}G free -> submitting $J"
    ray job submit --no-wait --submission-id $J --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' \
      -- bash -c 'RESUME=1 MIN_START_GB=60 bash /home/bh-aiteam/train_relcart20_v4.sh F 0' 2>&1 | grep -E "submitted|rror"
    for i in $(seq 1 60); do
      s=$(curl -s -m 60 "$RAY_ADDRESS/api/jobs/$J/logs" | python3 -c "import json,sys;print(json.load(sys.stdin)['logs'])" 2>/dev/null | tr '\r' '\n' | grep -E "RESUME from|ot_train.py:4[0-9]+ step:|Traceback|refusing" | head -2)
      echo "$s" | grep -qE "ot_train.py:4[0-9]+ step:|Traceback|refusing" && { log "$s"; break; }
      sleep 20
    done
    exit 0
  fi
  sleep 60
done
