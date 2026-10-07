#!/bin/bash
# [2026-10-01 user] waits for build_chain_r384.sh (CHAIN_R384_DONE) on the 4090, then submits the R384 REL-only domain-6 run
# through Ray on physical GPU0 (same settings as R312C-RELCART20-RELONLY-D6-S600K) and confirms the first logged step.
NODE=bh-aiteam@100.64.0.2; export RAY_ADDRESS=http://100.64.0.1:8265
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for CHAIN_R384_DONE"
while true; do
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=15 -J head-lp $NODE 'tail -3 ~/umi_bridge/chain_r384.log; pgrep -f "[b]uild_chain_r384.sh" >/dev/null && echo RUNNING || echo NOT_RUNNING' 2>/dev/null)
  echo "$r" | grep -q CHAIN_R384_DONE && break
  if echo "$r" | grep -q NOT_RUNNING; then log "chain stopped without CHAIN_R384_DONE:"; echo "$r"; exit 1; fi
  sleep 60
done
log "chain done -> submitting"
J=r384-relonly-d6-s600k-$(date +%H%M)
ray job submit --no-wait --submission-id $J --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' \
  -- bash -c 'STEPS=600000 DECAY_STEPS=600000 SAVE_FREQ=10000 MIN_START_GB=60 RELONLY_PREFLIGHT=3 bash /home/bh-aiteam/train_relcart20_r384_relonly_d6.sh G 0' 2>&1 | grep -E "submitted|rror"
for i in $(seq 1 60); do
  s=$(curl -s -m 60 "$RAY_ADDRESS/api/jobs/$J/logs" | python3 -c "import json,sys;print(json.load(sys.stdin)['logs'])" 2>/dev/null | tr '\r' '\n')
  echo "$s" | grep -qE "ot_train.py:4[0-9]+ step:|Traceback|refusing|FAILED|gate" && break
  sleep 20
done
echo "$s" | grep -aE "\[domain\]|physical GPU|steps=|Auto-scal|cfg.steps|relonly-preflight\] batch 3|ot_train.py:4[0-9]+ step:|Traceback|refusing|gate" | head -8 | cut -c1-220
log "starting archiver"
cd ~; IDLE_EXIT=1000000 bash ~/umi_bridge/trackb_archive_v2_relonly.sh R384-RELCART20-RELONLY-D6-S600K >> ~/archive_R384-RELCART20-RELONLY-D6-S600K.log 2>&1
