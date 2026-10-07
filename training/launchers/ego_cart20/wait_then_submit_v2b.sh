#!/bin/bash
# [2026-10-01 user] run B (ego_cart20_v2b, rule C) with the SAME recipe on the SAME 4090 GPU1 right after run A
# (EGO-CART20V2-RELONLY-D6-100K) finishes. [21:2x user] A is stopped early at 45k; B starts right away.
# KEEP_NODE_STEPS=100000: the final B checkpoint stays on the node as the R312c FT init (ft_r312c_when_b_done.sh). Ray can't hold a job PENDING > 900 s, so this Mac waiter polls the node.
NODE=bh-aiteam@100.64.0.2; log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for run A to exit and GPU1 to be free"
while true; do
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=15 -J head-lp $NODE 'pgrep -f "[t]rain_rel16_relonly.py.*EGO-CART20V2-RELONLY-D6-100K" >/dev/null && echo A_RUNNING || echo A_DONE; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1' 2>/dev/null)
  if echo "$r" | grep -q A_DONE; then u=$(echo "$r" | tail -1 | tr -dc 0-9); [ -n "$u" ] && [ "$u" -lt 1000 ] && break; fi
  sleep 120
done
log "run A done, GPU1 free -> submitting run B"
EGO_NAME=ego_cart20_v2b RUN_NAME=EGO-CART20V2-RELONLY-D6-100K-V2B SKIP_UPLOAD=1 bash ~/ego_cart20/launch/submit_ego_cart20v2.sh 1
log "starting archiver for run B"
cd ~; STATE_EVERY=50000 KEEP_NODE_STEPS="100000" IDLE_EXIT=1000000 bash ~/umi_bridge/trackb_archive_v2_relonly.sh EGO-CART20V2-RELONLY-D6-100K-V2B >> ~/archive_EGO-CART20V2-RELONLY-D6-100K-V2B.log 2>&1
