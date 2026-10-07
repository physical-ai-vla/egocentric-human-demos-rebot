#!/bin/bash
# [2026-10-07 user option 3] NEW data only, same 93 origin-protocol episodes (82 train / 11 val), two state conventions:
#   A  start-anchored RELCART20 (anchor = pose at AUTO go)       -> HRA-RIGHTONLY-ROBOTCAM-A93START-LOSSMASK-D20-B8-300K
#   B  origin-anchored (pose in the fingertip-on-X frame F, anchor = I) -> HRA-RIGHTONLY-ROBOTCAM-A93ORIGIN-LOSSMASK-D20-B8-300K
set -u
A=$HOME/c8/hra_a100; P=$A/pipe; PY=$HOME/xvla-mac/bin/python
log() { echo "[$(date '+%F %T')] $*"; }
exp() {  # <processed> <out> <name>
  for i in $(seq 1 30); do [ -f $2/${3}_train/EXPORT.json ] && [ -f $2/${3}_val/EXPORT.json ] && return 0
    sp=""; for x in train val; do [ -f $2/${3}_$x/EXPORT.json ] || sp="$sp $x"; done; rm -rf $2/${3}_*.tmp
    for x in $sp; do [ -d $2/${3}_$x ] && rm -rf $2/${3}_$x; done      # a split without EXPORT.json is half-written: redo it from the shards
    (cd $HOME/ego_cart20 && $PY ego_cart20/scripts/export_lerobot_right_only_robotcam.py $1 $2 --name $3 --workers 4 --splits $sp >> $P/export_ab.log 2>&1); done
  [ -f $2/${3}_train/EXPORT.json ]
}
exp $A/processed_v2_a93 $A/lerobot_a93 ego_hra_a93_start_v1 || { log "export A failed"; exit 1; }
exp $A/processed_v2_origin $A/lerobot_a93o ego_hra_a93_origin_v1 || { log "export B failed"; exit 1; }
for x in lerobot_a93/ego_hra_a93_start_v1_train lerobot_a93o/ego_hra_a93_origin_v1_train; do log "$x: $(python3 -c "import json;e=json.load(open('$A/$x/EXPORT.json'));print(e['report']['episodes'],'eps',e['report']['rows'],'rows')")"; done
DS=ego_hra_a93_start_v1_train LD=$A/lerobot_a93/ego_hra_a93_start_v1_train SMR=HRA-RIGHTONLY-A93START-D20-SMOKE MRUN=HRA-RIGHTONLY-ROBOTCAM-A93START-LOSSMASK-D20-B8-300K bash $P/chain_generic.sh > $P/chain_a93_start.log 2>&1 &
until grep -qE "main run submitted|did not pass|NOT started|failed" $P/chain_a93_start.log; do sleep 30; done
log "A: $(grep -E 'SMOKE VERDICT|main run submitted' $P/chain_a93_start.log | cut -c1-160)"
DS=ego_hra_a93_origin_v1_train LD=$A/lerobot_a93o/ego_hra_a93_origin_v1_train SMR=HRA-RIGHTONLY-A93ORIGIN-D20-SMOKE MRUN=HRA-RIGHTONLY-ROBOTCAM-A93ORIGIN-LOSSMASK-D20-B8-300K bash $P/chain_generic.sh > $P/chain_a93_origin.log 2>&1
log "B: $(grep -E 'SMOKE VERDICT|main run submitted' $P/chain_a93_origin.log | cut -c1-160)"
log "AB DONE"
