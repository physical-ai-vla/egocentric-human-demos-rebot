#!/bin/bash
# [2026-10-07 user "val 없이 다 트레인 / 1번"] CT6 (start_pose_CT6.json) with ALL 86 episodes in train (val_frac 0); dataset only, no training submit
set -u; A=$HOME/c8/hra_a100; P=$A/pipe; PY=$HOME/xvla-mac/bin/python; O=$A/lerobot_ct6; NAME=ego_hra_ct6_86all_v1
log() { echo "[$(date '+%F %T')] $*"; }
(cd $HOME/ego_cart20 && PYTHONPATH=$HOME/ego_cart20 $PY $P/convert_origin.py $A/raw_v2_CT6 $A/processed_v2_CT6_all 0 2>&1 | tail -1)
cp $A/processed_v2_robotcam/metadata.json $A/processed_v2_CT6_all/metadata.json 2>/dev/null || true
for i in $(seq 1 30); do [ -f $O/${NAME}_train/EXPORT.json ] && break
  rm -rf $O/${NAME}_*.tmp; [ -d $O/${NAME}_train ] && rm -rf $O/${NAME}_train
  (cd $HOME/ego_cart20 && $PY ego_cart20/scripts/export_lerobot_right_only_robotcam.py $A/processed_v2_CT6_all $O --name $NAME --workers 4 --splits train >> $P/export_CT6_all.log 2>&1); done
[ -f $O/${NAME}_train/EXPORT.json ] || { log "export failed"; exit 1; }
log "export: $(python3 -c "import json;e=json.load(open('$O/${NAME}_train/EXPORT.json'));print(e['report']['episodes'],'eps',e['report']['rows'],'rows')")"
