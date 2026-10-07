#!/bin/bash
# [2026-10-07 user "B 결과가 좋으면 새 start pose로 CT5 재생성"] CT6 = CT5 table-aware + export with start_pose_CT6.json (6 cm back, 4 cm down); dataset only, no training submit
set -u; A=$HOME/c8/hra_a100; P=$A/pipe; PY=$HOME/xvla-mac/bin/python; O=$A/lerobot_ct6; NAME=ego_hra_ct6_v1
log() { echo "[$(date '+%F %T')] $*"; }
(cd $HOME/ego_cart20 && PYTHONPATH=$HOME/ego_cart20 $PY $P/convert_origin.py $A/raw_v2_CT6 $A/processed_v2_CT6 2>&1 | tail -1 | grep -oE "'train': \{[^}]*\}, 'val': \{[^}]*\}")
cp $A/processed_v2_robotcam/metadata.json $A/processed_v2_CT6/metadata.json 2>/dev/null || true
for i in $(seq 1 30); do [ -f $O/${NAME}_train/EXPORT.json ] && [ -f $O/${NAME}_val/EXPORT.json ] && break
  sp=""; for x in train val; do [ -f $O/${NAME}_$x/EXPORT.json ] || sp="$sp $x"; done; rm -rf $O/${NAME}_*.tmp
  for x in $sp; do [ -d $O/${NAME}_$x ] && rm -rf $O/${NAME}_$x; done
  (cd $HOME/ego_cart20 && $PY ego_cart20/scripts/export_lerobot_right_only_robotcam.py $A/processed_v2_CT6 $O --name $NAME --workers 4 --splits $sp >> $P/export_CT6.log 2>&1); done
[ -f $O/${NAME}_train/EXPORT.json ] || { log "export CT6 failed"; exit 1; }
log "export: $(python3 -c "import json;e=json.load(open('$O/${NAME}_train/EXPORT.json'));print(e['report']['episodes'],'eps',e['report']['rows'],'rows')")"
