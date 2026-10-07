#!/bin/bash
# [2026-10-07 user "ct4는 지금 만들고 그걸 올려줘", one GPU] CT4: per-axis start-offset decay (xy 0.2-0.8, z 0-0.3, rot 0.1-0.6) + table-aware safety, 87 episodes (PASS 80 + PASS-CORRECTED 7), B-style state (anchor = F), C922-view wrist
set -u; A=$HOME/c8/hra_a100; P=$A/pipe; PY=$HOME/xvla-mac/bin/python; O=$A/lerobot_ct4; NAME=ego_hra_ct4_87_v1
log() { echo "[$(date '+%F %T')] $*"; }
(cd $HOME/ego_cart20 && PYTHONPATH=$HOME/ego_cart20 $PY $P/convert_origin.py $A/raw_v2_CT4 $A/processed_v2_CT4 2>&1 | tail -1 | grep -oE "'train': \{[^}]*\}, 'val': \{[^}]*\}")
cp $A/processed_v2_robotcam/metadata.json $A/processed_v2_CT4/metadata.json 2>/dev/null || true
for i in $(seq 1 30); do [ -f $O/${NAME}_train/EXPORT.json ] && [ -f $O/${NAME}_val/EXPORT.json ] && break
  sp=""; for x in train val; do [ -f $O/${NAME}_$x/EXPORT.json ] || sp="$sp $x"; done; rm -rf $O/${NAME}_*.tmp
  for x in $sp; do [ -d $O/${NAME}_$x ] && rm -rf $O/${NAME}_$x; done
  (cd $HOME/ego_cart20 && $PY ego_cart20/scripts/export_lerobot_right_only_robotcam.py $A/processed_v2_CT4 $O --name $NAME --workers 4 --splits $sp >> $P/export_CT4.log 2>&1); done
[ -f $O/${NAME}_train/EXPORT.json ] || { log "export CT4 failed"; exit 1; }
log "export: $(python3 -c "import json;e=json.load(open('$O/${NAME}_train/EXPORT.json'));print(e['report']['episodes'],'eps',e['report']['rows'],'rows')")"
L4=/home/bh-aiteam/train_hra_a93_lossmask_d20_b8.sh DS=${NAME}_train LD=$O/${NAME}_train SMR=HRA-RIGHTONLY-CT4-D20-SMOKE MRUN=HRA-RIGHTONLY-ROBOTCAM-CT4-LOSSMASK-D20-B8-300K bash $P/chain_generic.sh > $P/chain_CT4.log 2>&1
log "CT4: $(grep -E 'SMOKE VERDICT|main run submitted|NOT started|did not pass' $P/chain_CT4.log | cut -c1-200 | tr '\n' ' ')"; log "CT4 DONE"
