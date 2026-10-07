#!/bin/bash
# [2026-10-07 user "1번하자"] CT7: start 08 (medoid + back/right/down), 134 HRA_A100 eps -> CT gates (IK-fail / endpoint dropped) -> ALL train
# (val 0) -> LeRobot (measured-C922 wrist) -> 4090 smoke + main 300k on ONE GPU.
set -u; A=$HOME/c8/hra_a100; P=$A/pipe; PY=$HOME/xvla-mac/bin/python; O=$A/lerobot_ct7; NAME=ego_hra_ct7_all_v1
log() { echo "[$(date '+%F %T')] $*"; }
(cd $A && $PY $P/table_aware_C7.py > $P/table_aware_C7.log 2>&1); tail -2 $P/table_aware_C7.log
(cd $HOME/ego_cart20 && PYTHONPATH=$HOME/ego_cart20 $PY $P/convert_origin.py $A/raw_v2_CT7 $A/processed_v2_CT7 0 2>&1 | tail -1 | grep -oE "'train': \{[^}]*\}")
cp $A/processed_v2_robotcam/metadata.json $A/processed_v2_CT7/metadata.json 2>/dev/null || true
for i in $(seq 1 30); do [ -f $O/${NAME}_train/EXPORT.json ] && break
  rm -rf $O/${NAME}_*.tmp; [ -d $O/${NAME}_train ] && rm -rf $O/${NAME}_train
  (cd $HOME/ego_cart20 && $PY ego_cart20/scripts/export_lerobot_right_only_robotcam.py $A/processed_v2_CT7 $O --name $NAME --workers 4 --splits train >> $P/export_CT7.log 2>&1); done
[ -f $O/${NAME}_train/EXPORT.json ] || { log "export CT7 failed"; exit 1; }
log "export: $(python3 -c "import json;e=json.load(open('$O/${NAME}_train/EXPORT.json'));print(e['report']['episodes'],'eps',e['report']['rows'],'rows')")"
L4=/home/bh-aiteam/train_hra_a93_lossmask_d20_b8.sh DS=${NAME}_train LD=$O/${NAME}_train SMR=HRA-RIGHTONLY-CT7-D20-SMOKE MRUN=HRA-RIGHTONLY-ROBOTCAM-CT7-ALL-LOSSMASK-D20-B8-300K bash $P/chain_generic.sh > $P/chain_CT7.log 2>&1
log "CT7: $(grep -E 'SMOKE VERDICT|main run submitted|NOT started|did not pass|failed' $P/chain_CT7.log | cut -c1-200 | tr '\n' ' ')"
