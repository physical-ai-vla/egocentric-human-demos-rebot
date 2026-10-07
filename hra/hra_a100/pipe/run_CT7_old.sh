#!/bin/bash
# [2026-10-07 user "그럼 2번으로 하자"] CT7-OLD: start 08 (medoid + back/right/down), YESTERDAY (1006) sessions only = the 85 CT7 PASS /
# PASS-CORRECTED episodes (the 1007 low-start session excluded) -> ALL train (val 0) -> LeRobot (measured-C922 wrist) -> 4090 smoke + main 300k, ONE GPU.
set -u; A=$HOME/c8/hra_a100; P=$A/pipe; PY=$HOME/xvla-mac/bin/python; O=$A/lerobot_ct7_old; NAME=ego_hra_ct7_old85_v1
log() { echo "[$(date '+%F %T')] $*"; }
(cd $HOME/ego_cart20 && PYTHONPATH=$HOME/ego_cart20 $PY $P/convert_origin.py $A/raw_v2_CT7_old $A/processed_v2_CT7_old 0 2>&1 | tail -1 | grep -oE "'train': \{[^}]*\}")
cp $A/processed_v2_robotcam/metadata.json $A/processed_v2_CT7_old/metadata.json 2>/dev/null || true
for i in $(seq 1 30); do [ -f $O/${NAME}_train/EXPORT.json ] && break
  rm -rf $O/${NAME}_*.tmp; [ -d $O/${NAME}_train ] && rm -rf $O/${NAME}_train
  (cd $HOME/ego_cart20 && $PY ego_cart20/scripts/export_lerobot_right_only_robotcam.py $A/processed_v2_CT7_old $O --name $NAME --workers 4 --splits train >> $P/export_CT7_old.log 2>&1); done
[ -f $O/${NAME}_train/EXPORT.json ] || { log "export CT7-old failed"; exit 1; }
log "export: $(python3 -c "import json;e=json.load(open('$O/${NAME}_train/EXPORT.json'));print(e['report']['episodes'],'eps',e['report']['rows'],'rows')")"
L4=/home/bh-aiteam/train_hra_a93_lossmask_d20_b8.sh DS=${NAME}_train LD=$O/${NAME}_train SMR=HRA-RIGHTONLY-CT7OLD-D20-SMOKE MRUN=HRA-RIGHTONLY-ROBOTCAM-CT7-OLD85-LOSSMASK-D20-B8-300K bash $P/chain_generic.sh > $P/chain_CT7_old.log 2>&1
log "CT7-old: $(grep -E 'SMOKE VERDICT|main run submitted|NOT started|did not pass|failed' $P/chain_CT7_old.log | cut -c1-200 | tr '\n' ' ')"
