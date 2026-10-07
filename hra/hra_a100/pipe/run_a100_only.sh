#!/bin/bash
# [2026-10-07 user "믹스? 왜 합쳤지? ... 올드 못써"] NEW data ONLY: processed_v2_robotcam (106 HRA_A100 episodes: origin-anchored protocol,
# cube-PnP or origin-plane metric scale, rows from the AUTO go, robot-TCP labels) -> LeRobot (measured-C922 wrist) -> 4090 smoke + main.
set -u
A=$HOME/c8/hra_a100; P=$A/pipe; PY=$HOME/xvla-mac/bin/python; O=$A/lerobot_a100; NAME=ego_hra_a100_robotcam_v1
log() { echo "[$(date '+%F %T')] $*"; }
for i in $(seq 1 30); do [ -f $O/${NAME}_train/EXPORT.json ] && [ -f $O/${NAME}_val/EXPORT.json ] && break
  (cd $HOME/ego_cart20 && $PY ego_cart20/scripts/export_lerobot_right_only_robotcam.py $A/processed_v2_robotcam $O --name $NAME --workers 4 >> $P/export_a100_only.log 2>&1); done
[ -f $O/${NAME}_train/EXPORT.json ] || { log "LeRobot export failed"; exit 1; }
log "export: $(python3 -c "import json;e=json.load(open('$O/${NAME}_train/EXPORT.json'));print(e['report']['episodes'],'eps',e['report']['rows'],'rows')")"
DS=${NAME}_train LD=$O/${NAME}_train SMR=HRA-RIGHTONLY-A100-D20-SMOKE MRUN=HRA-RIGHTONLY-ROBOTCAM-A100-LOSSMASK-D20-B8-300K bash $P/chain_generic.sh > $P/chain_a100_only.log 2>&1
tail -3 $P/chain_a100_only.log; log "DONE"
