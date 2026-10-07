#!/bin/bash
# [2026-10-07 user "B 결과 좋으면 CT6로 학습 교체" + "5k 나오면 띄워줘"] wait for the CT6 export -> stop the CT5-86 main run (one GPU only) ->
# smoke + main 300k on the freed 4090 GPU -> when step 5000 is complete load it into the 8056 UI with the CT6 start pose + G anchor.
set -u; A=$HOME/c8/hra_a100; P=$A/pipe; O=$A/lerobot_ct6; NAME=ego_hra_ct6_v1; MRUN=HRA-RIGHTONLY-ROBOTCAM-CT6-LOSSMASK-D20-B8-300K
export RAY_ADDRESS=http://100.64.0.1:8265; log() { echo "[$(date '+%F %T')] $*"; }
while pgrep -f build_CT6.sh >/dev/null; do sleep 20; done
[ -f $O/${NAME}_train/EXPORT.json ] && [ -f $O/${NAME}_val/EXPORT.json ] || { log "CT6 export missing -> stop"; exit 1; }
log "CT6 export: $(python3 -c "import json;e=json.load(open('$O/${NAME}_train/EXPORT.json'));print(e['report']['episodes'],'eps',e['report']['rows'],'rows')")"
ray job stop hra-a100-main-4090-1244 2>&1 | tail -1; log "CT5-86 main run stopped (one GPU rule)"
L4=/home/bh-aiteam/train_hra_a93_lossmask_d20_b8.sh DS=${NAME}_train LD=$O/${NAME}_train SMR=HRA-RIGHTONLY-CT6-D20-SMOKE MRUN=$MRUN bash $P/chain_generic.sh > $P/chain_CT6.log 2>&1
log "CT6: $(grep -E 'SMOKE VERDICT|main run submitted|NOT started|did not pass' $P/chain_CT6.log | cut -c1-200 | tr '\n' ' ')"
grep -q "main run submitted" $P/chain_CT6.log || exit 1
for i in $(seq 1 120); do
  if [ ! -f $A/.ct6_ui_loaded ]; then
    cp -n $HOME/c8/hra_red/umi_start_pose.json $HOME/c8/hra_red/umi_start_pose.json.bak_preCT6_1007
    python3 -c "import json;s=json.load(open('$A/start_pose_CT6.json'));json.dump(dict(t=s['t'],note='CT6 start pose (copied from ~/c8/hra_a100/start_pose_CT6.json); previous in umi_start_pose.json.bak_preCT6_1007. '+s['note'],T_base_tcp=s['T_base_tcp']),open('$HOME/c8/hra_red/umi_start_pose.json','w'),indent=1)"
    if RUN=$MRUN DS_EXPECT=${NAME}_train V4_FIXED_ANCHOR_JSON=$A/G_anchor.json UMI_TCP_PITCH_DEG=0 UMI_TCP_OFFSET_MM=0,0,0 bash $HOME/umi_bridge/load_hra_rightonly_ckpt_to_ui.sh 5000 8056 > $P/ui_CT6_5k.log 2>&1; then
      touch $A/.ct6_ui_loaded; log "CT6 5k loaded on 8056: $(tail -1 $P/ui_CT6_5k.log)"; break; fi
  fi; sleep 120; done
log "run_CT6 DONE"
