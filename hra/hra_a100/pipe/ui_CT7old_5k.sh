#!/bin/bash
# [2026-10-07 user "5k 올려줄래?"] load CT7-old (08 start, 85 eps, 100k) step 5000 into 8056 once complete AND the UI is idle (never restart mid-run):
# G anchor, UMI pitch/offset 0; zoom / cube-stop keep their per-port values.
A=$HOME/c8/hra_a100
idle() { curl -s -m 5 localhost:8056/status | python3 -c "import json,sys;d=json.load(sys.stdin);sys.exit(0 if not d['ui']['running'] and not d['ui']['busy'] else 1)"; }
for i in $(seq 1 120); do
  ssh -o BatchMode=yes -J head-lp bh-aiteam@100.64.0.2 "grep -aq 'Checkpoint policy after step 5000' /home/bh-aiteam/holobrain-data/trainB/HRA-RIGHTONLY-ROBOTCAM-CT7-OLD85-LOSSMASK-D20-B8-100K.log 2>/dev/null" < /dev/null || { sleep 60; continue; }
  idle || { sleep 20; continue; }
  RUN=HRA-RIGHTONLY-ROBOTCAM-CT7-OLD85-LOSSMASK-D20-B8-100K DS_EXPECT=ego_hra_ct7_old85_v1_train V4_FIXED_ANCHOR_JSON=$A/G_anchor.json UMI_TCP_PITCH_DEG=0 UMI_TCP_OFFSET_MM=0,0,0 \
    bash $HOME/umi_bridge/load_hra_rightonly_ckpt_to_ui.sh 5000 8056 > $A/pipe/ui_CT7old_5k.log 2>&1 && { echo "[$(date +%T)] LOADED: $(tail -1 $A/pipe/ui_CT7old_5k.log)"; exit 0; }
  grep -q "not complete" $A/pipe/ui_CT7old_5k.log || { echo "[$(date +%T)] FAILED: $(tail -3 $A/pipe/ui_CT7old_5k.log)"; exit 1; }
  sleep 60; done; echo TIMEOUT
