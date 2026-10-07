#!/bin/bash
# [2026-10-07 user "ct5-86이지" (5k 나오면 띄워줘)] load CT5-86 step 5000 into 8056 once complete: G anchor, UMI pitch/offset 0, start = umi_start_pose.json (CT5 start, unchanged)
A=$HOME/c8/hra_a100
for i in $(seq 1 60); do
  RUN=HRA-RIGHTONLY-ROBOTCAM-CT5-86-LOSSMASK-D20-B8-300K DS_EXPECT=ego_hra_ct5_86_v1_train V4_FIXED_ANCHOR_JSON=$A/G_anchor.json UMI_TCP_PITCH_DEG=0 UMI_TCP_OFFSET_MM=0,0,0 \
    bash $HOME/umi_bridge/load_hra_rightonly_ckpt_to_ui.sh 5000 8056 > $A/pipe/ui_CT5_5k.log 2>&1 && { echo "[$(date +%T)] LOADED: $(tail -1 $A/pipe/ui_CT5_5k.log)"; exit 0; }
  tail -1 $A/pipe/ui_CT5_5k.log | grep -q "not complete" || { echo "[$(date +%T)] FAILED: $(tail -3 $A/pipe/ui_CT5_5k.log)"; exit 1; }
  sleep 60; done; echo TIMEOUT
