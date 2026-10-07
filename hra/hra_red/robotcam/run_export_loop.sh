#!/bin/bash
# [2026-10-06] resumable robotcam export, re-run until both splits have EXPORT.json (workers die on the SVT-AV1 dealloc bug:
# "Semaphore object deallocated while in use" in svt_av1_enc_deinit_handle; completed shards are kept via .complete markers)
O=$HOME/c8/hra_red/lerobot_robotcam
for i in $(seq 1 30); do
  [ -f $O/ego_hra_red_rightonly_robotcam_v2_train/EXPORT.json ] && [ -f $O/ego_hra_red_rightonly_robotcam_v2_val/EXPORT.json ] && { echo "done after $((i-1)) reruns"; exit 0; }
  echo "[$(date '+%T')] attempt $i, complete shards $(find $O -name '*.complete' | wc -l)"
  cd $HOME/ego_cart20 && $HOME/xvla-mac/bin/python ego_cart20/scripts/export_lerobot_right_only_robotcam.py $HOME/c8/hra_red/processed_robotcam $O --workers 4 >> $HOME/c8/hra_red/robotcam/export_loop_py.log 2>&1
done
echo "gave up"; exit 1
