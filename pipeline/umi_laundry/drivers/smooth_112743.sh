#!/bin/bash
# [2026-10-15] run_smoother over UMIalign_20261015_112743 (50 kept eps), 4 lanes; skips eps with smooth_qa.json. Honours trim.json.
S=~/ego_collector/datasets/human_handumi_raw/UMIalign/UMIalign_20261015_112743; L=~/c8/robotlike/logs/smooth_112743; cd ~/ego_cart20
one () { local e=$S/episode_$(printf %06d $1); [ -f $e/.complete ] || return
  [ -f $e/derived/umi_fusion/smooth_qa.json ] && { echo "SKIP ep $1"; return; }
  caffeinate -ims ~/ego_collector/.venv/bin/python -u ego_cart20/umi_fusion/run_smoother.py $e > $L/ep$1.log 2>&1 && echo "OK ep $1 $(date +%T)" || echo "FAIL ep $1 $(tail -1 $L/ep$1.log)"; }
lane () { for i in "$@"; do one $i; done; }
for k in 1 2 3 4; do lane $(seq $k 4 53) > $L/lane$k.log 2>&1 & done; wait; echo SMOOTH_DONE $(date +%T) > $L/done.txt
