#!/bin/bash
# [2026-10-14] run_smoother over UMIalign_20261014_112920 ep1-33; waits for each episode's Orbbec canonical (summary json = finished).
S=~/ego_collector/datasets/human_handumi_raw/UMIalign/UMIalign_20261014_112920; cd ~/ego_cart20
for i in $(seq 1 33); do e=$S/episode_$(printf %06d $i)
  until [ -f $e/derived/body_track/canonical_summary.json ]; do sleep 30; done
  [ -f $e/derived/umi_fusion/smooth_qa.json ] && { echo "SKIP ep $i"; continue; }
  caffeinate -ims ~/ego_collector/.venv/bin/python -u ego_cart20/umi_fusion/run_smoother.py $e > ~/c8/robotlike/logs/smooth_112920/ep$i.log 2>&1 && echo "OK ep $i $(date +%T)" || echo "FAIL ep $i $(tail -1 ~/c8/robotlike/logs/smooth_112920/ep$i.log)"
done; echo SMOOTH_DONE
