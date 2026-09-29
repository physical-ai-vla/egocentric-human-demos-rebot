#!/bin/bash
# HRL80 part B: SAME frozen v2k_retarget (r30 mode = v1 / v2-K-R30 / v2-Kr-R30, tr mode = v2-TR-R30) on all 60 HRL80 episodes.
cd ~/c8; L=~/c8/robotlike; V=~/umi_bridge/track_c/v2k
A="--episodes-json $L/hrl80_60_episodes.json --raw-root $HOME/ego_collector/datasets/human_handumi_raw/HRL80 --prefix HRL80 --runs $L/runs --export $L/export"
for k in 0 1 2 3; do
  ~/xvla-mac/bin/python $V/v2k_retarget.py --mode r30 $A --out $L/r30_hrl80 --part $k/4 > $L/logs/r30_hrl80_$k.log 2>&1 &
  ~/xvla-mac/bin/python $V/v2k_retarget.py --mode tr $A --out $L/tr_hrl80 --part $k/4 > $L/logs/tr_hrl80_$k.log 2>&1 &
done
wait
echo "retarget done $(date +%H:%M): r30 $(ls $L/r30_hrl80/*.json | wc -l) tr $(ls $L/tr_hrl80/*.json | wc -l)"
grep -lE "Traceback|Error" $L/logs/r30_hrl80_*.log $L/logs/tr_hrl80_*.log && echo "ERRORS in logs"
