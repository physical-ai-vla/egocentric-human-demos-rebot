#!/bin/bash
# [2026-10-15] run_smoother --rcb data_v003 --rc-prior tips_v001 --sigma-rc 0.0001 -> derived/umi_fusion_rcb3fix for the 93 training eps, 5 lanes.
L=~/c8/robotlike/logs/smooth_rcb3fix; cd ~/ego_cart20; cp ~/c8/robotlike/logs/smooth_rcb3/eps.txt $L/eps.txt
one () { local e=$1 n=$(basename $(dirname $1))_$(basename $1); [ -f $e/derived/umi_fusion_rcb3fix/smooth_qa.json ] && { echo "SKIP $n"; return; }
  caffeinate -ims ~/ego_collector/.venv/bin/python -u ego_cart20/umi_fusion/run_smoother.py $e --rcb data_v003 --rc-prior tips_v001 --sigma-rc 0.0001 --out-name umi_fusion_rcb3fix > $L/$n.log 2>&1 && echo "OK $n" || echo "FAIL $n"; }
lane () { while read e; do one $e; done; }
for k in 0 1 2 3 4; do awk -v k=$k "NR % 5 == k" $L/eps.txt > $L/part_$k; done
for p in $L/part_*; do lane < $p >> $L/lanes.log 2>&1 & done; wait; echo DONE $(date +%T) > $L/done.txt
