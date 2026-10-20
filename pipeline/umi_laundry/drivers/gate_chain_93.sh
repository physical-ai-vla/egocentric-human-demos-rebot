#!/bin/bash
# [2026-10-15] 93 training eps: AprilTag anchor gate -> smoother (rcb3 + tips r_c fixed + tag gate) -> Orbbec tag/tips acceptance check.
L=~/c8/robotlike/logs/gate_chain; E=~/c8/robotlike/logs/smooth_rcb3fix/eps.txt; cd ~/ego_collector
.venv/bin/python scripts/umi_fusion/orbbec_tag_gate.py $(cat $E) -j 6 > $L/tag_gate.log 2>&1; echo "GATE_DONE $(date +%T)" >> $L/chain.log
cd ~/ego_cart20
one () { local e=$1; [ -f $e/derived/umi_fusion_rcb3gate/smooth_qa.json ] && return
  ~/ego_collector/.venv/bin/python ego_cart20/umi_fusion/run_smoother.py $e --rcb data_v003 --rc-prior tips_v001 --sigma-rc 0.0001 --tag-gate --out-name umi_fusion_rcb3gate > /dev/null 2>&1 && echo "OK $e" || echo "FAIL $e"; }
for k in 0 1 2 3 4; do awk -v k=$k 'NR % 5 == k' $E | while read e; do one $e; done >> $L/smooth.log 2>&1 & done; wait; echo "SMOOTH_DONE $(date +%T)" >> $L/chain.log
cd ~/ego_collector
for s in 20261013_160900 20261014_112920 20261015_112743; do
  .venv/bin/python scripts/umi_fusion/orbbec_tag_check.py $(grep $s $E) --out ~/c8/robotlike/rcb3_check/gate93g_$s --old umi_fusion_rcb3fix --new umi_fusion_rcb3gate --step 4 > $L/check_$s.log 2>&1 &
done; wait; echo "CHAIN_DONE $(date +%T)" >> $L/chain.log
