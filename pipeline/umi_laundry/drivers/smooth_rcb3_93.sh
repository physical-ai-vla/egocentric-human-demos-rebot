#!/bin/bash
# [2026-10-15] run_smoother --rcb data_v003 into derived/umi_fusion_rcb3 for the 93 training episodes (not excluded), 5 lanes. Old outputs untouched.
L=~/c8/robotlike/logs/smooth_rcb3; cd ~/ego_cart20; R=~/ego_collector/datasets/human_handumi_raw/UMIalign
ls -d $R/UMIalign_20261013_160900/episode_* $R/UMIalign_20261014_112920/episode_* $R/UMIalign_20261015_112743/episode_* | while read e; do
  [ -f $e/.complete ] && ! grep -q '"exclude_from_training": true' $e/episode_meta.json && echo $e; done > $L/eps.txt
one () { local e=$1 n=$(basename $(dirname $1))_$(basename $1); [ -f $e/derived/umi_fusion_rcb3/smooth_qa.json ] && return
  caffeinate -ims ~/ego_collector/.venv/bin/python -u ego_cart20/umi_fusion/run_smoother.py $e --rcb data_v003 --out-name umi_fusion_rcb3 > $L/$n.log 2>&1 && echo "OK $n" || echo "FAIL $n"; }
export -f one; export L
cat $L/eps.txt | xargs -P 5 -I{} bash -c 'one {}' > $L/lanes.log 2>&1; echo DONE $(date +%T) > $L/done.txt
