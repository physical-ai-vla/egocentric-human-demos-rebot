#!/bin/bash
# [2026-10-06] HRA_A100 fisheye export (no --start-at-go: the origin hold must stay in for the 0,0,0 anchoring)
e=$1; s=$(basename $(dirname $e)); tag=${s#HRA_A100_}_$(basename $e | sed 's/episode_//')_right; d=$HOME/c8/hra_a100/export/$tag
[ -f $d/raw_video.mp4 ] && [ -f $d/imu_data.json ] && exit 0
mkdir -p $d; cd $HOME/ego_collector && ~/groot-infer-env/bin/python scripts/umi_export_orbslam.py "$e" right "$d" > $d/export.log 2>&1 && echo "OK $tag" || echo "FAIL $tag"
