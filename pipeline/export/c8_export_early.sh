#!/bin/bash
# [2026-09-23] C8 stage E: re-export the 294 legacy production episodes (5 sessions) x 2 sides with the SAME exporter and
# default flags as Track A's hu_export.sh (umi_export_orbslam.py, groot-infer-env). One job per episode-side, 6 parallel.
set -u
BASE=$HOME/ego_collector; RAW=$BASE/datasets/human_handumi_raw/Hpilot; OUT=$HOME/c8/export_early
job() {
  ep=$1; side=$2; S=$(basename $(dirname $ep)); E=$(basename $ep); tag=${S#Hpilot_}_${E#episode_}_$side; d=$OUT/$tag
  [ -f $d/raw_video.mp4 ] && [ -f $d/imu_data.json ] && { echo "SKIP $tag"; return; }
  mkdir -p $d
  if ( cd $BASE && ~/groot-infer-env/bin/python scripts/umi_export_orbslam.py "$ep" $side "$d" ) > $d/export.log 2>&1; then echo "OK $tag"; else echo "FAIL $tag"; fi
}
export -f job; export BASE OUT
for s in 20260915_161954 20260915_171744 20260916_090424 20260916_105608 20260916_112115 20260916_121003 20260916_121832 20260916_122225 20260916_125236 20260916_142318; do
  for ep in $(ls -d $RAW/Hpilot_$s/episode_* | sort); do echo "$ep left"; echo "$ep right"; done
done | xargs -P 6 -L 1 bash -c 'job $0 $1'
echo EXPORT_DONE
