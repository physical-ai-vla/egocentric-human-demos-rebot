#!/bin/bash
# [2026-10-15] Tag + export (robotlike_session_check_4090.sh steps 1-2) for UMIalign_20261015_112743
L=$HOME/c8/robotlike; EXP=$L/export; BASE=$HOME/ego_collector; NAME=UMIalign_20261015_112743; SESS=$BASE/datasets/human_handumi_raw/UMIalign/$NAME; TS=${NAME#*_}; LIST=$L/lists/$NAME.txt; : > $LIST
for ep in "$SESS"/episode_*; do [ -f "$ep/.complete" ] || continue
  grep -q '"exclude_from_training": true' "$ep/episode_meta.json" 2>/dev/null && continue
  for side in left right; do echo "${TS}_$(basename $ep | sed 's/^episode_//')_$side" >> $LIST; done; done
echo "$NAME sides $(wc -l < $LIST)"
job () { tag=$1; side=${tag##*_}; e=${tag%_*}; e=${e##*_}; d=$EXP/$tag
  [ -f $d/raw_video.mp4 ] && [ -f $d/imu_data.json ] && return 0; mkdir -p $d
  if ( cd $BASE && ~/groot-infer-env/bin/python scripts/umi_export_orbslam.py "$SESS/episode_$e" $side "$d" ) > $d/export.log 2>&1; then echo "  export OK $tag"; else echo "  export FAIL $tag"; fi; }
export -f job; export BASE EXP SESS; xargs -P 4 -n 1 bash -c 'job "$0"' < $LIST
m=0; for t in $(cat $LIST); do [ -f $EXP/$t/imu_data.json ] && [ -f $EXP/$t/raw_video.mp4 ] && [ -f $EXP/$t/orbslam_setting.yaml ] || { echo "MISSING $t"; m=$((m+1)); }; done; echo "EXPORT_DONE missing $m"
