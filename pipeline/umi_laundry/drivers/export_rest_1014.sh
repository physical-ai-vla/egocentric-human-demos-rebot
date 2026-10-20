#!/bin/bash
# [2026-10-14] Tag + export (same exporter/flags as robotlike_session_check_4090.sh steps 1-2) for UMIalign 144936 / 153819,
# then build the pending MASt3R list over 131556 / 144936 / 153819 (sides without a finished runs_restart ss1).
L=$HOME/c8/robotlike; EXP=$L/export; BASE=$HOME/ego_collector; R=$BASE/datasets/human_handumi_raw/UMIalign
for NAME in UMIalign_20261013_144936 UMIalign_20261013_153819; do SESS=$R/$NAME; TS=${NAME#*_}; LIST=$L/lists/$NAME.txt; : > $LIST
  for ep in "$SESS"/episode_*; do [ -f "$ep/.complete" ] || continue
    grep -q '"exclude_from_training": true' "$ep/episode_meta.json" 2>/dev/null && continue
    for side in left right; do echo "${TS}_$(basename $ep | sed 's/^episode_//')_$side" >> $LIST; done; done
  echo "$NAME sides $(wc -l < $LIST)"
  job () { tag=$1; side=${tag##*_}; e=${tag%_*}; e=${e##*_}; d=$EXP/$tag
    [ -f $d/raw_video.mp4 ] && [ -f $d/imu_data.json ] && return 0; mkdir -p $d
    if ( cd $BASE && ~/groot-infer-env/bin/python scripts/umi_export_orbslam.py "$SESS/episode_$e" $side "$d" ) > $d/export.log 2>&1; then echo "  export OK $tag"; else echo "  export FAIL $tag"; fi; }
  export -f job; export BASE EXP SESS; xargs -P 4 -n 1 bash -c 'job "$0"' < $LIST
done
P=$L/lists/UMIalign_rest_1014.pending.txt; : > $P
for NAME in UMIalign_20261013_131556 UMIalign_20261013_144936 UMIalign_20261013_153819; do while read t; do
  d=$L/runs_restart/$t; [ -f $d/ss1.csv ] && [ -f $d/ss1.json ] && [ ! -f $d/ss1.progress.json ] && continue
  [ -f $EXP/$t/imu_data.json ] && [ -f $EXP/$t/raw_video.mp4 ] && [ -f $EXP/$t/orbslam_setting.yaml ] && echo $t >> $P || echo "NO_EXPORT $t"; done < $L/lists/$NAME.txt; done
echo "PENDING $(wc -l < $P)"
