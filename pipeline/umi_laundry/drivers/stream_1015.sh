#!/bin/bash
# [2026-10-15] stream finished exports of UMIalign_20261015_112743 to 5090 NFS c8/robotlike/in/<tag>/ (3 files MASt3R reads), 4 parallel
L=$HOME/c8/robotlike; LIST=$L/lists/UMIalign_20261015_112743.txt; NFS=/srv/data/johann/c8/robotlike/in; RS=/opt/homebrew/bin/rsync
up () { t=$1; d=$HOME/c8/robotlike/export/$t; [ -f $d/raw_video.mp4 ] && [ -f $d/imu_data.json ] && [ -f $d/orbslam_setting.yaml ] || return 1
  [ -f $d/.up1015 ] && return 0
  $RS -a $d/raw_video.mp4 $d/imu_data.json $d/orbslam_setting.yaml bh-ai-5090@gpu-5090:$NFS/$t/ --rsync-path="mkdir -p $NFS/$t && rsync" && touch $d/.up1015 && echo "UP $t"; }
export -f up; export NFS RS
while :; do
  sleep 20; [ -f $LIST ] || continue
  cat $LIST | xargs -P 4 -n 1 bash -c 'up "$0"' 2>/dev/null
  n=0; for t in $(cat $LIST); do [ -f $L/export/$t/.up1015 ] && n=$((n+1)); done; echo "$(date +%T) uploaded $n / $(wc -l < $LIST)"
  [ $n -ge $(wc -l < $LIST) ] && grep -q EXPORT_DONE $L/logs/export_1015.log && break
done; echo STREAM_DONE
