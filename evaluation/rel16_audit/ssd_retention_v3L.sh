#!/bin/bash
# [2026-09-29] v3L copy (every 5k) of ssd_retention_head180v3.sh (user: every 10k to the SSD, node cleaned): weights every 10k, state newest + 600k
# [2026-09-28, CHANGED same day by user: keep ONLY 50k multiples on the SSD] SSD checkpoint retention for HEAD180-REL16V3-D600K, FROZEN before any result existed (user). The training save
# cadence stays 5k; this only thins the SSD archive after a checkpoint is fully archived (<step>/.archived).
#   keep weights: ONLY multiples of 50k (50k..600k, 12 ckpts) + anything listed in keep_extra. (Was: 5/10/20/40k, 10k/25k grid,
#                 2 rolling recent -- replaced 2026-09-28 by the user.) The node always keeps the newest ckpt with its state.
#   keep training_state: the newest archived one + 600k
# usage: ssd_retention_r312c.sh [--dry-run]   (loops every 30 min until the run's 600k is archived; log beside this file)
DEST=/Volumes/PortableSSD/rebot_ckpts_archive/trackb_HEAD180-REL16V3L-D600K
EXTRA=~/umi_bridge/rel16_audit/v3L_keep_extra.txt; LOG=~/umi_bridge/rel16_audit/ssd_retention_v3L.log
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
touch $EXTRA
keep() {   # $1 = step (int). [2026-09-28 user] only multiples of 50k (+ keep_extra); was 5/10/20/40k + 10k/25k grid
  local s=$1
  [ $((s % 5000)) -eq 0 ] && return 0   # [2026-09-29 user] keep every 5k
  grep -qx "$(printf %06d $s)" $EXTRA && return 0
  return 1
}
while true; do
  [ -d "$DEST" ] || { sleep 1800; continue; }
  steps=$(ls "$DEST" | grep -E '^[0-9]{6}$' | while read d; do [ -f "$DEST/$d/.archived" ] && echo $d; done | sort)
  recent=$(echo "$steps" | tail -2)
  recent_state=$(echo "$steps" | while read d; do [ -d "$DEST/$d/training_state" ] && echo $d; done | tail -2)
  for d in $steps; do
    s=$((10#$d))
    if ! keep $s; then                     # [2026-09-28 user] no rolling-recent exception: the node keeps the resume point
      echo "$(date '+%F %T') drop weights $d$([ $DRY = 1 ] && echo ' (dry-run)')" >> $LOG
      [ $DRY = 0 ] && rm -rf "$DEST/$d"
      continue
    fi
    if [ -d "$DEST/$d/training_state" ] && [ $s -ne 600000 ] && [ "$d" != "$(echo "$recent_state" | tail -1)" ]; then
      echo "$(date '+%F %T') drop training_state $d$([ $DRY = 1 ] && echo ' (dry-run)')" >> $LOG
      [ $DRY = 0 ] && rm -rf "$DEST/$d/training_state"
    fi
  done
  [ $DRY = 1 ] && exit 0
  [ -f "$DEST/600000/.archived" ] && { echo "$(date '+%F %T') 600k archived -- retention done" >> $LOG; exit 0; }
  sleep 1800
done
