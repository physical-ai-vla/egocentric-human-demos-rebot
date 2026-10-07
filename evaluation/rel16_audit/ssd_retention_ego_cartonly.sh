#!/bin/bash
# [2026-09-30 user] SSD checkpoint retention for EGO-RELCART20-REL16V4-CARTONLY-PRETRAIN, FROZEN before any result exists.
# Save cadence stays 5k; this only thins the SSD archive after a checkpoint is fully archived (<step>/.archived).
#   keep weights: 5k 10k 20k 30k 50k 75k 100k (PRIMARY) 150k 200k (SECONDARY / max) + keep_extra
#   keep training_state: 100k, 200k + the newest archived one (the node always keeps the newest ckpt with its state)
# usage: ssd_retention_ego_cartonly.sh [run name] [--dry-run]   (loops every 30 min until 200k is archived)
RUN=${1:-EGO-RELCART20-REL16V4-CARTONLY-PRETRAIN}
DEST=/Volumes/PortableSSD/rebot_ckpts_archive/trackb_$RUN
EXTRA=~/umi_bridge/rel16_audit/ego_cartonly_keep_extra.txt; LOG=~/umi_bridge/rel16_audit/ssd_retention_ego_cartonly.log
DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1
touch $EXTRA
KEEP="5000 10000 20000 30000 50000 75000 100000 150000 200000"; KEEP_STATE="100000 200000"
inlist () { for x in $2; do [ "$x" = "$1" ] && return 0; done; return 1; }
while true; do
  [ -d "$DEST" ] || { [ $DRY = 1 ] && { echo "no $DEST yet"; exit 0; }; sleep 1800; continue; }
  steps=$(ls "$DEST" | grep -E '^[0-9]{6}$' | while read d; do [ -f "$DEST/$d/.archived" ] && echo $d; done | sort)
  newest_state=$(echo "$steps" | while read d; do [ -d "$DEST/$d/training_state" ] && echo $d; done | tail -1)
  for d in $steps; do
    s=$((10#$d))
    if ! inlist $s "$KEEP" && ! grep -qx "$d" $EXTRA; then
      echo "$(date '+%F %T') drop weights $d$([ $DRY = 1 ] && echo ' (dry-run)')" >> $LOG
      [ $DRY = 0 ] && rm -rf "$DEST/$d"
      continue
    fi
    if [ -d "$DEST/$d/training_state" ] && ! inlist $s "$KEEP_STATE" && [ "$d" != "$newest_state" ]; then
      echo "$(date '+%F %T') drop training_state $d$([ $DRY = 1 ] && echo ' (dry-run)')" >> $LOG
      [ $DRY = 0 ] && rm -rf "$DEST/$d/training_state"
    fi
  done
  [ $DRY = 1 ] && exit 0
  [ -f "$DEST/200000/.archived" ] && { echo "$(date '+%F %T') 200k archived -- retention done" >> $LOG; exit 0; }
  sleep 1800
done
