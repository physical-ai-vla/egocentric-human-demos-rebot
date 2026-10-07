#!/bin/bash
# [2026-10-04 user "ckpt가 자동으로 로드가 안되는거 같아"] HRA right-only auto-loader for :8056 ONLY (the shared swapper
# auto_ui_d20_e5_v5.sh serves 8052/8053 and is not touched). Every 2 min: newest checkpoint of the HRA run (SSD archive with
# .archived, or a complete one on the 4090) -> load_hra_rightonly_ckpt_to_ui.sh if newer than what 8056 serves.
# Skips while the UI pin exists ($ST/8056.pin, set by the UI dropdown) or an episode is running. After a swap the previous
# local copy is deleted once the SSD archive has it. Log: ~/umi_bridge/auto_ui_hra_8056.log. Run under nohup.
P=8056; RUN=HRA-RIGHTONLY-LOSSMASK-D20-B8-300K
N=bh-aiteam@100.64.0.2; TB=/home/bh-aiteam/holobrain-data/trainB; SSD=/Volumes/PortableSSD/rebot_ckpts_archive/trackb_$RUN
ST=~/umi_bridge/.auto_ui_d20_e5; mkdir -p $ST; cd ~/umi_bridge
log() { echo "[$(date '+%F %T')] $*"; }
loc() { echo ~/holobrain-mac-model/ckpt_UI_${RUN}_$((10#$1 / 1000))k_rightonly; }
while true; do
  if [ ! -f $ST/$P.pin ]; then
    node=$(ssh -o BatchMode=yes -o ConnectTimeout=15 -J head-lp $N "ls $TB/$RUN/checkpoints 2>/dev/null | grep -E '^[0-9]{6}$'" < /dev/null 2>/dev/null)
    ssd=$(ls "$SSD" 2>/dev/null | grep -E '^[0-9]{6}$' | while read s; do [ -f "$SSD/$s/.archived" ] && echo $s; done)
    NEW=$(printf "%s\n%s\n" "$node" "$ssd" | grep -E '^[0-9]{6}$' | sort -u | tail -1)
    CUR=$(cat $ST/$P 2>/dev/null)
    if [ -n "$NEW" ] && [ "$CUR" != "$NEW" ]; then
      if curl -s -m 5 localhost:$P/status | grep -q '"running": *true'; then log ":$P busy (episode running), $NEW waits"
      elif ./load_hra_rightonly_ckpt_to_ui.sh $((10#$NEW)) $P > /tmp/claude_ld_$P.log 2>&1 < /dev/null \
           && curl -s -m 5 localhost:$P/ckpts | grep -q "_$((10#$NEW / 1000))k_rightonly"; then
        echo $NEW > $ST/$P; log ":$P loaded $RUN $NEW sha $(shasum -a 256 $(loc $NEW)/model.safetensors | cut -c1-16)"
        [ -n "$CUR" ] && echo "$CUR" >> $ST/hra_pending_delete
      else log ":$P $NEW not loaded yet: $(tail -1 /tmp/claude_ld_$P.log)"; fi
    fi
  fi
  # old local copies: delete once archived on the SSD and not served / pinned
  if [ -f $ST/hra_pending_delete ]; then
    : > $ST/hra_pending_delete.next
    while read S; do
      if [ "$(cat $ST/$P 2>/dev/null)" = "$S" ] || [ "$(cat $ST/$P.pin 2>/dev/null)" = "$S" ] || [ ! -f "$SSD/$S/.archived" ]; then
        echo "$S" >> $ST/hra_pending_delete.next
      else rm -rf "$(loc $S)"; log "deleted local copy $S (archived)"; fi
    done < $ST/hra_pending_delete
    mv $ST/hra_pending_delete.next $ST/hra_pending_delete
  fi
  sleep 120
done
