#!/bin/bash
# [2026-09-28] Move 5090 C-old pretrain checkpoints to the Mac PortableSSD (user: save every 100k, move to SSD).
# Usage: c8old5090_archiver.sh <run> <ray_job_id>
# Per finished checkpoint (pretrained_model + training_state, idle > 3 min): rsync -> <dest>/<step>.partial, per-file sha256 on both
# sides (model files AND training_state) must match -> rename to <dest>/<step> + <step>.sha256 + <step>.completed. KEEP_ON_5090 (space
# separated steps, default "300000") are never deleted from the 5090 (final ckpt stays on both until handoff). Then the 5090 copy of OLDER archived steps is deleted; the newest
# checkpoint always stays on the 5090 (resume point / handoff). Nothing is deleted unless its SSD copy is sha-verified.
# SSD free < 40 GB -> stop archiving (no deletion) and report.
RUN=$1; JOB=$2; R=/srv/data/johann/c8old5090/runs/$RUN/checkpoints
D=/Volumes/PortableSSD/rebot_ckpts_archive/5090_c8old_${RUN}; mkdir -p $D; LOG=$D/archiver.log
RS=/opt/homebrew/bin/rsync
log() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
log "archiver start run=$RUN job=$JOB dest=$D"
while :; do
  running=$(ray job status $JOB 2>/dev/null | grep -c "RUNNING")
  for s in $(ssh gpu-5090 "cd $R 2>/dev/null && for d in [0-9][0-9][0-9][0-9][0-9][0-9]; do [ -f \$d/pretrained_model/model.safetensors ] && [ -f \$d/training_state/training_step.json ] && [ -z \"\$(find \$d -newermt '-3 minutes' -type f)\" ] && echo \$d; done"); do
    [ -d $D/$s ] && continue
    free=$(df -g /Volumes/PortableSSD | tail -1 | awk '{print $4}'); [ "$free" -lt 40 ] && { log "SSD free ${free} GB < 40 -> not archiving $s"; continue; }
    log "copy $s"; rm -rf $D/$s.partial
    $RS -a gpu-5090:$R/$s/ $D/$s.partial/ || { log "rsync $s failed, retry next round"; continue; }
    a=$(ssh gpu-5090 "cd $R/$s && find . -type f | LC_ALL=C sort | xargs sha256sum"); b=$(cd $D/$s.partial && find . -type f | LC_ALL=C sort | xargs shasum -a 256)   # C-locale order on both sides (Linux vs macOS sort differ)
    if [ "$(echo "$a" | awk '{print $1,$2}')" = "$(echo "$b" | awk '{print $1,$2}')" ]; then
      mv $D/$s.partial $D/$s; echo "$b" > $D/$s.sha256; date "+%F %T" > $D/$s.completed; log "archived $s ($(du -sh $D/$s | cut -f1), $(echo "$b" | wc -l | tr -d ' ') files sha-verified)"
    else log "sha mismatch $s -> keep 5090 copy, retry next round"; continue; fi
  done
  # delete 5090 copies of archived steps except the newest checkpoint on the 5090
  newest=$(ssh gpu-5090 "cd $R 2>/dev/null && ls -d [0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | sort | tail -1")
  for s in $(ssh gpu-5090 "cd $R 2>/dev/null && ls -d [0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null"); do
    [ "$s" = "$newest" ] && continue; case " ${KEEP_ON_5090:-300000} " in *" $((10#$s)) "*) continue;; esac; [ -d $D/$s ] && [ -f $D/$s.sha256 ] && [ -f $D/$s.completed ] || continue
    ssh gpu-5090 "rm -rf $R/$s" && log "removed 5090 copy of $s (SSD copy verified)"
  done
  if [ "$running" = 0 ]; then
    pend=$(ssh gpu-5090 "cd $R 2>/dev/null && ls -d [0-9][0-9][0-9][0-9][0-9][0-9]" | while read s; do [ -d $D/$s ] || echo $s; done)
    [ -z "$pend" ] && { log "job not running and all checkpoints archived -> exit"; exit 0; }
  fi
  sleep 600
done
