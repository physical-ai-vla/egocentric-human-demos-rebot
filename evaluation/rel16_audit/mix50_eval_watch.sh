#!/bin/bash
# [2026-10-02 user] run mix50_eval.py on every MIX50 ego checkpoint the auto UI loads (and once on the raw-ego 50k final as reference).
# Waits for the robot100 val export first. Log: rel16_audit/mix50_eval_watch.log, results: rel16_audit/mix50_cosine.csv. nohup.
cd ~/holobrain-mac-model; DONE=~/umi_bridge/rel16_audit/.mix50_eval_done; touch $DONE
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q "EXPORT DONE" ~/c8/ego_cart20_v2b_robot100_export.log 2>/dev/null && grep -q "EXPORT DONE" ~/c8/ego_cart20_v2b_robotized_export.log; do sleep 60; done
while true; do
  for RUN in EGO-CART20V2-RELONLY-D20-B8-50K-V2B-MIX50 EGO-CART20V2-RELONLY-D20-B8-100K-V2B-MIX70 EGO-CART20V2-RELONLY-D20-B8-50K-V2B; do
    for d in $(ls -d ~/holobrain-mac-model/ckpt_UI_${RUN}_*k_pinklockwy 2>/dev/null); do
      tag=$(basename $d); grep -qx "$tag" $DONE && continue
      [ "$RUN" = EGO-CART20V2-RELONLY-D20-B8-50K-V2B ] && [ "${tag#*_50k_}" = "$tag" ] && continue      # raw ego: only the 50k final
      [ -f $d/model.safetensors ] || continue
      log "eval $tag"; ~/xvla-mac/bin/python ~/umi_bridge/rel16_audit/mix50_eval.py $d $tag 2>&1 | grep -E "^(robot|ego_|ROW)|Error|Traceback"
      echo "$tag" >> $DONE
    done
  done
  sleep 300
done
