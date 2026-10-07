#!/bin/bash
# [2026-10-01 user] when run B (EGO-CART20V2-RELONLY-D6-100K-V2B) finishes, fine-tune on R312c (REL-only D6 recipe) from B's
# 100000 checkpoint on the same 4090 GPU1: cosine DECAY 600k, trained to 300k. Recipe 6: steps < decay makes LeRobot auto-scale
# the decay down, so STEPS = DECAY = 600000 and this watcher STOPS the job externally once the 300000 checkpoint is fully saved
# (STOP_AT, default 300000). The FT archiver runs alongside (training_state every 50k, so 300000 keeps its state).
# B's 100000 is kept on the node by its archiver (KEEP_NODE_STEPS=100000) and hard-linked here to a stable base dir.
# Refuses (no FT) if B stops without a 100000 checkpoint. Run under nohup (background tool tasks die after 2 h).
NODE=bh-aiteam@100.64.0.2; J="-J head-lp"; export RAY_ADDRESS=http://100.64.0.1:8265
BRUN=EGO-CART20V2-RELONLY-D6-100K-V2B; FTRUN=FT-R312C-FROM-EGOV2B100K-RELONLY-D6-D600K; STOP_AT=${STOP_AT:-300000}
TB=/home/bh-aiteam/holobrain-data/trainB; FTBASE=/home/bh-aiteam/xvla_base_ego_v2b100k_d6
log() { echo "[$(date '+%F %T')] $*"; }
scp -q $J ~/ego_cart20/launch/train_ft_r312c_from_ego_relonly_d6.sh $NODE:train_ft_r312c_from_ego_relonly_d6.sh && ssh $J $NODE 'chmod +x ~/train_ft_r312c_from_ego_relonly_d6.sh; sha256sum ~/train_ft_r312c_from_ego_relonly_d6.sh'
log "waiting for $BRUN to finish (100000 checkpoint) and GPU1 to be free"
while true; do
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $J $NODE "pgrep -f '[t]rain_rel16_relonly.py.*$BRUN' >/dev/null && echo B_RUNNING || echo B_DONE; [ -s $TB/$BRUN/checkpoints/100000/pretrained_model/model.safetensors ] && echo HAS100K || echo NO100K; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1" 2>/dev/null)
  if echo "$r" | grep -q B_DONE; then
    echo "$r" | grep -q HAS100K || { sleep 60; r2=$(ssh $J $NODE "[ -s $TB/$BRUN/checkpoints/100000/pretrained_model/model.safetensors ] && echo HAS100K"); echo "$r2" | grep -q HAS100K || { log "B stopped WITHOUT a 100000 checkpoint -> no FT"; exit 1; }; }
    u=$(echo "$r" | tail -1 | tr -dc 0-9); [ -n "$u" ] && [ "$u" -lt 1000 ] && break
  fi
  sleep 120
done
log "B done -> FT base"
# a saved checkpoint's config.json lacks the draccus "type" key (base has "type": "xvla") -> --policy.path would fail to parse.
# weights hard-linked; config.json written as a NEW file with "type": "xvla" prepended (B's checkpoint is never modified).
ssh $J $NODE "set -e; S=$TB/$BRUN/checkpoints/100000/pretrained_model; if [ ! -e $FTBASE ]; then cp -al \$S $FTBASE; rm $FTBASE/config.json; python3 -c \"import json; c=json.load(open('\$S/config.json')); json.dump({'type': 'xvla', **c}, open('$FTBASE/config.json', 'w'), indent=4)\"; fi; ls $FTBASE; python3 -c \"import json;print('type', json.load(open('$FTBASE/config.json')).get('type'), 'domain',[s['config']['domain_id'] for s in json.load(open('$FTBASE/policy_preprocessor.json'))['steps'] if s.get('registry_name')=='xvla_add_domain_id'])\"" || { log "FT base copy failed"; exit 1; }
SID=ft-r312c-from-egov2b100k-d600k-$(date +%H%M)
ray job submit --no-wait --submission-id $SID --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' \
  -- bash -c "FT_BASE=$FTBASE RUN_NAME=$FTRUN STEPS=600000 DECAY_STEPS=600000 SAVE_FREQ=10000 MIN_START_GB=60 RELONLY_PREFLIGHT=3 bash /home/bh-aiteam/train_ft_r312c_from_ego_relonly_d6.sh F 1" 2>&1 | grep -E "submitted|rror"
for i in $(seq 1 90); do
  s=$(curl -s -m 60 "$RAY_ADDRESS/api/jobs/$SID/logs" | python3 -c "import json,sys;print(json.load(sys.stdin)['logs'])" 2>/dev/null | tr '\r' '\n')
  echo "$s" | grep -qE "ot_train.py:4[0-9]+ step:|Traceback|refusing|FAILED|gate:" && break
  sleep 20
done
echo "$s" | grep -aE "\[domain\]|physical GPU|steps=|Auto-scal|cfg.steps|relonly-preflight\] batch 3|PREFLIGHT|ot_train.py:4[0-9]+ step:|Traceback|refusing|gate" | head -12 | cut -c1-220
echo "$s" | grep -q "Auto-scal" && log "WARNING: LR auto-scaling message seen -- decay is NOT 600k"
log "starting FT archiver (background)"
cd ~; STATE_EVERY=50000 IDLE_EXIT=1000000 nohup bash ~/umi_bridge/trackb_archive_v2_relonly.sh $FTRUN >> ~/archive_$FTRUN.log 2>&1 < /dev/null &
log "waiting for checkpoint $STOP_AT to be fully saved, then stopping $SID"
CK=$(printf "%06d" $STOP_AT)
while true; do
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $J $NODE "pgrep -f '[t]rain_rel16_relonly.py.*$FTRUN' >/dev/null && echo RUNNING || echo GONE; D=$TB/$FTRUN/checkpoints/$CK; [ -s \$D/pretrained_model/model.safetensors ] && [ -s \$D/training_state/optimizer_state.safetensors ] && [ -s \$D/training_state/training_step.json ] && echo SAVED || echo NOT_YET" 2>/dev/null)
  echo "$r" | grep -q GONE && { log "FT process gone before $STOP_AT (crash or manual stop) -- not stopping anything"; exit 1; }
  echo "$r" | grep -q SAVED && break
  sleep 120
done
sleep 180   # let the save finish flushing (rng / scheduler files) before the stop
ray job stop $SID 2>&1 | tail -1
log "stopped $SID at checkpoint $STOP_AT (decay 600k schedule); the archiver moves $CK with training_state, resume = RESUME=1"
