#!/bin/bash
# [2026-10-02 user] when the domain-20 ego pretrain EGO-CART20V2-RELONLY-D20-50K-V2B finishes (050000), fine-tune on R312c
# (REL-only recipe) from that checkpoint on 4090 GPU1: STEPS = DECAY = 150000 (user), run to completion (no external stop).
# Uses the unchanged FT launcher train_ft_r312c_from_ego_relonly_d6.sh with EXPECT_DOMAIN=20 (its domain gate reads the
# FT base processor, which carries domain_id 20 from the pretrain). The 050000 checkpoint is kept on the node by the ego archiver
# (KEEP_NODE_STEPS=050000) and hard-linked to a stable base dir with a NEW config.json carrying "type": "xvla".
# Refuses (no FT) if the pretrain stops without a complete 050000. Run under nohup.
NODE=bh-aiteam@100.64.0.2; J="-J head-lp"; export RAY_ADDRESS=http://100.64.0.1:8265
BRUN=EGO-CART20V2-RELONLY-D20-50K-V2B; CK=050000; FTRUN=FT-R312C-FROM-EGOV2B50K-RELONLY-D20-S150K
TB=/home/bh-aiteam/holobrain-data/trainB; FTBASE=/home/bh-aiteam/xvla_base_ego_v2b50k_d20
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for $BRUN to finish ($CK checkpoint) and GPU1 to be free"
while true; do
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $J $NODE "pgrep -f '[t]rain_rel16_relonly.py.*$BRUN' >/dev/null && echo B_RUNNING || echo B_DONE; [ -s $TB/$BRUN/checkpoints/$CK/pretrained_model/model.safetensors ] && echo HASCK || echo NOCK; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1" 2>/dev/null)
  if echo "$r" | grep -q B_DONE; then
    echo "$r" | grep -q HASCK || { sleep 60; r2=$(ssh $J $NODE "[ -s $TB/$BRUN/checkpoints/$CK/pretrained_model/model.safetensors ] && echo HASCK"); echo "$r2" | grep -q HASCK || { log "pretrain stopped WITHOUT $CK -> no FT"; exit 1; }; }
    u=$(echo "$r" | tail -1 | tr -dc 0-9); [ -n "$u" ] && [ "$u" -lt 1000 ] && break
  fi
  sleep 120
done
log "pretrain done -> FT base"
ssh $J $NODE "set -e; S=$TB/$BRUN/checkpoints/$CK/pretrained_model; if [ ! -e $FTBASE ]; then cp -al \$S $FTBASE; rm $FTBASE/config.json; python3 -c \"import json; c=json.load(open('\$S/config.json')); json.dump({'type': 'xvla', **c}, open('$FTBASE/config.json', 'w'), indent=4)\"; fi; ls $FTBASE; python3 -c \"import json;print('type', json.load(open('$FTBASE/config.json')).get('type'), 'domain',[s['config']['domain_id'] for s in json.load(open('$FTBASE/policy_preprocessor.json'))['steps'] if s.get('registry_name')=='xvla_add_domain_id'])\"" || { log "FT base copy failed"; exit 1; }
SID=ft-r312c-from-egov2b50k-d20-s150k-$(date +%H%M)
ray job submit --no-wait --submission-id $SID --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' \
  -- bash -c "EXPECT_DOMAIN=20 FT_BASE=$FTBASE RUN_NAME=$FTRUN STEPS=150000 DECAY_STEPS=150000 SAVE_FREQ=10000 MIN_START_GB=60 RELONLY_PREFLIGHT=3 bash /home/bh-aiteam/train_ft_r312c_from_ego_relonly_d6.sh F 1" 2>&1 | grep -E "submitted|rror"
for i in $(seq 1 90); do
  s=$(curl -s -m 60 "$RAY_ADDRESS/api/jobs/$SID/logs" | python3 -c "import json,sys;print(json.load(sys.stdin)['logs'])" 2>/dev/null | tr '\r' '\n')
  echo "$s" | grep -qE "ot_train.py:4[0-9]+ step:|Traceback|refusing|FAILED|gate:" && break
  sleep 20
done
echo "$s" | grep -aE "\[domain\]|physical GPU|cfg.steps|Auto-scal|relonly-preflight\] batch 3|ot_train.py:4[0-9]+ step:|Traceback|refusing|gate" | head -12 | cut -c1-220
echo "$s" | grep -q "Auto-scal" && log "WARNING: LR auto-scaling message seen"
log "starting FT archiver"
cd ~; IDLE_EXIT=1000000 nohup bash ~/umi_bridge/trackb_archive_v2_relonly.sh $FTRUN >> ~/archive_$FTRUN.log 2>&1 < /dev/null &
log "done"
