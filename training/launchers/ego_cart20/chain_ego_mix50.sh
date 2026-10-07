#!/bin/bash
# [2026-10-02 user, rev GPU1 + MIX50 naming] robotized-wrist ego chain on GPU1 (replaces the raw-ego FT B): export done -> SHA256SUMS -> upload to the 4090 -> wait for GPU0 free (robot-only A
# R312C-RELCART20-RELONLY-D20-B8-150K finished) -> ego 50k (same B8 recipe, dataset ego_cart20_v2b_robotized_train) -> archiver
# (KEEP 050000) -> FT watcher (R312c 150k from its 050000, GPU0). Run under nohup.
NODE=bh-aiteam@100.64.0.2; J="-J head-lp"; export RAY_ADDRESS=http://100.64.0.1:8265
SRC=~/c8/ego_cart20_v2b_robotized_lerobot; NAME=ego_cart20_v2b_robotized; NODE_NAME=ego_cart20_v2b_mix50; DST=/home/bh-aiteam/holobrain-data/lerobot
RUN=EGO-CART20V2-RELONLY-D20-B8-50K-V2B-MIX50; ARUN=EGO-CART20V2-RELONLY-D20-B8-50K-V2B
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q "EXPORT DONE" ~/c8/ego_cart20_v2b_robotized_export.log; do sleep 60; done; log "export done"
for s in train val; do
  d=$SRC/${NAME}_$s; [ -f $d/SHA256SUMS ] || ( cd $d && LC_ALL=C find . -type f ! -name SHA256SUMS | LC_ALL=C sort | xargs shasum -a 256 > SHA256SUMS )
  /opt/homebrew/bin/rsync -a --partial -e "ssh -J head-lp" $d/ $NODE:$DST/${NODE_NAME}_$s/ || { log "upload $s failed"; exit 1; }
  ssh $J $NODE "cd $DST/${NODE_NAME}_$s && sha256sum -c --quiet SHA256SUMS && chmod -R a-w . && echo '$s SHA OK'" || { log "SHA check $s failed"; exit 1; }
done
log "uploaded; waiting for $ARUN (raw ego 50k) to finish and GPU1 to be free"
until ssh -o BatchMode=yes -o ConnectTimeout=15 $J $NODE "! pgrep -f '[t]rain_rel16_relonly.py.*trainB/$ARUN( |/|\$)' >/dev/null; [ \$? -eq 1 ] && [ \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) -lt 1000 ]"; do sleep 120; done
log "GPU1 free -> submitting $RUN"
ray job submit --no-wait --submission-id ego-mix50-d20-b8-50k-$(date +%H%M) --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' -- bash -c "ACCELERATE_MIXED_PRECISION=bf16 EGO_DS=${NODE_NAME}_train RUN_NAME=$RUN STEPS=50000 DECAY_STEPS=50000 SAVE_FREQ=5000 MIN_START_GB=60 RELONLY_PREFLIGHT=3 bash /home/bh-aiteam/train_ego_cart20v2_relonly_d20_b8.sh EGO 1" 2>&1 | grep -E "submitted|rror"
cd ~; nohup bash -c "STATE_STEPS=050000 KEEP_NODE_STEPS=050000 IDLE_EXIT=1000000 bash ~/umi_bridge/trackb_archive_v3_relonly.sh $RUN >> ~/archive_$RUN.log 2>&1" >/dev/null 2>&1 < /dev/null &
nohup bash ~/ego_cart20/launch/ft_r312c_when_ego_mix50_d20_done.sh >> ~/ego_cart20/launch/ft_r312c_d20_mix50.log 2>&1 < /dev/null &
log "pretrain submitted, archiver + FT watcher started"
