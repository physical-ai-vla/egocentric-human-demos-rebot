#!/bin/bash
# [2026-10-02 user] prepare the 100 %-robotized ego set (ROBOTIZE_P=1.0) on the 4090 for tomorrow: wait for the train export,
# frame-count check (48411 / 5671), SHA256SUMS, upload as ego_cart20_v2b_robot100_{train,val}, verify. No training is started.
NODE=bh-aiteam@100.64.0.2; J="-J head-lp"; SRC=~/c8/ego_cart20_v2b_robot100_lerobot; DST=/home/bh-aiteam/holobrain-data/lerobot
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q "EXPORT DONE" ~/c8/ego_cart20_v2b_robot100_train_export.log; do sleep 60; done
for s in train val; do
  n=$(python3 -c "import json;print(json.load(open('$SRC/ego_cart20_v2b_robot100_$s/meta/info.json'))['total_frames'])"); log "$s frames $n"
done
for s in train val; do
  d=$SRC/ego_cart20_v2b_robot100_$s; [ -f $d/SHA256SUMS ] || ( cd $d && LC_ALL=C find . -type f ! -name SHA256SUMS | LC_ALL=C sort | xargs shasum -a 256 > SHA256SUMS )
  /opt/homebrew/bin/rsync -a --partial -e "ssh -J head-lp" $d/ $NODE:$DST/ego_cart20_v2b_robot100_$s/ || { log "upload $s failed"; exit 1; }
  ssh $J $NODE "cd $DST/ego_cart20_v2b_robot100_$s && sha256sum -c --quiet SHA256SUMS && chmod -R a-w . && echo 'robot100 $s SHA OK'" || { log "SHA $s failed"; exit 1; }
done
log "robot100 ready on the node"
