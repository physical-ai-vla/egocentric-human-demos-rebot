#!/bin/bash
# [2026-10-01] Mac: copy ego_cart20_v2_{train,val} + launcher to the 4090, SHA256SUMS, then ray job submit (node 100.64.0.2, physical GPU arg),
# then the REL-only archiver.  Recipe: umi_bridge/RELCART20_RELONLY_D6_RECIPE.md.  Schedule (user 10-01): STEPS = DECAY = 100000, save 10k.
# usage: submit_ego_cart20v2.sh <physical gpu>
#   EGO_NAME=ego_cart20_v2 (default) | ego_cart20_v2b ; RUN_NAME=EGO-CART20V2-...; UPLOAD_ONLY=1 (copy + verify, no submit); SKIP_UPLOAD=1
set -euo pipefail
GPU=${1:?physical gpu}; NODE=bh-aiteam@100.64.0.2; J="-J head-lp"; export RAY_ADDRESS=http://100.64.0.1:8265
NAME=${EGO_NAME:-ego_cart20_v2}; SRC=$HOME/c8/${NAME}_lerobot; DST=/home/bh-aiteam/holobrain-data/lerobot; RUN=${RUN_NAME:-EGO-CART20V2-RELONLY-D6-100K}
[ "${SKIP_UPLOAD:-0}" = 1 ] || for s in train val; do
  d=$SRC/${NAME}_$s; [ -f $d/EXPORT.json ] || { echo "missing $d/EXPORT.json"; exit 1; }
  [ -f $d/SHA256SUMS ] || ( cd $d && LC_ALL=C find . -type f ! -name SHA256SUMS | LC_ALL=C sort | xargs shasum -a 256 > SHA256SUMS )
  rsync -a -e "ssh $J" $d/ $NODE:$DST/${NAME}_$s/
  ssh $J $NODE "cd $DST/${NAME}_$s && sha256sum -c --quiet SHA256SUMS && chmod -R a-w . && echo '$s SHA OK'"
done
[ "${UPLOAD_ONLY:-0}" = 1 ] && { echo "upload only: done"; exit 0; }
scp -q $J $HOME/ego_cart20/launch/train_ego_cart20v2_relonly_d6.sh $NODE:train_ego_cart20v2_relonly_d6.sh; ssh $J $NODE 'chmod +x ~/train_ego_cart20v2_relonly_d6.sh; sha256sum ~/train_ego_cart20v2_relonly_d6.sh'
U=$(ssh $J $NODE "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -dc 0-9; echo; df --output=avail -BG /home | tail -1")
echo "GPU$GPU used / free: $U"
SID=$(echo "$RUN" | tr 'A-Z' 'a-z')-$(date +%H%M)
ray job submit --no-wait --submission-id $SID --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' \
  -- bash -c "EGO_DS=${NAME}_train RUN_NAME=$RUN STEPS=100000 DECAY_STEPS=100000 SAVE_FREQ=10000 MIN_START_GB=60 RELONLY_PREFLIGHT=3 bash /home/bh-aiteam/train_ego_cart20v2_relonly_d6.sh EGO $GPU" 2>&1 | grep -E "submitted|rror"
echo "$SID" > $HOME/ego_cart20/launch/LAST_SUBMISSION
for i in $(seq 1 90); do
  s=$(curl -s -m 60 "$RAY_ADDRESS/api/jobs/$SID/logs" | python3 -c "import json,sys;print(json.load(sys.stdin)['logs'])" 2>/dev/null | tr '\r' '\n')
  echo "$s" | grep -qE "ot_train.py:4[0-9]+ step:|Traceback|refusing|FAILED|gate:" && break
  sleep 20
done
echo "$s" | grep -aE "\[domain\]|\[data\]|physical GPU|steps=|Auto-scal|cfg.steps|relonly-preflight|PREFLIGHT|ot_train.py:4[0-9]+ step:|Traceback|refusing|gate" | head -20 | cut -c1-240
