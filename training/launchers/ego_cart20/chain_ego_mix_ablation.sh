#!/bin/bash
# [2026-10-02 user] MIX ablation chain (replaces chain_ego_mix50.sh): GPU1 raw ego 50k -> MIX50 ego 50k -> MIX70 ego 50k; then ONE
# R312c FT 150k from the better 50k (higher robot-frame k8 cosine, mean of L/R, from rel16_audit/mix50_cosine.csv; within 0.05 the
# higher ego_raw k8 wins), on the first GPU with < 1 GB in use. Same B8 recipe (bs 8, AdamW8bit, AMP bf16, domain 20) everywhere.
NODE=bh-aiteam@100.64.0.2; J="-J head-lp"; export RAY_ADDRESS=http://100.64.0.1:8265; DST=/home/bh-aiteam/holobrain-data/lerobot; TB=/home/bh-aiteam/holobrain-data/trainB
log() { echo "[$(date '+%F %T')] $*"; }
upload() {  # <local export dir> <local name> <node name>
  for s in train val; do
    d=$1/${2}_$s; [ -f $d/SHA256SUMS ] || ( cd $d && LC_ALL=C find . -type f ! -name SHA256SUMS | LC_ALL=C sort | xargs shasum -a 256 > SHA256SUMS )
    /opt/homebrew/bin/rsync -a --partial -e "ssh -J head-lp" $d/ $NODE:$DST/${3}_$s/ || return 1
    ssh $J $NODE "cd $DST/${3}_$s && sha256sum -c --quiet SHA256SUMS && chmod -R a-w . && echo '$3 $s SHA OK'" || return 1
  done
}
running() { ssh -o BatchMode=yes -o ConnectTimeout=15 $J $NODE "pgrep -f '[t]rain_rel16_relonly.py.*trainB/$1( |/|\$)' >/dev/null"; }   # 0 running, 1 not, 255 ssh
gpu_free() { [ "$(ssh -o BatchMode=yes -o ConnectTimeout=15 $J $NODE "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1" | tr -dc 0-9)" -lt 1000 ] 2>/dev/null; }
wait_done() { until running $1; [ $? -eq 1 ] && gpu_free 1; do sleep 120; done; }
pretrain() {  # <run> <node dataset>
  ray job submit --no-wait --submission-id $(echo $1 | tr 'A-Z' 'a-z' | cut -c1-40)-$(date +%H%M) --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' \
    -- bash -c "ACCELERATE_MIXED_PRECISION=bf16 EGO_DS=${2}_train RUN_NAME=$1 STEPS=50000 DECAY_STEPS=50000 SAVE_FREQ=5000 MIN_START_GB=60 RELONLY_PREFLIGHT=3 bash /home/bh-aiteam/train_ego_cart20v2_relonly_d20_b8.sh EGO 1" 2>&1 | grep -E "submitted|rror"
  cd ~; nohup bash -c "STATE_STEPS=050000 KEEP_NODE_STEPS=050000 IDLE_EXIT=1000000 bash ~/umi_bridge/trackb_archive_v3_relonly.sh $1 >> ~/archive_$1.log 2>&1" >/dev/null 2>&1 < /dev/null &
  until running $1; do sleep 60; done; log "$1 running on GPU1"
}
R50=EGO-CART20V2-RELONLY-D20-B8-50K-V2B-MIX50; R70=EGO-CART20V2-RELONLY-D20-B8-50K-V2B-MIX70
until grep -q "EXPORT DONE" ~/c8/ego_cart20_v2b_robotized_export.log; do sleep 60; done
upload ~/c8/ego_cart20_v2b_robotized_lerobot ego_cart20_v2b_robotized ego_cart20_v2b_mix50 || { log "MIX50 upload failed"; exit 1; }; log "MIX50 uploaded"
wait_done EGO-CART20V2-RELONLY-D20-B8-50K-V2B; log "raw ego 50k done, GPU1 free"
pretrain $R50 ego_cart20_v2b_mix50
until grep -q "EXPORT DONE" ~/c8/ego_cart20_v2b_mix70_export.log; do sleep 60; done
upload ~/c8/ego_cart20_v2b_mix70_lerobot ego_cart20_v2b_mix70 ego_cart20_v2b_mix70 || { log "MIX70 upload failed"; exit 1; }; log "MIX70 uploaded"
wait_done $R50; log "MIX50 50k done"
pretrain $R70 ego_cart20_v2b_mix70
wait_done $R70; log "MIX70 50k done"
# pick by the eval watcher's 50k rows (auto UI :8056 / :8058 load the 50k, mix50_eval_watch.sh scores them)
CSV=~/umi_bridge/rel16_audit/mix50_cosine.csv; t0=$(date +%s)
until [ $(grep -c "ckpt_UI_${R50}_50k\|ckpt_UI_${R70}_50k" $CSV 2>/dev/null) -ge 2 ] || [ $(( $(date +%s) - t0 )) -gt 10800 ]; do sleep 120; done
PICK=$(python3 - "$CSV" "$R50" "$R70" <<'PY'
import csv, sys
rows = {r["ckpt"]: r for r in csv.DictReader(open(sys.argv[1]))}
sc = {}
for run in sys.argv[2:]:
    r = rows.get(f"ckpt_UI_{run}_50k_pinklockwy")
    if r: sc[run] = ((float(r["robot_Lk8"]) + float(r["robot_Rk8"])) / 2, (float(r["ego_raw_Lk8"]) + float(r["ego_raw_Rk8"])) / 2)
if not sc: print("NONE"); sys.exit()
a, b = sys.argv[2], sys.argv[3]
if a in sc and b in sc:
    best = a if (sc[a][0] - sc[b][0] > 0.05 or (abs(sc[a][0] - sc[b][0]) <= 0.05 and sc[a][1] >= sc[b][1])) else b
else:
    best = next(iter(sc))
print(best, " ".join(f"{k}:robot={v[0]:.3f},ego_raw={v[1]:.3f}" for k, v in sc.items()))
PY
)
log "pick: $PICK"; RUN=${PICK%% *}; [ "$RUN" = NONE ] && { log "no 50k eval rows -> no FT, decide by hand"; exit 1; }
TAG=${RUN##*-}; FTRUN=FT-R312C-FROM-EGOV2B${TAG}-RELONLY-D20-B8-150K; FTBASE=/home/bh-aiteam/xvla_base_ego_v2b$(echo $TAG | tr 'A-Z' 'a-z')_d20
ssh $J $NODE "set -e; S=$TB/$RUN/checkpoints/050000/pretrained_model; [ -e $FTBASE ] || { cp -al \$S $FTBASE; rm $FTBASE/config.json; python3 -c \"import json; c=json.load(open('\$S/config.json')); json.dump({'type': 'xvla', **c}, open('$FTBASE/config.json', 'w'), indent=4)\"; }; ls $FTBASE | head -3" || { log "FT base failed"; exit 1; }
until gpu_free 1 || gpu_free 0; do sleep 120; done; G=1; gpu_free 1 || G=0
ray job submit --no-wait --submission-id ft-r312c-from-egov2b$(echo $TAG | tr 'A-Z' 'a-z')-d20-b8-150k-$(date +%H%M) --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' \
  -- bash -c "ACCELERATE_MIXED_PRECISION=bf16 EXPECT_DOMAIN=20 FT_BASE=$FTBASE RUN_NAME=$FTRUN STEPS=150000 DECAY_STEPS=150000 SAVE_FREQ=5000 MIN_START_GB=60 RELONLY_PREFLIGHT=3 bash /home/bh-aiteam/train_ft_r312c_from_ego_relonly_d20_b8.sh F $G" 2>&1 | grep -E "submitted|rror"
cd ~; nohup bash -c "IDLE_EXIT=1000000 bash ~/umi_bridge/trackb_archive_v3_relonly.sh $FTRUN >> ~/archive_$FTRUN.log 2>&1" >/dev/null 2>&1 < /dev/null &
log "FT $FTRUN submitted on GPU$G from $RUN 050000"
