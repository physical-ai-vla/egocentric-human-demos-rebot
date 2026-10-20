#!/bin/bash
# [2026-10-13 17:00 user] pause Soft-Fold (950k), MASt3R the UMIalign sessions on GPU1 (+GPU0 if it frees), then resume training from 950k
export RAY_ADDRESS=http://100.64.0.1:8265
cd ~/ego_collector
for s in UMIalign_20261013_160900 UMIalign_20261013_131556 UMIalign_20261013_144936 UMIalign_20261013_153819; do
  M3_GPUS="0 1" RUNNER=restart SKIP_CHECK=1 bash scripts/robotlike_session_check_4090.sh datasets/human_handumi_raw/UMIalign/$s
done
echo "[$(date '+%m-%d %H:%M')] MASt3R done -> resume Soft-Fold from 950k on GPU1"
until [ "$(ssh -o BatchMode=yes bh-aiteam@100.64.0.2 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1" | tr -dc 0-9)" -lt 2000 ]; do sleep 30; done
ray job submit --submission-id sf-rel32bin-1000k-resume950k --entrypoint-resources '{"node:100.64.0.2": 0.001}' --no-wait -- bash -c "ACCELERATE_MIXED_PRECISION=bf16 RESUME=1 RUN_NAME=SOFTFOLD-REL32BIN-RELONLY-D21-B8-1000K STEPS=1000000 SAVE_FREQ=50000 bash /home/bh-aiteam/softfold/code_rel/launch/train_softfold_rel32bin_4090.sh 1"
sleep 120
cd ~ && JUMP="" ONESHOT=0 STATE_EVERY=100000 KEEP_NODE_STEPS="650000" nohup ~/bh-egocentric/training/archivers/trackb_archive_v4_relonly_bce.sh SOFTFOLD-REL32BIN-RELONLY-D21-B8-1000K >> ~/archive_SOFTFOLD-REL32BIN-RELONLY-D21-B8-1000K.log 2>&1 &
echo "[$(date '+%m-%d %H:%M')] training resumed + archiver restarted"
