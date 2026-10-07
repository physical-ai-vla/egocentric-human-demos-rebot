#!/bin/bash
# [2026-10-07] A/B trainings on the 4090 with the size-agnostic preflight launcher copy (train_hra_a93_lossmask_d20_b8.sh)
set -u; A=$HOME/c8/hra_a100; P=$A/pipe; export L4=/home/bh-aiteam/train_hra_a93_lossmask_d20_b8.sh
log() { echo "[$(date '+%F %T')] $*"; }
DS=ego_hra_a93_start_v1_train LD=$A/lerobot_a93/ego_hra_a93_start_v1_train SMR=HRA-RIGHTONLY-A93START-D20-SMOKE MRUN=HRA-RIGHTONLY-ROBOTCAM-A93START-LOSSMASK-D20-B8-300K bash $P/chain_generic.sh > $P/chain_a93_start.log 2>&1 &
until grep -qE "main run submitted|did not pass|NOT started|failed" $P/chain_a93_start.log; do sleep 30; done
log "A: $(grep -E 'SMOKE VERDICT|main run submitted' $P/chain_a93_start.log | cut -c1-200 | tr '\n' ' ')"
DS=ego_hra_a93_origin_v1_train LD=$A/lerobot_a93o/ego_hra_a93_origin_v1_train SMR=HRA-RIGHTONLY-A93ORIGIN-D20-SMOKE MRUN=HRA-RIGHTONLY-ROBOTCAM-A93ORIGIN-LOSSMASK-D20-B8-300K bash $P/chain_generic.sh > $P/chain_a93_origin.log 2>&1
log "B: $(grep -E 'SMOKE VERDICT|main run submitted' $P/chain_a93_origin.log | cut -c1-200 | tr '\n' ' ')"
log "AB TRAIN DONE"
