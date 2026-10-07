#!/bin/bash
# [2026-10-07 user "ct5꺼주고 ct6돌려줘"] CT6 (start 6 cm back / 4 cm down) ALL-86 train set -> smoke + main 300k on ONE 4090 GPU (the one CT5-86 freed)
set -u; A=$HOME/c8/hra_a100; P=$A/pipe; O=$A/lerobot_ct6; NAME=ego_hra_ct6_86all_v1
L4=/home/bh-aiteam/train_hra_a93_lossmask_d20_b8.sh DS=${NAME}_train LD=$O/${NAME}_train SMR=HRA-RIGHTONLY-CT6ALL-D20-SMOKE MRUN=HRA-RIGHTONLY-ROBOTCAM-CT6-86ALL-LOSSMASK-D20-B8-300K bash $P/chain_generic.sh > $P/chain_CT6_all.log 2>&1
echo "[$(date '+%F %T')] CT6: $(grep -E 'SMOKE VERDICT|main run submitted|NOT started|did not pass|failed' $P/chain_CT6_all.log | cut -c1-200 | tr '\n' ' ')"
