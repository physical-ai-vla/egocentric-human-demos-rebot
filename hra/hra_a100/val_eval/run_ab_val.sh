#!/bin/bash
# [2026-10-07 user "A B 둘다 한번 해줘"] held-out val loss (11 val episodes) of A (start) and B (origin) checkpoints on Mac MPS, bf16, seed 0, train subset 1000
cd ~/c8/hra_a100/val_eval; PY=~/xvla-mac/bin/python; L=~/c8/hra_a100
for st in 005000 015000 030000 045000 010000 020000 025000 035000 040000 050000; do
  $PY val_loss_ab.py --steps $st --seeds 0 --train_n 1000 --run HRA-RIGHTONLY-ROBOTCAM-A93START-LOSSMASK-D20-B8-300K --val $L/lerobot_a93/ego_hra_a93_start_v1_val --train $L/lerobot_a93/ego_hra_a93_start_v1_train --ckroot ckA --out resA.json 2>&1 | grep -E "DONE|Error|Traceback|data\]" >> runA.log
  [ "$st" = 050000 ] && continue
  $PY val_loss_ab.py --steps $st --seeds 0 --train_n 1000 --run HRA-RIGHTONLY-ROBOTCAM-A93ORIGIN-LOSSMASK-D20-B8-300K --val $L/lerobot_a93o/ego_hra_a93_origin_v1_val --train $L/lerobot_a93o/ego_hra_a93_origin_v1_train --ckroot ckB --out resB.json 2>&1 | grep -E "DONE|Error|Traceback|data\]" >> runB.log
done
echo ALL_DONE >> runA.log
