#!/bin/bash
# [2026-09-28] R312-center UMI76 REL16 = HEAD180 + FRONT132 (pan=center only, grade A, 22/order), user 2026-09-28.
# r675rbp_umi.zarr, global rot180 (as B663), QA-v2 grade A, order-balanced, right-wrist-compatible preferred.
# Selection: ~/umi_bridge/umi76/r380_front200_v1.json (deterministic). Contract: ~/umi_bridge/umi76/UMI76_CONTRACT.md
set -e
PY=/home/bh-aiteam/miniforge3/envs/holobrain/bin/python
D=/home/bh-aiteam/holobrain-data/lerobot
B=/home/bh-aiteam/umi_bridge
OUT=$D/r312c_umi76_rel16_v1
FREE=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9)
[ "$FREE" -lt 40 ] && { echo "only ${FREE}G free on /home, refusing to convert"; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists, refusing to overwrite"; exit 1; }
echo "free on /home: ${FREE}G"
cd $B/umi76
CONV_WRITER_THREADS=4 nice -n 19 env PYTHONPATH=/home/bh-aiteam/lerobot-seeed/src:/home/bh-aiteam/universal_manipulation_interface $PY \
  umi76_to_lerobot.py \
  --source zarr=$B/r150_umi.zarr,lerobot=$D/rebot_3stack_R150_headview \
  --source zarr=$B/r30_umi.zarr,lerobot=$D/rebot_3stack_R30_day4_headview \
  --source zarr=$B/r675rbp_umi.zarr,lerobot=$D/rebot_3stack_center675_s96,global_transform=rot180,episodes_json=$B/r675_rbp.json,select=$B/umi76/r312c_front132_v1.json \
  --out $OUT \
  --repo-id rebot/r312c_umi76_rel16_v1 \
  --action-mode umi --state-mode slim20 --gripper-target next \
  > $B/convert_r312c_umi76.log 2>&1
echo BUILD_R312C_DONE >> $B/convert_r312c_umi76.log
