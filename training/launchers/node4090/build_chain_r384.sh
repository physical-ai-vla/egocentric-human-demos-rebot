#!/bin/bash
# [2026-10-01 user] R384 = R312c (HEAD180 + FRONT132) + FRONT far 21 + phase1 51 (user-reviewed exclusions), FRONT rot180.
# Same commands as build_r312c_umi76.sh + chain_r312c_relcart20_v4.sh; only the select file and the output names differ.
set -e -o pipefail
PY=/home/bh-aiteam/miniforge3/envs/holobrain/bin/python; D=/home/bh-aiteam/holobrain-data/lerobot; B=/home/bh-aiteam/umi_bridge
export PYTHONPATH=/home/bh-aiteam/lerobot-seeed/src:/home/bh-aiteam/universal_manipulation_interface UMI_ROOT=/home/bh-aiteam/universal_manipulation_interface
L=$B/chain_r384.log; : > $L
FREE=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9); [ "$FREE" -lt 40 ] && { echo "only ${FREE}G free" >> $L; exit 1; }
for o in r384_umi76_rel16_v1 r384_umi76_rel16_v3c r384_umi76_rel16_v3d r384_relcart20_rel16_v4; do [ -e $D/$o ] && { echo "$D/$o exists, refusing" >> $L; exit 1; }; done
cd $B/umi76
echo "[$(date '+%F %T')] convert (HEAD R150+R30, FRONT r675 rot180 select r384_front204_v1)" >> $L
CONV_WRITER_THREADS=4 nice -n 19 $PY umi76_to_lerobot.py \
  --source zarr=$B/r150_umi.zarr,lerobot=$D/rebot_3stack_R150_headview \
  --source zarr=$B/r30_umi.zarr,lerobot=$D/rebot_3stack_R30_day4_headview \
  --source zarr=$B/r675rbp_umi.zarr,lerobot=$D/rebot_3stack_center675_s96,global_transform=rot180,episodes_json=$B/r675_rbp.json,select=$B/umi76/r384_front204_v1.json \
  --out $D/r384_umi76_rel16_v1 --repo-id rebot/r384_umi76_rel16_v1 \
  --action-mode umi --state-mode slim20 --gripper-target next >> $L 2>&1
echo "[$(date '+%F %T')] derive_v2 C" >> $L
nice -n 19 $PY run_derive_v2_ro_parent.py --parent $D/r384_umi76_rel16_v1 --out $D/r384_umi76_rel16_v3c --variant C >> $L 2>&1
echo "[$(date '+%F %T')] gate_v2 (GRIP=continuous)" >> $L
GRIP=continuous nice -n 19 $PY gate_v2.py $D/r384_umi76_rel16_v1 $D/r384_umi76_rel16_v3c >> $L 2>&1
echo "[$(date '+%F %T')] derive_v3d" >> $L
nice -n 19 $PY derive_v3d.py --parent $D/r384_umi76_rel16_v3c --out $D/r384_umi76_rel16_v3d >> $L 2>&1
echo "[$(date '+%F %T')] derive_relcart20" >> $L
nice -n 19 $PY derive_relcart20.py --parent $D/r384_umi76_rel16_v3d --out $D/r384_relcart20_rel16_v4 >> $L 2>&1
(cd $D/r384_relcart20_rel16_v4 && find . -type f ! -path "./videos/*" -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > $B/umi76/r384_relcart20_rel16_v4.SHA256SUMS
echo "[$(date '+%F %T')] CHAIN_R384_DONE" >> $L
