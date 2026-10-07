#!/bin/bash
# [2026-09-28] R312-center chain: convert (nice 19) -> converter gate -> derive v2A/v2B -> v2 gates. Stops at the first failure.
set -e -o pipefail
B=/home/bh-aiteam/umi_bridge; D=/home/bh-aiteam/holobrain-data/lerobot; PY=/home/bh-aiteam/miniforge3/envs/holobrain/bin/python
export PYTHONPATH=/home/bh-aiteam/lerobot-seeed/src:/home/bh-aiteam/universal_manipulation_interface UMI_ROOT=/home/bh-aiteam/universal_manipulation_interface
L=$B/chain_r312c.log; : > $L
echo "[$(date '+%F %T')] convert" >> $L
bash ~/build_r312c_umi76.sh >> $L 2>&1
grep -q BUILD_R312C_DONE $B/convert_r312c_umi76.log
echo "[$(date '+%F %T')] gate_r312c" >> $L
cd $B/umi76 && nice -n 19 $PY gate_r312c.py >> $L 2>&1; grep -q "ALL R312c GATES PASS" $L
echo "[$(date '+%F %T')] derive A/B" >> $L
nice -n 19 $PY derive_v2.py --parent $D/r312c_umi76_rel16_v1 --out $D/r312c_umi76_rel16_v2A --variant A >> $L 2>&1
nice -n 19 $PY derive_v2.py --parent $D/r312c_umi76_rel16_v1 --out $D/r312c_umi94_rel16_v2B --variant B >> $L 2>&1
nice -n 19 $PY gate_v2.py $D/r312c_umi76_rel16_v1 $D/r312c_umi76_rel16_v2A $D/r312c_umi94_rel16_v2B >> $L 2>&1
grep -q "18/18 PASS" $L
echo "[$(date '+%F %T')] CHAIN_R312C_DONE" >> $L
