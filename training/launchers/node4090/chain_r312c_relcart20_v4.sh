#!/bin/bash
# [2026-09-30 user] RELCART20 v4 = the RELCART20-v3 recipe on R312-center (HEAD180 + FRONT132 pan=center, rot180), 312 eps / 174,012 frames.
# Same chain as R180: r312c_umi76_rel16_v1 -> derive_v2 --variant C (continuous leader gripper) -> gate_v2 -> derive_v3d (dq12 + aux.q_t,
# built-in FK gate) -> derive_relcart20 (state20, QA1 built in). Scripts unchanged. Stops at the first failure.
# [2026-09-30] derive_v2 runs through run_derive_v2_ro_parent.py (copy2 + chmod 644, as derive_v3d/relcart20 do): the R312c parent is frozen read-only.
set -e -o pipefail
B=/home/bh-aiteam/umi_bridge; D=/home/bh-aiteam/holobrain-data/lerobot; PY=/home/bh-aiteam/miniforge3/envs/holobrain/bin/python
export PYTHONPATH=/home/bh-aiteam/lerobot-seeed/src:/home/bh-aiteam/universal_manipulation_interface UMI_ROOT=/home/bh-aiteam/universal_manipulation_interface
L=$B/chain_r312c_relcart20_v4.log; : > $L
cd $B/umi76
echo "[$(date '+%F %T')] derive_v2 C" >> $L
nice -n 19 $PY run_derive_v2_ro_parent.py --parent $D/r312c_umi76_rel16_v1 --out $D/r312c_umi76_rel16_v3c --variant C >> $L 2>&1
echo "[$(date '+%F %T')] gate_v2 (GRIP=continuous)" >> $L
GRIP=continuous nice -n 19 $PY gate_v2.py $D/r312c_umi76_rel16_v1 $D/r312c_umi76_rel16_v3c >> $L 2>&1
echo "[$(date '+%F %T')] derive_v3d" >> $L
nice -n 19 $PY derive_v3d.py --parent $D/r312c_umi76_rel16_v3c --out $D/r312c_umi76_rel16_v3d >> $L 2>&1
echo "[$(date '+%F %T')] derive_relcart20" >> $L
nice -n 19 $PY derive_relcart20.py --parent $D/r312c_umi76_rel16_v3d --out $D/r312c_relcart20_rel16_v4 >> $L 2>&1
(cd $D/r312c_relcart20_rel16_v4 && find . -type f ! -path "./videos/*" -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > $B/umi76/r312c_relcart20_rel16_v4.SHA256SUMS
echo "[$(date '+%F %T')] CHAIN_R312C_RELCART20_V4_DONE" >> $L
