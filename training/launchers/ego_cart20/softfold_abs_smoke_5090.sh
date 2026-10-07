#!/bin/bash
# abs30 smoke: export the first N finished raw episodes (read-only links into the main raw tree) + 60-step GPU smoke.
set -e -o pipefail
R=/srv/data/johann/relonly; SF=/srv/data/johann/softfold; CODE=$SF/code; PY=$R/venv/bin/python; export PYTHONPATH=$R/code/lerobot-seeed/src
N=${N:-20}; RAW2=$SF/raw_abs_smoke; LR=$SF/lerobot
rm -rf $RAW2 $LR/sfabs_smoke_*; mkdir -p $RAW2
for d in $(ls $SF/raw | grep ^sf_ | head -n $N); do [ -f $SF/raw/$d/raw_episode.json ] && ln -s $SF/raw/$d $RAW2/$d; done
$PY $CODE/ego_cart20/scripts/export_softfold_abs.py $RAW2 $LR --name sfabs_smoke --workers 10 --val-folders none 2>&1 | grep -v "moov atom\|it/s\]" | tail -3
( cd $LR/sfabs_smoke_train && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
ACCELERATE_MIXED_PRECISION=bf16 SF_DS=sfabs_smoke_train RUN_NAME=FT-SOFTFOLD-SMOKE-ABS-60 STEPS=60 SAVE_FREQ=60 XVLA_FREEZE=30 \
  bash $CODE/launch/train_softfold_abs_5090.sh --log_freq=10 || true
f=$SF/runs/FT-SOFTFOLD-SMOKE-ABS-60.log
tr '\r' '\n' < $f | grep -E "xvla-recipe|step:|End of training|Error|Traceback" | tail -14
nvidia-smi --query-gpu=memory.used --format=csv,noheader
