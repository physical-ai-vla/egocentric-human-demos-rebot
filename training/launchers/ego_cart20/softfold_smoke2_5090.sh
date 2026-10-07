#!/bin/bash
# smoke-2: links the first N finished raw episodes of the main build (read-only use), converts/exports them to a separate
# tree and runs the 60-step GPU smoke (preflight needs >= 20001 rows).  Never touches the main build's files.
set -e -o pipefail
R=/srv/data/johann/relonly; SF=/srv/data/johann/softfold; CODE=$SF/code; PY=$R/venv/bin/python; export PYTHONPATH=$R/code/lerobot-seeed/src
N=${N:-20}; RAW2=$SF/raw_smoke2; PROC2=$SF/proc_smoke2; LR=$SF/lerobot_smoke
until [ "$(ls $SF/raw | grep -c ^sf_)" -ge "$N" ]; do sleep 30; done
rm -rf $RAW2 $PROC2 $LR/sfsmoke2_*; mkdir -p $RAW2
for d in $(ls $SF/raw | grep ^sf_ | head -n $N); do ln -s $SF/raw/$d $RAW2/$d; done
$PY $CODE/ego_cart20/scripts/convert_dataset_softfold.py $RAW2 $PROC2 --workers 8 --val-folders none
$PY $CODE/ego_cart20/scripts/validate_dataset.py $PROC2 > $SF/validate_smoke2.log 2>&1 && echo "VALIDATION PASS" || { tail -5 $SF/validate_smoke2.log; exit 1; }
$PY $CODE/ego_cart20/scripts/export_lerobot_softfold.py $PROC2 $LR --name sfsmoke2 --workers 8 --splits train 2>&1 | grep -v "moov atom\|it/s\]" | tail -2
( cd $LR/sfsmoke2_train && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
ACCELERATE_MIXED_PRECISION=bf16 RELONLY_PREFLIGHT=3 SF_DS=../lerobot_smoke/sfsmoke2_train RUN_NAME=SOFTFOLD-SMOKE2-60 STEPS=60 SAVE_FREQ=60 MIN_START_GB=150 \
  bash $R/code/train_softfold_5090_b8.sh --log_freq=10
