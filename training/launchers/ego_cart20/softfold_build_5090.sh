#!/bin/bash
# [2026-10-07 user] Soft-Fold (Facebear/XVLA-Soft-Fold, 1542 hdf5, ~450 GB) -> RELCART20 REL-only LeRobot on the 5090.
# Streaming: each hdf5 is downloaded, exported to the ego_cart20 raw format (320x240 videos) and DELETED, P at a time,
# so the 5090 never holds the full raw set.  Then convert (max_raw_gap 0.10, folder-level val) -> validate -> LeRobot export
# -> SHA256SUMS.  Chunk 32 (user 2026-10-07).  Resumable: finished raw episodes / shards are skipped.  LIMIT=N = smoke on the first N files.
set -e -o pipefail
R=/srv/data/johann/relonly; SF=/srv/data/johann/softfold; CODE=$SF/code
PY=$R/venv/bin/python; export PYTHONPATH=$R/code/lerobot-seeed/src
NAME=${NAME:-softfold_cart20_h32_v1}; P=${P:-8}; LIMIT=${LIMIT:-0}; TAG=${TAG:-}
H5=$SF/hdf5; RAW=$SF/raw$TAG; PROC=$SF/proc_${NAME}$TAG; LR=$SF/lerobot$TAG
mkdir -p $H5 $RAW $LR
echo "[$(date '+%F %T')] softfold build NAME=$NAME P=$P LIMIT=$LIMIT RAW=$RAW"
# 1. file list (HF API)
curl -s https://huggingface.co/api/datasets/Facebear/XVLA-Soft-Fold | $PY -c "
import json,sys; d=json.load(sys.stdin)
for s in sorted(x['rfilename'] for x in d['siblings'] if x['rfilename'].endswith('.hdf5')): print(s)" > $SF/files.txt
[ "$LIMIT" -gt 0 ] && { head -n $LIMIT $SF/files.txt > $SF/files_use.txt; } || cp $SF/files.txt $SF/files_use.txt
echo "files: $(wc -l < $SF/files_use.txt)"
# 2. download -> raw export -> delete, P parallel
one() {
  f=$1; d=$(dirname $f); b=$(basename $f .hdf5); id="sf_${d}_${b/episode_/ep}"
  [ -f $RAW/$id/raw_episode.json ] && { echo "{\"episode\": \"$id\", \"status\": \"exists\"}"; return 0; }
  mkdir -p $H5/$d; t=$H5/$f
  for i in 1 2 3 4 5; do curl -sfL --retry 5 --retry-delay 10 -C - -o $t.part "https://huggingface.co/datasets/Facebear/XVLA-Soft-Fold/resolve/main/$f" && break; sleep 20; done
  mv $t.part $t
  $PY $CODE/ego_cart20/sources/softfold_export.py --out $RAW --delete-source $t 2>&1 | grep '^{' || true
}
export -f one; export RAW H5 PY CODE
xargs -P $P -I{} bash -c 'one {}' < $SF/files_use.txt > $SF/export_log$TAG.jsonl
echo "[$(date '+%F %T')] export: $(grep -c '"ok"\|"exists"' $SF/export_log$TAG.jsonl) ok, $(grep -c '"error"' $SF/export_log$TAG.jsonl || true) error"
grep '"error"' $SF/export_log$TAG.jsonl | head -5 || true
[ "${RAW_ONLY:-0}" = 1 ] && { echo "[$(date '+%F %T')] RAW_ONLY=1 -> stop after raw export ($(ls $RAW | grep -c ^sf_) episodes)"; echo "RAW DONE"; exit 0; }
# 3. convert + validate
rm -rf $PROC
$PY $CODE/ego_cart20/scripts/convert_dataset_softfold.py $RAW $PROC --workers 24 ${VAL_FOLDERS:+--val-folders $VAL_FOLDERS}
$PY $CODE/ego_cart20/scripts/validate_dataset_softfold.py $PROC > $SF/validate_${NAME}$TAG.log 2>&1 && echo "VALIDATION PASS" || { tail -5 $SF/validate_${NAME}$TAG.log; echo "VALIDATION FAIL"; exit 1; }
# 4. LeRobot export
SPLITS="train val"; [ -s $PROC/val_manifest.jsonl ] || SPLITS="train"
$PY $CODE/ego_cart20/scripts/export_lerobot_softfold.py $PROC $LR --name $NAME --workers 16 --splits $SPLITS 2>&1 | grep -v "moov atom\|it/s\]" | tail -3
for s in $SPLITS; do ( cd $LR/${NAME}_$s && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS ); done
# 5. load check (LeRobotDataset, one sample)
$PY - <<PYEOF
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("rebot/${NAME}_train", root="$LR/${NAME}_train"); x = ds[len(ds) // 2]
print("LOAD OK", len(ds), ds.meta.total_episodes, {k: tuple(v.shape) for k, v in x.items() if hasattr(v, "shape")}, x["task"])
PYEOF
echo "[$(date '+%F %T')] BUILD DONE $LR"
