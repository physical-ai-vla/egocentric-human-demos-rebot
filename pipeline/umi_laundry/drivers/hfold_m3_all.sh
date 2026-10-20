#!/bin/bash
# [2026-10-12] MASt3R poses (4090 GPU0) for every kept Hfold session, sequentially.
cd ~/ego_collector
for s in Hfold_20261012_143136 Hfold_20261012_152646 Hfold_20261012_160608 Hfold_20261012_163312 Hfold_20261012_163800; do
  SKIP_CHECK=1 bash scripts/robotlike_session_check_4090.sh datasets/human_handumi_raw/Hfold/$s
done
echo ALL_DONE
