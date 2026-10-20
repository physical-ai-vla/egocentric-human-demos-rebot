#!/bin/bash
# Orbbec laundry_v1 + canonical tracking for every usable UMIalign episode (Mac CPU), newest session first
cd ~/ego_collector
for s in UMIalign_20261013_160900 UMIalign_20261013_144936 UMIalign_20261013_153819 UMIalign_20261013_131556; do
  for e in datasets/human_handumi_raw/UMIalign/$s/episode_*; do
    [ -f $e/.complete ] || continue
    grep -q '"exclude_from_training": true' $e/episode_meta.json && continue
    [ -f $e/derived/body_track/canonical.npz ] && continue
    [ -f $e/derived/body_track/track.npz ] || .venv/bin/python scripts/handumi_body_track.py --profile laundry_v1 --episode $e --save-npz --debug-frames 0 > /dev/null 2>&1
    caffeinate -ims .venv/bin/python -u scripts/handumi_canonical_track.py --episode $e --debug-frames 0 > /dev/null 2>&1 && echo "OK $e" || echo "FAIL $e"
  done
done
echo TRACK_ALL_DONE
