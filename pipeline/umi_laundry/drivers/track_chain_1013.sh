#!/bin/bash
cd ~/ego_collector
S=datasets/human_handumi_raw/UMIalign/UMIalign_20261013_131556
.venv/bin/python scripts/handumi_body_track.py --profile laundry_v1 --session $S --save-npz --debug-frames 0
for e in $S/episode_*; do [ -f $e/.complete ] || continue; caffeinate -ims .venv/bin/python -u scripts/handumi_canonical_track.py --episode $e --debug-frames 0; done
echo TRACK_DONE
