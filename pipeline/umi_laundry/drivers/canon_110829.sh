#!/bin/bash
cd ~/ego_collector
for e in 1 2 3; do caffeinate -ims .venv/bin/python -u scripts/handumi_canonical_track.py --episode datasets/human_handumi_raw/UMIpilot/UMIpilot_20261013_110829/episode_00000$e --debug-dir ~/Desktop/Hfold_review_20261012/tracker_debug/canonical --debug-frames 2; done
echo CANON_DONE
