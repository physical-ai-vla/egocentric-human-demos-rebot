#!/bin/bash
# [2026-10-14] Resume Orbbec body_track(laundry_v1) + canonical for UMIalign_20261014_112920 after the 14:07 Mac reboot. 2 lanes.
cd ~/ego_collector; S=datasets/human_handumi_raw/UMIalign/UMIalign_20261014_112920
one () { local e=$S/episode_$(printf %06d $1)
  [ -f $e/derived/body_track/track.npz ] || caffeinate -ims .venv/bin/python -u scripts/handumi_body_track.py --episode $e --profile laundry_v1 --save-npz || { echo "FAIL body $1"; return; }
  [ -f $e/derived/body_track/canonical.npz ] || caffeinate -ims .venv/bin/python -u scripts/handumi_canonical_track.py --episode $e || { echo "FAIL canon $1"; return; }
  echo "OK ep $1 $(date +%T)"; }
lane () { for i in "$@"; do one $i; done; }
lane 14 16 18 20 22 24 26 28 30 32 > ~/c8/robotlike/logs/UMIalign_20261014_112920.orbbec_resumeA.log 2>&1 &
lane 15 17 19 21 23 25 27 29 31 33 > ~/c8/robotlike/logs/UMIalign_20261014_112920.orbbec_resumeB.log 2>&1 &
wait; echo ORBBEC_RESUME_DONE
