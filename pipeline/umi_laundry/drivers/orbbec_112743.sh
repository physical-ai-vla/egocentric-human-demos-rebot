#!/bin/bash
# [2026-10-15] Orbbec body_track(laundry_v1) + canonical for UMIalign_20261015_112743 (50 kept eps), 3 lanes; skips eps with outputs.
cd ~/ego_collector; S=datasets/human_handumi_raw/UMIalign/UMIalign_20261015_112743; L=~/c8/robotlike/logs
one () { local e=$S/episode_$(printf %06d $1); [ -f $e/.complete ] || return
  [ -f $e/derived/body_track/track.npz ] || caffeinate -ims .venv/bin/python -u scripts/handumi_body_track.py --episode $e --profile laundry_v1 --save-npz --debug-frames 0 || { echo "FAIL body $1"; return; }
  [ -f $e/derived/body_track/canonical.npz ] || caffeinate -ims .venv/bin/python -u scripts/handumi_canonical_track.py --episode $e || { echo "FAIL canon $1"; return; }
  echo "OK ep $1 $(date +%T)"; }
lane () { for i in "$@"; do one $i; done; echo LANE_DONE; }
lane $(seq 1 3 53) > $L/UMIalign_20261015_112743.orbbecA.log 2>&1 &
lane $(seq 2 3 53) > $L/UMIalign_20261015_112743.orbbecB.log 2>&1 &
lane $(seq 3 3 53) > $L/UMIalign_20261015_112743.orbbecC.log 2>&1 &
wait; echo ORBBEC_DONE $(date +%T) > $L/UMIalign_20261015_112743.orbbec_done.txt
