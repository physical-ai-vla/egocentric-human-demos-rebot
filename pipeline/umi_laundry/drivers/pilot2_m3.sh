#!/bin/bash
# [2026-10-12] wait for the running restart job (GPU0), then restart_v1 MASt3R on the new pilot session
export RAY_ADDRESS=http://100.64.0.1:8265; SID=$(cat $HOME/c8/robotlike/restart_sid.txt)
while ray job status $SID 2>/dev/null | grep -q RUNNING; do sleep 30; done
cd ~/ego_collector && RUNNER=restart SKIP_CHECK=1 bash scripts/robotlike_session_check_4090.sh datasets/human_handumi_raw/UMIpilot/UMIpilot_20261012_180033
echo PILOT2_DONE
