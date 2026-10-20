#!/bin/bash
# wait for the running Hfold session job, then MASt3R the newest UMIpilot session (4090 GPU0)
while pgrep -f "robotlike_session_check_4090.sh datasets/human_handumi_raw/Hfold" >/dev/null; do sleep 15; done
cd ~/ego_collector
for s in "$@"; do SKIP_CHECK=1 bash scripts/robotlike_session_check_4090.sh $s; done
echo ALL_DONE
