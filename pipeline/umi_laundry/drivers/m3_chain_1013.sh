#!/bin/bash
cd ~/ego_collector
for s in UMIpilot/UMIpilot_20261013_110829 UMIalign/UMIalign_20261013_131556 UMIalign/UMIalign_20261013_125447; do
  RUNNER=restart SKIP_CHECK=1 bash scripts/robotlike_session_check_4090.sh datasets/human_handumi_raw/$s
done
echo CHAIN_DONE
