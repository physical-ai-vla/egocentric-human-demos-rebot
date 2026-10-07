#!/bin/bash
cd ~/umi_bridge/rel16_audit
while pgrep -f "[s]tep2_predict.py" >/dev/null; do sleep 20; done
QUOTA='{"grip_go": 20}' OUT=~/umi_bridge/rel16_audit/step2b_grip.npz ~/xvla-mac/bin/python step2_predict.py > step2b.log 2>&1
