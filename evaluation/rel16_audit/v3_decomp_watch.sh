#!/bin/bash
# [2026-09-29] Adds policy_decomp (err(k), bias vs D_draw) + the 3-way IK decomposition to every HEAD180-REL16V3 checkpoint the
# running v3_gate_watch.sh evaluates (that script is not edited while it runs). Waits for its signed-error line per step.
LOG=~/umi_bridge/rel16_audit/v3_gate_watch.log
for K in $(seq 60 10 600); do
  N=v2eval_ckpt_HEAD180-REL16V3_${K}k
  until grep -q "signed error\] $N.npz" $LOG 2>/dev/null; do sleep 120; done
  sleep 30
  { echo "=== decomp $((K * 1000)) $(date '+%F %T')"
    ~/xvla-mac/bin/python ~/umi_bridge/rel16_audit/policy_decomp.py ~/umi_bridge/rel16_audit/$N.npz HEAD180-REL16V3 $((K * 1000)) ~/umi_bridge/rel16_audit/policy_decomp.csv 2>&1
    ~/xvla-mac/bin/python ~/umi_bridge/learned_ik/eval_ik_backends.py pred ~/umi_bridge/rel16_audit/$N.npz ~/umi_bridge/learned_ik/evals/decomp_ckpt_HEAD180-REL16V3_${K}k.json 2>&1 | grep -E "^\[(policy|N0)\]"
  } >> ~/umi_bridge/rel16_audit/v3_decomp_watch.log
done
