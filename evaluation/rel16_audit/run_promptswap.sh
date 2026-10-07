#!/bin/bash
cd ~/umi_bridge/rel16_audit
while pgrep -f "[r]un_step2b.sh" >/dev/null; do sleep 20; done
~/xvla-mac/bin/python promptswap_joint.py ~/holobrain-mac-model/ckpt_C-old_R150ft_250000 ~/holobrain-mac-model/ckpt_E0_R150_CstateD_fp32_250000 > promptswap.log 2>&1
