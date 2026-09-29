#!/bin/bash
# C-old v2 hybrid-all ABLATION (TR 689 + Kr_fallback 138 = 827) from the same frozen master, same builder / writer / checks.
set -e; cd ~/c8
~/xvla-mac/bin/python c8oldv2_build_dataset.py hybrid gate 2>&1 | grep -vE "jaxls|INFO|Warning|warn\(" | tail -2
~/xvla-mac/bin/python c8oldv2_build_dataset.py hybrid rows 2>&1 | grep -vE "jaxls|INFO|Warning|warn\(" | tail -2
C8OLD_ROWS=$HOME/c8/c8oldv2_hybrid_rows C8OLD_OUT=$HOME/c8/c8oldv2_hybrid_data C8OLD_SHARDS=$HOME/c8/c8oldv2_hybrid_shards \
C8OLD_SPLITS=c8oldv2_hybrid_train,c8oldv2_hybrid_val C8OLD_SCHEMA=track_c_c_old_dataset/v2_ablation \
C8OLD_SOURCE="~/c8/robotlike/hybrid_master/MASTER.json, retarget_method in {TR, Kr_fallback} (ablation: quality vs quantity)" \
C8OLD_MAIN_TARGET="hybrid master top-1 pseudo q: TR (v2-TR-R30) where feasible else Kr_fallback (v2-Kr-R30)" \
C8OLD_CD_TARGET="observation.ee.tcp_tgt = measured human relative TCP attached to the segment's anchor (TR window q_ref(0) | Kr seed) @ diag(C^T, 1)" \
~/xvla-mac/bin/python write_stage.py 2>&1 | grep -vE "Warning|warn\(|swscaler|it/s\]|%\|" | tail -4
~/xvla-mac/bin/python c8oldv2_validate.py hybrid 2>&1 | grep -vE "Warning|warn\(|swscaler|torchcodec|objc" | tail -30
