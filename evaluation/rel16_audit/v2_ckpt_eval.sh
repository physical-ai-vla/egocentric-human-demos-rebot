#!/bin/bash
# [2026-09-28] offline gate for a REL16-v2 checkpoint BEFORE any real-robot run (user): prompt-swap + grip_go.
# usage: v2_ckpt_eval.sh <local ckpt dir> <A|B> [HEAD]   (pause it while the robot UI runs: it shares the MPS)
#   HEAD = the Stage-1 re-check for a PRELIM PASS: HEAD episodes only, moving 20 + rest_go 10 (n 30) + grip_go 15
set -e
CK=${1:?ckpt}; TW=${2:?A or B}
case "$TW" in
  A) ROOT=~/holobrain-data/lerobot/r312c_umi76_rel16_v2A; SM=umi76 ;;
  B) ROOT=~/holobrain-data/lerobot/r312c_umi94_rel16_v2B; SM=umi94 ;;
  C) ROOT=~/holobrain-data/lerobot/r180_umi76_rel16_v3d; SM=umi76 ;;   # [2026-09-29] REL16-v3 (continuous gripper, dq aux; preds sliced to 20)
  D) ROOT=~/holobrain-data/lerobot/r180_umi76_rel16_v3d; SM=umi76; export FEED_ROOT=~/holobrain-data/lerobot/r180_cart20_rel16_v3d ;;   # [2026-09-29] CART20: same picks as C, cart20 state fed
  E) ROOT=~/holobrain-data/lerobot/r180_umi76_rel16_v3d; SM=umi76; export FEED_ROOT=~/holobrain-data/lerobot/r180_relcart20_rel16_v3d ;;   # [2026-09-29] RELCART20: same picks as C, relcart20 state fed
  F) ROOT=~/holobrain-data/lerobot/r180_umi76_rel16_v3L; SM=umi76 ;;   # [2026-09-29] v3L intent labels: scored against its OWN (leader) GT
  G) ROOT=~/holobrain-data/lerobot/r180_umi76_rel16_v3L; SM=umi76; export FEED_ROOT=~/holobrain-data/lerobot/r180_relcart20_rel16_v3L ;;   # [2026-09-29] RELCART20 intent
esac
cd ~/umi_bridge/rel16_audit
if [ "${3:-}" = "HEAD" ]; then POOLV=HEAD; Q='{"moving": 20, "rest_go": 10, "grip_go": 15}'; SUF=_HEAD30
else POOLV=; Q='{"moving": 15, "rest_go": 10, "grip_go": 15}'; SUF=; fi
OUT=~/umi_bridge/rel16_audit/v2eval_$(basename $CK)$SUF.npz
V4_CKPT=$CK ROOT=$ROOT V4_STATE_MODE=$SM GRIP_BINARY=1 POOL=$POOLV OUT=$OUT QUOTA="$Q" \
  ~/xvla-mac/bin/python step2_predict.py > ${OUT%.npz}.log 2>&1
~/xvla-mac/bin/python analyze.py $OUT | tee ${OUT%.npz}.txt
