#!/bin/bash
# [2026-09-29] CART20 copy of v3_gate_watch.sh: same evaluator (variant D = same picks as C, cart20 state fed), every 10k from 10k,
# plus policy_decomp (err(k), bias vs D_draw) and the 3-way IK decomposition after each eval.
# [2026-09-28] Run the v2 offline gate (v2_ckpt_eval.sh) on R312C-REL16V2-B checkpoints as the archiver lands them.
# Waits for <step>/.archived on the SSD, copies the weights locally, and never starts while the robot UI (:8025)
# is running a loop, because the eval shares the Mac MPS and would slow the robot's inference.
DEST=/Volumes/PortableSSD/rebot_ckpts_archive/trackb_HEAD180-CART20-REL16V3-D600K
NODE=bh-aiteam@100.64.0.2; NCK=/home/bh-aiteam/holobrain-data/trainB/HEAD180-CART20-REL16V3-D600K/checkpoints
STEPS=${STEPS:-"010000 020000 030000 040000 050000 060000 070000 080000 090000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 210000 220000 230000 240000 250000 260000 270000 280000 290000 300000 310000 320000 330000 340000 350000 360000 370000 380000 390000 400000 410000 420000 430000 440000 450000 460000 470000 480000 490000 500000 510000 520000 530000 540000 550000 560000 570000 580000 590000 600000"}   # [2026-09-29 user] every 10k from 50k
LOG=~/umi_bridge/rel16_audit/cart20_gate_watch.log
for s in $STEPS; do
  # [2026-09-29] do not wait for the SSD: the archiver keeps the NEWEST checkpoint on the node while training runs, so with a
  # 50k cadence the SSD copy lands only one checkpoint later. Pull the weights straight from the node (read-only, the node
  # copy stays) once they exist AND training has logged a step past it (the save is complete); fall back to the SSD copy.
  K=$((10#$s / 1000)); L=~/holobrain-mac-model/ckpt_HEAD180-CART20-REL16V3_${K}k
  until [ -f "$DEST/$s/.archived" ] || ssh -o BatchMode=yes -J head-lp $NODE "test -s $NCK/$s/pretrained_model/model.safetensors && grep -a 'step:' $NCK/../../HEAD180-CART20-REL16V3-D600K.log | tail -1 | grep -qE 'step:([0-9]+)K' && [ \$(grep -a 'step:' $NCK/../../HEAD180-CART20-REL16V3-D600K.log | tail -1 | grep -oE 'step:[0-9]+' | tr -dc 0-9) -gt $((K + 1)) ]" 2>/dev/null; do sleep 120; done
  if [ -f "$DEST/$s/.archived" ]; then /opt/homebrew/bin/rsync -a "$DEST/$s/pretrained_model/" "$L/"
  else /opt/homebrew/bin/rsync -a -e "ssh -o BatchMode=yes -J head-lp" "$NODE:$NCK/$s/pretrained_model/" "$L/"; fi
  ~/xvla-mac/bin/python - "$L" <<'EOS'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "config.json"; c = json.loads(p.read_text())
if "type" not in c: p.write_text(json.dumps({"type": "xvla", **c}, indent=2))
EOS
  while curl -s -m 5 localhost:8025/status | grep -q '"running": *true' || curl -s -m 5 localhost:8026/status | grep -q '"running": *true'; do sleep 60; done   # [2026-09-29] both robot UIs (v3 :8025, cart20 :8026)
  echo "=== $s $(date '+%F %T')" >> $LOG
  ~/umi_bridge/rel16_audit/v2_ckpt_eval.sh "$L" D 2>&1 | sed -n '/\[v2 gate\]/,$p;/\[2\] motion/,/draw-to-draw/p' >> $LOG
  ~/xvla-mac/bin/python ~/umi_bridge/rel16_audit/signed_err.py ~/umi_bridge/rel16_audit/v2eval_$(basename $L).npz ~/holobrain-data/lerobot/r180_umi76_rel16_v3d 2>&1 | grep -v "jaxls\|pyroki\|INFO" >> $LOG   # [2026-09-29] L/R signed base-frame error from the start
  ~/xvla-mac/bin/python ~/umi_bridge/rel16_audit/policy_decomp.py ~/umi_bridge/rel16_audit/v2eval_$(basename $L).npz HEAD180-CART20-REL16V3 $((10#$s)) ~/umi_bridge/rel16_audit/policy_decomp.csv >> $LOG 2>&1
  ~/xvla-mac/bin/python ~/umi_bridge/learned_ik/eval_ik_backends.py pred ~/umi_bridge/rel16_audit/v2eval_$(basename $L).npz ~/umi_bridge/learned_ik/evals/decomp_$(basename $L).json 2>&1 | grep -E "^\[(policy|N0)\]" >> $LOG
  [ $((10#$s % 50000)) -ne 0 ] && rm -rf "$L"      # [2026-09-29] keep Mac copies only for 50k multiples (results stay in the log + npz)
done
echo "watch done $(date '+%F %T')" >> $LOG
