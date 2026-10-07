#!/bin/bash
# [2026-09-28] Run the v2 offline gate (v2_ckpt_eval.sh) on R312C-REL16V2-B checkpoints as the archiver lands them.
# Waits for <step>/.archived on the SSD, copies the weights locally, and never starts while the robot UI (:8025)
# is running a loop, because the eval shares the Mac MPS and would slow the robot's inference.
DEST=/Volumes/PortableSSD/rebot_ckpts_archive/trackb_R312C-REL16V2-B-D600K
NODE=bh-aiteam@100.64.0.2; NCK=/home/bh-aiteam/holobrain-data/trainB/R312C-REL16V2-B-D600K/checkpoints
STEPS=${STEPS:-"100000 150000 200000 250000 300000 350000 400000 450000 500000 550000 600000"}   # [2026-09-28 user] 50k units only
LOG=~/umi_bridge/rel16_audit/v2b_gate_watch.log
for s in $STEPS; do
  # [2026-09-29] do not wait for the SSD: the archiver keeps the NEWEST checkpoint on the node while training runs, so with a
  # 50k cadence the SSD copy lands only one checkpoint later. Pull the weights straight from the node (read-only, the node
  # copy stays) once they exist AND training has logged a step past it (the save is complete); fall back to the SSD copy.
  K=$((10#$s / 1000)); L=~/holobrain-mac-model/ckpt_R312C-REL16V2-B_${K}k
  until [ -f "$DEST/$s/.archived" ] || ssh -o BatchMode=yes -J head-lp $NODE "test -s $NCK/$s/pretrained_model/model.safetensors && grep -a 'step:' $NCK/../../R312C-REL16V2-B-D600K.log | tail -1 | grep -qE 'step:([0-9]+)K' && [ \$(grep -a 'step:' $NCK/../../R312C-REL16V2-B-D600K.log | tail -1 | grep -oE 'step:[0-9]+' | tr -dc 0-9) -gt $((K + 1)) ]" 2>/dev/null; do sleep 120; done
  if [ -f "$DEST/$s/.archived" ]; then /opt/homebrew/bin/rsync -a "$DEST/$s/pretrained_model/" "$L/"
  else /opt/homebrew/bin/rsync -a -e "ssh -o BatchMode=yes -J head-lp" "$NODE:$NCK/$s/pretrained_model/" "$L/"; fi
  ~/xvla-mac/bin/python - "$L" <<'EOS'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "config.json"; c = json.loads(p.read_text())
if "type" not in c: p.write_text(json.dumps({"type": "xvla", **c}, indent=2))
EOS
  while curl -s -m 5 localhost:8025/status | grep -q '"running": *true'; do sleep 60; done
  echo "=== $s $(date '+%F %T')" >> $LOG
  ~/umi_bridge/rel16_audit/v2_ckpt_eval.sh "$L" B 2>&1 | sed -n '/\[v2 gate\]/,$p;/\[2\] motion/,/draw-to-draw/p' >> $LOG
done
echo "watch done $(date '+%F %T')" >> $LOG
