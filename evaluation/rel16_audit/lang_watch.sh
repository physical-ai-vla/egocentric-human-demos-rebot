#!/bin/bash
# [2026-09-29] Per-checkpoint language/gripper diagnostics (offline_diag.py) for one run, from the SSD archive (or the node once the
# training has logged past the step): T2 at episode starts every checkpoint (D_prompt, D_noise, decoder-hidden prompt diff,
# 6-prompt first-move cos); T1 grip-state test at the first checkpoint and every 50k. Temp local copy, deleted after.
# Waits while a robot UI (:8025/:8026) runs a loop; one diagnostic at a time on the Mac (lock dir).
# usage: lang_watch.sh <RUN> <label> <state umi76|cart20> "<steps in k>"
RUN=$1; LAB=$2; ST=$3; KS=$4
SSD=/Volumes/PortableSSD/rebot_ckpts_archive/trackb_$RUN; N=bh-aiteam@100.64.0.2; NCK=/home/bh-aiteam/holobrain-data/trainB/$RUN/checkpoints
LOG=~/umi_bridge/rel16_audit/lang_watch_$LAB.log; LOCK=/tmp/offline_diag.lock; first=1
for K in $KS; do
  S=$(printf %06d $((K * 1000))); L=~/holobrain-mac-model/ckpt_LANG_${LAB}_${K}k
  until [ -f "$SSD/$S/.archived" ] || ssh -o BatchMode=yes -o ConnectTimeout=15 -J head-lp $N "test -s $NCK/$S/pretrained_model/model.safetensors && grep -a 'step:' $NCK/../../$RUN.log | tail -1 | grep -oE 'step:[0-9]+K' | tr -dc 0-9 | awk '{exit !(\$1+0 > $K + 1)}'" 2>/dev/null; do sleep 180; done
  if [ -f "$SSD/$S/.archived" ]; then /opt/homebrew/bin/rsync -a "$SSD/$S/pretrained_model/" "$L/"
  else /opt/homebrew/bin/rsync -a -e "ssh -o BatchMode=yes -J head-lp" "$N:$NCK/$S/pretrained_model/" "$L/" || { echo "fetch $K failed" >> $LOG; continue; }; fi
  ~/xvla-mac/bin/python -c "import json,pathlib,sys;p=pathlib.Path('$L')/'config.json';c=json.loads(p.read_text());'type' in c or p.write_text(json.dumps({'type':'xvla',**c},indent=2))"
  until mkdir $LOCK 2>/dev/null; do sleep 30; done
  while curl -s -m 5 localhost:8025/status | grep -q '"running": *true' || curl -s -m 5 localhost:8026/status | grep -q '"running": *true'; do sleep 60; done
  T1=1; { [ $first = 1 ] || [ $((K % 50)) = 0 ]; } && T1=0
  O=~/umi_bridge/rel16_audit/offline_diag_${LAB}_${K}k.json
  echo "=== $LAB ${K}k $(date '+%F %T') T1=$([ $T1 = 0 ] && echo on || echo off)" >> $LOG
  SKIP_T1=$T1 T2_STARTS=1 V4_CKPT=$L ~/xvla-mac/bin/python ~/umi_bridge/rel16_audit/offline_diag.py $O --state $ST 2>&1 | grep -E "^\[T1|^    |^\[T2 median" >> $LOG
  rmdir $LOCK
  ~/xvla-mac/bin/python - "$O" "$LAB" "$K" <<'EOS' >> $LOG
import csv, json, os, sys
import numpy as np
o, lab, k = sys.argv[1], sys.argv[2], int(sys.argv[3]); d = json.load(open(o)); T2 = d["T2"]
med = lambda key: float(np.median([r[key] for r in T2]))
row = dict(run=lab, k=k, D_prompt_A16=med("A16_prompt_mm"), D_noise_A16=med("A16_draw_mm"), ratio=med("A16_prompt_mm") / max(med("A16_draw_mm"), 1e-9),
           dec_hidden=med("dec_in"), enc_txt=med("enc_txt"), dir_cos_min=med("A16_dir_cos_min"))
f = os.path.expanduser("~/umi_bridge/rel16_audit/lang_curve.csv"); new = not os.path.exists(f)
with open(f, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(row)); new and w.writeheader(); w.writerow({a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()})
print("[lang]", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()})
EOS
  first=0; rm -rf "$L"
done
