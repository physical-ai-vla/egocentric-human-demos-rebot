#!/bin/bash
# [2026-10-02 user: "학습 경과를 잘 봐줘"] every 10 min: FT (raw ego 50k -> R312c) vs A (R312c from scratch, same B8 recipe, stopped at 60k)
# at matched steps (mean loss over the last 1k steps), + R384 for reference; alerts on a dead job / NaN. -> rel16_audit/ft_watch.{log,csv}
N=bh-aiteam@100.64.0.2; TB=/home/bh-aiteam/holobrain-data/trainB; OUT=~/umi_bridge/rel16_audit/ft_watch.csv
FT=FT-R312C-FROM-EGOV2BB850K-RELONLY-D20-B8-300K; A=R312C-RELCART20-RELONLY-D20-B8-150K; R=R384-RELCART20-RELONLY-D20-B8-300K
[ -f $OUT ] || echo "time,ft_step,ft_loss_1k,a_loss_same_step,r384_step,r384_loss_1k,ft_running" > $OUT
while true; do
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -J head-lp $N "python3 - <<'PY'
import re, subprocess
def curve(run):
    t = open('$TB/' + run + '.log', errors='ignore').read().replace('\r', '\n')
    pts = []
    for m in re.finditer(r'ot_train.py:\d+ step:([0-9.]+)(K?) .*?loss:([0-9.naNinf]+)', t):
        s = float(m.group(1)) * (1000 if m.group(2) else 1); pts.append((s, m.group(3)))
    return pts
def mean_upto(pts, s):
    v = [float(l) for st, l in pts if s - 1000 < st <= s and l.replace('.', '', 1).isdigit()]
    return round(sum(v) / len(v), 4) if v else ''
ft, a, r = curve('$FT'), curve('$A'), curve('$R')
fs = ft[-1][0] if ft else 0; rs = r[-1][0] if r else 0
nan = any(not l.replace('.', '', 1).isdigit() for _, l in ft[-20:])
run = subprocess.call(['pgrep', '-f', 'train_rel16_relonly.py.*trainB/$FT( |/|\$)'], stdout=subprocess.DEVNULL) == 0
print(f'{int(fs)},{mean_upto(ft, fs)},{mean_upto(a, fs) if fs <= 60000 else \"\"},{int(rs)},{mean_upto(r, rs)},{int(run)},{int(nan)}')
PY" 2>/dev/null)
  if [ -n "$r" ]; then
    nan=${r##*,}; row=${r%,*}; echo "$(date '+%F %T'),$row" >> $OUT
    [ "$nan" = 1 ] && echo "[$(date '+%F %T')] ALERT: NaN/inf loss in $FT"
    [ "${row##*,}" = 0 ] && echo "[$(date '+%F %T')] ALERT: $FT process not running (finished or died)"
  fi
  sleep 600
done
