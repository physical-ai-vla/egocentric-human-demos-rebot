#!/bin/bash
# TR smoke must pass -> wait for the R30 full run -> TR full on old 259 -> held-out R120 comparison of all versions.
SP=${TMPDIR:-/tmp}/scratchpad
cnt () { ls "$1"/*.json 2>/dev/null | wc -l | tr -d ' '; }
while pgrep -f "[v]2k_retarget.py --mode tr --episodes-json $SP" >/dev/null; do sleep 30; done
ok=$(python3 -c "
import json,glob
rs=[json.load(open(p)) for p in glob.glob('$SP/tr_test/*.json')]
print(int(len(rs)==2 and all('error' not in r and r.get('segments') for r in rs)))")
[ "$ok" = 1 ] || { echo "TR SMOKE FAILED -- TR full NOT launched"; tail -5 $SP/tr_smoke.log; exit 1; }
echo "TR smoke OK $(date +%H:%M)"; grep -vE "jaxls|INFO|Warning|warn\(" $SP/tr_smoke.log | tail -4
until [ "$(cnt ~/c8/robotlike/r30_old)" -ge 259 ] && ! pgrep -f "[v]2k_retarget.py --mode r30" >/dev/null; do sleep 60; done
echo "R30 full done $(date +%H:%M); launching TR full"; mkdir -p ~/c8/robotlike/tr_old; cd ~/c8
for k in 0 1 2 3 4 5 6 7; do nohup ~/xvla-mac/bin/python ~/umi_bridge/track_c/v2k/v2k_retarget.py --mode tr --episodes-json ~/c8/robotlike/c8old_259_episodes.json --raw-root ~/ego_collector/datasets/human_handumi_raw/Hpilot --runs ~/c8/runs_m3 --export ~/c8/export --export ~/c8/export_early --out ~/c8/robotlike/tr_old --part $k/8 > ~/c8/robotlike/logs/tr_old_$k.log 2>&1 & done
sleep 60
until [ "$(cnt ~/c8/robotlike/tr_old)" -ge 259 ] && ! pgrep -f "[v]2k_retarget.py --mode tr --episodes-json /Users" >/dev/null; do sleep 60; done
echo "TR full done $(date +%H:%M)"
~/xvla-mac/bin/python ~/umi_bridge/track_c/v2k/v2k_compare.py --eval-ref heldout120 --group r30=$HOME/c8/robotlike/r30_old --group tr=$HOME/c8/robotlike/tr_old --out ~/c8/robotlike/compare_R30_heldout.json > ~/c8/robotlike/logs/compare_R30_heldout.txt 2>&1
echo "compare done $(date +%H:%M)"
