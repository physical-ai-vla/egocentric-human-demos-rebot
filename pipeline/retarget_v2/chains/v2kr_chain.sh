#!/bin/bash
# v2-K (old 259) done -> proper v2-Kr (frozen lambda 0.5) on the same 259 -> v1/v2-K/v2-Kr compare. Patterns use [v] so
# pgrep never matches a shell that carries them.
cnt () { ls "$1"/*.json 2>/dev/null | wc -l | tr -d ' '; }
until [ "$(cnt ~/c8/robotlike/v2k_old)" -ge 259 ] && ! pgrep -f "[v]2k_retarget.py --episodes-json" >/dev/null; do sleep 60; done
echo "v2-K done $(date +%H:%M); launching proper v2-Kr"; mkdir -p ~/c8/robotlike/v2kr_old; cd ~/c8
for k in 0 1 2 3 4 5 6 7; do nohup ~/xvla-mac/bin/python ~/umi_bridge/track_c/v2k/v2k_retarget.py --mode v2kr --episodes-json ~/c8/robotlike/c8old_259_episodes.json --raw-root ~/ego_collector/datasets/human_handumi_raw/Hpilot --runs ~/c8/runs_m3 --export ~/c8/export --export ~/c8/export_early --out ~/c8/robotlike/v2kr_old --part $k/8 > ~/c8/robotlike/logs/v2kr_old_$k.log 2>&1 & done
sleep 60
until [ "$(cnt ~/c8/robotlike/v2kr_old)" -ge 259 ] && ! pgrep -f "[v]2k_retarget.py --mode v2kr" >/dev/null; do sleep 60; done
echo "v2-Kr done $(date +%H:%M)"; grep -h "v2kr frozen" ~/c8/robotlike/logs/v2kr_old_0.log
~/xvla-mac/bin/python ~/umi_bridge/track_c/v2k/v2k_compare.py --group old259_v2k=$HOME/c8/robotlike/v2k_old --group old259_v2kr=$HOME/c8/robotlike/v2kr_old --parity-offline ~/c8/robotlike/baseline_offline --out ~/c8/robotlike/v2k_vs_v2kr_old259.json > ~/c8/robotlike/logs/v2k_vs_v2kr.txt 2>&1
~/xvla-mac/bin/python ~/umi_bridge/track_c/v2k/v2k_tail.py ~/c8/robotlike/v2kr_old > ~/c8/robotlike/logs/v2kr_tail.txt 2>&1
echo "compare done $(date +%H:%M)"
