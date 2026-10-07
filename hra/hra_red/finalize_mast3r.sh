#!/bin/bash
# [2026-10-03 user] HRA_red MASt3R split over 5080 + 5090: wait for ALL jobs, verify 200/200 complete outputs, duplicates, missing
# (one retry on the 5090), then pull to the Mac with the session script (todo empty -> pull only), which writes the
# "SKIP_CHECK=1: poses only" line the export chain waits for.  The chain therefore cannot start on a partial set.
set -u
export RAY_ADDRESS=http://100.64.0.1:8265; R=/srv/data/johann/c8/robotlike
log() { echo "[$(date '+%F %T')] $*"; }
term() { ray job status $1 2>&1 | grep -oiE "succeeded|failed|stopped" | tail -1; }
for j in $(cat $HOME/c8/hra_red/mast3r_jobs.txt); do until [ -n "$(term $j)" ]; do sleep 60; done; log "$j: $(term $j) ($(ray job logs $j 2>&1 | grep -c '^OK ') OK, $(ray job logs $j 2>&1 | grep -c '^FAIL ') FAIL)"; done
check() {
  ssh gpu-5090 "O=$R/runs_resident; L=$R/lists; : > \$L/HRA_missing.txt; n=0
    for t in \$(cat \$L/HRA_red_20261003_152502.txt); do if [ -f \$O/\$t/ss1.csv ] && [ -f \$O/\$t/ss1.json ] && [ -f \$O/\$t.vram ]; then n=\$((n+1)); else echo \$t >> \$L/HRA_missing.txt; fi; done
    echo \$n \$(wc -l < \$L/HRA_missing.txt)"
}
read n miss <<< "$(check)"; log "complete $n / 200, missing $miss"
dups=$(for j in $(cat $HOME/c8/hra_red/mast3r_jobs.txt); do ray job logs $j 2>&1 | grep '^OK ' | awk '{print $2}'; done | sort | uniq -d | wc -l | tr -d ' ')
log "episodes processed by more than one job: $dups"
if [ "$miss" -gt 0 ]; then
  S=rl-m3-hra-red-5090retry-$(date +%H%M%S); log "retrying $miss on the 5090 as $S"
  ray job submit --submission-id $S --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.5": 0.01}' --no-wait -- bash -c "ls $R/in/ > /dev/null; bash $R/robotlike_resident_5090.sh $R/lists/HRA_missing.txt $R/runs_resident $R/in" > /dev/null 2>&1
  until [ -n "$(term $S)" ]; do sleep 60; done; read n miss <<< "$(check)"; log "after retry: complete $n / 200, missing $miss"
fi
[ "$n" = 200 ] || { log "NOT all 200 complete -> chain NOT released (missing list: $R/lists/HRA_missing.txt)"; exit 1; }
log "200/200 complete -> pull via the session script"
cd $HOME/ego_collector && SIDES=right SKIP_CHECK=1 bash scripts/robotlike_session_check.sh datasets/human_handumi_raw/HRA_red/HRA_red_20261003_152502 >> $HOME/c8/robotlike/logs/HRA_red_152502_full.log 2>&1
log "pulled: $(ls $HOME/c8/robotlike/runs | grep -c '^red_20261003_152502_.*_right$') local run dirs; chain released: $(grep -c 'SKIP_CHECK=1: poses only' $HOME/c8/robotlike/logs/HRA_red_152502_full.log)"
