#!/bin/bash
# [2026-10-03] the 5090b list (120 -> 61) runs DIRECTLY on the 5090 (no Ray; started 18:52:05, log rl-m3-hra-red-5090b-direct-1852.log).
# Wait for it to end (RESIDENT_DONE or its process gone), then the finalize gate (Ray jobs 5090rev + 5080b, 200/200, retry once).
L=/srv/data/johann/c8/robotlike/rl-m3-hra-red-5090b-direct-1852.log
log() { echo "[$(date '+%F %T')] $*"; }
while :; do
  st=$(ssh -o ConnectTimeout=20 gpu-5090 "grep -c RESIDENT_DONE $L; pgrep -f 'run_perframe_resident.py --list /srv/data/johann/c8/robotlike/lists/HRA_red_20261003_152502.5090b.txt' | wc -l" 2>/dev/null | tr '\n' ' ')
  set -- $st; [ "${1:-0}" -ge 1 ] && break; [ -n "${2:-}" ] && [ "${2}" = 0 ] && { log "direct 5090b process gone without RESIDENT_DONE"; break; }
  sleep 60
done
log "direct 5090b ended: $(ssh gpu-5090 "grep -c '^OK ' $L") OK, $(ssh gpu-5090 "grep -c '^FAIL ' $L") FAIL"
exec bash $HOME/c8/hra_red/finalize_mast3r.sh
