#!/bin/bash
# [2026-10-03] the queued 5090b job was failed by Ray's 900 s job-start timeout while the 5090 was busy with 5090rev.
# Submit the same disjoint list (120 -> 61) the moment 5090rev ends, add it to the job list, then run the finalize gate.
set -u
export RAY_ADDRESS=http://100.64.0.1:8265; R=/srv/data/johann/c8/robotlike
log() { echo "[$(date '+%F %T')] $*"; }
first=$(awk '{print $1}' $HOME/c8/hra_red/mast3r_jobs.txt)
until ray job status $first 2>&1 | grep -qiE "succeeded|failed|stopped"; do sleep 30; done
log "$first ended -> resubmitting the 5090b list"
S=rl-m3-hra-red-5090b2-$(date +%H%M%S)
ray job submit --submission-id $S --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.5": 0.01}' --no-wait -- bash -c "ls $R/in/ > /dev/null; bash $R/robotlike_resident_5090.sh $R/lists/HRA_red_20261003_152502.5090b.txt $R/runs_resident $R/in" 2>&1 | grep -E "submitted|rror"
# job list for the finalize gate: drop the failed 5090b, add the resubmission
grep -v . /dev/null; echo "$(awk '{print $1, $2}' $HOME/c8/hra_red/mast3r_jobs.txt) $S" > $HOME/c8/hra_red/mast3r_jobs.txt; log "jobs: $(cat $HOME/c8/hra_red/mast3r_jobs.txt)"
exec bash $HOME/c8/hra_red/finalize_mast3r.sh
