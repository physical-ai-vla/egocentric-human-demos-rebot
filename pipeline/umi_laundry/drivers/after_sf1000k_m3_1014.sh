#!/bin/bash
# [2026-10-14 user: "우선 1000부터 끝내자"] Soft-Fold 950k->1000k runs on 4090 GPU0 first (GPU1 = tobey X-VLA bread 2M);
# when it ends, MASt3R the remaining UMIalign sessions on GPU0 (pool/runner fixed 10-14: kf buffer 512, give-up after 3 FAILs)
export RAY_ADDRESS=http://100.64.0.1:8265
until ray job status sf-rel32bin-1000k-resume950k-gpu0 2>/dev/null | grep -qiE "succeeded|failed|stopped"; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] Soft-Fold job: $(ray job status sf-rel32bin-1000k-resume950k-gpu0 2>/dev/null | grep -oiE 'succeeded|failed|stopped' | tail -1)"
cd ~/ego_collector
for s in UMIalign_20261013_131556 UMIalign_20261013_144936 UMIalign_20261013_153819; do
  M3_GPUS="0" RUNNER=restart SKIP_CHECK=1 bash scripts/robotlike_session_check_4090.sh datasets/human_handumi_raw/UMIalign/$s
done
echo "[$(date '+%m-%d %H:%M')] MASt3R UMIalign done"
