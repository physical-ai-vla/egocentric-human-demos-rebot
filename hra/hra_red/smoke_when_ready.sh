#!/bin/bash
# [2026-10-03 user] after the HRA chain: QC gate -> upload train split to the 4090 -> wait for a FREE 4090 GPU (never stop a run)
# -> 300-step smoke of train_hra_rightonly_lossmask_d20_b8.sh (D20, bs 8, preflight 3) -> report.  The main run is NOT started.
set -u
export RAY_ADDRESS=http://100.64.0.1:8265; N=bh-aiteam@gpu-4090; OUT=$HOME/c8/hra_red; DS=ego_hra_red_rightonly_v1_train
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for the chain"; until grep -q "CHAIN DONE" $OUT/chain.log; do sleep 120; done
python3 - <<'PY' || { log "QC GATE FAILED -> no upload, no smoke (see scale_qc_summary.json)"; exit 1; }
import json, pathlib
q = json.load(open(pathlib.Path.home() / "c8/hra_red/scale_qc_summary.json")); fails = []
if q.get("exported_ok", 0) < 160: fails.append(f"exported_ok {q.get('exported_ok')} < 160")
if not q.get("centre_resid_cm_p50") or q["centre_resid_cm_p50"] >= 1.0: fails.append(f"centre resid p50 {q.get('centre_resid_cm_p50')} cm >= 1")
s = q.get("s_pnp_p5_p50_p95") or [0, 0, 0]
if not s[0] or s[2] / s[0] > 1.6: fails.append(f"s_pnp p95/p5 {s} not one cluster (> 1.6)")
a = q.get("action_0p8s_right_translation_cm_p50_p95") or [0, 0]
if not (5 <= a[1] <= 30): fails.append(f"0.8 s action p95 {a[1]} cm outside 5-30 (ego v2 p95 14.6)")
h = q.get("home_to_cube_tcp_net_m_p5_p50_p95") or [0, 0, 0]
if not (0.35 <= h[1] <= 0.8) or h[2] - h[0] > 0.35: fails.append(f"HOME->cube net {h} m (want p50 0.35-0.8, p5-p95 width <= 0.35)")
print("QC GATE", "PASS" if not fails else "FAIL", fails, flush=True); raise SystemExit(1 if fails else 0)
PY
log "QC gate PASS -> upload"
L=$OUT/lerobot/$DS; ( cd $L && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS )
/opt/homebrew/bin/rsync -a $L/ $N:/home/bh-aiteam/holobrain-data/lerobot/$DS/ || { log "upload failed"; exit 2; }
ssh $N "cd /home/bh-aiteam/holobrain-data/lerobot/$DS && sha256sum -c --quiet SHA256SUMS" && log "upload verified" || { log "remote sha check failed"; exit 2; }
log "waiting for a free 4090 GPU (< 1 GB on 3 checks in a row)"; ok=0; G=
while [ $ok -lt 3 ]; do
  for g in 0 1; do u=$(ssh -o ConnectTimeout=20 $N "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g" 2>/dev/null | tr -dc 0-9)
    if [ -n "$u" ] && [ "$u" -lt 1000 ]; then [ "$G" = "$g" ] && ok=$((ok+1)) || { G=$g; ok=1; }; break; fi; done
  [ -z "${u:-}" ] || [ "$u" -ge 1000 ] && ok=0; sleep 120
done
log "4090 GPU $G free -> smoke"
SM=hra-rightonly-smoke-$(date +%H%M)
ray job submit --no-wait --submission-id $SM --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' -- bash -c \
  "ACCELERATE_MIXED_PRECISION=bf16 RUN_NAME=HRA-RIGHTONLY-LOSSMASK-D20-SMOKE STEPS=300 DECAY_STEPS=300 SAVE_FREQ=100 MIN_START_GB=60 RELONLY_PREFLIGHT=3 bash /home/bh-aiteam/train_hra_rightonly_lossmask_d20_b8.sh HRA $G --log_freq=10" 2>&1 | grep -E "submitted|rror"
for i in $(seq 1 180); do st=$(ray job status $SM 2>&1 | grep -oiE "succeeded|failed|stopped" | tr a-z A-Z | head -1); [ -n "$st" ] && break; sleep 20; done
ray job logs $SM 2>&1 | tr '\r' '\n' > $OUT/smoke_$SM.log
grep -aE "\[domain\]|\[data\]|lossmask-preflight|relonly-preflight|\[relonly-lossmask\] batch|step:|Checkpoint|End of training|Traceback|Error|nan" $OUT/smoke_$SM.log | head -60 | cut -c1-220
R=/home/bh-aiteam/holobrain-data/trainB/HRA-RIGHTONLY-LOSSMASK-D20-SMOKE
ssh $N "ls $R/checkpoints/; /home/bh-aiteam/miniforge3/envs/holobrain/bin/python -c \"from safetensors.torch import load_file;import glob;f=sorted(glob.glob('$R/checkpoints/000300/pretrained_model/*.safetensors'))[0];sd=load_file(f);import torch;print('ckpt load OK', f, len(sd), 'tensors, finite', all(torch.isfinite(v.float()).all().item() for v in sd.values()))\""
log "smoke status $st (log $OUT/smoke_$SM.log). Main run NOT started."
