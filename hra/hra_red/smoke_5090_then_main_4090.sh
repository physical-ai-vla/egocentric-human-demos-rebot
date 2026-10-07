#!/bin/bash
# [2026-10-03 user] after the HRA chain: QC gate -> upload to the 5090 -> wait until the 5090 is FREE (MASt3R done; never stop a job)
# -> 300-step smoke (train_hra_rightonly_5090_b8.sh: D20, bs 8, AdamW8bit, bf16, loss-mask, preflight 6) -> automatic contract
# verdict -> ONLY if every check passes: upload to the 4090, wait for a FREE 4090 GPU (never stop a run), then the MAIN run
# HRA-RIGHTONLY-LOSSMASK-D20-B8-300K on the 4090 (train_hra_rightonly_lossmask_d20_b8.sh, steps = decay = 300k, save 5k)
# + the Mac-side SSD archiver.  (user: "학습 준비되면 ... 300k step" then "본학습은 4090에서 하자")
set -u
export RAY_ADDRESS=http://100.64.0.1:8265; H5=gpu-5090; R=/srv/data/johann/relonly; OUT=$HOME/c8/hra_red; DS=ego_hra_red_rightonly_v1_train
L5=$R/code/train_hra_rightonly_5090_b8.sh
log() { echo "[$(date '+%F %T')] $*"; }
sub() { ray job submit --no-wait --submission-id $1 --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.5": 0.001}' -- bash -c "$2" 2>&1 | grep -E "submitted|rror"; }
log "waiting for the chain"; until grep -q "CHAIN DONE" $OUT/chain.log; do sleep 120; done
python3 - <<'PY' || { log "QC GATE FAILED -> no upload, no smoke, no main (see scale_qc_summary.json)"; exit 1; }
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
log "QC gate PASS -> upload to the 5090"
LD=$OUT/lerobot/$DS; ( cd $LD && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS )
/opt/homebrew/bin/rsync -a $LD/ $H5:$R/data/$DS/ || { log "upload failed"; exit 2; }
ssh $H5 "cd $R/data/$DS && sha256sum -c --quiet SHA256SUMS" && log "upload verified" || { log "remote sha check failed"; exit 2; }
log "waiting for the 5090 to be free (< 1 GB on 3 checks in a row)"; ok=0
while [ $ok -lt 3 ]; do u=$(ssh -o ConnectTimeout=20 $H5 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | tr -dc 0-9)
  if [ -n "$u" ] && [ "$u" -lt 1000 ]; then ok=$((ok+1)); else ok=0; fi; sleep 60; done
SMR=HRA-RIGHTONLY-SMOKE-D20-5090; SM=hra-rightonly-smoke-5090-$(date +%H%M)
ssh $H5 "rm -rf $R/runs/$SMR $R/runs/$SMR.log"
log "5090 free -> smoke $SM"
sub $SM "ACCELERATE_MIXED_PRECISION=bf16 RUN_NAME=$SMR STEPS=300 DECAY_STEPS=300 SAVE_FREQ=300 MIN_START_GB=200 RELONLY_PREFLIGHT=6 bash $L5 --log_freq=10"
st=; for i in $(seq 1 240); do st=$(ray job status $SM 2>&1 | grep -oiE "succeeded|failed|stopped" | tr a-z A-Z | head -1); [ -n "$st" ] && break; sleep 15; done
ray job logs $SM 2>&1 | tr '\r' '\n' > $OUT/smoke_5090.log
grep -aE "\[domain\]|\[data\]|\[8bit\]|lossmask-preflight|relonly-preflight|step:|End of training|Traceback|Error" $OUT/smoke_5090.log | head -50 | cut -c1-230
CK=$(ssh $H5 "$R/venv/bin/python -c \"import glob,torch;from safetensors.torch import load_file;f=sorted(glob.glob('$R/runs/$SMR/checkpoints/000300/pretrained_model/*.safetensors'));sd=load_file(f[0]);print('CKPT_OK' if all(torch.isfinite(v.float()).all().item() for v in sd.values()) else 'CKPT_NONFINITE', len(sd))\"" 2>&1 | tail -1)
VERDICT=$(python3 - "$OUT/smoke_5090.log" "$st" "$CK" <<'PY'
import re, sys, statistics as S
log, st, ck = open(sys.argv[1], errors="ignore").read(), sys.argv[2], sys.argv[3]; f = []
pre = [l for l in log.splitlines() if "preflight]" in l]
if not pre: f.append("no preflight lines")
if any("FAIL" in l for l in pre): f.append("preflight FAIL: " + next(l for l in pre if "FAIL" in l)[:160])
if not any("masked dims" in l and "PASS" in l for l in pre): f.append("no masked-dims grad PASS line")
if not any("supervised_dims_per_sample" in l for l in pre): f.append("no effective mask-sum line")
loss = [float(x) for x in re.findall(r"\bloss:\s*([0-9.eE+-]+|nan|inf)", log)]
if re.search(r"loss:\s*(nan|inf)|rel (nan|inf)", log, re.I): f.append("non-finite loss")
if len(loss) < 10: f.append(f"only {len(loss)} logged losses")
elif not S.median(loss[-5:]) < S.median(loss[:5]): f.append(f"loss did not fall: first5 {S.median(loss[:5]):.4f} last5 {S.median(loss[-5:]):.4f}")
if "End of training" not in log: f.append("no End of training")
if st != "SUCCEEDED": f.append(f"job {st}")
if not ck.startswith("CKPT_OK"): f.append(f"checkpoint: {ck}")
print("PASS" if not f else "FAIL " + " | ".join(f))
if loss: print(f"losses first5 {loss[:5]} last5 {loss[-5:]}", file=sys.stderr)
PY
)
log "SMOKE VERDICT: $VERDICT (job $st, $CK)"
case "$VERDICT" in PASS*) ;; *) log "smoke did not pass -> main run NOT started"; exit 1;; esac
ssh $H5 "rm -rf $R/runs/$SMR"
N4=bh-aiteam@gpu-4090; MRUN=HRA-RIGHTONLY-LOSSMASK-D20-B8-300K
log "smoke PASS -> upload to the 4090"
/opt/homebrew/bin/rsync -a $LD/ $N4:/home/bh-aiteam/holobrain-data/lerobot/$DS/ || { log "4090 upload failed"; exit 2; }
ssh $N4 "cd /home/bh-aiteam/holobrain-data/lerobot/$DS && sha256sum -c --quiet SHA256SUMS" && log "4090 upload verified" || { log "4090 sha check failed"; exit 2; }
log "waiting for a free 4090 GPU (< 1 GB on 3 checks in a row; running jobs are never stopped)"; ok=0; G=
while [ $ok -lt 3 ]; do
  fr=
  for g in 0 1; do u=$(ssh -o ConnectTimeout=20 $N4 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g" 2>/dev/null | tr -dc 0-9)
    if [ -n "$u" ] && [ "$u" -lt 1000 ]; then fr=$g; break; fi; done
  if [ -n "$fr" ]; then [ "$G" = "$fr" ] && ok=$((ok+1)) || { G=$fr; ok=1; }; else ok=0; G=; fi
  sleep 120
done
MAIN=hra-rightonly-d20-b8-300k-4090-$(date +%H%M)
log "4090 GPU $G free -> main $MAIN"
ray job submit --no-wait --submission-id $MAIN --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' -- bash -c \
  "ACCELERATE_MIXED_PRECISION=bf16 RUN_NAME=$MRUN STEPS=300000 DECAY_STEPS=300000 SAVE_FREQ=5000 MIN_START_GB=60 RELONLY_PREFLIGHT=3 bash /home/bh-aiteam/train_hra_rightonly_lossmask_d20_b8.sh HRA $G" 2>&1 | grep -E "submitted|rror"
s=; for i in $(seq 1 60); do s=$(ray job logs $MAIN 2>&1 | tr '\r' '\n'); echo "$s" | grep -qE "ot_train.py:[0-9]+ step:|Traceback|refusing|gate:|mismatch" && break; sleep 20; done
echo "$s" | grep -aE "\[domain\]|\[data\]|\[precision\]|8bit\]|lossmask-preflight|relonly-preflight|cfg.steps|Auto-scal|step:|refusing|gate|mismatch|Traceback" | head -16 | cut -c1-200
cd ~; IDLE_EXIT=1000000 nohup bash ~/umi_bridge/trackb_archive_v2_relonly.sh $MRUN >> ~/archive_$MRUN.log 2>&1 < /dev/null & disown
log "main run submitted on the 4090: $MAIN (GPU $G); archiver -> ~/archive_$MRUN.log; UI: ~/umi_bridge/load_hra_rightonly_ckpt_to_ui.sh <step> 8056"
