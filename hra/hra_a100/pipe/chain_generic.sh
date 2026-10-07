#!/bin/bash
# [2026-10-07] GENERIC (DS / LD / SMR / MRUN env) copy of the ROBOTCAM variant of ~/c8/hra_red/smoke_then_main_4090.sh (untouched): dataset ego_hra_red_rightonly_robotcam_v2_train
# (labels in ROBOT TCP via hand-eye M, right wrist rendered by the MEASURED robot C922). Same launcher / recipe; HRA_DS override.
# Waits for the Mac export (pid file) instead of the MASt3R chain; no scale-QC gate (same episodes / rows as v1, already gated).
# GPU: the 4090 GPU freed by stopping R30-RELCART20-RELONLY-D20-B8-600K (user 2026-10-06).
export RAY_ADDRESS=http://100.64.0.1:8265; N4=bh-aiteam@gpu-4090; OUT=$HOME/c8/hra_a100; DS=${DS:?}; LD=${LD:?}
L4=${L4:-/home/bh-aiteam/train_hra_rightonly_lossmask_d20_b8.sh}; TB=/home/bh-aiteam/holobrain-data/trainB; PY4=/home/bh-aiteam/miniforge3/envs/holobrain/bin/python
SMR=${SMR:?}; MRUN=${MRUN:?}
log() { echo "[$(date '+%F %T')] $*"; }
sub() { ray job submit --no-wait --submission-id $1 --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' -- bash -c "$2" 2>&1 | grep -E "submitted|rror"; }
python3 - "$LD" <<'PY' || { log "EXPORT check failed"; exit 1; }
import json, pathlib, sys
f = pathlib.Path(sys.argv[1]) / "EXPORT.json"; e = json.load(open(f))
e["variant"] = e.get("variant") or "robotcam"; e["schema"] = "ego_cart20_right_only_lerobot/v1"; json.dump(e, open(f, "w"), indent=1)
print("export", e["report"]["episodes"], "episodes", e["report"]["rows"], "rows")
PY
log "export OK -> upload to the 4090"
LD=${LD:?}; ( cd $LD && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS )
/opt/homebrew/bin/rsync -a $LD/ $N4:/home/bh-aiteam/holobrain-data/lerobot/$DS/ || { log "upload failed"; exit 2; }
ssh $N4 "cd /home/bh-aiteam/holobrain-data/lerobot/$DS && sha256sum -c --quiet SHA256SUMS" && log "upload verified" || { log "remote sha check failed"; exit 2; }
log "waiting for a free 4090 GPU (< 1 GB on 3 checks in a row)"; ok=0; G=
while [ $ok -lt 3 ]; do
  fr=
  for g in 0 1; do u=$(ssh -o ConnectTimeout=20 $N4 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g" 2>/dev/null | tr -dc 0-9)
    if [ -n "$u" ] && [ "$u" -lt 1000 ]; then fr=$g; break; fi; done
  if [ -n "$fr" ]; then [ "$G" = "$fr" ] && ok=$((ok+1)) || { G=$fr; ok=1; }; else ok=0; G=; fi
  sleep 120
done
ssh $N4 "rm -rf $TB/$SMR $TB/$SMR.log"
SM=hra-a100-smoke-4090-$(date +%H%M); log "4090 GPU $G free -> smoke $SM"
sub $SM "ACCELERATE_MIXED_PRECISION=bf16 RUN_NAME=$SMR STEPS=300 DECAY_STEPS=300 SAVE_FREQ=300 MIN_START_GB=60 RELONLY_PREFLIGHT=6 HRA_DS=$DS bash $L4 HRA $G --log_freq=10"
st=; for i in $(seq 1 360); do st=$(ray job status $SM 2>&1 | grep -oiE "succeeded|failed|stopped" | tr a-z A-Z | head -1); [ -n "$st" ] && break; sleep 15; done
ray job logs $SM 2>&1 | tr '\r' '\n' > $OUT/pipe/smoke_$SMR.log
grep -aE "\[domain\]|\[data\]|\[8bit\]|lossmask-preflight|relonly-preflight|step:|End of training|Traceback|Error" $OUT/pipe/smoke_$SMR.log | head -50 | cut -c1-230
CK=$(ssh $N4 "$PY4 -c \"import glob,torch;from safetensors.torch import load_file;f=sorted(glob.glob('$TB/$SMR/checkpoints/000300/pretrained_model/*.safetensors'));sd=load_file(f[0]);print('CKPT_OK' if all(torch.isfinite(v.float()).all().item() for v in sd.values()) else 'CKPT_NONFINITE', len(sd))\"" 2>&1 | tail -1)
VERDICT=$(python3 - "$OUT/pipe/smoke_$SMR.log" "$st" "$CK" <<'PY'
import re, sys, statistics as S
log, st, ck = open(sys.argv[1], errors="ignore").read(), sys.argv[2], sys.argv[3]; f = []
pre = [l for l in log.splitlines() if "preflight]" in l]
if not pre: f.append("no preflight lines")
if any("FAIL" in l for l in pre): f.append("preflight FAIL: " + next(l for l in pre if "FAIL" in l)[:160])
if not any("masked dims" in l and "PASS" in l for l in pre): f.append("no masked-dims grad PASS line")
if not any("supervised_dims_per_sample" in l for l in pre): f.append("no effective mask-sum line")
loss = [float(x) for x in re.findall(r"\bloss:\s*([0-9.eE+-]+)", log)]
if re.search(r"loss:\s*(nan|inf)|rel (nan|inf)", log, re.I): f.append("non-finite loss")
if len(loss) < 10: f.append(f"only {len(loss)} logged losses")
elif not S.median(loss[-5:]) < S.median(loss[:5]): f.append(f"loss did not fall: first5 {S.median(loss[:5]):.4f} last5 {S.median(loss[-5:]):.4f}")
if "End of training" not in log: f.append("no End of training")
if st != "SUCCEEDED": f.append(f"job {st}")
if not ck.startswith("CKPT_OK"): f.append(f"checkpoint: {ck}")
print("PASS" if not f else "FAIL " + " | ".join(f))
PY
)
log "SMOKE VERDICT (4090 GPU $G): $VERDICT (job $st, $CK)"
case "$VERDICT" in PASS*) ;; *) log "smoke did not pass -> main run NOT started"; exit 1;; esac
ssh $N4 "rm -rf $TB/$SMR"
u=$(ssh $N4 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G" | tr -dc 0-9)
[ -n "$u" ] && [ "$u" -lt 1000 ] || { log "GPU $G got taken during the smoke (${u} MiB) -> main NOT started; rerun this script"; exit 1; }
MAIN=hra-a100-main-4090-$(date +%H%M); log "starting main $MAIN on 4090 GPU $G"
sub $MAIN "ACCELERATE_MIXED_PRECISION=bf16 RUN_NAME=$MRUN STEPS=300000 DECAY_STEPS=300000 SAVE_FREQ=5000 MIN_START_GB=60 RELONLY_PREFLIGHT=3 HRA_DS=$DS bash $L4 HRA $G"
s=; for i in $(seq 1 60); do s=$(ray job logs $MAIN 2>&1 | tr '\r' '\n'); echo "$s" | grep -qE "ot_train.py:[0-9]+ step:|Traceback|refusing|gate:|mismatch" && break; sleep 20; done
echo "$s" | grep -aE "\[domain\]|\[data\]|\[precision\]|8bit\]|lossmask-preflight|relonly-preflight|cfg.steps|Auto-scal|step:|refusing|gate|mismatch|Traceback" | head -16 | cut -c1-200
cd ~; IDLE_EXIT=1000000 nohup bash ~/umi_bridge/trackb_archive_v2_relonly.sh $MRUN >> ~/archive_$MRUN.log 2>&1 < /dev/null & disown
log "main run submitted on the 4090: $MAIN (GPU $G); archiver -> ~/archive_$MRUN.log; UI: ~/umi_bridge/load_hra_rightonly_ckpt_to_ui.sh <step> 8056"
