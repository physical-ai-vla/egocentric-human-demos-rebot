#!/bin/bash
# [2026-09-28] C-old B1-old recipe on the 5090 mirror env. IDENTICAL to c8old_chain.sh train()/common_env except paths
# (+ VRAM sampling, GPU guard, unbuffered log). Usage: c8old5090_train.sh <run> <repo_id> <dataset_root> <steps> <decay> <save_freq> [extra args]
# STOP_AT=<step> (RESUMED_AT=<ckpt step> when resuming; log lines print 1K for 1000-1999, so the stop counts log lines): keep --steps as configured (LeRobot rescales warmup/decay when steps < decay) and stop after step <STOP_AT> is logged
#   (a checkpoint step is waited for when STOP_AT is a multiple of save_freq).
# RESUME_FROM=<ckpt dir>: resume that checkpoint (optimizer / scheduler / rng) via resume_bi_c8old.py (other positional args ignored except <run>).
# GUARD (no auto-restart, ever): any NEW compute PID on the GPU that is not this run -> stop, exit 6; free VRAM < MIN_FREE_MIB (512) -> stop, exit 7;
#   CUDA OOM -> exit 4; NaN/Inf -> exit 3. Env: LOGF (100), EE_LOG_EVERY (1000), EEF_TARGET_SOURCE (measured), MIN_FREE_MIB.
E=/srv/data/johann/c8old5090; PY=$E/venv/bin/python; X=$E/xvla; N=$1 REPO=$2 ROOT=$3 STEPS=$4 DEC=$5 SAVE=$6; shift 6
export HF_HOME=$E/hf HF_HUB_OFFLINE=1 HUMANIK_DELTA=1 HUMANIK_LEAD=5 HUMANIK_ROBOT=1 HUMANIK_TARGET=cmd
export XVLA_EE_AUX=1 EE_AUX_SOURCE=state EE_AUX_SCALE=10 EE_AUX_LAMBDA=2.0 EE_FK_LAMBDA=20 EE_LOG_EVERY=${EE_LOG_EVERY:-1000}
# rebot_fk_torch.py hard-codes /home/bh-aiteam/rebot_ee (sys.path insert + REBOT_URDF setdefault): point both at the hash-identical copy
export PYTHONPATH=$E/rebot_ee REBOT_URDF=$E/rebot_ee/reBot_B601_DM_dualarm.urdf PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XVLA_TF32=1 XVLA_FUSED_ADAM=1 EEF_TARGET_SOURCE=${EEF_TARGET_SOURCE:-measured}
MIN_FREE=${MIN_FREE_MIB:-512}; L=$E/runs/$N.log
mkdir -p $E/runs; [ -e $L ] && { echo "log $L exists -> refuse"; exit 9; }
echo "GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"; df -h /srv/data | tail -1
PRE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u | tr '\n' ' '); echo "pre-existing compute PIDs (allowed): $PRE"
( while :; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader >> $E/runs/$N.vram.csv; sleep 10; done ) & VP=$!
if [ -n "$RESUME_FROM" ]; then
  $PY $X/resume_bi_c8old.py --config_path=$RESUME_FROM/pretrained_model/train_config.json --resume=true > $L 2>&1 &
else
  $PY $X/train_bi.py --dataset.repo_id=$REPO --dataset.root=$ROOT --policy.path=lerobot/xvla-base --policy.dtype=float32 \
    --output_dir=$E/runs/$N --job_name=$N --steps=$STEPS --save_freq=$SAVE --log_freq=${LOGF:-100} --policy.scheduler_decay_steps=$DEC --seed=1000 \
    --batch_size=4 --num_workers=8 --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false \
    --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
    --policy.normalization_mapping='{"STATE":"IDENTITY","ACTION":"IDENTITY","VISUAL":"IDENTITY"}' \
    --rename_map '{"observation.images.global":"observation.images.image","observation.images.left_wrist":"observation.images.image2","observation.images.right_wrist":"observation.images.image3"}' \
    "$@" > $L 2>&1 &
fi
P=$!
mine() { local p=$1; while [ -n "$p" ] && [ "$p" -gt 1 ]; do [ "$p" = "$P" ] && return 0; p=$(awk '{print $4}' /proc/$p/stat 2>/dev/null); done; return 1; }
stop() { kill $P 2>/dev/null; sleep 10; kill -9 $P 2>/dev/null; pkill -9 -P $P 2>/dev/null; kill $VP 2>/dev/null; }
while kill -0 $P 2>/dev/null; do
  if tail -c 20000 $L | grep -aE "step:[0-9]+ " | tail -3 | grep -qiE "loss:(nan|inf)|grdn:(nan|inf)"; then echo "NAN/INF in $N -> stop"; stop; exit 3; fi
  for q in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
    case " $PRE " in *" $q "*) continue;; esac
    mine $q || { echo "GUARD_FAIL foreign compute PID $q ($(ps -o user=,args= -p $q | cut -c1-120)) at $(date +%T) -> stop, no restart"; stop; exit 6; }
  done
  fr=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{print $1-$2}')
  [ "$fr" -lt "$MIN_FREE" ] && { echo "GUARD_FAIL free VRAM ${fr} MiB < ${MIN_FREE} at $(date +%T) -> stop, no restart"; stop; exit 7; }
  if [ -n "$STOP_AT" ] && [ "$(grep -acE "step:[0-9.]+K? smpl" $L)" -ge "$(( (STOP_AT - ${RESUMED_AT:-0}) / ${LOGF:-100} ))" ] && { [ $((STOP_AT % SAVE)) -ne 0 ] || grep -aq "Checkpoint policy after step $STOP_AT\b" $L; }; then
    sleep 20; echo "STOP_AT $STOP_AT reached -> stop"; stop
    echo "stopped at $STOP_AT; peak VRAM total $(awk -F, '{print $2}' $E/runs/$N.vram.csv | sort -n | tail -1)"; exit 0; fi
  sleep 15
done
wait $P; rc=$?; kill $VP 2>/dev/null
grep -qi "out of memory" $L && { echo "CUDA OOM -> FAIL (no restart)"; exit 4; }
echo "train rc $rc; peak VRAM total $(awk -F, '{print $2}' $E/runs/$N.vram.csv | sort -n | tail -1)"; exit $rc
