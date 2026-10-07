#!/bin/bash
# [2026-09-28, user] RESUME C-old R90 (r90ft600k) from checkpoint 020000 after the 13:46 pause. Same env as c8old_r90_ft.sh, the C-old
# code (resume_bi_c8old.py -> c8old/xvla/train_bi.py, humanik_delta md5 asserted), the SAVED config (600k / decay 400k / seed 1000 /
# 90 ep / log_freq 200) with --resume=true -> optimizer / scheduler / RNG / step restored from 020000/training_state. Appends to the
# original log. Usage (Ray, node-pinned, entrypoint_num_gpus 0): bash c8old_r90_ft_resume.sh <gpu>
set -u
GPU=$1; B=/home/bh-aiteam/c8old; W=/home/bh-aiteam/workspace/bh_rebot_LeRobot; PY=$W/.venv/bin/python; N=r90ft600k
OUT=$B/runs/$N; CK=$OUT/checkpoints/020000; LOGF=$B/runs/$N.log
fail() { echo "R90 RESUME FAIL: $*"; exit 1; }
[ "$(md5sum < $B/xvla/humanik_delta.py | cut -c1-32)" = "6b63986b890ea815b3ea977e09205552" ] || fail "c8old humanik_delta md5 changed"
[ "$(readlink $OUT/checkpoints/last)" = "020000" ] || fail "checkpoints/last is not 020000"
[ "$($PY -c "import json;print(json.load(open('$CK/training_state/training_step.json'))['step'])")" = "20000" ] || fail "training_step != 20000"
$PY -c "import json;c=json.load(open('$CK/pretrained_model/train_config.json'));assert c['steps']==600000 and c['policy']['scheduler_decay_steps']==400000 and c['seed']==1000 and len(c['dataset']['episodes'])==90 and c['policy']['pretrained_path'].endswith('ft_init_pretrain300k') and c['output_dir']=='$OUT', c" || fail "saved config mismatch"
M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU) || fail "no GPU $GPU"; [ "$M" -lt 1500 ] || fail "GPU $GPU busy (${M} MiB)"
export HF_HOME=/home/bh-aiteam/.cache/huggingface HF_HUB_OFFLINE=1 HUMANIK_DELTA=1 HUMANIK_LEAD=5 HUMANIK_ROBOT=1 HUMANIK_TARGET=cmd
export XVLA_EE_AUX=1 EE_AUX_SOURCE=state EE_AUX_SCALE=10 EE_AUX_LAMBDA=2.0 EE_FK_LAMBDA=20 EE_LOG_EVERY=200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XVLA_TF32=1 XVLA_FUSED_ADAM=1 EEF_TARGET_SOURCE=legacy CUDA_VISIBLE_DEVICES=$GPU
echo "===== RESUME from 020000 $(date '+%F %T') gpu=$GPU =====" >> $LOGF
$PY $B/xvla/resume_bi_c8old.py --config_path=$CK/pretrained_model/train_config.json --resume=true >> $LOGF 2>&1 &
P=$!
( while kill -0 $P 2>/dev/null; do
    T=$(tail -c 60000 $LOGF | tr '\r' '\n'); S=$(echo "$T" | grep -a 'step:' | grep -a 'loss:' | tail -1 | sed 's/.*step:/step:/' | awk '{print $1, $5, $6, $7}')
    E=$(echo "$T" | grep -a 'batch [0-9]*: FK' | tail -1 | grep -oE 'L p50 [0-9.]+|R p50 [0-9.]+|aux_loss [0-9.e+-]+|fk_loss [0-9.e+-]+' | tr '\n' ' ')
    R=$(echo "$T" | grep -aoE '[0-9.]+step/s' | tail -1 | tr -dc '0-9.'); G=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $GPU)
    echo "$(date '+%F %T') | $S | $E(FK mm) | ${R} step/s $(awk "BEGIN{print ${R:-0}*4}") samples/s | gpu $G" >> $B/runs/$N.status.log; sleep 60; done ) &
wait $P; rc=$?; echo "R90 RESUME train exited rc $rc"; grep -qi "out of memory" $LOGF && fail "cuda oom"
[ $rc -eq 0 ] && touch $B/runs/$N.done; exit $rc
