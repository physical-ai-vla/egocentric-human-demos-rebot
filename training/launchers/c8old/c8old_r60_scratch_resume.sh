#!/bin/bash
# [2026-09-28 20:5x, user] RESUME SCRATCH R60 (r60scratch600k) from its latest checkpoint (paused 2026-09-29 10:3x by the user); save_freq stays 50k.
# (template: c8old_r90_ft_resume.sh) Same env as c8old_r90_ft.sh, the C-old
# code (resume_bi_c8old.py -> c8old/xvla/train_bi.py, humanik_delta md5 asserted), the SAVED config (600k / decay 400k / seed 1000 /
# 90 ep / log_freq 200) with --resume=true -> optimizer / scheduler / RNG / step restored from 020000/training_state. Appends to the
# original log. Usage (Ray, node-pinned, entrypoint_num_gpus 0): bash c8old_r90_ft_resume.sh <gpu>
set -u
GPU=$1; B=/home/bh-aiteam/c8old; W=/home/bh-aiteam/workspace/bh_rebot_LeRobot; PY=$W/.venv/bin/python; N=r60scratch600k
OUT=$B/runs/$N; CK=$OUT/checkpoints/$(readlink $OUT/checkpoints/last); LOGF=$B/runs/$N.log
fail() { echo "R60-SCRATCH RESUME FAIL: $*"; exit 1; }
[ "$(md5sum < $B/xvla/humanik_delta.py | cut -c1-32)" = "6b63986b890ea815b3ea977e09205552" ] || fail "c8old humanik_delta md5 changed"
# [2026-09-28] code-hash freeze: the exact c8old/xvla files every C-old / scratch FT ran with (unchanged since 2026-09-26 10:52)
( cd $B/xvla && md5sum -c --quiet - ) <<'MD5' || fail "c8old/xvla code hash changed"
70ea32ae91bc028d25d0a485d7802bb3  train_bi.py
6b63986b890ea815b3ea977e09205552  humanik_delta.py
ad3de96c1fe89275a5accbad00205c82  eef_delta.py
2d49b7031a360bdfbab177002595d02d  contrastive_sampler.py
9a732c0bcd1c5e692e875fb63772758d  keep_indices.py
56c8f538e0adf25695e37ed1b90d5610  aug_defaults.py
0a50a52bcea34102122c0becf68d1a4a  lr_groups.py
186c39c81138837f2c160b486ed3529f  rebot_fk_torch.py
MD5
CKS=$(readlink $OUT/checkpoints/last); [ -n "$CKS" ] || fail "no checkpoints/last"
[ "$($PY -c "import json;print(json.load(open('$CK/training_state/training_step.json'))['step'])")" = "$((10#$CKS))" ] || fail "training_step != $CKS"
$PY -c "import json;c=json.load(open('$CK/pretrained_model/train_config.json'));assert c['steps']==600000 and c['policy']['scheduler_decay_steps']==400000 and c['seed']==1000 and len(c['dataset']['episodes'])==60 and c['policy']['pretrained_path']=='lerobot/xvla-base' and c['output_dir']=='$OUT', c" || fail "saved config mismatch"
$PY -c "import json;c=json.load(open('$CK/pretrained_model/train_config.json'));F=json.load(open('$B/r150_nested_subset_v1.json'));R30=set(json.load(open('$B/r30_episodes.json'))['episodes']);e=c['dataset']['episodes'];assert e==F['ladder']['R60']['episodes'] and R30<=set(e), 'R60 list / R30 contract'" || fail "R60 episode list or R30 contract mismatch in the saved config"
M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU) || fail "no GPU $GPU"; [ "$M" -lt 1500 ] || fail "GPU $GPU busy (${M} MiB)"
export HF_HOME=/home/bh-aiteam/.cache/huggingface HF_HUB_OFFLINE=1 HUMANIK_DELTA=1 HUMANIK_LEAD=5 HUMANIK_ROBOT=1 HUMANIK_TARGET=cmd
export XVLA_EE_AUX=1 EE_AUX_SOURCE=state EE_AUX_SCALE=10 EE_AUX_LAMBDA=2.0 EE_FK_LAMBDA=20 EE_LOG_EVERY=200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XVLA_TF32=1 XVLA_FUSED_ADAM=1 EEF_TARGET_SOURCE=legacy CUDA_VISIBLE_DEVICES=$GPU
echo "===== RESUME from $CKS (save_freq 50000) $(date '+%F %T') gpu=$GPU =====" >> $LOGF
$PY $B/xvla/resume_bi_c8old.py --config_path=$CK/pretrained_model/train_config.json --resume=true --save_freq=50000 >> $LOGF 2>&1 &
P=$!
( while kill -0 $P 2>/dev/null; do
    T=$(tail -c 60000 $LOGF | tr '\r' '\n'); S=$(echo "$T" | grep -a 'step:' | grep -a 'loss:' | tail -1 | sed 's/.*step:/step:/' | awk '{print $1, $5, $6, $7}')
    E=$(echo "$T" | grep -a 'batch [0-9]*: FK' | tail -1 | grep -oE 'L p50 [0-9.]+|R p50 [0-9.]+|aux_loss [0-9.e+-]+|fk_loss [0-9.e+-]+' | tr '\n' ' ')
    R=$(echo "$T" | grep -aoE '[0-9.]+step/s' | tail -1 | tr -dc '0-9.'); G=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $GPU)
    echo "$(date '+%F %T') | $S | $E(FK mm) | ${R} step/s $(awk "BEGIN{print ${R:-0}*4}") samples/s | gpu $G" >> $B/runs/$N.status.log; sleep 60; done ) &
wait $P; rc=$?; echo "R60-SCRATCH RESUME train exited rc $rc"; grep -qi "out of memory" $LOGF && fail "cuda oom"
[ $rc -eq 0 ] && touch $B/runs/$N.done; exit $rc
