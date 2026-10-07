#!/bin/bash
# [2026-10-02 user, option 1] extend R384 and FT from steps = decay = 150k to 300k IN PLACE: wait for the next checkpoint to be
# fully saved (weights + optimizer + training_step == step), stop the Ray job, keep the node copy (archiver off meanwhile), rewrite
# checkpoints/last/pretrained_model/train_config.json (steps, policy.scheduler_decay_steps, scheduler.num_decay_steps -> 300000;
# backup *.orig150k), resume with RESUME=1 (+ AMP bf16), verify "RESUME from" + cfg.steps 300K, restart the archiver. nohup.
export RAY_ADDRESS=http://100.64.0.1:8265; N=bh-aiteam@100.64.0.2; TB=/home/bh-aiteam/holobrain-data/trainB
log() { echo "[$(date '+%F %T')] $*"; }
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=20 -J head-lp $N "$@" < /dev/null; }
extend() {  # <run> <job id> <next ckpt step> <resume cmd>
  RUN=$1; JOB=$2; CK=$(printf "%06d" $3); CMD=$4
  log "$RUN: waiting for $CK"
  until rsh "d=$TB/$RUN/checkpoints/$CK; [ -s \$d/pretrained_model/model.safetensors ] && [ -s \$d/training_state/optimizer_state.safetensors ] && [ -s \$d/training_state/scheduler_state.json ] && [ \"\$(tr -dc 0-9 < \$d/training_state/training_step.json)\" = $3 ]"; do sleep 30; done
  sleep 60   # let rng / remaining files flush
  for pid in $(ps -axo pid,command | awk -v r="$RUN" '$2=="bash" && $3 ~ /trackb_archive_v3_relonly.sh$/ && $4==r {print $1}'); do pkill -P $pid; kill $pid; log "$RUN archiver $pid stopped"; done
  for pid in $(ps -axo pid,command | awk -v r="$RUN" '$2=="bash" && $3=="-c" && index($0, "trackb_archive_v3_relonly.sh " r " ") {print $1}'); do kill $pid; done
  ray job stop $JOB 2>&1 | tail -1
  until rsh "! pgrep -f '[t]rain_rel16_relonly.py.*trainB/$RUN( |/|\$)' >/dev/null"; do sleep 10; done
  rsh "readlink $TB/$RUN/checkpoints/last; python3 - <<'PY'
import json, shutil
p = '$TB/$RUN/checkpoints/last/pretrained_model/train_config.json'; shutil.copy(p, p + '.orig150k')
c = json.load(open(p)); c['steps'] = 300000; c['policy']['scheduler_decay_steps'] = 300000; c['scheduler']['num_decay_steps'] = 300000
json.dump(c, open(p, 'w'), indent=4); print('config ->', c['steps'], c['policy']['scheduler_decay_steps'], c['scheduler'])
PY"
  SID=$(echo $RUN | tr 'A-Z' 'a-z' | cut -c1-30)-to300k-$(date +%H%M)
  ray job submit --no-wait --submission-id $SID --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' -- bash -c "$CMD" 2>&1 | grep -E "submitted|rror"
  for i in $(seq 1 60); do s=$(ray job logs $SID 2>&1 | tr '\r' '\n'); echo "$s" | grep -qE "ot_train.py:4[0-9]+ step:|Traceback|refusing|gate|FAILED" && break; sleep 20; done
  echo "$s" | grep -aE "RESUME from|physical GPU|cfg.steps|Auto-scal|8bit\] optimizer state loaded|step:|refusing|gate|FAILED|Traceback" | head -6 | sed "s/^/   [$RUN] /" | cut -c1-200
  cd ~; nohup bash -c "IDLE_EXIT=1000000 bash ~/umi_bridge/trackb_archive_v3_relonly.sh $RUN >> ~/archive_$RUN.log 2>&1" >/dev/null 2>&1 < /dev/null &
  log "$RUN: resumed as $SID, archiver restarted"
}
extend R384-RELCART20-RELONLY-D20-B8-150K r384-relonly-d20-b8-150k-1639 65000 \
  "ACCELERATE_MIXED_PRECISION=bf16 RUN_NAME=R384-RELCART20-RELONLY-D20-B8-150K RESUME=1 STEPS=300000 DECAY_STEPS=300000 SAVE_FREQ=5000 MIN_START_GB=60 bash /home/bh-aiteam/train_relcart20_v4_relonly_d20_b8_r384.sh R 0" &
extend FT-R312C-FROM-EGOV2BB850K-RELONLY-D20-B8-150K ft-r312c-from-egov2bb850k-d20-b8-150k-1651 60000 \
  "ACCELERATE_MIXED_PRECISION=bf16 EXPECT_DOMAIN=20 FT_BASE=/home/bh-aiteam/xvla_base_ego_v2bb850k_d20 RUN_NAME=FT-R312C-FROM-EGOV2BB850K-RELONLY-D20-B8-150K RESUME=1 STEPS=300000 DECAY_STEPS=300000 SAVE_FREQ=5000 MIN_START_GB=60 bash /home/bh-aiteam/train_ft_r312c_from_ego_relonly_d20_b8.sh F 1" &
wait; log "both extended"
