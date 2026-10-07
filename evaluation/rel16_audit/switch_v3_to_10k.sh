#!/bin/bash
# [2026-09-29 user] once HEAD180-REL16V3-D600K has written 050000 (full state), stop it and resume from 050000 with save_freq 10000.
# The archiver is stopped FIRST (it would otherwise see "training stopped" and archive + delete the resume point, as on 09-28)
# and restarted after the resume is verified, with KEEP_NODE_STEPS=050000.
# [13:15] remote login shell is zsh: `[ a \\> b ]` was a syntax error (rc 2) so the wait never ended -> numeric awk test.
RUN=HEAD180-REL16V3-D600K; N=bh-aiteam@100.64.0.2; CK=/home/bh-aiteam/holobrain-data/trainB/$RUN/checkpoints; LOGN=/home/bh-aiteam/holobrain-data/trainB/$RUN.log
LOG=~/umi_bridge/rel16_audit/switch_v3_to_10k.log; export RAY_ADDRESS=http://100.64.0.1:8265
r() { ssh -o BatchMode=yes -o ConnectTimeout=20 -J head-lp $N "$@"; }
log() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
log "waiting for $CK/050000 with training_state and the log past 50.3K"
until r "test -s $CK/050000/training_state/optimizer_state.safetensors && test -s $CK/050000/pretrained_model/model.safetensors && grep -a 'step:' $LOGN | tail -1 | grep -oE 'step:[0-9.]+K' | tr -dc 0-9. | awk '{exit !(\$1+0 > 50.2)}'" 2>/dev/null; do sleep 60; done
log "050000 complete; stopping the archiver, then the training job"
pkill -f "trackb_archive_v2.sh $RUN"; sleep 2
ray job stop head180-rel16v3-d600k >> $LOG 2>&1
until ! r "pgrep -f '[t]rain_rel16aux.py.*trainB/$RUN(/| |\$)' >/dev/null"; do sleep 5; done
r "readlink $CK/last; ls $CK" | tee -a $LOG
r "test -s $CK/050000/training_state/optimizer_state.safetensors && [ \$(readlink $CK/last) = 050000 ]" || { log "ABORT: 050000 full state / last pointer not as expected -- not resuming automatically"; exit 1; }
log "resuming from 050000 with SAVE_FREQ=10000"
ray job submit --no-wait --submission-id head180-rel16v3-d600k-resume50k-s10k --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' -- bash -c 'RESUME=1 SAVE_FREQ=10000 MIN_START_GB=60 bash /home/bh-aiteam/train_umi_v3.sh C 1' >> $LOG 2>&1
until r "grep -a -q 'RESUME from 050000' $LOGN 2>/dev/null; tail -c 200000 $LOGN | tr '\r' '\n' | grep -aq 'step:5[0-9.]*K'" 2>/dev/null; do sleep 30; done
sleep 60
r "tail -c 400000 $LOGN | tr '\r' '\n' | grep -a -o \"'save_freq': [0-9]*\" | tail -1; grep -a 'step:' $LOGN | tail -1 | cut -c1-120" | tee -a $LOG
r "tail -c 400000 $LOGN | tr '\r' '\n' | grep -a -q \"'save_freq': 10000\"" || log "WARNING: save_freq 10000 not visible in the resumed config"
log "restarting the archiver with KEEP_NODE_STEPS=050000"
cd ~ && KEEP_NODE_STEPS=050000 nohup caffeinate -i ~/umi_bridge/trackb_archive_v2.sh $RUN >> ~/archive_$RUN.log 2>&1 &
log "switch done"
