#!/bin/bash
# [2026-10-03 user] v2 (copy of auto_ui_d20_e5.sh): + :8054 COTRAIN on the 5090 (4th RUNS field '5090' = node, loader
# load_relonly_5090_run_ckpt_to_ui.sh, no SSD archive -> old local copies deleted right away), + UI checkpoint selector pins
# ($ST/<port>.pin -> port skipped), + never delete a local copy some port is serving or pinned to.
# [2026-10-02 user] "ckpt 나올때마다 ui에 띄워줘": every 5 min, for each D20 5-epoch run, find the newest checkpoint (4090 node or
# SSD archive) and, if newer than what its port serves and the UI is not mid-run, load it with load_relonly_run_ckpt_to_ui.sh.
# Deploy env = the current UI contract (step mode exec_k 4, streaming off, Pink joint5 unlocked [B], fp32, dwell 1.0, MIT bridge).
# After a successful swap the PREVIOUS local copy of the same run (ckpt_UI_<run>_<k>k_pinklockwy, a cache of the SSD archive) is
# deleted to keep Mac disk bounded. Swaps are logged to ~/umi_bridge/ui_swap_log.tsv. Run under nohup.
N=bh-aiteam@100.64.0.2; TB=/home/bh-aiteam/holobrain-data/trainB; SSDR=/Volumes/PortableSSD/rebot_ckpts_archive
cd ~/umi_bridge
unset V4_STREAM V4_STREAM_SYNC V4_STREAM_ROWS V4_STREAM_SPEED V4_ADAPTIVE_K V4_TE V4_DEADBAND_XY_MM V4_DEADBAND_Z_MM V4_N_ACTION
# [2026-10-02 user] bf16 inference + one cycle = k1..k4 as a continuous path (pass-through, full reach test on k4 only)
# [2026-10-02 user] pure step mode: streaming off, no pass-through, no prefetch; one inference -> k1..k8 each with the full reach test
# [2026-10-03 user "ui와 추론 5090 통해서"] policy forward on the 5090 inference server (xvla_infer_server.py, Ray job)
# [2026-10-03 16:57 user "5090 추론하고 curobo 꺼주고 mps로"] remote inference OFF (was: export V4_REMOTE_INFER=http://100.64.0.5:8791)
unset V4_REMOTE_INFER V4_REMOTE_CKPT
export IK_BACKEND=pink V4_JAW_TORQUE=0 V4_ROBOT=http://localhost:8021 EXEC_K=1 V4_N_ACTION=4 V4_PASS_THROUGH=0 V4_PREFETCH=0 V4_STREAM=0 V4_DENOISE_STEPS=5 DWELL=1.0 V4_DTYPE=bf16 V4_PINK_LOCK= V4_YAW_MAX_STEP_DEG=0 DOMAIN_EXPECT=20
# port:run:dataset[:ik backend] (default pink)
# [2026-10-02 16:4x user] ONLY two UIs, both pink: :8052 R384, :8053 MIX70 pretrain. (earlier: :8054 (raw-ego FT B) cancelled; :8056 MIX50 / :8058 MIX70 ego pretrain; :8057 / :8059 = R312c FT from MIX50 / MIX70 (only the better one is trained)
RUNS="8052:R384-RELCART20-RELONLY-D20-B8-300K:r384_relcart20_rel16_v4 8053:FT-R312C-FROM-EGOV2BB850K-RELONLY-D20-B8-300K:r312c_relcart20_rel16_v4"  # [2026-10-03 18:3x user] 8054 off (co-train stopped)
H5=bh-ai-5090@100.64.0.5; R5=/srv/data/johann/relonly/runs
ST=~/umi_bridge/.auto_ui_d20_e5   # per-port state files (bash 3.2: no associative arrays)
mkdir -p $ST
log() { echo "[$(date '+%F %T')] $*"; }
while true; do
  for e in $RUNS; do
    P=${e%%:*}; r=${e#*:}; RUN=${r%%:*}; DS=${r#*:}; IKB=pink; NODE=4090
    case "$DS" in *:*) NODE=${DS#*:}; DS=${DS%%:*} ;; esac
    [ -f $ST/$P.pin ] && continue      # user picked a checkpoint in the UI
    if [ "$NODE" = 5090 ]; then PFX=5090_; LOADER=./load_relonly_5090_run_ckpt_to_ui.sh
      node=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $H5 "cd $R5/$RUN/checkpoints 2>/dev/null && for c in [0-9]*; do [ -f \$c/pretrained_model/model.safetensors ] && echo \$c; done" < /dev/null 2>/dev/null)
    else PFX=trackb_; LOADER=./load_relonly_run_ckpt_to_ui.sh
      node=$(ssh -o BatchMode=yes -o ConnectTimeout=15 -J head-lp $N "ls $TB/$RUN/checkpoints 2>/dev/null | grep -E '^[0-9]{6}$'" < /dev/null 2>/dev/null); fi
    ssd=$(ls "$SSDR/$PFX$RUN" 2>/dev/null | grep -E '^[0-9]{6}$' | while read s; do [ -f "$SSDR/$PFX$RUN/$s/.archived" ] && echo $s; done)
    NEW=$(printf "%s\n%s\n" "$node" "$ssd" | grep -E '^[0-9]{6}$' | sort -u | tail -1)
    [ -z "$NEW" ] && continue
    CUR=$(cat $ST/$P 2>/dev/null); [ "$CUR" = "$NEW" ] && continue
    curl -s -m 5 localhost:$P/status | grep -q '"running": *true' && { log ":$P busy (episode running), $RUN $NEW waits"; continue; }
    T0=$(date +%s)
    # step passed in DECIMAL: the loader printf %06d reads 005000 as OCTAL (= 2560)
    if IK_BACKEND=$IKB RUN=$RUN DS_EXPECT=$DS $LOADER $((10#$NEW)) $P > /tmp/claude_ld_$P.log 2>&1 < /dev/null; then
      K=$((10#$NEW / 1000)); D=ckpt_UI_${RUN}_${K}k_pinklockwy; H=$(shasum -a 256 ~/holobrain-mac-model/$D/model.safetensors | cut -c1-16)
      printf "%s\tport=%s\trun_name=%s\tcheckpoint_step=%s\tcheckpoint_sha=%s\tload_time=%ss\tik_backend=%s\tjoint_lock=NONE\tmode=STEP k1-k4 full dwell (no stream/pass/prefetch) denoise 5\tdwell=1.0s\tdtype=bf16\tdomain_id=20\tckpt_dir=%s\n" "$(date '+%F %T')" $P $RUN $NEW $H $(( $(date +%s)-T0 )) $IKB $D >> ui_swap_log.tsv
      [ -n "$CUR" ] && echo "$RUN $CUR" >> $ST/pending_delete
      echo $NEW > $ST/$P; log ":$P loaded $RUN $NEW (${K}k) sha $H"
    else
      log ":$P $RUN $NEW not loaded yet: $(tail -1 /tmp/claude_ld_$P.log)"
    fi
  done
  # previous local UI copies are deleted only after the SSD archive has them (the v3 archiver seeds weights from them)
  if [ -f $ST/pending_delete ]; then
    : > $ST/pending_delete.next
    while read R S; do
      serving=0; for e in $RUNS; do p=${e%%:*}; r=${e#*:}; [ "${r%%:*}" = "$R" ] || continue
        [ "$(cat $ST/$p 2>/dev/null)" = "$S" ] && serving=1; [ "$(cat $ST/$p.pin 2>/dev/null)" = "$S" ] && serving=1; done
      five=0; for e in $RUNS; do r=${e#*:}; [ "${r%%:*}" = "$R" ] && case "$r" in *:5090) five=1 ;; esac; done
      if [ $serving = 1 ]; then echo "$R $S" >> $ST/pending_delete.next
      elif [ $five = 1 ] || [ -f "$SSDR/trackb_$R/$S/.archived" ]; then rm -rf ~/holobrain-mac-model/ckpt_UI_${R}_$((10#$S / 1000))k_pinklockwy; log "deleted local UI copy $R $S (archived)"
      else echo "$R $S" >> $ST/pending_delete.next; fi
    done < $ST/pending_delete
    mv $ST/pending_delete.next $ST/pending_delete
  fi
  sleep 120   # [2026-10-02 user] 2-minute check (was 300)
done
