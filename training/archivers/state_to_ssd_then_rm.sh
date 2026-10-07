#!/bin/bash
# [2026-10-02 user] cleanup step 2: for stopped runs whose node checkpoint is already archived on the SSD as WEIGHTS ONLY,
# copy the node training_state to the SSD archive (sha256-verified file by file), re-verify the SSD weights against the node,
# then delete the node checkpoint dir (and a dangling `last`). Nothing is deleted unless every check passes.
N=bh-aiteam@100.64.0.2; TB=/home/bh-aiteam/holobrain-data/trainB; A=/Volumes/PortableSSD/rebot_ckpts_archive
rsh () { ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -J head-lp $N "$@" < /dev/null; }
log() { echo "[$(date '+%F %T')] $*"; }
for x in R312C-RELCART20-RELONLY-D6-S600K:160000 FT-R312C-FROM-EGOV2B100K-RELONLY-D6-D600K:040000 R312C-RELCART20-REL16V4-D600K:085000 HEAD180-REL16V3-D600K:090000; do
  RUN=${x%:*}; S=${x#*:}; R=$TB/$RUN/checkpoints/$S; D=$A/trackb_$RUN/$S
  [ -f $D/.archived ] || { log "$RUN $S: SSD copy not marked archived -- skip"; continue; }
  w_ssd=$(shasum -a 256 $D/pretrained_model/model.safetensors | cut -c1-64); w_node=$(rsh "sha256sum $R/pretrained_model/model.safetensors" | cut -c1-64)
  [ -n "$w_node" ] && [ "$w_ssd" = "$w_node" ] || { log "$RUN $S: weights sha mismatch SSD $w_ssd node $w_node -- skip"; continue; }
  if [ ! -d $D/training_state ]; then
    rm -rf $D/training_state.part
    log "$RUN $S: copying training_state"
    /opt/homebrew/bin/rsync -rt --partial --timeout=300 -e "ssh -o BatchMode=yes -o ServerAliveInterval=15 -J head-lp" $N:$R/training_state/ $D/training_state.part/ || { log "$RUN $S: rsync failed -- skip"; continue; }
    nsum=$(rsh "cd $R && sha256sum training_state/* | sort -k2"); lsum=$(cd $D && for f in training_state.part/*; do shasum -a 256 "$f"; done | sed 's#training_state.part/#training_state/#' | sort -k2)
    [ "$nsum" = "$lsum" ] || { log "$RUN $S: training_state sha mismatch -- partial kept, node untouched"; continue; }
    st=$(python3 -c "import json;print(json.load(open('$D/training_state.part/training_step.json'))['step'])")
    [ "$st" = "$((10#$S))" ] || { log "$RUN $S: training_step $st != $S -- skip"; continue; }
    mv $D/training_state.part $D/training_state && echo "$nsum" >> $D/SHA256SUMS
  fi
  rsh "rm -rf $R; L=$TB/$RUN/checkpoints/last; [ -L \$L ] && [ ! -e \$L ] && rm \$L; df -BG / | tail -1" | sed "s#^#[$RUN $S removed from node] #"
  log "$RUN $S: SSD now has $(du -sh $D | cut -f1) (pretrained_model + training_state)"
done
log "done"
