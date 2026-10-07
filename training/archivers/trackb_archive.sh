#!/bin/bash
# Mac-side checkpoint archiver: move each evaluated checkpoint to the SSD, whole, and verify it.
#
# The first version copied only `pretrained_model`. When the disk filled and both 400k runs died, there was
# no resumable checkpoint anywhere -- the node keeps training_state for the newest checkpoint alone, and that
# was the one that died mid-write. So the optimiser state now comes too.
#
# It cannot come for ALL of them: training_state is 6.4 GB against 3.5 GB of weights, and 80 checkpoints x 2
# runs would be 1.6 TB against an 0.9 TB disk. Weights are kept for every checkpoint (that is the learning
# curve and the deployable artefact); the optimiser state is kept for the newest KEEP_STATE only, which is
# all a resume ever needs.
#
# usage: trackb_archive.sh <run-name> [dest-root]
set -u
RUN=${1:?run name}
DEST=${2:-/Volumes/PortableSSD/rebot_ckpts_archive/trackb_$RUN}
KEEP_STATE=${KEEP_STATE:-2}
NODE=bh-aiteam@100.64.0.2
REMOTE=/home/bh-aiteam/holobrain-data/trainB/$RUN/checkpoints
EVALS=/home/bh-aiteam/umi_bridge/evals/$RUN
RSYNC=/opt/homebrew/bin/rsync
mkdir -p "$DEST"

lsize () { find "$1" -type f -exec stat -f%z {} + 2>/dev/null | awk '{t+=$1} END{print t+0}'; }

while true; do
    steps=$(ssh -o ConnectTimeout=15 $NODE "ls $REMOTE 2>/dev/null | grep -E '^[0-9]{6}$' | sort -n")
    [ -z "$steps" ] && { sleep 120; continue; }
    newest=$(echo "$steps" | tail -1)
    # How much room is left on the node decides whether a checkpoint may wait for its evaluation.
    # Normally it waits, so the evaluator still has it locally. Under pressure it does not: the disk filling
    # is what killed both 400k runs, and a checkpoint on the SSD can always be evaluated later, while a
    # training save that runs out of room cannot be recovered at all.
    free_gb=$(ssh -o ConnectTimeout=15 $NODE "df -BG --output=avail /home/bh-aiteam | tail -1 | tr -dc '0-9'")
    # Checkpoints no longer wait to be evaluated before they move. Evaluation ran on the node and
    # needed a GPU; both of the node's GPUs are held by the two trainings, so those jobs never started
    # and the checkpoints piled up behind them until the disk was full. Scoring now happens on this
    # machine, from the copy this loop makes.
    urgent=1
    for s in $steps; do
        [ "$s" = "$newest" ] && continue
        if [ "$urgent" = "0" ]; then
            ssh $NODE "test -f $EVALS/$s.json" || continue
        fi
        [ -f "$DEST/$s/.archived" ] && continue
        echo "[$(date +%H:%M:%S)] pulling $RUN/$s"
        mkdir -p "$DEST/$s"
        ok=1
        for part in pretrained_model training_state; do
            ssh $NODE "test -d $REMOTE/$s/$part" || continue
            $RSYNC -rt "$NODE:$REMOTE/$s/$part" "$DEST/$s/" || { ok=0; break; }
            rsz=$(ssh $NODE "du -sb $REMOTE/$s/$part | cut -f1")
            lsz=$(lsize "$DEST/$s/$part")
            [ "$rsz" = "$lsz" ] || { echo "  size mismatch on $part: remote=$rsz local=$lsz"; ok=0; break; }
        done
        [ "$ok" = "1" ] || { echo "  copy failed, keeping the node copy"; continue; }
        ssh $NODE "cat $EVALS/$s.json" > "$DEST/$s/eval.json" 2>/dev/null

        # integrity: the weights are there, and if the optimiser state came, it is complete and agrees
        # with the directory name. A checkpoint that fails this is not deleted from the node.
        if [ ! -s "$DEST/$s/pretrained_model/model.safetensors" ]; then
            echo "  no model.safetensors after copy -- keeping the node copy"; continue
        fi
        if [ -d "$DEST/$s/training_state" ]; then
            want=$((10#$s))
            got=$(python3 -c "import json,sys;print(json.load(open('$DEST/$s/training_state/training_step.json')).get('step','?'))" 2>/dev/null)
            if [ ! -s "$DEST/$s/training_state/optimizer_state.safetensors" ] && \
               [ ! -s "$DEST/$s/training_state/optimizer_param_groups.json" ]; then
                echo "  training_state has no optimiser state (the run died mid-save here) -- weights only"
                rm -rf "$DEST/$s/training_state"
            elif [ "$got" != "$want" ]; then
                echo "  training_step $got != $want -- dropping the state, keeping weights"
                rm -rf "$DEST/$s/training_state"
            fi
        fi

        touch "$DEST/$s/.archived"
        ssh $NODE "rm -rf $REMOTE/$s"
        echo "  archived $s ($(lsize "$DEST/$s" | awk '{printf "%.1f GB", $1/1e9}')) and removed from the node"

        # keep the optimiser state for the newest few only; the weights of every step stay
        keep=$(ls -d "$DEST"/[0-9]*/ 2>/dev/null | sort | tail -$KEEP_STATE | xargs -n1 basename 2>/dev/null)
        for d in $(ls -d "$DEST"/[0-9]*/ 2>/dev/null | xargs -n1 basename); do
            echo "$keep" | grep -qx "$d" && continue
            [ -d "$DEST/$d/training_state" ] && { rm -rf "$DEST/$d/training_state"; echo "  pruned state of $d"; }
        done
    done
    ssh -o ConnectTimeout=15 $NODE "pgrep -f 'lerobot_train.py.*$RUN' > /dev/null" || {
        ssh $NODE "test -f $EVALS/$newest.json" && continue
        echo "training finished and everything evaluated is archived"; break; }
    sleep ${POLL_S:-30}
done
