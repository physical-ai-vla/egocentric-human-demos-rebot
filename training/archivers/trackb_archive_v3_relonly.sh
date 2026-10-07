#!/bin/bash
# [2026-10-02 user] v3: copy of trackb_archive_v2_relonly.sh (untouched) that SEEDS pretrained_model/model.safetensors from the Mac UI
# copy (~/holobrain-mac-model/ckpt_UI_<run>_<k>k_pinklockwy, written by the UI loader) when its sha256 equals the node's, and
# stamps the node mtime on it so rsync skips the 3.3 GB weights. The final full-manifest sha256 check is unchanged.
# [2026-09-30] copy of trackb_archive_v2.sh (untouched: the v3 archiver runs it) whose running-check also matches train_rel16_relonly.py
# Mac-side checkpoint archiver, v2. Moves each checkpoint from the 4090 to the SSD and removes it from the node
# only after a sha256 match.
#
# v1 (trackb_archive.sh) died on 2026-09-25 19:03: a Mac<->4090 ssh timeout made its final
# `ssh ... pgrep || break` look like "training finished", it exited, 14 checkpoints (132 GB) piled up on the node
# and REL16 died at 120k with ENOSPC. So here the ssh transport failure (exit 255) is never read as an answer:
# it only means "retry later". "Training finished" needs a successful ssh AND pgrep saying no process.
#
# Per checkpoint:  copy into <step>.copying/ -> sha256 of every file, remote vs local -> rename to <step>/ ->
# touch <step>/.archived -> rm on the node. A <step>/ without .archived is a leftover and is redone.
# A checkpoint already archived (by v1 or by hand) but still on the node is verified against the node copy
# and then removed from the node.
#
# Optimiser state (6.6 GB against 3.3 GB of weights) is copied only for steps that are a multiple of
# STATE_EVERY, plus whatever STATE_STEPS lists. The newest checkpoint on the node always keeps its state (it
# is never archived while training runs), so a resume never depends on the SSD. At ~7 MB/s the link moves
# 10.6 GB in ~25 min, which is also how long 5k steps take: copying every state would leave no margin.
#
# usage: trackb_archive_v2.sh <run-name> [dest-dir]
#   STATE_EVERY=25000   copy training_state when step % STATE_EVERY == 0 (0 = never)
#   STATE_STEPS="115000 120000"   extra steps whose training_state is copied
#   ONESHOT=1           exit once the node has no numeric checkpoint left (backlog of a dead/finished run)
#   IDLE_EXIT=5         otherwise exit after this many consecutive polls with training confirmed not running
#   KEEP_NODE_STEPS="400000"  archive + verify these like any other, but leave the node copy (a paused run's
#                       resume point). They do not count as "left" for the exit test.
#   JUMP=head-lp        ssh jump host ("" = direct). On 2026-09-26 the Mac->4090 tailnet path fell back to the
#                       Tokyo DERP relay (0.9 MB/s); head-lp has a direct path to both ends (20 MB/s via the jump).
set -u
RUN=${1:?run name}
DEST=${2:-/Volumes/PortableSSD/rebot_ckpts_archive/trackb_$RUN}
STATE_EVERY=${STATE_EVERY:-25000}
STATE_STEPS=${STATE_STEPS:-}
ONESHOT=${ONESHOT:-0}
KEEP_NODE_STEPS=${KEEP_NODE_STEPS:-}
IDLE_EXIT=${IDLE_EXIT:-5}
POLL_S=${POLL_S:-60}
RETRY_S=${RETRY_S:-60}
NODE=bh-aiteam@100.64.0.2
REMOTE=/home/bh-aiteam/holobrain-data/trainB/$RUN/checkpoints
RSYNC=/opt/homebrew/bin/rsync
JUMP=${JUMP-head-lp}
SSH_CMD="ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes${JUMP:+ -J $JUMP}"
mkdir -p "$DEST" || exit 1

keep_on_node () { for x in $KEEP_NODE_STEPS; do [ "$((10#$x))" = "$((10#$1))" ] && return 0; done; return 1; }
log () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
# ServerAlive: a dead link fails in ~1 min instead of hanging. BatchMode: never block on a prompt.
rsh () { $SSH_CMD $NODE "$@"; }

want_state () {
    local s=$((10#$1))
    [ "$STATE_EVERY" -gt 0 ] && [ $((s % STATE_EVERY)) -eq 0 ] && return 0
    for x in $STATE_STEPS; do [ "$((10#$x))" = "$s" ] && return 0; done
    return 1
}

# sha256 manifest of the parts present, paths relative to the checkpoint dir, sorted
remote_manifest () {  # <step> <parts...>
    local s=$1; shift
    rsh "cd $REMOTE/$s && find $* -type f ! -name '.*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum"
}
local_manifest () {   # <dir> <parts...>
    local d=$1; shift
    (cd "$d" && find "$@" -type f ! -name '.*' -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256)
}

# [v3] seed the weights from the UI's local copy (same bytes, already pulled once over the slow link)
UI_ROOT=${UI_ROOT:-$HOME/holobrain-mac-model}
seed_weights () {  # <step>
    local s=$1 k=$((10#$1 / 1000)) src dst rs ls mt
    src="$UI_ROOT/ckpt_UI_${RUN}_${k}k_pinklockwy/model.safetensors"; dst="$DEST/$1.copying/pretrained_model/model.safetensors"
    [ -s "$src" ] || return 0
    # a leftover rsync --partial file is replaced; a complete copy (same size as the UI copy) is kept
    [ -s "$dst" ] && [ "$(stat -f %z "$dst")" = "$(stat -f %z "$src")" ] && return 0
    read -r rs mt <<< "$(rsh "sha256sum $REMOTE/$s/pretrained_model/model.safetensors | cut -c1-64; stat -c %Y $REMOTE/$s/pretrained_model/model.safetensors" | tr '\n' ' ')"
    [ -n "$rs" ] && [ -n "$mt" ] || return 0
    ls=$(shasum -a 256 "$src" | cut -c1-64)
    [ "$ls" = "$rs" ] || { log "  $s: UI copy sha differs from the node -- pulling normally"; return 0; }
    mkdir -p "$(dirname "$dst")"
    if cp "$src" "$dst.part" && [ "$(shasum -a 256 "$dst.part" | cut -c1-64)" = "$rs" ]; then
        mv "$dst.part" "$dst"; touch -t "$(date -r "$mt" +%Y%m%d%H%M.%S)" "$dst"
        log "  $s: weights seeded from the UI copy (sha match), rsync skips them"
    else rm -f "$dst.part"; fi
    return 0
}

# returns 0 done, 2 network (retry later), 1 a real integrity failure (node copy kept, needs a human)
archive_one () {
    local s=$1 final="$DEST/$1" tmp="$DEST/$1.copying" parts rc
    [ -f "$final/.archived" ] && keep_on_node "$s" && return 0     # kept resume point, already verified
    if [ -f "$final/.archived" ]; then
        # already on the SSD; only the node copy is left. Verify it matches before removing it.
        parts=$(cd "$final" && ls -d pretrained_model training_state 2>/dev/null | tr '\n' ' ')
        remote_manifest "$s" $parts > "$final/.sha.remote"; rc=$?
        [ $rc -eq 255 ] && return 2
        [ -f "$final/SHA256SUMS" ] || local_manifest "$final" $parts > "$final/SHA256SUMS"
        if [ $rc -ne 0 ] || ! cmp -s "$final/SHA256SUMS" "$final/.sha.remote"; then
            log "  $s: SSD copy and node copy differ -- keeping both, check by hand"; return 1
        fi
        rm -f "$final/.sha.remote"
        keep_on_node "$s" && return 0
        rsh "rm -rf $REMOTE/$s"; rc=$?; [ $rc -eq 255 ] && return 2
        log "  $s: was already archived; verified and removed from the node"; return 0
    fi

    log "pulling $RUN/$s"
    # [2026-09-30] keep a partial .copying across network retries (it used to be wiped here, so under a contended link every
    # retry restarted from zero and nothing ever finished -> 4090 disk-guard stops). rsync --partial resumes; the sha256 check below
    # still decides whether the copy is accepted.
    mkdir -p "$tmp"
    parts="pretrained_model"
    if want_state "$s"; then
        rsh "test -s $REMOTE/$s/training_state/optimizer_state.safetensors"; rc=$?
        [ $rc -eq 255 ] && return 2
        if [ $rc -eq 0 ]; then parts="pretrained_model training_state"
        else log "  $s: no complete optimiser state on the node -- weights only"; fi
    fi
    seed_weights "$s"
    for part in $parts; do
        $RSYNC -rt --partial --timeout=300 -e "$SSH_CMD" "$NODE:$REMOTE/$s/$part" "$tmp/" || { log "  rsync $part failed"; return 2; }
    done
    remote_manifest "$s" $parts > "$tmp/.sha.remote"; rc=$?
    [ $rc -eq 255 ] && return 2
    [ $rc -ne 0 ] && { log "  $s: remote sha256 failed (rc $rc)"; return 1; }
    local_manifest "$tmp" $parts > "$tmp/SHA256SUMS"
    if ! grep -q " pretrained_model/model.safetensors$" "$tmp/.sha.remote" || ! cmp -s "$tmp/SHA256SUMS" "$tmp/.sha.remote"; then
        log "  $s: sha256 mismatch or no model.safetensors -- node copy kept, partial copy discarded"; rm -rf "$tmp"; return 1
    fi
    rm -f "$tmp/.sha.remote"
    if [ -d "$tmp/training_state" ]; then
        got=$(python3 -c "import json;print(json.load(open('$tmp/training_state/training_step.json'))['step'])" 2>/dev/null)
        [ "$got" = "$((10#$s))" ] || { log "  $s: training_step $got != $s -- dropping the state"; rm -rf "$tmp/training_state"
            local_manifest "$tmp" pretrained_model > "$tmp/SHA256SUMS"; }
    fi
    [ -e "$final" ] && rm -rf "$final"      # leftover without .archived (e.g. v1's 3 MB 055000)
    mv "$tmp" "$final" && touch "$final/.archived" || return 1
    if keep_on_node "$s"; then
        log "  archived $s ($(du -sh "$final" | cut -f1), $(echo $parts)) -- node copy KEPT (resume point)"; return 0
    fi
    rsh "rm -rf $REMOTE/$s"; rc=$?; [ $rc -eq 255 ] && return 2
    log "  archived $s ($(du -sh "$final" | cut -f1), $(echo $parts)) and removed from the node"
    return 0
}

idle=0
while true; do
    steps=$(rsh "ls $REMOTE 2>/dev/null | grep -E '^[0-9]{6}\$' | sort -n"); rc=$?
    [ $rc -eq 255 ] && { log "ssh failed -- network, retrying in ${RETRY_S}s"; sleep $RETRY_S; continue; }
    # [l] keeps pgrep from matching the remote shell that carries this very pattern
    rsh "pgrep -f '([l]erobot_train|[t]rain_rel16aux|[t]rain_rel16_relonly)[.]py.*trainB/$RUN(/| |\$)' >/dev/null"; running=$?   # [2026-09-28] (/|...): a RESUMED run carries --config_path=trainB/$RUN/checkpoints/..., which the old pattern missed and archived the resume point
    if [ $running -ne 0 ] && [ $running -ne 1 ]; then
        log "running-check inconclusive (rc $running) -- retrying in ${RETRY_S}s"; sleep $RETRY_S; continue
    fi
    newest=$(echo "$steps" | tail -1)
    net=0
    for s in $steps; do
        # while training runs the newest is its resume point and may still be being written
        [ $running -eq 0 ] && [ "$s" = "$newest" ] && continue
        archive_one "$s"; rc=$?
        [ $rc -eq 2 ] && { net=1; break; }
    done
    [ $net -eq 1 ] && { log "network error mid-archive -- retrying in ${RETRY_S}s"; sleep $RETRY_S; continue; }

    if [ $running -eq 1 ]; then
        left=$(rsh "ls $REMOTE 2>/dev/null | grep -E '^[0-9]{6}\$'"); rc=$?
        [ $rc -eq 255 ] && { sleep $RETRY_S; continue; }
        n=0; for x in $left; do keep_on_node "$x" || n=$((n + 1)); done; left=$n
        if [ "$ONESHOT" = "1" ]; then
            [ "${left:-1}" = "0" ] && { log "ONESHOT: node has no checkpoint left, done"; exit 0; }
        else
            idle=$((idle + 1))
            [ $idle -ge $IDLE_EXIT ] && [ "${left:-1}" = "0" ] && {
                log "training confirmed not running for $idle polls and everything is archived"; exit 0; }
        fi
    else
        idle=0
    fi
    sleep $POLL_S
done
