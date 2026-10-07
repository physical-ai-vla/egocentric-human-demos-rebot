#!/bin/bash
# Load a B663 twin checkpoint from the SSD archive into the real-robot smoke UI.
#
# Same job as load_ckpt_to_ui.sh, but the source is the SSD archive, not the training node --
# the archiver deletes a checkpoint from the node once it has verified the copy, so anything
# older than the last few milestones only exists on the SSD.
#
# The UI holds the model in memory, so swapping means a restart: ~1 minute during which the
# robot keeps its pose but takes no commands. Any running loop is stopped first on purpose.
#
# usage: load_ssd_ckpt_to_ui.sh <step> <REL32|DELTA32|REL16|DELTA16> [port]
#
# The port lets the two twins run side by side (REL on 8016, DELTA on 8017). Both talk to the SAME
# robot_service on 8020, so only ONE of them may drive the arm at a time -- start a loop on one and leave
# the other idle. Running both loops together would interleave commands from two policies.
set -u
STEP=$(printf "%06d" "${1:?step}")
TWIN=${2:?REL32 or DELTA32}
PORT=${3:-8016}
# Each twin carries its whole serving contract, not just the action mode. Setting these by hand is how a
# 76-D model gets fed a 20-D state or a 16-step model gets decoded as 32 -- both of which run without error.
case "$TWIN" in
  REL32)   MODE=umi;   SMODE=slim20; CH=32; RUNDIR=B663-REL32;          ARCH=trackb_B663-REL32 ;;
  DELTA32) MODE=delta; SMODE=slim20; CH=32; RUNDIR=B663-DELTA32;        ARCH=trackb_B663-DELTA32 ;;
  REL16)   MODE=umi;   SMODE=umi76;  CH=16; RUNDIR=B180H-UMI76-REL16;   ARCH=trackb_B180H-UMI76-REL16 ;;
  DELTA16) MODE=delta; SMODE=umi76;  CH=16; RUNDIR=B180H-UMI76-DELTA16; ARCH=trackb_B180H-UMI76-DELTA16 ;;
  *) echo "twin must be REL32 | DELTA32 | REL16 | DELTA16"; exit 1 ;;
esac
SRC=/Volumes/PortableSSD/rebot_ckpts_archive/$ARCH/$STEP/pretrained_model
LOCAL=$HOME/holobrain-mac-model/ckpt_${RUNDIR}_$(( 10#$STEP / 1000 ))k
# The archiver deletes a checkpoint from the training node once the SSD copy verifies, so old steps live
# only on the SSD -- but the NEWEST step can be the other way round: still on the node, not yet archived.
# Fall back to the node rather than failing, otherwise the freshest checkpoint is the one you cannot serve.
if [ ! -f "$SRC/model.safetensors" ]; then
  NODE_SRC=bh-aiteam@100.64.0.2:/home/bh-aiteam/holobrain-data/trainB/$RUNDIR/checkpoints/$STEP/pretrained_model
  echo "not on the SSD, pulling from the training node: $STEP $TWIN"
  ssh -o ConnectTimeout=10 bh-aiteam@100.64.0.2 "test -f /home/bh-aiteam/holobrain-data/trainB/$RUNDIR/checkpoints/$STEP/pretrained_model/model.safetensors" \
    || { echo "on neither the SSD nor the node: $STEP $TWIN"; exit 1; }
  SRC=$NODE_SRC
fi

/opt/homebrew/bin/rsync -a "${SRC%/}/" "$LOCAL/" || { echo "rsync FAILED"; exit 1; }
du -sh "$LOCAL"

# the trainer writes config.json without draccus's `type` key, which every stock loader needs
$HOME/xvla-mac/bin/python - "$LOCAL" <<'EOS'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "config.json"
c = json.loads(p.read_text())
if "type" not in c:
    p.write_text(json.dumps({"type": "xvla", **c}, indent=2))
    print("injected type=xvla")
print({k: c.get(k) for k in ("chunk_size", "max_state_dim", "max_action_dim", "action_mode")})
EOS

curl -s -m 10 -X POST http://localhost:$PORT/stop > /dev/null 2>&1
curl -s -m 10 -X POST http://localhost:$PORT/prompt_ab_stop > /dev/null 2>&1
# kill by listening socket, never by name: a pkill -f on the script name takes out the OTHER twin's UI too
kill $(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t) 2>/dev/null
sleep 3
cd $HOME/holobrain-mac-model || exit 1
V4_CKPT=$LOCAL V4_ACTION_MODE=$MODE V4_GRIPPER=${V4_GRIPPER:-predict} \
  V4_CLAMP_MM=${V4_CLAMP_MM:-0} V4_CLAMP_DEG=${V4_CLAMP_DEG:-0} \
  V4_CHUNK=${V4_CHUNK:-$CH} V4_STATE_MODE=$SMODE V4_EXEC_K=${V4_EXEC_K:-8} V4_DTYPE=${V4_DTYPE:-fp32} \
  V4_N_ACTION=${V4_N_ACTION:-1} V4_GLOBAL_ROT180=${V4_GLOBAL_ROT180:-0} V4_GLOBAL_MIRROR=${V4_GLOBAL_MIRROR:-0} V4_UI_PORT=$PORT \
  nohup $HOME/xvla-mac/bin/python mac_v4_smoke_ui.py > $HOME/v4_smoke_ui_$PORT.log 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 5
  code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://localhost:$PORT/ || true)
  [ "$code" = "200" ] && break
done
grep -a "\[v4\]" $HOME/v4_smoke_ui_$PORT.log | tail -2
echo "UI http://localhost:$PORT  ckpt=$LOCAL  action_mode=$MODE  state_mode=$SMODE  chunk=${V4_CHUNK:-$CH}  exec_k=${V4_EXEC_K:-8}  (http=$code)"
