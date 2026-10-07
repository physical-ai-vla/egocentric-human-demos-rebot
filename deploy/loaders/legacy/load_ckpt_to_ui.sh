#!/bin/bash
# Wait for a long-run milestone, pull it to the Mac, and load it into the real-robot smoke UI.
#
# The UI holds the model in memory, so swapping checkpoints means a restart: ~1 minute during which the robot
# keeps its pose but takes no commands. Any running loop is stopped first on purpose -- a restart mid-loop
# would leave the arm mid-chunk.
#
# usage: load_ckpt_to_ui.sh <step> [run]
set -u
STEP=$(printf "%06d" "${1:?step}")
RUN=${2:-v4base400k}
NODE=bh-aiteam@100.64.0.2
REMOTE=/home/bh-aiteam/holobrain-data/trainB/$RUN/checkpoints/$STEP/pretrained_model
LOCAL=$HOME/holobrain-mac-model/ckpt_${RUN}_$STEP

until ssh -o ConnectTimeout=10 $NODE "test -f $REMOTE/model.safetensors"; do
  echo "waiting for $RUN step $STEP ..."
  sleep 60
done
echo "checkpoint $STEP present, pulling"
rsync -a "$NODE:$REMOTE/" "$LOCAL/" || { echo "rsync FAILED"; exit 1; }
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

curl -s -m 10 -X POST http://localhost:8016/stop > /dev/null 2>&1
curl -s -m 10 -X POST http://localhost:8016/prompt_ab_stop > /dev/null 2>&1
kill $(lsof -nP -iTCP:8016 -sTCP:LISTEN -t) 2>/dev/null
sleep 3
cd $HOME/holobrain-mac-model || exit 1
V4_CKPT=$LOCAL V4_GRIPPER=${V4_GRIPPER:-predict} V4_CLAMP_MM=${V4_CLAMP_MM:-5} V4_CLAMP_DEG=${V4_CLAMP_DEG:-1.5} \
  nohup $HOME/xvla-mac/bin/python mac_v4_smoke_ui.py > $HOME/v4_smoke_ui.log 2>&1 < /dev/null &
for i in $(seq 1 24); do
  sleep 5
  code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://localhost:8016/ || true)
  [ "$code" = "200" ] && break
done
grep -a "\[v4\]" $HOME/v4_smoke_ui.log | tail -1
echo "UI on http://localhost:8016 with $LOCAL (http=$code)"
