#!/bin/bash
# [2026-09-29] Load a HEAD180-CART20-REL16V3-D600K checkpoint into the robot UI: the REL16-v3 deployment contract with ONE
# difference, the state: V4_STATE_MODE=cart20 (infer_core_v4.cart20_state, live==dataset parity 5.6e-7). Copy of load_v3_ckpt_to_ui.sh.
#   state cart20 (L/R TCP pos+rot6d + openness) | action REL (umi) chunk 16 | only dims 0..19 reach the controller (dq 20..31 ignored, verified)
#   gripper: continuous g = clip(leader/45, 0, 1) -> binary at V4_GRIP_THRESH = 0.6 (= cmd 27/45, the validated boundary)
#            OPEN -> cmd 42, CLOSE -> cmd 0 (gripper_contract_v2.json)
#   exec_k 16 (EXEC_K=<k> overrides, e.g. the 2026-09-29 exec_k 4 test) | n_action 1 | dwell 1.5 s | clamp off | GLOBAL_ROT180 0 (live cube test) | mirror 0
#   IK: solver-internal clip OFF (2026-09-29 decision; IK_MJD=0.35 gives the old N0), outer DQ_MAX 0.6 + LO/HI kept,
#       every waypoint logged to ~/v4_ik_waypoints.jsonl (DQ_MAX / limit hits)
# usage: load_cart20_ckpt_to_ui.sh <step, e.g. 10000> [port=8026]
set -u
STEP=$(printf "%06d" "${1:?step}"); PORT=${2:-8026}; IK_MJD=${IK_MJD:-none}; DWELL=${DWELL:-1.5}; EXEC_K=${EXEC_K:-16}; RUN=HEAD180-CART20-REL16V3-D600K
K=$((10#$STEP / 1000)); LOCAL=$HOME/holobrain-mac-model/ckpt_UI_HEAD180-CART20-REL16V3_${K}k   # UI-only copy: the eval watcher deletes its own copies
SSD=/Volumes/PortableSSD/rebot_ckpts_archive/trackb_$RUN/$STEP/pretrained_model
NODE=bh-aiteam@100.64.0.2:/home/bh-aiteam/holobrain-data/trainB/$RUN/checkpoints/$STEP/pretrained_model
# [2026-09-29] completeness guard: SSD copy only with .archived + SHA256SUMS match; node copy only after training logged a step
# past it, and the transferred model.safetensors must match the node's sha256.
if [ ! -f "$LOCAL/model.safetensors" ]; then
  mkdir -p "$LOCAL"
  if [ -f "$SSD/../.archived" ]; then
    ( cd "$SSD/.." && grep "pretrained_model/model.safetensors" SHA256SUMS | shasum -a 256 -c - >/dev/null 2>&1 ) || { echo "SSD copy of $STEP fails its SHA256SUMS"; rm -rf "$LOCAL"; exit 1; }
    /opt/homebrew/bin/rsync -a "$SSD/" "$LOCAL/"
  else
    LAST=$(ssh -o BatchMode=yes -J head-lp bh-aiteam@100.64.0.2 "grep -a 'step:' /home/bh-aiteam/holobrain-data/trainB/$RUN.log | tail -1 | grep -oE 'step:[0-9]+' | tr -dc 0-9")
    [ -n "$LAST" ] && [ "$LAST" -gt $((K + 1)) ] || { echo "node checkpoint $STEP may still be being written (training at ${LAST:-?}K)"; rm -rf "$LOCAL"; exit 1; }
    /opt/homebrew/bin/rsync -a -e "ssh -o BatchMode=yes -J head-lp" "$NODE/" "$LOCAL/" || { echo "checkpoint $STEP not found on the node"; rm -rf "$LOCAL"; exit 1; }
    R=$(ssh -o BatchMode=yes -J head-lp bh-aiteam@100.64.0.2 "sha256sum ${NODE#*:}/model.safetensors | cut -c1-64"); Lh=$(shasum -a 256 "$LOCAL/model.safetensors" | cut -c1-64)
    [ "$R" = "$Lh" ] || { echo "transfer sha mismatch for $STEP"; rm -rf "$LOCAL"; exit 1; }
  fi
fi
$HOME/xvla-mac/bin/python - "$LOCAL" <<'EOS' || { echo "checkpoint identity/shape guard FAILED -- UI not started"; exit 1; }
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "config.json"; c = json.loads(p.read_text())
if "type" not in c: p.write_text(json.dumps({"type": "xvla", **c}, indent=2))
assert c.get("max_action_dim") == 32 and c.get("max_state_dim") == 20, (c.get("max_action_dim"), c.get("max_state_dim"))
# run identity guard: shape alone cannot tell this run from another 32/76 checkpoint
tc = json.loads((pathlib.Path(sys.argv[1]) / "train_config.json").read_text())
assert tc["output_dir"].rstrip("/").endswith("trainB/HEAD180-CART20-REL16V3-D600K"), f"not a HEAD180-CART20-REL16V3-D600K checkpoint: {tc['output_dir']}"
assert tc["dataset"]["root"].rstrip("/").endswith("r180_cart20_rel16_v3d"), f"trained on {tc['dataset']['root']}"
print("run identity OK:", tc["output_dir"], "| dataset", tc["dataset"]["root"].split("/")[-1], "| steps", tc["steps"], "| save", tc["save_freq"])
EOS
curl -s -m 10 -X POST http://localhost:$PORT/stop > /dev/null 2>&1
kill $(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t) 2>/dev/null; sleep 3
cd $HOME/holobrain-mac-model || exit 1
V4_CKPT=$LOCAL V4_ACTION_MODE=umi V4_STATE_MODE=cart20 V4_GRIPPER=binary V4_GRIP_THRESH=0.6 \
  V4_CLAMP_MM=0 V4_CLAMP_DEG=0 V4_CHUNK=16 V4_EXEC_K=$EXEC_K V4_DWELL_TIMEOUT_S=$DWELL V4_DTYPE=fp32 V4_N_ACTION=1 \
  V4_GLOBAL_ROT180=0 V4_GLOBAL_MIRROR=0 V4_UI_PORT=$PORT V4_IK_MAX_JOINT_DELTA=$IK_MJD V4_DQ_MAX=0.6 \
  nohup $HOME/xvla-mac/bin/python mac_v4_smoke_ui.py > $HOME/v4_smoke_ui_$PORT.log 2>&1 < /dev/null &
for i in $(seq 1 30); do sleep 5; c=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://localhost:$PORT/ || true); [ "$c" = 200 ] && break; done
curl -s -m 5 http://localhost:$PORT/status | $HOME/xvla-mac/bin/python -c "import json,sys;print(json.load(sys.stdin)['model'])"
echo "UI http://localhost:$PORT  CART20-REL16-v3 $STEP  grip threshold 0.6  IK solver clip $IK_MJD  DQ_MAX 0.6  exec_k $EXEC_K  dwell ${DWELL}s"
