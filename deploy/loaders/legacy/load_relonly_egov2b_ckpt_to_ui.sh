#!/bin/bash
# [2026-10-01 user] REL-only domain-6 run (R312C-RELCART20-RELONLY-D6-S600K, 4090) -- copy of load_relonly_v4_ckpt_to_ui.sh;
# (originally:) REL-only ablation (R312C-RELCART20-RELONLY-V4, 5090) into the robot UI. Same deploy contract as the v4 pink UI
# (:8036: relcart20 state, umi action, chunk 16, exec_k 4, continuous gripper, dwell 1.0, joint5 lock) PLUS V4_RELONLY=1
# (channels 20:32 hard-zeroed at the transformer input/output, as in training). Pure Pink only (pinkdq refused by infer_core_v4).
# usage: load_relonly_v4_ckpt_to_ui.sh <step> [port=8037]
set -u
STEP=$(printf "%06d" "${1:?step}"); PORT=${2:-8037}; EXEC_K=${EXEC_K:-4}; DWELL=${DWELL:-1.0}; RUN=EGO-CART20V2-RELONLY-D6-100K-V2B
K=$((10#$STEP / 1000)); LOCAL=$HOME/holobrain-mac-model/ckpt_UI_${RUN}_${K}k_pinklockwy
N=bh-aiteam@100.64.0.2; R=/home/bh-aiteam/holobrain-data/trainB; SRC=$R/$RUN/checkpoints/$STEP/pretrained_model
SSD=/Volumes/PortableSSD/rebot_ckpts_archive/trackb_$RUN/$STEP
if [ -f "$SSD/.archived" ]; then want=$(grep " pretrained_model/model.safetensors$" "$SSD/SHA256SUMS" | cut -c1-64)
else
  # completeness: "Checkpoint policy after step N" followed by a later step line in the run log
  ssh -o BatchMode=yes -J head-lp $N "awk -v n=$((10#$STEP)) '{gsub(/\r/,\"\")} s==0 && match(\$0,/Checkpoint policy after step [0-9]+/){if(substr(\$0,RSTART+29,RLENGTH-29)+0==n)s=1;next} s==1 && (/ot_train\.py:[0-9]+ step:/ || /End of training/){f=1;exit} END{exit f?0:1}' $R/$RUN.log" < /dev/null \
    || ssh -o BatchMode=yes -J head-lp $N "/home/bh-aiteam/miniforge3/envs/holobrain/bin/python -c \"import json,os,struct,sys; d='$R/$RUN/checkpoints/$STEP'; st=json.load(open(d+'/training_state/training_step.json'))['step']; p=d+'/pretrained_model/model.safetensors'; h=open(p,'rb'); k=struct.unpack('<Q',h.read(8))[0]; hd=json.loads(h.read(k)); e=max(v['data_offsets'][1] for kk,v in hd.items() if kk!='__metadata__'); ok=(st==$((10#$STEP)) and 8+k+e==os.path.getsize(p)); print('[loader] stopped-run proof', st, ok); sys.exit(0 if ok else 1)\"" < /dev/null \
    || { echo "5090 checkpoint $STEP not complete yet"; exit 1; }
  want=$(ssh -o BatchMode=yes -J head-lp $N "sha256sum $SRC/model.safetensors" < /dev/null | cut -c1-64)
fi
[ -f "$LOCAL/model.safetensors" ] && [ "$(shasum -a 256 "$LOCAL/model.safetensors" | cut -c1-64)" != "$want" ] && { echo "stale local copy -- re-fetching"; rm -rf "$LOCAL"; }
if [ ! -f "$LOCAL/model.safetensors" ]; then
  rm -rf "$LOCAL.part"; mkdir -p "$LOCAL.part"
  if [ -f "$SSD/.archived" ]; then /opt/homebrew/bin/rsync -a "$SSD/pretrained_model/" "$LOCAL.part/"
  else rm -rf "$LOCAL.part"; ~/umi_bridge/pget_ckpt.sh "$SRC" "$LOCAL.part" 8 < /dev/null || { echo "fetch failed"; exit 1; }; fi
  [ "$(shasum -a 256 "$LOCAL.part/model.safetensors" | cut -c1-64)" = "$want" ] || { echo "sha mismatch"; rm -rf "$LOCAL.part"; exit 1; }
  mv "$LOCAL.part" "$LOCAL"
fi
$HOME/xvla-mac/bin/python - "$LOCAL" <<'EOS' || { echo "identity guard FAILED -- UI not started"; exit 1; }
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "config.json"; c = json.loads(p.read_text())
if "type" not in c: p.write_text(json.dumps({"type": "xvla", **c}, indent=2))
assert c.get("max_action_dim") == 32 and c.get("max_state_dim") == 20
tc = json.loads((pathlib.Path(sys.argv[1]) / "train_config.json").read_text())
assert tc["output_dir"].rstrip("/").endswith("trainB/EGO-CART20V2-RELONLY-D6-100K-V2B"), tc["output_dir"]
assert tc["dataset"]["root"].rstrip("/").endswith("ego_cart20_v2b_train"), tc["dataset"]["root"]
print("run identity OK:", tc["output_dir"], "| steps", tc["steps"])
EOS
curl -s -m 10 -X POST http://localhost:$PORT/stop > /dev/null 2>&1
kill $(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t) 2>/dev/null; sleep 3
cd $HOME/holobrain-mac-model || exit 1
V4_RELONLY=1 V4_CKPT=$LOCAL V4_ACTION_MODE=umi V4_STATE_MODE=relcart20 V4_GRIPPER=continuous V4_GRIP_THRESH=0.6 \
  V4_CLAMP_MM=0 V4_CLAMP_DEG=0 V4_CHUNK=16 V4_EXEC_K=$EXEC_K V4_DWELL_TIMEOUT_S=$DWELL V4_DTYPE=${V4_DTYPE:-fp32} V4_N_ACTION=${V4_N_ACTION:-1} \
  V4_GLOBAL_ROT180=0 V4_GLOBAL_MIRROR=0 V4_UI_PORT=$PORT V4_IK_MAX_JOINT_DELTA=none IK_BACKEND=${IK_BACKEND:-pink} V4_PINK_LOCK=${V4_PINK_LOCK-joint5} \
  V4_DQ_MAX=${DQ_MAX:-0.6} V4_FK_MAX_MM=${FK_MAX_MM:-60} UI_TAG=pinklockwy \
  nohup $HOME/xvla-mac/bin/python mac_v4_smoke_ui.py > $HOME/v4_smoke_ui_$PORT.log 2>&1 < /dev/null &
for i in $(seq 1 30); do sleep 5; [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://localhost:$PORT/)" = 200 ] && break; done
curl -s -m 5 http://localhost:$PORT/status | $HOME/xvla-mac/bin/python -c "import json,sys;print(json.load(sys.stdin)['model'])"
grep -m3 -iE "relonly|zero" $HOME/v4_smoke_ui_$PORT.log
echo "UI http://localhost:$PORT  REL-only R312c $STEP  exec_k $EXEC_K dwell ${DWELL}s IK ${IK_BACKEND:-pink} lock joint5 gripper continuous"
