#!/bin/bash
# [2026-10-03 user] HRA_red RIGHT-ONLY approach prior (4090 run HRA-RIGHTONLY-LOSSMASK-D20-B8-300K; SSD archive or node) -> the Mac UI.
# Same deploy contract as load_relonly_run_ckpt_to_ui.sh (relcart20 state, umi action, chunk 16, exec_k 4, pink + joint5 lock,
# V4_RELONLY=1 zeroing 20:32) PLUS the one-arm contract the model was trained with (infer_core_v4 V4_ARM_ONLY=right):
#   left state = dummy (identity, gL 0), left wrist image = black, left arm + both jaws HOLD (their outputs had no loss),
#   task = "Approach to the red cube" (the only training instruction).
# usage: load_hra_rightonly_ckpt_to_ui.sh <step> [port=8056]      env: RUN (default the 300k run), EXEC_K (4), DWELL (1.0),
#        IK_PROFILE=lock (default: the stacking deploy contract, pink + joint5 lock) | free (pink, joint5 UNLOCKED, full ori at the
#        pink default weights pos 1.0 / ori 0.5, posture-to-seed 1e-3 = continuity) -- user: do not treat lock as the answer, compare both.
#        Hard stops (V4_HRA_*): 6 s, <5 mm x3 chunks, right TCP 80 mm between inferences, 450 mm net travel; output guard 250 mm / 60 deg.
# [2026-10-04 user "다 빼줘"] every HRA stop / refusal is OFF by default (time, still, 80 mm/cycle, 450 mm travel, output guard,
#   left/jaw refusal gate); set HRA_* env (>0, HRA_GATE=1) to turn one back on. The left arm + jaws are still HELD by the core.
# [2026-10-04 user "ckpt가 자동으로 로드가 안되는거 같아"] the UI gets RUN / DS_EXPECT / CKSEL_LOADER (= this script) so its
#   checkpoint dropdown lists the SSD archive and reloads through THIS loader; auto_ui_hra_8056.sh loads each new checkpoint.
# Ports (user): 8053 = stacking / cotrain models, 8056 = HRA right-only. The loader refuses 8053.
# Robot interface = the one the 8053 UI uses (2026-10-04): MIT bridge robot_service_mit.py on :8021, V4_JAW_TORQUE=0, step mode.
# Safety (UI + core, only with V4_ARM_ONLY=right): red MODE banner; /run latches left arm + jaws and every /execute_step that
# deviates is REFUSED and stops the run; output guard refuses chunks > 250 mm / 60 deg; approach stop after V4_HRA_MAX_RUN_S (60 s ROBOT time: step mode runs ~0.2 s of demo per cycle; 6 s stopped after 1-4 cycles, 10-04)
# or predicted right move < V4_HRA_STILL_MM (5 mm) for V4_HRA_STILL_N (3) chunks; stacking prompt buttons disabled.
set -u
STEP=$(printf "%06d" "${1:?step}"); PORT=${2:-8056};
[ "$PORT" = 8053 ] && { echo "8053 is the stacking / cotrain UI -- use 8056 (or another free port) for HRA"; exit 1; }
 EXEC_K=${EXEC_K:-4}; DWELL=${DWELL:-1.0}
RUN=${RUN:-HRA-RIGHTONLY-LOSSMASK-D20-B8-300K}; DS_EXPECT=${DS_EXPECT:-ego_hra_red_rightonly_v1_train}   # [2026-10-07] overridable (A93 runs)
K=$((10#$STEP / 1000)); LOCAL=$HOME/holobrain-mac-model/ckpt_UI_${RUN}_${K}k_rightonly
# [user 2026-10-03: main run on the 4090] checkpoints live on the 4090 (trainB) until the archiver moves them to the SSD
N=bh-aiteam@100.64.0.2; R=/home/bh-aiteam/holobrain-data/trainB; SRC=$R/$RUN/checkpoints/$STEP/pretrained_model
SSD=/Volumes/PortableSSD/rebot_ckpts_archive/trackb_$RUN/$STEP
if [ -f "$SSD/.archived" ]; then want=$(grep " pretrained_model/model.safetensors$" "$SSD/SHA256SUMS" | cut -c1-64)
else
  ssh -o BatchMode=yes -J head-lp $N "awk -v n=$((10#$STEP)) '{gsub(/\r/,\"\")} s==0 && match(\$0,/Checkpoint policy after step [0-9]+/){if(substr(\$0,RSTART+29,RLENGTH-29)+0==n)s=1;next} s==1 && (/ot_train\.py:[0-9]+ step:/ || /End of training/){f=1;exit} END{exit f?0:1}' $R/$RUN.log" < /dev/null \
    || { echo "4090 checkpoint $STEP of $RUN not complete yet"; exit 1; }
  want=$(ssh -o BatchMode=yes -J head-lp $N "sha256sum $SRC/model.safetensors" < /dev/null | cut -c1-64)
fi
[ -n "$want" ] || { echo "no checkpoint $STEP of $RUN (node or SSD)"; exit 1; }
[ -f "$LOCAL/model.safetensors" ] && [ "$(shasum -a 256 "$LOCAL/model.safetensors" | cut -c1-64)" != "$want" ] && { echo "stale local copy -- re-fetching"; rm -rf "$LOCAL"; }
if [ ! -f "$LOCAL/model.safetensors" ]; then
  rm -rf "$LOCAL.part"; mkdir -p "$LOCAL.part"
  if [ -f "$SSD/.archived" ]; then /opt/homebrew/bin/rsync -a "$SSD/pretrained_model/" "$LOCAL.part/"
  else rm -rf "$LOCAL.part"; ~/umi_bridge/pget_ckpt.sh "$SRC" "$LOCAL.part" 8 < /dev/null || { echo "fetch failed"; exit 1; }; fi
  [ "$(shasum -a 256 "$LOCAL.part/model.safetensors" | cut -c1-64)" = "$want" ] || { echo "sha mismatch"; rm -rf "$LOCAL.part"; exit 1; }
  mv "$LOCAL.part" "$LOCAL"
fi
RUN=$RUN DS_EXPECT=$DS_EXPECT $HOME/xvla-mac/bin/python - "$LOCAL" <<'EOS' || { echo "identity guard FAILED -- UI not started"; exit 1; }
import json, os, pathlib, sys
d = pathlib.Path(sys.argv[1]); p = d / "config.json"; c = json.loads(p.read_text())
if "type" not in c: p.write_text(json.dumps({"type": "xvla", **c}, indent=2))
assert c.get("max_action_dim") == 32 and c.get("max_state_dim") == 20, (c.get("max_action_dim"), c.get("max_state_dim"))
tc = json.loads((d / "train_config.json").read_text())
assert tc["output_dir"].rstrip("/").endswith("/" + os.environ["RUN"]), tc["output_dir"]
assert tc["dataset"]["root"].rstrip("/").endswith(os.environ["DS_EXPECT"]), tc["dataset"]["root"]
pp = json.loads((d / "policy_preprocessor.json").read_text())
dom = [st["config"]["domain_id"] for st in pp["steps"] if st.get("registry_name") == "xvla_add_domain_id"]
assert dom == [20], dom
print("run identity OK:", tc["output_dir"], "| domain", dom, "| steps", tc["steps"])
EOS
# [2026-10-04 user "5080b 추론"] REMOTE_INFER=http://100.64.0.3:8791 -> the policy forward runs on the 5080b server
# (xvla_infer_server_nfs.py, Ray job); the checkpoint must be on the shared NFS (/srv/data = /mnt/shared on the 5080b): uploaded here once.
# the choice sticks per port (dropdown reloads and auto_ui_hra_8056.sh keep it): REMOTE_INFER=<url> sets it, REMOTE_INFER= clears it
RF=$HOME/umi_bridge/.auto_ui_d20_e5/$PORT.remote
if [ -n "${REMOTE_INFER+x}" ]; then if [ -n "$REMOTE_INFER" ]; then echo "$REMOTE_INFER" > $RF; else rm -f $RF; fi
else REMOTE_INFER=$(cat $RF 2>/dev/null); fi
RCK=""
if [ -n "$REMOTE_INFER" ]; then
  RNAME=$(basename "$LOCAL"); RDIR=/srv/data/johann/ui_infer/ckpts/$RNAME; case "$REMOTE_INFER" in *100.64.0.5*) RDEF=/srv/data/johann/ui_infer/ckpts ;; *) RDEF=/mnt/shared/johann/ui_infer/ckpts ;; esac   # 5090 = the NFS server itself
  RCK=${REMOTE_CKPT_ROOT:-$RDEF}/$RNAME
  if ! ssh -o BatchMode=yes bh-ai-5090@100.64.0.5 "test -f $RDIR/model.safetensors" < /dev/null; then
    # [2026-10-04 user "서버끼리 받는게 더 빠를텐데"] the 4090 mounts the NFS (/mnt/shared): copy node -> NFS there (~40 s) instead of
    # pushing 3.3 GB over the slow Mac link; the Mac upload stays as the fallback (checkpoint already moved to the SSD)
    ssh -o BatchMode=yes -J head-lp $N "set -e; S=$R/$RUN/checkpoints/$STEP/pretrained_model; D=/mnt/shared/johann/ui_infer/ckpts/$RNAME; \
      test -f \$S/model.safetensors; rm -rf \$D.part; cp -r \$S \$D.part; python3 -c \"import json;p='\$D.part/config.json';c=json.load(open(p));json.dump({'type':'xvla',**{k:v for k,v in c.items() if k!='type'}},open(p,'w'),indent=2)\"; mv \$D.part \$D" < /dev/null \
      && echo "copied $RNAME 4090 -> NFS" || echo "4090 -> NFS copy not possible, uploading from the Mac"
  fi
  if ! ssh -o BatchMode=yes bh-ai-5090@100.64.0.5 "test -f $RDIR/model.safetensors" < /dev/null; then
    echo "uploading $RNAME to the NFS for remote inference ..."
    /opt/homebrew/bin/rsync -a "$LOCAL/" bh-ai-5090@100.64.0.5:$RDIR.part/ < /dev/null \
      && ssh -o BatchMode=yes bh-ai-5090@100.64.0.5 "mv $RDIR.part $RDIR" < /dev/null || { echo "remote upload failed"; exit 1; }
  fi
fi
# [2026-10-04] preload on the inference server BEFORE swapping the UI: a cold checkpoint is read from the NFS (up to ~5 min) and the
# single-threaded server blocks every /infer meanwhile (the UI times out at 30 s)
if [ -n "$REMOTE_INFER" ]; then
  echo "preloading $RCK on $REMOTE_INFER ..."; t0=$(date +%s)
  curl -s -m 900 -X POST "$REMOTE_INFER/load" -H 'Content-Type: application/json' -d "{\"ckpt\":\"$RCK\",\"dtype\":\"${V4_DTYPE:-bf16}\",\"denoise\":${DENOISE_STEPS:-10}}" | grep -q '"ok":true' \
    && echo "  server ready ($(( $(date +%s) - t0 )) s)" || { echo "remote preload FAILED -- UI not swapped"; exit 1; }
fi
curl -s -m 10 -X POST http://localhost:$PORT/stop > /dev/null 2>&1
# [2026-10-04 user] keep the operator's live control settings (k / prefetch / RTC / pass / dwell / speed / interp) across a reload
MODEQ=$(curl -s -m 5 http://localhost:$PORT/status | python3 -c "import json,sys
try:
    c=json.load(sys.stdin)['model']['ctl']; b=lambda v:'1' if v else '0'
    print(f\"k={c['k']}&stream={b(c['stream'])}&speed={c['speed_pct']}&interp={b(c['goal_interp'])}&remote={b(c['remote'])}&prefetch={b(c['prefetch'])}&rtc={b(c['rtc'])}&pass={b(c['pass'])}&dwell={c['dwell_s']}&pflead={c.get('prefetch_lead', -1)}&passlast={b(c.get('pass_last', True))}&plan={b(c.get('plan', False))}&planema={c.get('plan_ema', 0.35)}&planxf={c.get('plan_xfade', 4)}&plansm={c.get('plan_smooth_deg', 3)}&planmin={c.get('plan_min_exec', 6)}\")   # [2026-10-07] PLAN details persist across reloads too
except Exception: pass" 2>/dev/null)
kill $(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t) 2>/dev/null; sleep 3
cd $HOME/holobrain-mac-model || exit 1
IKF=$HOME/umi_bridge/.auto_ui_d20_e5/$PORT.ikprofile   # [2026-10-06] the IK profile sticks per port like the remote choice
if [ -n "${IK_PROFILE:-}" ]; then echo "$IK_PROFILE" > $IKF; else IK_PROFILE=$(cat $IKF 2>/dev/null); fi
IK_PROFILE=${IK_PROFILE:-${V4_HRA_IK_PROFILE:-lock}}
UPF=$HOME/umi_bridge/.auto_ui_d20_e5/$PORT.umipitch   # [2026-10-06] HandUMI camera-axis TCP offset (deg), sticks per port; UMI_TCP_PITCH_DEG=0 clears
if [ -n "${UMI_TCP_PITCH_DEG+x}" ]; then echo "$UMI_TCP_PITCH_DEG" > $UPF; else UMI_TCP_PITCH_DEG=$(cat $UPF 2>/dev/null); fi
UMI_TCP_PITCH_DEG=${UMI_TCP_PITCH_DEG:-0}   # a UI dropdown reload keeps the profile the UI was started with
UOF=$HOME/umi_bridge/.auto_ui_d20_e5/$PORT.umioffset   # [2026-10-06] HandUMI TCP point in the robot TCP frame "x,y,z" mm (hand-eye -94.8,2.4,38.4); UMI_TCP_OFFSET_MM=0,0,0 clears
if [ -n "${UMI_TCP_OFFSET_MM+x}" ]; then echo "$UMI_TCP_OFFSET_MM" > $UOF; else UMI_TCP_OFFSET_MM=$(cat $UOF 2>/dev/null); fi
UMI_TCP_OFFSET_MM=${UMI_TCP_OFFSET_MM:-0,0,0}
WZF=$HOME/umi_bridge/.auto_ui_d20_e5/$PORT.wristzoom   # [2026-10-07 user] right-wrist model-input zoom sticks per port (UI /wrist_zoom writes it); WRIST_ZOOM=1 clears
if [ -n "${WRIST_ZOOM+x}" ]; then echo "$WRIST_ZOOM" > $WZF; else WRIST_ZOOM=$(cat $WZF 2>/dev/null); fi
export V4_WRIST_ZOOM=${WRIST_ZOOM:-1.0}
CSF=$HOME/umi_bridge/.auto_ui_d20_e5/$PORT.cubestop   # [2026-10-07 user] cube-size stop (px, raw right wrist; 0 = off), sticks per port
if [ -n "${CUBE_STOP_PX+x}" ]; then echo "$CUBE_STOP_PX" > $CSF; else CUBE_STOP_PX=$(cat $CSF 2>/dev/null); fi
export V4_CUBE_STOP_PX=${CUBE_STOP_PX:-0}
case "$IK_PROFILE" in
  lock) LOCKV=${V4_PINK_LOCK-joint5}; ORIV=${V4_PINK_ORI:-full} ;;
  free) LOCKV=; ORIV=${V4_PINK_ORI:-full} ;;
  *) echo "IK_PROFILE must be lock | free"; exit 1 ;;
esac
V4_ROBOT=${V4_ROBOT:-http://localhost:8021} V4_JAW_TORQUE=${V4_JAW_TORQUE:-0} V4_STREAM=0 V4_PREFETCH=0 V4_PASS_THROUGH=0 \
V4_HRA_GATE=${HRA_GATE:-0} V4_HRA_MODE=1 V4_ARM_ONLY=right V4_ARM_ONLY_GRIP=hold V4_TASK="Approach to the red cube" \
V4_HRA_MAX_RUN_S=${HRA_MAX_RUN_S:-0} V4_HRA_STILL_MM=${HRA_STILL_MM:-5} V4_HRA_STILL_N=${HRA_STILL_N:-0} V4_ARM_ONLY_MAX_PRED_MM=${HRA_MAX_PRED_MM:-0} V4_ARM_ONLY_MAX_PRED_DEG=${HRA_MAX_PRED_DEG:-0} \
V4_HRA_MAX_CYCLE_MM=${HRA_MAX_CYCLE_MM:-0} V4_HRA_MAX_TRAVEL_MM=${HRA_MAX_TRAVEL_MM:-0} \
V4_RELONLY=1 V4_CKPT=$LOCAL V4_ACTION_MODE=umi V4_STATE_MODE=relcart20 V4_GRIPPER=hold \
  V4_CLAMP_MM=0 V4_CLAMP_DEG=0 V4_CHUNK=16 V4_EXEC_K=$EXEC_K V4_DWELL_TIMEOUT_S=$DWELL V4_DTYPE=${V4_DTYPE:-bf16} V4_N_ACTION=${V4_N_ACTION:-1} \
  V4_GLOBAL_ROT180=0 V4_GLOBAL_MIRROR=0 V4_UI_PORT=$PORT V4_IK_MAX_JOINT_DELTA=none IK_BACKEND=${IK_BACKEND:-pink} V4_PINK_LOCK=$LOCKV V4_PINK_ORI=$ORIV V4_PINK_POSTURE=${V4_PINK_POSTURE:-1e-3} V4_HRA_IK_PROFILE=$IK_PROFILE \
  RUN=$RUN DS_EXPECT=$DS_EXPECT CKSEL_LOADER=$HOME/umi_bridge/load_hra_rightonly_ckpt_to_ui.sh \
  V4_REMOTE_INFER=$REMOTE_INFER V4_REMOTE_CKPT=$RCK V4_DENOISE_STEPS=${DENOISE_STEPS:-10} V4_PREFETCH_LEAD=${PREFETCH_LEAD:--1} V4_UMI_TCP_PITCH_DEG=$UMI_TCP_PITCH_DEG V4_UMI_TCP_OFFSET_MM=$UMI_TCP_OFFSET_MM V4_PASS_LAST=${PASS_LAST:-1} V4_CYCLE_SETTLE_S=${CYCLE_SETTLE_S:-0} V4_DQ_MAX=${DQ_MAX:-0.6} V4_FK_MAX_MM=${FK_MAX_MM:-60} UI_TAG=hra_rightonly_$IK_PROFILE \
  nohup $HOME/xvla-mac/bin/python mac_v4_smoke_ui.py > $HOME/v4_smoke_ui_$PORT.log 2>&1 < /dev/null &
for i in $(seq 1 30); do sleep 5; [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://localhost:$PORT/)" = 200 ] && break; done
curl -s -m 5 http://localhost:$PORT/status | $HOME/xvla-mac/bin/python -c "import json,sys;print(json.load(sys.stdin)['model'])"
grep -m3 -E "ARM_ONLY|relonly" $HOME/v4_smoke_ui_$PORT.log
[ -n "$MODEQ" ] && echo "restored control settings: $(curl -s -m 5 -X POST "http://localhost:$PORT/mode?$MODEQ")"
[ -n "$REMOTE_INFER" ] && echo "remote inference ON ($REMOTE_INFER): $(curl -s -m 10 -X POST "http://localhost:$PORT/mode?remote=1")"
echo "UI http://localhost:$PORT  HRA right-only $RUN $STEP  IK $IK_PROFILE  task 'Approach to the red cube'  exec_k $EXEC_K  left arm + jaws HOLD"
