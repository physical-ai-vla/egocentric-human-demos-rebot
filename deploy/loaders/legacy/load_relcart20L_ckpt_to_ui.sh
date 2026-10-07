#!/bin/bash
# [2026-09-29] RELCART20 copy of load_cart20_ckpt_to_ui.sh: V4_STATE_MODE=relcart20 (anchor reset on /run), default EXEC_K 4 (spec §17).
# [2026-09-29] Load a HEAD180-CART20-REL16V3-D600K checkpoint into the robot UI: the REL16-v3 deployment contract with ONE
# difference, the state: V4_STATE_MODE=relcart20 (infer_core_v4.cart20_state, live==dataset parity 5.6e-7). Copy of load_v3_ckpt_to_ui.sh.
#   state cart20 (L/R TCP pos+rot6d + openness) | action REL (umi) chunk 16 | only dims 0..19 reach the controller (dq 20..31 ignored, verified)
#   gripper: continuous g = clip(leader/45, 0, 1) -> binary at V4_GRIP_THRESH = 0.6 (= cmd 27/45, the validated boundary)
#            OPEN -> cmd 42, CLOSE -> cmd 0 (gripper_contract_v2.json)
#   exec_k 16 (EXEC_K=<k> overrides, e.g. the 2026-09-29 exec_k 4 test) | n_action 1 | dwell 1.5 s | clamp off | GLOBAL_ROT180 0 (live cube test) | mirror 0
#   IK: solver-internal clip OFF (2026-09-29 decision; IK_MJD=0.35 gives the old N0), outer DQ_MAX 0.6 + LO/HI kept,
#       every waypoint logged to ~/v4_ik_waypoints.jsonl (DQ_MAX / limit hits)
# usage: load_relcart20_ckpt_to_ui.sh <step, e.g. 10000> [port=8026]
set -u
STEP=$(printf "%06d" "${1:?step}"); PORT=${2:-8031}; IK_MJD=${IK_MJD:-none}; DWELL=${DWELL:-1.5}; GRIPPER=${GRIPPER:-binary};   # binary (thresh 0.6) | continuous (cmd = clip(g,0,1)*45)
 IK_BACKEND=${IK_BACKEND:-numerical}; EXEC_K=${EXEC_K:-1}; RUN=HEAD180-RELCART20-REL16V3L-D600K
K=$((10#$STEP / 1000)); LOCAL=$HOME/holobrain-mac-model/ckpt_UI_HEAD180-RELCART20-REL16V3L_${K}k${UI_TAG:+_$UI_TAG}   # UI-only copy: the eval watcher deletes its own copies
SSD=/Volumes/PortableSSD/rebot_ckpts_archive/trackb_$RUN/$STEP/pretrained_model
NODE=bh-aiteam@100.64.0.2:/home/bh-aiteam/holobrain-data/trainB/$RUN/checkpoints/$STEP/pretrained_model
# [2026-09-29] completeness guard: SSD copy only with .archived + SHA256SUMS match; node copy only after training logged a step
# past it, and the transferred model.safetensors must match the node's sha256.
# [2026-09-29] one download per checkpoint: UIs of the same run share it (UI_TAG dirs are APFS clones of the untagged copy)
DL=${LOCAL%_$UI_TAG}; [ -z "$UI_TAG" ] && DL=$LOCAL
# [2026-09-30] stale-copy guard: a restarted run reuses the run name (and so passes the identity check), and a local copy left from the
# earlier run would be served silently (it happened: :8034 ran the aborted run's 5k). Any existing local copy must match the SOURCE sha
# (SSD SHA256SUMS if archived, else the node file); on mismatch it is deleted and re-fetched.
src_sha () {
  if [ -f "$SSD/../.archived" ]; then grep " pretrained_model/model.safetensors$" "$SSD/../SHA256SUMS" | cut -c1-64
  else ssh -o BatchMode=yes -J head-lp bh-aiteam@100.64.0.2 "sha256sum ${NODE#*:}/model.safetensors 2>/dev/null" | cut -c1-64; fi
}
for d in "$DL" "$LOCAL"; do
  [ -f "$d/model.safetensors" ] || continue
  want=$(src_sha); have=$(shasum -a 256 "$d/model.safetensors" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$have" ] || { echo "stale local copy $d (have ${have:0:16}, source ${want:0:16}) -- deleting and re-fetching"; rm -rf "$d"; }
done
if [ -n "$UI_TAG" ] && [ ! -f "$LOCAL/model.safetensors" ]; then
  until mkdir "$DL.lock" 2>/dev/null; do sleep 10; done
  [ -f "$DL/model.safetensors" ] || { UI_TAG= "$0" "$@" --download-only; }
  rmdir "$DL.lock"
  [ -f "$DL/model.safetensors" ] && { rm -rf "$LOCAL"; cp -cR "$DL" "$LOCAL"; }
fi
if [ ! -f "$LOCAL/model.safetensors" ]; then
  mkdir -p "$LOCAL"
  if [ -f "$SSD/../.archived" ]; then
    ( cd "$SSD/.." && grep "pretrained_model/model.safetensors" SHA256SUMS | shasum -a 256 -c - >/dev/null 2>&1 ) || { echo "SSD copy of $STEP fails its SHA256SUMS"; rm -rf "$LOCAL"; exit 1; }
    /opt/homebrew/bin/rsync -a "$SSD/" "$LOCAL/"
  else
    # [2026-09-30 user] completeness = the log has "Checkpoint policy after step N" AND a later training step line (the save is
    # synchronous, so the next step line means every file is written). Was: log step > K+1 (waited ~2k steps for the K-rounded display).
    ssh -o BatchMode=yes -J head-lp bh-aiteam@100.64.0.2 "awk -v n=$((10#$STEP)) -f - /home/bh-aiteam/holobrain-data/trainB/$RUN.log" <<'AWK' \
      || ssh -o BatchMode=yes -J head-lp bh-aiteam@100.64.0.2 "/home/bh-aiteam/miniforge3/envs/holobrain/bin/python - ${NODE#*:}/.. $((10#$STEP))" <<'PYCHK' \
      || { echo "node checkpoint $STEP not complete yet (no training step logged after its save, and no stopped-run proof)"; rm -rf "$LOCAL"; exit 1; }
{ gsub(/\r/, "") }
s == 0 && match($0, /Checkpoint policy after step [0-9]+/) { if (substr($0, RSTART + 29, RLENGTH - 29) + 0 == n) s = 1; next }
s == 1 && /ot_train\.py:[0-9]+ step:/ { f = 1; exit }
END { exit f ? 0 : 1 }
AWK
# [2026-09-30] stopped-run proof (the disk-guard can stop a run right after a save, so no later step line exists):
# training_state/training_step.json == N and model.safetensors header+data size == file size.
import json, os, struct, sys
d, n = sys.argv[1], int(sys.argv[2])
st = json.load(open(os.path.join(d, "training_state", "training_step.json")))["step"]
p = os.path.join(d, "pretrained_model", "model.safetensors"); h = open(p, "rb"); k = struct.unpack("<Q", h.read(8))[0]
hd = json.loads(h.read(k)); end = max(v["data_offsets"][1] for kk, v in hd.items() if kk != "__metadata__")
ok = st == n and 8 + k + end == os.path.getsize(p)
print(f"[loader] stopped-run proof for {n}: training_step {st}, safetensors complete {8 + k + end == os.path.getsize(p)} -> {ok}")
sys.exit(0 if ok else 1)
PYCHK
    rm -rf "$LOCAL"   # [2026-09-29] parallel byte-range fetch (8 streams; one ssh stream is latency-bound at ~3-4 MB/s) + sha check
    ~/umi_bridge/pget_ckpt.sh "${NODE#*:}" "$LOCAL" 8 || { echo "fetch/sha failed for $STEP"; rm -rf "$LOCAL" "$LOCAL.part"; exit 1; }
  fi
fi
[ "${3:-}" = "--download-only" ] && exit 0
$HOME/xvla-mac/bin/python - "$LOCAL" <<'EOS' || { echo "checkpoint identity/shape guard FAILED -- UI not started"; exit 1; }
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "config.json"; c = json.loads(p.read_text())
if "type" not in c: p.write_text(json.dumps({"type": "xvla", **c}, indent=2))
assert c.get("max_action_dim") == 32 and c.get("max_state_dim") == 20, (c.get("max_action_dim"), c.get("max_state_dim"))
# run identity guard: shape alone cannot tell this run from another 32/76 checkpoint
tc = json.loads((pathlib.Path(sys.argv[1]) / "train_config.json").read_text())
assert tc["output_dir"].rstrip("/").endswith("trainB/HEAD180-RELCART20-REL16V3L-D600K"), f"not a HEAD180-RELCART20-REL16V3L-D600K checkpoint: {tc['output_dir']}"
assert tc["dataset"]["root"].rstrip("/").endswith("r180_relcart20_rel16_v3L"), f"trained on {tc['dataset']['root']}"
print("run identity OK:", tc["output_dir"], "| dataset", tc["dataset"]["root"].split("/")[-1], "| steps", tc["steps"], "| save", tc["save_freq"])
EOS
curl -s -m 10 -X POST http://localhost:$PORT/stop > /dev/null 2>&1
kill $(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t) 2>/dev/null; sleep 3
cd $HOME/holobrain-mac-model || exit 1
V4_CKPT=$LOCAL V4_ACTION_MODE=umi V4_STATE_MODE=relcart20 V4_GRIPPER=$GRIPPER V4_GRIP_THRESH=0.6 \
  V4_CLAMP_MM=0 V4_CLAMP_DEG=0 V4_CHUNK=16 V4_EXEC_K=$EXEC_K V4_DWELL_TIMEOUT_S=$DWELL V4_DTYPE=fp32 V4_N_ACTION=1 \
  V4_GLOBAL_ROT180=0 V4_GLOBAL_MIRROR=0 V4_UI_PORT=$PORT V4_IK_MAX_JOINT_DELTA=$IK_MJD IK_BACKEND=$IK_BACKEND V4_DQ_MAX=0.6 \
  nohup $HOME/xvla-mac/bin/python mac_v4_smoke_ui.py > $HOME/v4_smoke_ui_$PORT.log 2>&1 < /dev/null &
for i in $(seq 1 30); do sleep 5; c=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://localhost:$PORT/ || true); [ "$c" = 200 ] && break; done
curl -s -m 5 http://localhost:$PORT/status | $HOME/xvla-mac/bin/python -c "import json,sys;print(json.load(sys.stdin)['model'])"
echo "UI http://localhost:$PORT  RELCART20-REL16-v3L(intent) $STEP  (anchor = first observation after /run)  gripper $GRIPPER  IK solver clip $IK_MJD  DQ_MAX 0.6  exec_k $EXEC_K  dwell ${DWELL}s  IK $IK_BACKEND"
