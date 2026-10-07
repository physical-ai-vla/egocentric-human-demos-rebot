#!/bin/bash
# DORMANT (2026-09-26, user): kept as a tool for later; DO NOT RUN before the 300k pretrain is finished and the user decides.
# [2026-09-26] R150 TRANSFER-SPEED PROBE (contract: C-OLD CHECKPOINT SELECTION POST-HOC ADDENDUM). NOT a generalization test:
# the R150 eval episodes (probe val-10) are inside the 150-episode FT data, so this measures adaptation speed only.
# A probe = the FIRST 50k steps of the full R150 FT, bit-for-bit the same recipe as the automatic chain's r150ft600k:
# same steps (600k) / decay (400k) / seed / dataset / optimizer / scheduler / env, weights-only init (fresh optimizer and scheduler),
# stopped externally once checkpoint 050000 (with training_state) is complete, so the winner can be resumed to 600k unchanged.
# Usage (Ray entrypoint, GPU pinned by the Mac probe supervisor, which picks a physically idle GPU; Track B pins GPU0 outside Ray):
#   bash c8old_probe.sh <tag> <init pretrained_model dir> <gpu index>
set -u
TAG=$1 INIT=$2 GPU=$3
B=/home/bh-aiteam/c8old; W=/home/bh-aiteam/workspace/bh_rebot_LeRobot; PY=$W/.venv/bin/python; X=$B/xvla
R150=/home/bh-aiteam/holobrain-data/lerobot/rebot_3stack_R150_headview; N=probe_$TAG; OUT=$B/runs/$N; LOGF=$B/runs/$N.log
REF=$B/runs/r150ft600k/train_config.json; STOP_AT=050000; PS=$B/probe_status.json
pst() { $PY - "$TAG" "$@" <<'P'
import json, sys, time, os
p = "/home/bh-aiteam/c8old/probe_status.json"; d = json.load(open(p)) if os.path.exists(p) else {}
t = d.setdefault(sys.argv[1], {})
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1); t[k] = v
t["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S"); json.dump(d, open(p, "w"), indent=1)
P
}
fail() { echo "PROBE $TAG FAIL: $*"; pst state=FAIL "reason=$*"; exit 1; }
echo "probe $TAG init=$INIT gpu=$GPU host=$(hostname)"
# ---- pre-launch asserts
[ -f $INIT/model.safetensors ] && [ -f $INIT/config.json ] || fail "init is not a pretrained_model dir"
[ -f $INIT.md5 ] && [ "$(md5sum < $INIT/model.safetensors | cut -c1-32)" = "$(cat $INIT.md5)" ] || fail "init md5 mismatch"
[ -f $REF ] || fail "reference config $REF missing (automatic r150ft600k must have started)"
[ -e $OUT ] && fail "$OUT exists (probes are never overwritten)"
M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU) || fail "no GPU $GPU"
[ "$M" -lt 1500 ] || fail "GPU $GPU busy (${M} MiB used)"
pst state=RUNNING gpu=$GPU init=$INIT "init_md5=$(cat $INIT.md5)"
# ---- identical to c8old_chain.sh common_env + r150_ft stage
export HF_HOME=/home/bh-aiteam/.cache/huggingface HF_HUB_OFFLINE=1 HUMANIK_DELTA=1 HUMANIK_LEAD=5 HUMANIK_ROBOT=1 HUMANIK_TARGET=cmd
export XVLA_EE_AUX=1 EE_AUX_SOURCE=state EE_AUX_SCALE=10 EE_AUX_LAMBDA=2.0 EE_FK_LAMBDA=20 EE_LOG_EVERY=1000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XVLA_TF32=1 XVLA_FUSED_ADAM=1 EEF_TARGET_SOURCE=legacy
export CUDA_VISIBLE_DEVICES=$GPU
$PY $X/train_bi.py --dataset.repo_id=rebot/rebot_3stack_R150_headview --dataset.root=$R150 --policy.path=$INIT --policy.dtype=float32 \
  --output_dir=$OUT --job_name=$N --steps=600000 --save_freq=10000 --log_freq=100 --policy.scheduler_decay_steps=400000 --seed=1000 \
  --batch_size=4 --num_workers=8 --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
  --policy.normalization_mapping='{"STATE":"IDENTITY","ACTION":"IDENTITY","VISUAL":"IDENTITY"}' \
  --rename_map '{"observation.images.global":"observation.images.image","observation.images.left_wrist":"observation.images.image2","observation.images.right_wrist":"observation.images.image3"}' \
  > $LOGF 2>&1 &
P=$!
killtrain() { kill $P 2>/dev/null; sleep 10; kill -9 $P 2>/dev/null; }
# ---- config parity vs the automatic FT (only init path / output / job name may differ)
for i in $(seq 1 60); do [ -f $OUT/train_config.json ] && break; kill -0 $P 2>/dev/null || break; sleep 10; done
[ -f $OUT/train_config.json ] || { killtrain; fail "no train_config.json (see $LOGF)"; }
$PY - $REF $OUT/train_config.json $INIT <<'P' || { killtrain; fail "config parity vs r150ft600k FAILED (see log)"; }
import json, sys
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict): o.update(flat(v, p + k + "."))
        else: o[p + k] = v
    return o
a, b = flat(json.load(open(sys.argv[1]))), flat(json.load(open(sys.argv[2]))); init = sys.argv[3]
ALLOW = {"output_dir", "job_name", "policy.pretrained_path", "policy.path", "wandb.run_id", "checkpoint_path"}
diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
bad = {k: v for k, v in diff.items() if k not in ALLOW}
for k in ("steps", "seed", "batch_size", "resume", "policy.scheduler_decay_steps", "policy.optimizer_lr", "dataset.repo_id", "dataset.root"):
    print(f"  {k:32s} ref {a.get(k)!r:40.40s} probe {b.get(k)!r:.40s}")
print("  allowed diffs:", {k: v[1] for k, v in diff.items() if k in ALLOW})
ok = not bad and b.get("steps") == 600000 and b.get("policy.scheduler_decay_steps") == 400000 and b.get("seed") == 1000 and not b.get("resume")
if bad: print("  DISALLOWED DIFFS:", bad)
print("CONFIG PARITY", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
P
# fresh optimizer / scheduler: resume == false is asserted above; the printed config contains the word "resume", so no log grep
pst config_parity=PASS
# ---- run to 050000, prune training_state of earlier checkpoints, stop
while kill -0 $P 2>/dev/null; do
  if tail -c 20000 $LOGF | tr '\r' '\n' | grep -a "ot_train.py:488" | tail -3 | grep -qiE "loss:(nan|inf)|grdn:(nan|inf)"; then killtrain; fail "nan/inf in training log"; fi
  L=$(readlink $OUT/checkpoints/last 2>/dev/null | xargs -r basename)
  for ck in 010000 020000 030000 040000; do [ -n "$L" ] && [ "$L" \> "$ck" ] && rm -rf $OUT/checkpoints/$ck/training_state; done
  if [ "$L" = "$STOP_AT" ] && [ -d $OUT/checkpoints/$STOP_AT/training_state ]; then
    sleep 20; killtrain; echo "stopped at $STOP_AT"; break; fi
  pst "step=$(tail -c 4000 $LOGF | tr '\r' '\n' | grep -aoE 'step:[0-9.]+[KM]?' | tail -1)"
  sleep 30
done
TS=$OUT/checkpoints/$STOP_AT/training_state
[ -d $TS ] || fail "process ended before $STOP_AT (see $LOGF)"
ls $TS | tee /dev/stderr | grep -q "optimizer" || fail "050000 training_state has no optimizer state"
pst state=STOPPED_AT_50K "ts_files=$(ls $TS | tr '\n' ' ')" "model50k_md5=$(md5sum < $OUT/checkpoints/$STOP_AT/pretrained_model/model.safetensors | cut -c1-32)"
touch $B/runs/$N.stopped50k; echo "PROBE $TAG DONE"
