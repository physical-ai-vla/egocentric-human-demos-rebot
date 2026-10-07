#!/bin/bash
# [2026-09-28, user] SCRATCH R90 (= the scratch R120 launcher with the R90 episode list; stop at 300k — user 09-28 11:4x). Orig: SCRATCH R120 = c8old_r120_ft.sh with ONE change: init lerobot/xvla-base (B1-old recipe, same code / env / episode list
# / seed / steps 600k / decay 400k as C-old R120; the Mac supervisor stops it at 400k). Physically idle GPU only (Track B pins GPU0 outside Ray).
# [2026-09-28, user] C-old R90 fine-tune (= the R120 launcher with the R90 episode list; stop at 300k — user 09-28 11:4x). Orig: C-old R120 fine-tune = the automatic r150ft600k with ONE change: the episode set (R150 -> frozen nested R120,
# r150_nested_subset_v1.json sha256 8996a20f…, R120 = collection sets 0-19 = R150 minus session 20260908_134759).
# Same init (pretrain 300k-final, ft_init_pretrain300k, md5 004974bb…), weights only (fresh optimizer / scheduler), steps 600k,
# decay 400k, seed 1000, B1 recipe, EEF_TARGET_SOURCE=legacy, LEAD 5 / chunk 30, save 10k, same env. Config parity vs
# r150ft600k/train_config.json is asserted right after launch (allowed diffs: dataset.episodes, output_dir, job_name).
# Usage (Ray entrypoint pinned to node:100.64.0.2, entrypoint_num_gpus 0): bash c8old_r120_ft.sh <gpu index>
set -u
GPU=$1; B=/home/bh-aiteam/c8old; W=/home/bh-aiteam/workspace/bh_rebot_LeRobot; PY=$W/.venv/bin/python; X=$B/xvla
R150=/home/bh-aiteam/holobrain-data/lerobot/rebot_3stack_R150_headview; FI=/home/bh-aiteam/.cache/huggingface/hub/models--lerobot--xvla-base/snapshots/cdb7964e4fe842935d671bfab5a5ebe00a96648c; N=r90scratch600k
OUT=$B/runs/$N; LOGF=$B/runs/$N.log; REF=$B/r150ft600k_train_config_ref.json; EPS=$(cat $B/r90_episodes.txt)
fail() { echo "R90-SCRATCH FAIL: $*"; exit 1; }
# [2026-09-28] code-hash freeze: the exact c8old/xvla files every C-old / scratch FT ran with (unchanged since 2026-09-26 10:52)
( cd $B/xvla && md5sum -c --quiet - ) <<'MD5' || fail "c8old/xvla code hash changed"
70ea32ae91bc028d25d0a485d7802bb3  train_bi.py
6b63986b890ea815b3ea977e09205552  humanik_delta.py
ad3de96c1fe89275a5accbad00205c82  eef_delta.py
2d49b7031a360bdfbab177002595d02d  contrastive_sampler.py
9a732c0bcd1c5e692e875fb63772758d  keep_indices.py
56c8f538e0adf25695e37ed1b90d5610  aug_defaults.py
0a50a52bcea34102122c0becf68d1a4a  lr_groups.py
186c39c81138837f2c160b486ed3529f  rebot_fk_torch.py
MD5
[ "$(sha256sum < $B/r150_nested_subset_v1.json | cut -c1-64)" = "8996a20fc88a8e4bb32dad4b1260556552c911159a06c128eb5eeaa4b52b5744" ] || fail "subset freeze hash changed"
[ "$($PY -c "import json;print(json.load(open('$B/r150_nested_subset_v1.json'))['ladder']['R90']['episodes']==json.loads(open('$B/r90_episodes.txt').read()))")" = "True" ] || fail "r90_episodes.txt != freeze"
[ "$(md5sum < $FI/model.safetensors | cut -c1-32)" = "0bed971480d94ddfe002560bde13a59a" ] || fail "xvla-base md5 changed"
[ -f $REF ] || fail "no r150ft600k reference config ($REF = r150ft600k 020000 train_config.json, copied 2026-09-27)"
[ -e $OUT ] && fail "$OUT exists"
M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU) || fail "no GPU $GPU"; [ "$M" -lt 1500 ] || fail "GPU $GPU busy (${M} MiB)"
export HF_HOME=/home/bh-aiteam/.cache/huggingface HF_HUB_OFFLINE=1 HUMANIK_DELTA=1 HUMANIK_LEAD=5 HUMANIK_ROBOT=1 HUMANIK_TARGET=cmd
export XVLA_EE_AUX=1 EE_AUX_SOURCE=state EE_AUX_SCALE=10 EE_AUX_LAMBDA=2.0 EE_FK_LAMBDA=20 EE_LOG_EVERY=200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XVLA_TF32=1 XVLA_FUSED_ADAM=1 EEF_TARGET_SOURCE=legacy CUDA_VISIBLE_DEVICES=$GPU
$PY $X/train_bi.py --dataset.repo_id=rebot/rebot_3stack_R150_headview --dataset.root=$R150 "--dataset.episodes=$EPS" --policy.path=lerobot/xvla-base --policy.dtype=float32 \
  --output_dir=$OUT --job_name=$N --steps=600000 --save_freq=10000 --log_freq=200 --policy.scheduler_decay_steps=400000 --seed=1000 \
  --batch_size=4 --num_workers=8 --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
  --policy.normalization_mapping='{"STATE":"IDENTITY","ACTION":"IDENTITY","VISUAL":"IDENTITY"}' \
  --rename_map '{"observation.images.global":"observation.images.image","observation.images.left_wrist":"observation.images.image2","observation.images.right_wrist":"observation.images.image3"}' \
  > $LOGF 2>&1 &
P=$!
killtrain() { kill $P 2>/dev/null; sleep 10; kill -9 $P 2>/dev/null; }
# [2026-09-27 v2] parity from the full config LeRobot prints at startup (train_config.json only appears at the first checkpoint)
for i in $(seq 1 60); do grep -aq "sampler:" $LOGF 2>/dev/null && grep -aq "ot_train.py:212 {" $LOGF && break; kill -0 $P 2>/dev/null || break; sleep 10; done
grep -aq "ot_train.py:212 {" $LOGF || { killtrain; fail "no startup config in $LOGF"; }
CFG_DUMP=$OUT.startup_config.json $PY $B/cfg_parity.py $B/ref_B1old_R150_startup.log $LOGF "output_dir,job_name,dataset.episodes,steps,save_freq,log_freq" vs-B1old || { killtrain; fail "config parity vs B1-old FAILED"; }
CFG_DUMP= $PY $B/cfg_parity.py $B/runs/r150ft600k.log $LOGF "output_dir,job_name,dataset.episodes,policy.pretrained_path,policy.output_features.action.shape,log_freq" vs-R150FT || { killtrain; fail "config parity vs r150ft600k FAILED"; }
# one-line status every 60 s: time | step loss grad lr | aux fk (+ FK mm) | step/s samples/s | gpu mem util
( while kill -0 $P 2>/dev/null; do
    T=$(tail -c 60000 $LOGF | tr '\r' '\n'); S=$(echo "$T" | grep -a 'step:' | grep -a 'loss:' | tail -1 | sed 's/.*step:/step:/' | awk '{print $1, $5, $6, $7}')
    E=$(echo "$T" | grep -a 'batch [0-9]*: FK' | tail -1 | grep -oE 'L p50 [0-9.]+|R p50 [0-9.]+|aux_loss [0-9.e+-]+|fk_loss [0-9.e+-]+' | tr '\n' ' ')
    R=$(echo "$T" | grep -aoE '[0-9.]+step/s' | tail -1 | tr -dc '0-9.'); G=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $GPU)
    echo "$(date '+%F %T') | $S | $E(FK mm) | ${R} step/s $(awk "BEGIN{print ${R:-0}*4}") samples/s | gpu $G" >> $B/runs/$N.status.log; sleep 60; done ) &
grep -a -m1 "sampler:" $LOGF
$PY - $OUT.startup_config.json $FI $B/runs/$N.meta.json <<'PYEOF'
import json, sys, time, hashlib
c = json.load(open(sys.argv[1])); init = sys.argv[2]
meta = dict(run=c["job_name"], written=time.strftime("%F %T"), init_path=c["policy"].get("pretrained_path"),
            init_model_md5=hashlib.md5(open(init + "/model.safetensors", "rb").read()).hexdigest(),
            optimizer="FRESH (resume=false, checkpoint_path=None: weights-only init via --policy.path; no optimizer / scheduler / rng state loaded)",
            resume=c.get("resume"), checkpoint_path=c.get("checkpoint_path"), seed=c.get("seed"), steps=c.get("steps"),
            scheduler_decay_steps=c["policy"].get("scheduler_decay_steps"), terminal_step=300000, primary_step=250000,
            episodes=len(c["dataset"].get("episodes") or []), subset="r150_nested_subset_v1.json R90 (sha256 8996a20f…)")
assert meta["resume"] is False and meta["checkpoint_path"] is None, meta
json.dump(meta, open(sys.argv[3], "w"), indent=1); print("META", json.dumps(meta))
PYEOF
wait $P; rc=$?; echo "R90-SCRATCH train exited rc $rc"; grep -qi "out of memory" $LOGF && fail "cuda oom"
[ $rc -eq 0 ] && touch $B/runs/$N.done; exit $rc
