#!/bin/bash
# [2026-09-25] Mac-side launcher for the Track C-old overnight chain. Stops at the first failed gate (writes the reason).
C8=$HOME/c8; R=bh-aiteam@100.64.0.2; RB=/home/bh-aiteam/c8old; LOG=$C8/c8old_runs/launch.log; mkdir -p $C8/c8old_runs
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
fail() { say "LAUNCH STOPPED: $*"; printf '# Track C-old overnight chain — NOT STARTED\n\nStopped before training: %s\n\nLast PASS: see launch.log\n' "$*" > $C8/c8old_runs/SUMMARY.md; exit 1; }
say "waiting for the dataset write stage"
while pgrep -f write_stage.py >/dev/null; do sleep 30; done
grep -q "c8old_val" $C8/c8old_write.log && [ -f $C8/c8old_data/c8old_provenance.json ] || fail "dataset write did not finish (see c8old_write.log)"
say "dataset written: $(python3 -c "import json;print(json.load(open('$C8/c8old_data/c8old_provenance.json'))['report'])")"
$HOME/xvla-mac/bin/python $C8/c8old_dataset_check.py > $C8/c8old_runs/dataset_check.log 2>&1
grep -v -E "objc|Warning" $C8/c8old_runs/dataset_check.log | tail -8 | tee -a $LOG
grep -q "^CHECKS PASS" $C8/c8old_runs/dataset_check.log || fail "post-build dataset checks FAILED"
$HOME/xvla-mac/bin/python $C8/c8old_gripper_gate.py > $C8/c8old_runs/gripper_gate.log 2>&1
grep -v -E "objc|Warning" $C8/c8old_runs/gripper_gate.log | tail -3 | tee -a $LOG
grep -q "^GRIPPER GATE PASS" $C8/c8old_runs/gripper_gate.log || fail "gripper contract gate FAILED (see c8old_runs/gripper_gate.log)"
python3 -c "import json,sys; sys.exit(0 if json.load(open('$C8/c8old_data/c8old_provenance.json'))['gripper'].startswith('A:') else 1)" || fail "dataset provenance gripper mapping is not A"
say "copying c8old_train / c8old_val to the 4090"
ssh $R "mkdir -p $RB/data"; /opt/homebrew/bin/rsync -a $C8/c8old_data/c8old_train $C8/c8old_data/c8old_val $C8/c8old_data/c8old_provenance.json $R:$RB/data/ || fail "dataset rsync failed"
L=$(cd $C8/c8old_data && find c8old_train c8old_val -type f | LC_ALL=C sort | xargs md5 -r | awk '{print $1"  "$2}' | md5 -q)
RM=$(ssh $R "cd $RB/data && find c8old_train c8old_val -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32")
[ "$L" = "$RM" ] || fail "dataset hash mismatch Mac $L vs 4090 $RM"
say "dataset hash identical on Mac and 4090: $L"
scp -q $C8/c8old/c8old_chain.sh $R:$RB/ && scp -q $C8/c8old/humanik_delta.py $R:$RB/xvla/humanik_delta.py || fail "code copy failed"
CM=$(ssh $R "md5sum < $RB/xvla/humanik_delta.py | cut -c1-32; md5sum < $RB/xvla/train_bi.py | cut -c1-32; git -C /home/bh-aiteam/workspace/bh_rebot_LeRobot rev-parse HEAD 2>/dev/null || echo no-git; md5sum < /home/bh-aiteam/.cache/huggingface/hub/models--lerobot--xvla-base/snapshots/cdb7964e4fe842935d671bfab5a5ebe00a96648c/model.safetensors | cut -c1-32")
set -- $CM; HD=$1 TB=$2 GIT=$3 BASE=$4
[ "$HD" = "$(md5 -q $C8/c8old/humanik_delta.py)" ] || fail "humanik_delta md5 mismatch after copy"
[ "$BASE" = "0bed971480d94ddfe002560bde13a59a" ] || fail "xvla-base weights hash changed ($BASE)"
python3 - <<P
import json, time
prov = dict(created=time.strftime("%F %T"), contract="TRACK_C_PSEUDO_JOINT_CONTRACT.md §22b / §24", humanik_delta_md5="$HD", train_bi_md5="$TB",
            lerobot_workspace_git="$GIT", xvla_base_md5="$BASE", dataset_hash="$L", dataset=json.load(open("$C8/c8old_data/c8old_provenance.json")),
            seed=1000, batch=4, dtype="float32", optimizer="B1 (lerobot xvla preset, fused AdamW)",
            pretrain=dict(steps=300000, decay=300000, save=10000, init="lerobot/xvla-base", EEF_TARGET_SOURCE="measured", HUMANIK_TARGET="cmd (action = pseudo q)"),
            r150_ft=dict(steps=600000, decay=400000, save=10000, init="pretrain300k weights only (fresh optimizer / scheduler)", dataset="rebot_3stack_R150_headview (150 ep)",
                         EEF_TARGET_SOURCE="legacy", primary_comparison_step=250000, baseline="ckpt_E0_R150_CstateD_fp32_250000"),
            env="HUMANIK_DELTA=1 HUMANIK_LEAD=5 HUMANIK_ROBOT=1 XVLA_EE_AUX=1 EE_AUX_SOURCE=state EE_AUX_SCALE=10 EE_AUX_LAMBDA=2.0 EE_FK_LAMBDA=20 XVLA_TF32=1 XVLA_FUSED_ADAM=1",
            gripper="A: state raw = -270 * open_fraction, action = raw / -6 (R150 wrist-video verified: raw 0 = closed, -270 = open; gripper gate 2026-09-26)",
            gripper_distribution=json.load(open("$C8/c8old_runs/gripper_gate/gate.json"))["distribution"],
            gripper_ego_anchors_ok=json.load(open("$C8/c8old_runs/gripper_gate/gate.json"))["ego_anchor_direction_ok"],
            gates=dict(regression="PASS (T0/T1 legacy invariance, measured equivalence; GPU noise floor)", dataset_gate="PASS (561 x 65 x 2: pos max 12.59 mm, rot max 5.0 deg)", loader_checks="PASS", gripper_gate="PASS (c8old_runs/gripper_gate/gate.json)"))
json.dump(prov, open("$C8/c8old_runs/provenance.json", "w"), indent=1)
P
scp -q $C8/c8old_runs/provenance.json $R:$RB/provenance.json
ssh $R "cd $RB && python3 -c \"import json,time; json.dump(dict(regression='PASS', dataset_gate='PASS', dataset_build='PASS', loader_smoke='PASS', dataset_hash='$L', last_update=time.strftime('%F %T')), open('stage_status.json','w'), indent=1)\""
JOB=$(python3 - <<'P'
import json, urllib.request, time
sid = f"c8old-chain-{time.strftime('%m%d%H%M')}"
body = dict(entrypoint="bash /home/bh-aiteam/c8old/c8old_chain.sh", submission_id=sid, entrypoint_num_gpus=1, entrypoint_resources={"node:100.64.0.2": 0.001})
print(json.loads(urllib.request.urlopen(urllib.request.Request("http://100.64.0.1:8265/api/jobs/", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=30).read())["job_id"])
P
) || fail "ray submit failed"
say "submitted Ray job $JOB"
for M in ego r150; do
  nice -n 10 $HOME/xvla-mac/bin/python $C8/c8old_eval_cache.py $M > $C8/c8old_runs/eval_cache_$M.log 2>&1
  grep -q "CACHE_OK $M" $C8/c8old_runs/eval_cache_$M.log && say "eval frame cache $M OK ($(grep 'max |d|' $C8/c8old_runs/eval_cache_$M.log))" || say "WARN eval frame cache $M failed -> evaluator uses the slow loader path"
done
nohup caffeinate -dimsu $HOME/xvla-mac/bin/python $C8/c8old_supervisor.py $JOB > $C8/c8old_runs/supervisor.out 2>&1 &
say "supervisor started (pid $!) under caffeinate"
