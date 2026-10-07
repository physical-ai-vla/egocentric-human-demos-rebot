#!/bin/bash
# [2026-10-07 user "slam 끝나면 바로 데이터셋 만들고 학습해줘"] HRA_A100: wait for the fisheye MASt3R runs (4090) -> pull + stage
# (A100_<tag> names, the right_only exporter's tag convention) -> right_only export (cube-PnP metric scale) -> sanity gate ->
# trim rows before the AUTO "go" -> robot-TCP labels (hand-eye M, as robotcam_v2) -> convert -> MIX with HRA_red robotcam_v2
# (166 + A100) -> LeRobot (measured-C922 wrist render) -> 4090: smoke + main MIX run; then RESUME the stopped OLD-only robotcam_v2
# run from its 40k checkpoint on the other GPU (OLD vs OLD+NEW ablation).
set -u
A=$HOME/c8/hra_a100; P=$A/pipe; PY=$HOME/xvla-mac/bin/python; NFS=/srv/data/johann/c8/hra_a100
RAWS=$HOME/ego_collector/datasets/human_handumi_raw/HRA_A100
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for SLAM"
until [ "$(ssh -o ConnectTimeout=20 gpu-5090 "cd $NFS; echo \$(( \$(ls runs/*/ss1.csv 2>/dev/null | wc -l) + \$(ls runs/*.giveup 2>/dev/null | wc -l) ))")" -ge 107 ]; do sleep 120; done
log "SLAM done: $(ssh gpu-5090 "cd $NFS; echo ok \$(ls runs/*/ss1.csv | wc -l) giveup \$(ls runs/*.giveup 2>/dev/null | wc -l)")"
# 1. pull + stage
mkdir -p $A/runs; /opt/homebrew/bin/rsync -a --include='*/' --include='ss1.csv' --include='ss1.json' --include='ss1.calib.yaml' --exclude='*' gpu-5090:$NFS/runs/ $A/runs/
n=0; for d in $A/runs/*/; do t=$(basename $d); [ -f $d/ss1.csv ] || continue; echo "MASt3R-SLAM resident 4090 per-process" > $d/runner.txt
  for L in $HOME/c8/robotlike/runs $HOME/c8/stage_p3/runs; do ln -sfn $d $L/A100_$t; done; ln -sfn $A/export/$t $HOME/c8/robotlike/export/A100_$t; n=$((n+1)); done
log "staged $n runs"
# 2. export (right_only, per session) + sanity
mkdir -p $A/raw
for s in $RAWS/HRA_A100_*; do [ "$(ls $s | grep -c '^episode_')" -gt 0 ] || continue
  (cd $HOME/ego_cart20 && PYTHONPATH=$HOME/ego_cart20 $PY $P/export_a100.py $s $A/raw 2>&1 | grep -vE "jaxls|INFO|Warning|warn\(" | tail -2); done   # occlusion-aware cube blob
$PY - <<PYEOF
import json, glob
L = []
for f in sorted(glob.glob("$A/raw/export_log_*.json")): L += json.load(open(f))
for e in L:   # IMU-VI fallback episodes have no usable PnP approach distance: travel_ratio gate N/A (ratio := 1)
    if e.get("scale_fallback") == "imu_vi" and e.get("status") == "ok" and e.get("net_displacement_m") is not None:
        e["dist_m_range"] = [0.0, float(e["net_displacement_m"])]; e["centre_resid_m"] = e.get("centre_resid_m") or 0.0; e["n_valid_pnp"] = e.get("n_valid_pnp") or 0
json.dump(L, open("$A/raw/export_log_all.json", "w"), indent=1); print("export logs merged", len(L), "ok", sum(e.get("status") == "ok" for e in L), "imu fallback", sum(e.get("scale_fallback") == "imu_vi" and e.get("status") == "ok" for e in L))
PYEOF
(cd $HOME/ego_cart20 && $PY -m ego_cart20.right_only sanity $A/raw --export-log $A/raw/export_log_all.json 2>&1 | tail -2)
# 3. trim at go, robot TCP, convert, mix
$PY $P/trim_go.py $A/raw
(cd $HOME/ego_cart20 && PYTHONPATH=$HOME/ego_cart20 $PY $P/make_raw_robotcam_args.py $A/raw $A/raw_robotcam | tail -1)
(cd $HOME/ego_cart20 && $PY -m ego_cart20.right_only convert $A/raw_robotcam $A/processed_robotcam --val-frac 0.1 2>&1 | tail -1)
$PY $P/make_mix.py $HOME/c8/hra_red/processed_robotcam $A/processed_robotcam $A/processed_mix
# 4. LeRobot export (resumable; SVT-AV1 dealloc crashes -> rerun)
O=$A/lerobot_mix; NAME=ego_hra_mix_robotcam_v1
for i in $(seq 1 30); do [ -f $O/${NAME}_train/EXPORT.json ] && [ -f $O/${NAME}_val/EXPORT.json ] && break
  (cd $HOME/ego_cart20 && $PY ego_cart20/scripts/export_lerobot_right_only_robotcam.py $A/processed_mix $O --name $NAME --workers 4 >> $P/export_lerobot.log 2>&1); done
[ -f $O/${NAME}_train/EXPORT.json ] || { log "LeRobot export failed"; exit 1; }
log "LeRobot export done: $(python3 -c "import json;e=json.load(open('$O/${NAME}_train/EXPORT.json'));print(e['report']['episodes'],'eps',e['report']['rows'],'rows')")"
# 5. MIX smoke + main (waits for a free 4090 GPU)
DS=${NAME}_train LD=$O/${NAME}_train SMR=HRA-RIGHTONLY-MIX-D20-SMOKE MRUN=HRA-RIGHTONLY-ROBOTCAM-MIX-LOSSMASK-D20-B8-300K bash $P/chain_generic.sh > $P/chain_mix.log 2>&1
tail -3 $P/chain_mix.log
G=$(grep -oE "starting main .* on 4090 GPU [01]" $P/chain_mix.log | grep -oE "[01]$")
[ -n "$G" ] || { log "MIX main not started (see chain_mix.log); OLD resume skipped"; exit 1; }
# 6. RESUME OLD-only robotcam_v2 (stopped at 43k, last ckpt 40k) on the other GPU
G2=$((1 - G)); export RAY_ADDRESS=http://100.64.0.1:8265
ray job submit --no-wait --submission-id hra-robotcam-old-resume-$(date +%H%M) --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.2": 0.001}' -- bash -c "ACCELERATE_MIXED_PRECISION=bf16 RESUME=1 RUN_NAME=HRA-RIGHTONLY-ROBOTCAM-LOSSMASK-D20-B8-300K MIN_START_GB=60 HRA_DS=ego_hra_red_rightonly_robotcam_v2_train bash /home/bh-aiteam/train_hra_rightonly_lossmask_d20_b8.sh HRA $G2" 2>&1 | grep -E "submitted|rror"
cd ~; IDLE_EXIT=1000000 nohup bash ~/umi_bridge/trackb_archive_v2_relonly.sh HRA-RIGHTONLY-ROBOTCAM-LOSSMASK-D20-B8-300K >> ~/archive_HRA-RIGHTONLY-ROBOTCAM-LOSSMASK-D20-B8-300K.log 2>&1 < /dev/null & disown
log "ALL SUBMITTED: MIX main on GPU $G, OLD resume on GPU $G2"
