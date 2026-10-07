#!/bin/bash
# [2026-09-23] C8 stage M on the RTX 5080 ONLY (node:100.64.0.3, gpu-5080b; /srv/data of the 5090 NFS-mounted at /mnt/shared).
# Same canonical config as the 5090 benchmark: same m3slam env (byte copy at johann/envs/m3slam, pip-freeze md5 checked),
# same repo + build_compat_5090 patch (sm_120, shared .so files), same checkpoints (MD5SUMS), run_perframe.py video-direct, ss1 seed0.
# No 5090 fallback: exits 2 unless this host's GPU is an RTX 5080.
# Usage: c8_run_5080.sh <side list> <out dir> [<input root, default c8/val/c8> <output name, default ss1>]. Input dir =
# <input root>/<tag with its first '_' -> '/' when the root is the Track A val dir, else tag>. One side at a time (1 GPU, 1 worker), flush per side, skips done sides.
# VRAM (v2, 2026-09-23): nvidia-smi sampled every 0.5 s -> <out>/<tag>.vram: tree peak MiB = SUM over this run's whole process
# tree (MASt3R-SLAM runs its backend in a torch.multiprocessing child; v1 tracked only the parent and read a constant 6784),
# gpu peak MiB = whole GPU, base MiB = whole GPU just before the side (other users / display).
# Memory rule (user, 2026-09-23): proc peak > 15.5 GB (15872 MiB) or OOM -> STOP the list (never change algorithm settings).
set -u
L=$1; OUT=$2; S=/mnt/shared/johann; C8=$S/c8; IN=${3:-$C8/val/c8}; NAME=${4:-ss1}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *5080*) ;; *) echo "NOT_5080: '$GPU' -> refuse"; exit 2;; esac
export M3_REPO=$S/mast3r_slam_official M3_BASE=$C8
export PYTHONPATH=$M3_REPO:$M3_REPO/thirdparty/mast3r:$S/asmk PYTHONNOUSERSITE=1
PY=$S/envs/m3slam/bin/python
mkdir -p $OUT; cd $C8/work5080
echo "HOST $(hostname) GPU $GPU  pip-freeze md5 $($PY -m pip freeze 2>/dev/null | md5sum | cut -c1-32) (5090 env a766d4894a95b5b4cb2029ba2fcf36bb)"
n=0
for T in $(cat $L); do
  out=$OUT/$T/$NAME.csv; LG=$OUT/$T.log; [ "$NAME" != ss1 ] && LG=$OUT/$T.$NAME.log
  D=$IN/$T; [ -d "$D" ] || D=$IN/${T/_//}
  [ -f $out ] && { echo "SKIP $T"; continue; }
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 15000 ]; do echo "wait: 5080 has <15 GB free (other user)"; sleep 60; done
  t0=$(date +%s)
  $PY $S/trackA_mast3r_pose_v1/run_perframe.py --video $D/raw_video.mp4 --setting $D/orbslam_setting.yaml \
      --subsample 1 --out $out > $LG 2>&1 &
  P=$!; pk=0; gk=0; base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  tree() { local q=$1; echo $q; for c in $(pgrep -P $q); do tree $c; done; }
  while kill -0 $P 2>/dev/null; do
    pids=" $(tree $P | tr '\n' ' ') "
    u=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v L="$pids" 'index(L, " "$1" "){s+=$2} END{print s+0}')
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u" -gt $pk ] && pk=$u; [ "$g" -gt $gk ] && gk=$g; sleep 0.5
  done
  wait $P; rc=$?; dt=$(( $(date +%s) - t0 ))
  echo "tree_peak_MiB $pk gpu_peak_MiB $gk base_MiB $base rc $rc sec $dt" > ${LG%.log}.vram
  if [ $rc -eq 0 ] && [ -f $out ]; then echo "OK $T $(grep '^done' $LG | tail -1)  vram tree $pk / gpu $gk (base $base) MiB  ${dt}s"
  else echo "FAIL $T rc $rc  vram tree $pk / gpu $gk (base $base) MiB"; tail -4 $LG; fi
  if grep -q -i "out of memory" $LG || [ $pk -gt 15872 ]; then echo "MEM_STOP $T tree peak $pk MiB (>15.5 GB or OOM)"; exit 4; fi
  n=$((n+1))
done
echo C8_RUN_DONE $n
