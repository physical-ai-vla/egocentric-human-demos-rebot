#!/bin/bash
# [2026-09-28] robot_like_v1 MASt3R on the RTX 5080 with run_perframe_resident.py (resident_v1, validated in contract §21:
# per-side body verbatim, model loaded once, RNG restored per side; 3/5 bitwise, 5/5 identical coverage/lost/kf/IMU-VI/
# Phase-3 segments). Same env + local-SSD checkpoint copy as c8_rest490_resident.sh. Canonical settings unchanged.
# usage: robotlike_resident_5080.sh <list on NFS> <out dir> <in root>
S=${SHARED_ROOT}; C8=$S/c8; V=$S/trackA_mast3r_pose_v1
export M3_REPO=$S/mast3r_slam_official M3_BASE=$C8 PYTHONPATH=$S/mast3r_slam_official:$S/mast3r_slam_official/thirdparty/mast3r:$S/asmk PYTHONNOUSERSITE=1
case "$(nvidia-smi --query-gpu=name --format=csv,noheader)" in *5080*) ;; *) echo NOT_5080; exit 2;; esac
echo "HOST $(hostname) GPU $(nvidia-smi --query-gpu=name --format=csv,noheader) runner resident_v1"
cd $C8/work5080_local
$S/envs/m3slam/bin/python $V/run_perframe_resident.py --list "$1" --in-root "$3" --out-dir "$2" --mem-stop-mib 15872
echo RESIDENT_DONE
