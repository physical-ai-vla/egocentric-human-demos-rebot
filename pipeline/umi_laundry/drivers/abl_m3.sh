#!/bin/bash
# [2026-10-12] mask / chunk ablation on Hfold 143136 ep1 (4090 GPU0), queued after the UMIpilot MASt3R job
L=$HOME/c8/robotlike; EXP=$L/export; NFS=/srv/data/johann/c8/robotlike; MNT=/mnt/shared/johann/c8/robotlike; LIST=$L/lists/abl_143136.txt
export RAY_ADDRESS=http://100.64.0.1:8265
while pgrep -f "umipilot_m3.sh|robotlike_session_check_4090" >/dev/null; do sleep 20; done
while read t; do /opt/homebrew/bin/rsync -a --relative "$EXP/./$t/raw_video.mp4" "$EXP/./$t/orbslam_setting.yaml" "$EXP/./$t/imu_data.json" gpu-5090:$NFS/in/; done < $LIST
/opt/homebrew/bin/rsync -a $LIST gpu-5090:$NFS/lists/
SID=rl4090-abl143136-$(date +%H%M%S)
ray job submit --submission-id $SID --entrypoint-resources '{"node:100.64.0.2": 0.001}' --no-wait -- bash -c "ls $MNT/in/* > /dev/null; CUDA_VISIBLE_DEVICES=0 bash $MNT/robotlike_resident_4090.sh $MNT/lists/abl_143136.txt $MNT/runs_resident $MNT/in"
while :; do st=$(ray job status $SID 2>/dev/null | grep -oiE "(succeeded|failed|stopped)" | tail -1); [ -n "$st" ] && break; sleep 60; done
ray job logs $SID > $L/logs/$SID.log 2>&1; echo "job $SID $st"
mkdir -p $L/runs_abl; while read t; do mkdir -p $L/runs_abl/$t; /opt/homebrew/bin/rsync -a gpu-5090:$NFS/runs_resident/$t/ss1.csv $L/runs_abl/$t/ 2>/dev/null; done < $LIST
echo ABL_DONE
