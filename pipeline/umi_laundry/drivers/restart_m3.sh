#!/bin/bash
# [2026-10-12] restart_v1 runner on the 3 episodes (4090 GPU0), queued after the ablation job
L=$HOME/c8/robotlike; NFS=/srv/data/johann/c8/robotlike; MNT=/mnt/shared/johann/c8/robotlike; V=/mnt/shared/johann/trackA_mast3r_pose_v1
export RAY_ADDRESS=http://100.64.0.1:8265
while pgrep -f "abl_m3.sh" >/dev/null; do sleep 20; done
/opt/homebrew/bin/rsync -a $L/lists/restart_v1.txt gpu-5090:$NFS/lists/
SID=rl4090-restart-$(date +%H%M%S)
ray job submit --submission-id $SID --entrypoint-resources '{"node:100.64.0.2": 0.001}' --no-wait -- bash -c "ls $MNT/in/* > /dev/null; CUDA_VISIBLE_DEVICES=0 RUNNER_PY=$V/run_perframe_restart.py bash $MNT/robotlike_resident_4090.sh $MNT/lists/restart_v1.txt $MNT/runs_restart $MNT/in"
while :; do st=$(ray job status $SID 2>/dev/null | grep -oiE "(succeeded|failed|stopped)" | tail -1); [ -n "$st" ] && break; sleep 60; done
ray job logs $SID > $L/logs/$SID.log 2>&1; echo "job $SID $st"
mkdir -p $L/runs_restart; while read t; do mkdir -p $L/runs_restart/$t; /opt/homebrew/bin/rsync -a gpu-5090:$NFS/runs_restart/$t/ $L/runs_restart/$t/ 2>/dev/null; done < $L/lists/restart_v1.txt
echo RESTART_DONE
