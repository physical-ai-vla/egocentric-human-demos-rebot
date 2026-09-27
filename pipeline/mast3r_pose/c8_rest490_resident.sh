#!/bin/bash
# [2026-09-24] Phase 2 (rest 490 = 380 main + 110 early) from the START with run_perframe_resident.py (resident_v1) on the
# 5080. Validated vs run_perframe.py on 5 chosen sides (contract §21). Checkpoints from the local SSD copy (MD5 OK,
# bitwise-neutral). Per-process rest outputs made before the switch are archived in runs_perproc_rest_archive/.
S=${SHARED_ROOT}; C8=$S/c8; V=$S/trackA_mast3r_pose_v1
export M3_REPO=$S/mast3r_slam_official M3_BASE=$C8 PYTHONPATH=$S/mast3r_slam_official:$S/mast3r_slam_official/thirdparty/mast3r:$S/asmk PYTHONNOUSERSITE=1
case "$(nvidia-smi --query-gpu=name --format=csv,noheader)" in *5080*) ;; *) echo NOT_5080; exit 2;; esac
echo "HOST $(hostname) GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd $C8/work5080_local
# gate: the per-side log redirection (dup2 only) must leave the normal validation side bitwise identical to the resident run
T=$(head -1 $C8/resident_val5.txt); echo $T > $C8/logcheck1.txt; rm -rf $C8/resident_val/logcheck
$S/envs/m3slam/bin/python $V/run_perframe_resident.py --list $C8/logcheck1.txt --in-root $C8/val/c8 --out-dir $C8/resident_val/logcheck | grep -E "^(OK|FAIL|STOP)"
if cmp -s $C8/resident_val/logcheck/$T/ss1.csv $C8/resident_val/resident/$T/ss1.csv; then echo "LOGCHECK BITWISE_IDENTICAL $T -> start rest490"
else echo "LOGCHECK DIFFERENT $T -> rest490 NOT started"; exit 5; fi
grep -c "" $C8/resident_val/logcheck/$T.log | sed 's/^/LOGCHECK log lines /'
$S/envs/m3slam/bin/python $V/run_perframe_resident.py --list $C8/rest_490sides.txt --in-root $C8/val/c8 --out-dir $C8/runs --mem-stop-mib 15872
