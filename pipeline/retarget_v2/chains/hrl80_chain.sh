#!/bin/bash
# HRL80_20260928_101010: resident MASt3R done -> offline check + report -> per-process vs resident parity (2 sides) ->
# robot-like 4-group table -> HRL80 v1/v2-K-R30/v2-Kr-R30 and v2-TR-R30 retarget.
JOB=rl-m3res-hrl80-101010-142957; S=$HOME/ego_collector/datasets/human_handumi_raw/HRL80/HRL80_20260928_101010; L=$HOME/c8/robotlike
while :; do st=$(ray job status $JOB 2>/dev/null | grep -oiE "(succeeded|failed|stopped)" | tail -1); [ -n "$st" ] && break; sleep 120; done
echo "resident job $st $(date +%H:%M)"; ray job logs $JOB > $L/logs/$JOB.log 2>&1
echo "OK $(grep -c '^OK ' $L/logs/$JOB.log)  FAIL $(grep -c '^FAIL ' $L/logs/$JOB.log)  STOP $(grep -c '^STOP ' $L/logs/$JOB.log)"
bash $HOME/ego_collector/scripts/robotlike_session_check.sh $S > $L/logs/HRL80_20260928_101010.run2.log 2>&1; tail -3 $L/logs/HRL80_20260928_101010.run2.log
# parity: 2 sides first done per-process, re-run with resident_v1 into a separate dir
M=${SHARED_ROOT}/c8/robotlike; ssh <gpu-node> "printf '20260928_101010_000003_left\n20260928_101010_000003_right\n' > ${DATA_ROOT}/c8/robotlike/lists/parity_resident.txt"
PJ=rl-m3res-parity-$(date +%H%M%S)
ray job submit --submission-id $PJ --entrypoint-num-gpus 1 --entrypoint-resources '{"node:<gpu-node-ip>": 0.01}' --no-wait -- bash -c "ls $M/in/* > /dev/null; bash $M/robotlike_resident_5080.sh $M/lists/parity_resident.txt $M/runs_resident_parity $M/in" > /dev/null 2>&1
while :; do st=$(ray job status $PJ 2>/dev/null | grep -oiE "(succeeded|failed|stopped)" | tail -1); [ -n "$st" ] && break; sleep 60; done
for t in 20260928_101010_000003_left 20260928_101010_000003_right; do
  ssh <gpu-node> "cmp -s ${DATA_ROOT}/c8/robotlike/runs/$t/ss1.csv ${DATA_ROOT}/c8/robotlike/runs_resident_parity/$t/ss1.csv && echo 'PARITY $t BITWISE_IDENTICAL' || python3 -c \"
import csv
def rd(p):
    R=list(csv.DictReader(open(p))); return R
a=rd('${DATA_ROOT}/c8/robotlike/runs/$t/ss1.csv'); b=rd('${DATA_ROOT}/c8/robotlike/runs_resident_parity/$t/ss1.csv')
va=[r['is_lost']=='false' for r in a]; vb=[r['is_lost']=='false' for r in b]
print('PARITY $t NOT bitwise: rows',len(a),len(b),'valid',sum(va),sum(vb),'mask_identical',va==vb)\""
done
# robot-like 4-group table (live replay for new = hrl80_live, old = baseline_live)
cd $HOME/ego_collector && ~/xvla-mac/bin/python scripts/robotlike_compare_groups.py --old-live $L/baseline_live --old-offline $L/baseline_offline --new-live $L/hrl80_live --new-session $S --out $L/compare_groups_final.json > $L/logs/compare_groups_final.txt 2>&1; cat $L/logs/compare_groups_final.txt
echo "hrl80 chain done $(date +%H:%M)"
