#!/bin/bash
# [2026-10-15] after rp_finalize FINAL_OK: remove the 4 MASt3R pods, copy session 112743 runs to Mac + SSD, md5-verify
RP=/private/tmp/claude-501/-Users-jeonghwanlee/ff3a1aa0-9c08-43f9-88b4-c7a8ed748192/scratchpad/rp.sh; LOGR=/srv/data/johann/runpod_bundle/rp_finalize_1015.log
while :; do s=$(ssh -o ConnectTimeout=30 bh-ai-5090@gpu-5090 "grep -h -e FINAL_OK -e FINAL_INCOMPLETE $LOGR" 2>/dev/null); [ -n "$s" ] && break; sleep 120; done
echo "$(date +%T) $s"; echo "$s" | grep -q FINAL_OK || { echo "NOT_OK: pods kept"; exit 0; }
for id in cjmdoaw2h0pxy9 wtypssnsaxazh8 ow9c4ogmg8sjwh kktylsw6v4dyhj; do $RP pod delete $id > /dev/null 2>&1 || $RP pod remove $id > /dev/null 2>&1; done
echo "pods left: $($RP pod list 2>/dev/null | python3 -c 'import sys,json;print([p["name"] for p in json.load(sys.stdin)])')"
L=$HOME/c8/robotlike/runs_restart; SSD=/Volumes/PortableSSD/umialign_mast3r/UMIalign_20261015_112743; LIST=$HOME/c8/robotlike/lists/UMIalign_20261015_112743.txt
/opt/homebrew/bin/rsync -a --files-from=<(sed 's#$#/#' $LIST) -r bh-ai-5090@gpu-5090:/srv/data/johann/c8/robotlike/runs_restart/ $L/
mkdir -p $SSD; /opt/homebrew/bin/rsync -a $L/20261015_112743_* $SSD/
c=0; bad=0; for t in $(cat $LIST); do [ -f $L/$t/ss1.csv ] && [ ! -f $L/$t/ss1.progress.json ] && c=$((c+1)); [ "$(md5 -q $L/$t/ss1.csv 2>/dev/null)" = "$(md5 -q $SSD/$t/ss1.csv 2>/dev/null)" ] || bad=$((bad+1)); done
echo "$(date +%T) MAC_SSD_DONE mac $c / $(wc -l < $LIST) mismatch $bad"
