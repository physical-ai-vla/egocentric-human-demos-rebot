#!/bin/bash
# [2026-09-28] robot_like_v1 offline check for one recorded session, end to end, in the background:
#   export (ORB-SLAM format, same exporter + flags as C8) -> 5090 NFS -> MASt3R-SLAM on the RTX 5080 via Ray
#   (c8_run_5080.sh, canonical C8 config, pinned node:100.64.0.3) -> pull ss1.csv -> robotlike_offline_check.py
#   -> <episode>/derived/robot_like/offline_check.json + <session>/robot_like_offline_report.json
# MASt3R is ~5.5 min per wrist on the 5080, so a session of N episodes needs ~11*N GPU-minutes: this is a per-session
# report, not per-episode feedback. Re-running skips everything already done (exports, sides with ss1.csv, checks).
# Never touches the raw episode except derived/robot_like/. Old C8 data is not read or written.
#
# usage: robotlike_session_check.sh <raw session dir>          e.g. datasets/human_handumi_raw/HRL80/HRL80_20260928_101500
# env (2026-10-03, single-arm takes): SIDES="right" (default "left right"), LIMIT=N (first N kept episodes only, pilots),
#      SKIP_CHECK=1 (stop after the poses: the robot check is bimanual), LIST_SUFFIX=_pilot (separate list/todo files)
set -u
SESS=$(cd "${1:?session dir}" && pwd); NAME=$(basename "$SESS")
L=$HOME/c8/robotlike; EXP=$L/export; RUNS=$L/runs; mkdir -p $EXP $RUNS $L/lists $L/logs
# RUNNER=kfdump (2026-10-03): resident runner + keyframe pointmaps (ss1.kf.npz) for known-size scale; own run dirs
[ "${RUNNER:-resident}" = kfdump ] && { RUNS=$L/runs_kfdump; mkdir -p $RUNS; }
NFS=/srv/data/johann/c8/robotlike; MNT=/mnt/shared/johann/c8/robotlike       # same dir seen from the 5090 / the 5080
export RAY_ADDRESS=${RAY_ADDRESS:-http://100.64.0.1:8265}
BASE=$HOME/ego_collector
log () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 1. tags (C8 convention <session ts>_<episode no>_<side>) for every KEPT episode
TS=${NAME#*_}; LIST=$L/lists/$NAME${LIST_SUFFIX:-}.txt; : > $LIST; NS=0
for ep in "$SESS"/episode_*; do
  [ -f "$ep/.complete" ] || continue
  [ -n "${LIMIT:-}" ] && [ $NS -ge $LIMIT ] && break; NS=$((NS + 1))
  for side in ${SIDES:-left right}; do echo "${TS}_$(basename $ep | sed 's/^episode_//')_$side" >> $LIST; done
done
N=$(wc -l < $LIST | tr -d ' '); [ "$N" -gt 0 ] || { log "no KEEP episodes in $SESS"; exit 1; }
log "$NAME: $NS kept episodes, $N sides (${SIDES:-left right})"

# 2. export, 4 in parallel (umi_export_orbslam.py, groot-infer-env, default flags == C8 c8_export.sh).
# A shell function + `export -f`, as in c8_export.sh: macOS xargs -I caps the command at 255 bytes.
job () {
  tag=$1; side=${tag##*_}; e=${tag%_*}; e=${e##*_}; d=$EXP/$tag
  [ -f $d/raw_video.mp4 ] && [ -f $d/imu_data.json ] && return 0
  mkdir -p $d
  if ( cd $BASE && ~/groot-infer-env/bin/python scripts/umi_export_orbslam.py "$SESS/episode_$e" $side "$d" ) > $d/export.log 2>&1; then
    echo "  export OK $tag"; else echo "  export FAIL $tag (see $d/export.log)"; fi
}
export -f job; export BASE EXP SESS
xargs -P 4 -n 1 bash -c 'job "$0"' < $LIST
missing=$(while read t; do [ -f $EXP/$t/imu_data.json ] || echo $t; done < $LIST)
[ -n "$missing" ] && log "export missing for: $(echo $missing)"

# 3. upload inputs to the 5090 NFS (only the three files MASt3R reads)
ssh gpu-5090 "mkdir -p $NFS/in $NFS/runs $NFS/lists" || { log "5090 unreachable"; exit 3; }
while read tag; do
  [ -f $EXP/$tag/imu_data.json ] || continue
  /opt/homebrew/bin/rsync -a --relative "$EXP/./$tag/raw_video.mp4" "$EXP/./$tag/imu_data.json" "$EXP/./$tag/orbslam_setting.yaml" gpu-5090:$NFS/in/ \
    || log "  upload FAIL $tag"
done < $LIST
# 4. MASt3R on the 5080 via Ray (the only GPU path; see ray-submit rule), then wait. The entrypoint lists the input
# dir first: a lookup that ran before an upload (e.g. an aborted earlier job) leaves a negative NFS dentry on the 5080 that
# made c8_run_5080.sh miss a side that was really there (2026-09-28). Sides that still fail get ONE resubmission.
submit_and_wait () {  # <todo list>
  local todo=$1 SID st
  /opt/homebrew/bin/rsync -a $todo gpu-5090:$NFS/lists/$(basename $todo)
  SID=rl-m3-$(echo $NAME | tr '[:upper:]_' '[:lower:]-')-$(date +%H%M%S)
  log "submitting $(wc -l < $todo | tr -d ' ') sides to the 5080 as $SID"
  # RUNNER=resident (default since 2026-09-28): run_perframe_resident.py, contract §21-validated, ~55 s/side faster;
  # RUNNER=perproc: c8_run_5080.sh (one process per side, the C8 ref208 runner). Outputs go to separate dirs.
  if [ "${RUNNER:-resident}" = kfdump ]; then RUN_CMD="bash $MNT/robotlike_resident_kfdump_5080.sh $MNT/lists/$(basename $todo) $MNT/runs_kfdump $MNT/in"
  elif [ "${RUNNER:-resident}" = perproc ]; then RUN_CMD="bash /mnt/shared/johann/c8/c8_run_5080.sh $MNT/lists/$(basename $todo) $MNT/runs $MNT/in"
  else RUN_CMD="bash $MNT/robotlike_resident_5080.sh $MNT/lists/$(basename $todo) $MNT/runs_resident $MNT/in"; fi
  ray job submit --submission-id $SID --entrypoint-num-gpus 1 --entrypoint-resources '{"node:100.64.0.3": 0.01}' --no-wait \
    -- bash -c "ls -la $MNT/in/ > /dev/null; ls $MNT/in/* > /dev/null; $RUN_CMD" \
    > $L/logs/$SID.submit 2>&1 || { log "ray submit failed: $(tail -3 $L/logs/$SID.submit)"; return 4; }
  while :; do
    st=$(ray job status $SID 2>/dev/null | grep -oiE "(pending|running|succeeded|failed|stopped)" | tail -1 | tr '[:lower:]' '[:upper:]')
    case "$st" in SUCCEEDED|FAILED|STOPPED) break ;; esac
    sleep 60
  done
  ray job logs $SID > $L/logs/$SID.log 2>&1
  log "MASt3R job $SID: $st  ($(grep -c '^OK ' $L/logs/$SID.log) OK, $(grep -c '^FAIL ' $L/logs/$SID.log) FAIL, $(grep -cE '^(MEM_STOP|STOP)' $L/logs/$SID.log) STOP)"
}
pull () {
  while read tag; do
    [ -f $RUNS/$tag/ss1.csv ] && continue
    mkdir -p $RUNS/$tag
    if [ "${RUNNER:-resident}" = kfdump ]; then
      /opt/homebrew/bin/rsync -a gpu-5090:$NFS/runs_kfdump/$tag/ss1.csv gpu-5090:$NFS/runs_kfdump/$tag/ss1.kf.npz $RUNS/$tag/ 2>/dev/null \
        && echo runs_kfdump > $RUNS/$tag/runner.txt; continue
    fi
    for d in runs runs_resident; do                       # per-process (C8 runner) or resident_v1 output
      /opt/homebrew/bin/rsync -a gpu-5090:$NFS/$d/$tag/ss1.csv $RUNS/$tag/ 2>/dev/null && { echo $d > $RUNS/$tag/runner.txt; break; }
    done
  done < $LIST
}
pull
todo=$L/lists/$NAME${LIST_SUFFIX:-}.todo.txt; : > $todo
while read tag; do [ -f $RUNS/$tag/ss1.csv ] || { [ -f $EXP/$tag/imu_data.json ] && echo $tag >> $todo; }; done < $LIST   # no input -> never submitted
if [ -s $todo ]; then
  submit_and_wait $todo; pull
  retry=$L/lists/$NAME${LIST_SUFFIX:-}.retry.txt; : > $retry
  while read tag; do [ -f $RUNS/$tag/ss1.csv ] || echo $tag >> $retry; done < $todo
  [ -s $retry ] && { log "retrying $(wc -l < $retry | tr -d ' ') side(s) once"; submit_and_wait $retry; pull; }
fi
log "ss1.csv present for $(while read t; do [ -f $RUNS/$t/ss1.csv ] && echo x; done < $LIST | wc -l | tr -d ' ') / $N sides"

[ "${SKIP_CHECK:-0}" = 1 ] && { log "SKIP_CHECK=1: poses only, robot check skipped"; exit 0; }
# 6. robot check (CPU, pyroki IK) + session report
~/xvla-mac/bin/python $BASE/scripts/robotlike_offline_check.py --session "$SESS" --runs $RUNS --export $EXP 2>&1 \
  | grep -vE "jaxls|INFO|Warning|warn\(" | tee $L/logs/$NAME.check.log
~/xvla-mac/bin/python - "$SESS" <<'EOF'
import json, pathlib, sys, collections
S = pathlib.Path(sys.argv[1]); recs = [json.loads(p.read_text()) for p in sorted(S.glob("episode_*/derived/robot_like/offline_check.json"))]
v = collections.Counter(r["verdict"] for r in recs); ff = collections.Counter()
for r in recs: ff.update(r.get("funnel_first_fail", {}))
segs = sum(r.get("n_segments", 0) for r in recs); ok = ff.get("null", 0) + ff.get("None", 0) + ff.get(None, 0)   # JSON turns the None key into "null"
rep = dict(schema="robot_like_offline_session/v1", session=S.name, episodes=len(recs), verdicts=dict(v),
           segments=segs, robot_feasible_segments=ok, robot_feasible_frac=round(ok / segs, 4) if segs else 0.0,
           segment_first_fail={str(k): n for k, n in ff.items()},
           per_episode={pathlib.Path(r["episode"]).name: dict(verdict=r["verdict"], feasible=r.get("robot_feasible_frac"),
                        d_min_m=r.get("collision_min_m"), margin_deg=r.get("joint_margin_min_deg")) for r in recs})
(S / "robot_like_offline_report.json").write_text(json.dumps(rep, indent=1, default=float))
print(f"SESSION {S.name}: {len(recs)} episodes {dict(v)} | robot-feasible segments {ok}/{segs} | first fails {rep['segment_first_fail']}")
EOF
log "done"
