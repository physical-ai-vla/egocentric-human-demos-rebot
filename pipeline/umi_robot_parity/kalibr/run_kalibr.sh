#!/usr/bin/env bash
# 카메라-IMU 공동 캘리브레이션 (Kalibr kalibr_calibrate_imu_camera: R_CI, t_CI, 시간 오프셋 동시 추정) — 원커맨드.
#   ~/vio/kalibr/run_kalibr.sh <take_dir> [--force-export] [-- <추가 kalibr 인자...>]
# 1) Mac: scripts/kalibr_bag_export.py --check  ->  <take>/kalibr/export/{calib.bag,camchain,imu,aprilgrid}.yaml
# 2) rsync -> gpu-4090:/home/bh-aiteam/kalibr_runs/<take>/   (디스크 여유 30 GB 미만이면 중단)
# 3) ray job submit (4090 핀, CPU 전용, docker kalibr:noetic, GPU 미사용)  -> 끝날 때까지 대기
# 4) 결과(camchain-imucam.yaml, results-imucam.txt, report-imucam.pdf, imu.yaml, 로그) -> <take>/kalibr/ ; 원격 bag 삭제
# 시간 prior: export_meta.json time_prior (보드-자이로 상관)를 Kalibr에 주입 (KALIBR_OWN_PRIOR=1 이면 Kalibr 자체 상관)
# ACC_DELAY_MS: 가속도계가 자이로보다 늦는 양 (가속도 샘플을 그만큼 앞당겨 씀), KALIBR_BZ_CAM=2: 재투영 robust(Blake-Zisserman),
# KALIBR_OUT_TAG=_x: 결과를 <take>/kalibr_x 에 (실험 병렬 실행용)
# 환경변수: KALIBR_MODE=imucam(기본) | camonly (cam0 카메라 단독 Kalibr = 보드/코너 검출 점검, 결과 <take>/kalibr/camonly/)
#          KALIBR_BLACK_TAG_BORDER (기본 1), KALIBR_TAG_ROT180 (기본 1) = 인쇄된 aprilgrid_6x6_25mm.pdf 보드용 패치 (patch_kalibr.py),
#          KALIBR_MAX_HOURS (기본 3), KEEP_REMOTE_BAG=1,
#          PRE_OFFSET_MS (기본 0: IMU 스탬프에 미리 더할 오프셋; Kalibr 시간 prior가 엉뚱하면 33.25 등으로 재시도)
set -euo pipefail
EGO=/Users/jeonghwanlee/ego_collector
PY=$EGO/.venv/bin/python
RSYNC=/opt/homebrew/bin/rsync
NODE=bh-aiteam@gpu-4090
REMOTE_ROOT=/home/bh-aiteam/kalibr_runs
MIN_FREE_GB=30
export RAY_ADDRESS=${RAY_ADDRESS:-http://100.64.0.1:8265}

[ $# -ge 1 ] || { echo "사용법: $0 <take_dir> [--force-export] [-- kalibr 인자...]"; exit 1; }
TAKE=$(cd "$1" && pwd); shift
FORCE=0; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --force-export) FORCE=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "알 수 없는 인자: $1"; exit 1 ;;
  esac
done
TAG=${KALIBR_OUT_TAG:-}
NAME=$(basename "$TAKE")$TAG
OUT=$TAKE/kalibr$TAG; EXP=$OUT/export; RDIR=$REMOTE_ROOT/$NAME
ACC=${ACC_DELAY_MS:-0}
mkdir -p "$OUT"
LOG=$OUT/run_kalibr_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "== Kalibr 카메라-IMU 캘리브레이션: $NAME  ($(date '+%F %T'))"

# 1) export
PRE=${PRE_OFFSET_MS:-0}
OLD_PRE=$([ -f "$EXP/export_meta.json" ] && "$PY" -c "import json,sys; print(float(json.load(open(sys.argv[1]))['pre_offset_ms']))" "$EXP/export_meta.json" || echo none)
OLD_ACC=$([ -f "$EXP/export_meta.json" ] && "$PY" -c "import json,sys; print(float(json.load(open(sys.argv[1])).get('acc_delay_ms', 0.0)))" "$EXP/export_meta.json" || echo none)
if [ $FORCE = 1 ] || [ ! -f "$EXP/calib.bag" ] || [ "$OLD_PRE" != "$(python3 -c "print(float('$PRE'))")" ] || [ "$OLD_ACC" != "$(python3 -c "print(float('$ACC'))")" ]; then
  echo "== [1/4] bag 변환 (pre_offset ${PRE} ms, acc_delay ${ACC} ms)"
  (cd "$EGO" && "$PY" scripts/kalibr_bag_export.py "$TAKE" --check --pre-offset-ms "$PRE" --acc-delay-ms "$ACC" --out "$EXP")
else
  echo "== [1/4] 기존 bag 재사용 ($EXP/calib.bag, 다시 만들려면 --force-export)"
fi
BAG_GB=$(( $(stat -f %z "$EXP/calib.bag") / 1000000000 + 1 ))

# 2) disk + rsync
echo "== [2/4] 4090 디스크 확인 + 업로드 (bag ${BAG_GB} GB)"
FREE_GB=$(ssh -o ConnectTimeout=15 $NODE "df -BG --output=avail / | tail -1 | tr -dc 0-9")
echo "   4090 / 여유 ${FREE_GB} GB"
if [ $(( FREE_GB - BAG_GB )) -lt $MIN_FREE_GB ]; then echo "!! 업로드 후 여유가 ${MIN_FREE_GB} GB 미만이 됩니다 — 중단"; exit 3; fi
ssh $NODE "mkdir -p $RDIR"
$RSYNC -a --partial --info=progress2 -z "$EXP/" "$NODE:$RDIR/"
$RSYNC -a /Users/jeonghwanlee/vio/kalibr/remote_kalibr.sh /Users/jeonghwanlee/vio/kalibr/patch_kalibr.py "$NODE:$RDIR/"
MODE=${KALIBR_MODE:-imucam}
# time-shift prior: board-vs-gyro xcorr from export_meta (Kalibr's own correlation prior can jump by seconds); KALIBR_OWN_PRIOR=1 to disable
PRIORS=""
if [ "${KALIBR_OWN_PRIOR:-0}" != 1 ]; then
  PRIORS=$("$PY" -c "import json,sys; p=json.load(open(sys.argv[1])).get('time_prior',{}); v=[p[k]['kalibr_prior_s'] for k in sorted(p)]; print(','.join(str(x) for x in v) if v and all(x==x for x in v) else '')" "$EXP/export_meta.json" 2>/dev/null || true)
fi
echo "   Kalibr 시간 prior: ${PRIORS:-Kalibr 자체 상관}"

# 3) ray submit (CPU only; no GPU requested, docker without --gpus)
# keep earlier local results out of the way (a failed run must not look like a success)
OLD=$(ls "$OUT"/calib-*imucam* "$OUT"/calib-imu.yaml "$OUT"/kalibr_job*.log 2>/dev/null | grep -v camonly || true)
[ "$MODE" = camonly ] && OLD=$(ls -d "$OUT"/camonly "$OUT"/kalibr_job_camonly.log 2>/dev/null || true)
if [ -n "$OLD" ]; then PREV=$OUT/prev_$(date +%Y%m%d_%H%M%S); mkdir -p "$PREV"; mv $OLD "$PREV"/; echo "   이전 결과 -> $PREV"; fi
echo "== [3/4] ray job submit (4090, CPU, mode=$MODE)"
SUB=kalibr_${NAME}_$(date +%H%M%S)
ray job submit --no-wait --submission-id "$SUB" --entrypoint-resources '{"node:100.64.0.2": 0.001}' \
  --runtime-env-json "{\"env_vars\": {\"KALIBR_MODE\": \"$MODE\", \"KALIBR_BLACK_TAG_BORDER\": \"${KALIBR_BLACK_TAG_BORDER:-1}\", \"KALIBR_TAG_ROT180\": \"${KALIBR_TAG_ROT180:-1}\", \"KALIBR_TIMESHIFT_PRIORS\": \"$PRIORS\", \"KALIBR_BZ_CAM\": \"${KALIBR_BZ_CAM:--1}\", \"KALIBR_MAX_HOURS\": \"${KALIBR_MAX_HOURS:-3}\"}}" \
  -- bash "$RDIR/remote_kalibr.sh" "$RDIR" "${EXTRA[@]+"${EXTRA[@]}"}" >/dev/null
echo "   제출됨: $SUB  (대시보드 http://100.64.0.1:8265 , 로그: ray job logs -f $SUB)"
while :; do
  ST=$(curl -s "$RAY_ADDRESS/api/jobs/$SUB" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' 2>/dev/null || echo UNKNOWN)
  case "$ST" in SUCCEEDED|FAILED|STOPPED) break ;; esac
  sleep 20
done   # (no 'ray job logs -f': it lingered ~7 min after the job ended)
ray job logs "$SUB" 2>&1 | tr '\r' '\n' | grep -v "Progress \|^\s*$" > "$OUT/kalibr_job_$MODE.log" || true
echo "   작업 상태: $ST"
grep -E "Extracted corners|Time shift camera to imu|^[-0-9.e]+$|Reprojection error|Gyroscope error|Accelerometer error|timeshift|Optimization failed|Traceback|Error:" "$OUT/kalibr_job_$MODE.log" | head -40 || true

# 4) pull results (small files only) and drop the remote bag
echo "== [4/4] 결과 가져오기 -> $OUT"
$RSYNC -a --exclude 'calib.bag' --include 'calib-*' --include 'export_meta.json' --include '*.log' --include 'camonly/' --include 'camonly/camonly-*' --exclude '*' "$NODE:$RDIR/" "$OUT/" || true
# $OUT/export_meta.json = snapshot of the export Kalibr actually ran on (pre_offset_ms etc.); kalibr_to_camimu.py reads this one first
if [ "${KEEP_REMOTE_BAG:-0}" != 1 ]; then ssh $NODE "rm -f $RDIR/calib.bag" && echo "   원격 bag 삭제 ($RDIR/calib.bag)"; fi
ls -la "$OUT" | grep -E "calib-|\.log|export_meta" || true
SH=$(grep -A1 "Time shift camera to imu (t_imu" "$OUT/kalibr_job_$MODE.log" 2>/dev/null | grep -E "^-?[0-9]+(\.[0-9]+)?(e[-+]?[0-9]+)?$" || true)
for x in $SH; do
  if python3 -c "import sys; sys.exit(0 if abs(float('$x')) > 0.2 else 1)"; then
    echo "!! Kalibr 시간 오프셋 prior가 ${x} s — 움직임(회전) 부족으로 상관 추정 실패. 회전을 더 크게 다시 찍거나 PRE_OFFSET_MS=33.25 로 재시도"
  fi
done
if [ "$ST" != SUCCEEDED ]; then echo "!! Kalibr 작업 실패 ($ST) — $OUT/kalibr_job_$MODE.log 확인"; exit 4; fi
if [ "$MODE" = camonly ]; then
  grep -A3 "reprojection error" "$OUT/camonly/camonly-results-cam.txt" 2>/dev/null || { echo "!! camonly 결과 없음 — $OUT/kalibr_job_camonly.log 확인"; exit 4; }
  echo "== 완료 (camonly: 재투영 오차가 ~1 px 이하이면 보드 정의/코너 검출 OK)"
elif [ -f "$OUT/calib-camchain-imucam.yaml" ]; then
  echo "== 완료. v005 변환(검토 후): (cd $EGO && $PY scripts/kalibr_to_camimu.py $OUT ; 저장은 --write)"
else
  echo "!! camchain-imucam.yaml 없음 — $OUT/kalibr_job_$MODE.log 확인"; exit 4
fi
