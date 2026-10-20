#!/usr/bin/env bash
# Runs ON the 4090 (Ray entrypoint). Kalibr in the existing kalibr:noetic image (CPU only, no --gpus), throw-away container (--rm).
#   bash remote_kalibr.sh <run_dir> [extra kalibr args...]
# run_dir holds calib.bag camchain.yaml imu.yaml aprilgrid.yaml patch_kalibr.py (rsynced from the Mac). Results land in run_dir (chowned back).
# env: KALIBR_MODE=imucam (default, kalibr_calibrate_imu_camera) | camonly (kalibr_calibrate_cameras on cam0 = board/corner sanity check)
#      KALIBR_BLACK_TAG_BORDER (default 1 = our printed board), KALIBR_TAG_ROT180 (default 1 = our printed board), KALIBR_MAX_HOURS (3)
set -euo pipefail
RUN_DIR=$1; shift || true
EXTRA="$*"
MODE=${KALIBR_MODE:-imucam}
BORDER=${KALIBR_BLACK_TAG_BORDER:-1}
ROT180=${KALIBR_TAG_ROT180:-1}
MAX_H=${KALIBR_MAX_HOURS:-3}
IMAGE=kalibr:noetic
NAME=kalibr_$(basename "$RUN_DIR")_$$
UID_GID="$(id -u):$(id -g)"
cd "$RUN_DIR"
echo "[remote] $(hostname) $(date '+%F %T') mode=$MODE run_dir=$RUN_DIR border=$BORDER rot180=$ROT180 priors='${KALIBR_TIMESHIFT_PRIORS:-kalibr}' bz=${KALIBR_BZ_CAM:--1} extra='$EXTRA'"
df -h / | tail -1
docker image inspect "$IMAGE" >/dev/null || { echo "[remote] 이미지 $IMAGE 없음"; exit 2; }
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
# drop results of earlier runs so a failed run can never hand back stale files
if [ "$MODE" = camonly ]; then rm -rf camonly; else rm -f calib-camchain-imucam.yaml calib-imu.yaml calib-results-imucam.txt calib-report-imucam.pdf; fi
if [ "$MODE" = camonly ]; then
  CMD="mkdir -p camonly && cd camonly && ln -sf ../calib.bag camonly.bag && rosrun kalibr kalibr_calibrate_cameras --bag camonly.bag --topics /cam0/image_raw --models pinhole-equi --target ../aprilgrid.yaml --dont-show-report --bag-freq 3 $EXTRA"
else
  CMD="rosrun kalibr kalibr_calibrate_imu_camera --bag calib.bag --cam camchain.yaml --imu imu.yaml --target aprilgrid.yaml --dont-show-report --timeoffset-padding 0.1 $EXTRA"
fi
rc=0
timeout "${MAX_H}h" docker run --rm --name "$NAME" -e KALIBR_BLACK_TAG_BORDER="$BORDER" -e KALIBR_TAG_ROT180="$ROT180" -e KALIBR_TIMESHIFT_PRIORS="${KALIBR_TIMESHIFT_PRIORS:-}" -e KALIBR_BZ_CAM="${KALIBR_BZ_CAM:--1}" -e MPLBACKEND=Agg \
  -v "$RUN_DIR":/data --entrypoint bash "$IMAGE" -c "
set -e
python3 /data/patch_kalibr.py
source /catkin_ws/devel/setup.bash
cd /data
set +e
$CMD 2>&1
rc=\$?
chown -R $UID_GID /data
exit \$rc
" || rc=$?
echo "[remote] kalibr 종료 코드 $rc  $(date '+%F %T')"
ls -la "$RUN_DIR"
exit $rc
