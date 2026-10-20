"""Runs INSIDE the (throw-away, --rm) kalibr:noetic container before calibration. Makes Kalibr read OUR printed AprilGrid
(ego_collector/outputs/aprilgrid_6x6_25mm.pdf):
  1. blackTagBorder from $KALIBR_BLACK_TAG_BORDER (printed board = 1 bit; Kalibr default 2) in the imu-cam and camera-only calibrators.
  2. $KALIBR_TAG_ROT180=1: our tags are drawn in the standard AprilTag/OpenCV orientation, while kalibr_create_target_pdf draws every tag
     rotated 180 deg (verified 10-16: OpenCV corner0 of each tag is top-left on our sheet, bottom-right on Kalibr's). Same id layout
     (id0 bottom-left, +x then rows up). Kalibr therefore assigns each tag's 4 corners rotated by 180 deg -> ~70 px residuals and a
     diverging camera calibration. Fix: after corner extraction, move every observed corner (r, c) of a tag to (1-r, 1-c) within the tag."""
import re
from pathlib import Path

PY = Path("/catkin_ws/src/kalibr/aslam_offline_calibration/kalibr/python")
BORDER = '{ind}options.blackTagBorder = int(__import__("os").environ.get("KALIBR_BLACK_TAG_BORDER", "2"))\n'

for f in (PY / "kalibr_imu_camera_calibration/IccSensors.py", PY / "kalibr_camera_calibration/CameraCalibrator.py"):
    s = f.read_text(); m = re.search(r"\n([ \t]*)options = acv_april\.AprilgridOptions\(\)[ \t]*\n", s)
    assert m, f"AprilgridOptions not found in {f}"
    s = s[:m.end()] + BORDER.format(ind=m.group(1)) + s[m.end():]; f.write_text(s); print(f"[patch] blackTagBorder <- env in {f.name}")

TE = PY / "kalibr_common/TargetExtractor.py"
assert "def extractCornersFromDataset" in TE.read_text(), "extractCornersFromDataset missing"
TE.write_text(TE.read_text() + '''

# ---- ego_collector patch (kalibr_runs/patch_kalibr.py): tags printed in standard orientation -> rotate corner ids 180 deg per tag
import os as _os
_orig_extractCornersFromDataset = extractCornersFromDataset
def _rot180_tag_corners(obs, ncols):
    idx = [int(i) for i in obs.getCornersIdx()]; pts = obs.getCornersImageFrame()
    moved = {}
    for k, i in enumerate(idx):
        r, c = divmod(i, ncols)
        moved[(2 * (r // 2) + 1 - r % 2) * ncols + 2 * (c // 2) + 1 - c % 2] = np.array(pts[k], dtype=float).ravel()
    for i in idx: obs.removeImagePoint(i)
    for j, p in moved.items(): obs.updateImagePoint(j, p)
def extractCornersFromDataset(*args, **kwargs):
    obs = _orig_extractCornersFromDataset(*args, **kwargs)
    if _os.environ.get("KALIBR_TAG_ROT180", "0") == "1" and len(obs):
        ncols = int(_os.environ.get("KALIBR_GRID_POINT_COLS", "0")) or int(obs[0].target().cols())
        for o in obs: _rot180_tag_corners(o, ncols)
        print("  [ego patch] tag corners rotated 180 deg for %d observations (point cols %d)" % (len(obs), ncols))
    return obs
''')
print(f"[patch] rot180 wrapper appended to {TE.name}")

# 3. $KALIBR_TIMESHIFT_PRIORS="-0.050,-0.045" (s, per camera in camchain order): replaces Kalibr's time-shift prior (un-demeaned full
#    correlation over a 100 knot/s camera spline; gave -5.39 s on cam0 of 20261016_190404_calib although board-vs-gyro xcorr says 40-50 ms).
IS = PY / "kalibr_imu_camera_calibration/IccSensors.py"
s = IS.read_text(); key = "    def findTimeshiftCameraImuPrior(self, imu, verbose=False):\n"
assert key in s, "findTimeshiftCameraImuPrior not found"
s = s.replace(key, key + """        _pri = __import__("os").environ.get("KALIBR_TIMESHIFT_PRIORS", "").strip()
        if _pri:
            _vals = [float(v) for v in _pri.split(",")]; _i = getattr(IccCamera, "_ego_prior_idx", 0); IccCamera._ego_prior_idx = _i + 1
            self.timeshiftCamToImuPrior = _vals[min(_i, len(_vals) - 1)]
            print("  [ego patch] time shift prior (t_imu = t_cam + shift) set externally for camera %d: %.4f s" % (_i, self.timeshiftCamToImuPrior))
            return
""", 1)
IS.write_text(s); print("[patch] external time-shift prior hook in IccSensors.py")

# 4. $KALIBR_BZ_CAM (e.g. 2 = dof of a reprojection error): Blake-Zisserman robust M-estimator on reprojection errors, hard-coded -1 (off)
#    in kalibr_calibrate_imu_camera. Down-weights outlier corner observations (190404: reproj mean 3 px vs median 1.2 px).
SC = PY / "kalibr_calibrate_imu_camera"; s = SC.read_text()
assert "blakeZisserCam=-1," in s, "blakeZisserCam=-1 not found"
SC.write_text(s.replace("blakeZisserCam=-1,", 'blakeZisserCam=float(__import__("os").environ.get("KALIBR_BZ_CAM", "-1")),', 1)); print("[patch] blakeZisserCam <- env KALIBR_BZ_CAM")
