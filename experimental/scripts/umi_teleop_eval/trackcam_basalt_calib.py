"""Basalt camera + camera-IMU calibration of the UMI tracking fisheye from a trackcam_imu --board recording (2026-10-15).
Fixed AprilGrid (outputs/aprilgrid_6x6_25mm.json, measure the printed tag!), UMI moved in front of it.

  1. export EuRoC (frames as saved, IMU host clock + the per-session offset from trackcam_imu_v001 as a starting point; Basalt
     re-estimates the residual time offset), mav0/cam1 -> cam0 symlink (Basalt's loader always reads two cameras)
  2. basalt_calibrate      --cam-types kb4   -> <out>/calib_cam/calibration.json (intrinsics)
  3. basalt_calibrate_imu  (needs step 2)    -> <out>/calib_imu/calibration.json (T_imu_cam incl. LEVER ARM, cam_time_offset_ns, biases)
  4. summary: intrinsics, T_imu_cam translation (cm) + rotation vs our R_CI, time offset, and the tool's reprojection report
    python scripts/trackcam_basalt_calib.py datasets/trackcam_imu/<ts> [--tag-size 0.025]"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from trackcam_basalt_eval import export, BAS   # noqa: E402


def run(cmd, cwd):
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(BAS / "lib")); r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    (Path(cwd) / "log.txt").write_text(r.stdout + "\n" + r.stderr); return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec"); ap.add_argument("--tag-size", type=float, default=0.025); ap.add_argument("--reuse-export", action="store_true"); a = ap.parse_args()
    E = Path(a.rec).resolve(); out = E / "basalt_calib"; out.mkdir(exist_ok=True)
    off = float(yaml.safe_load((ROOT / "configs/calibration/trackcam_imu_v001.yaml").read_text())["offset_ms"])
    if not (a.reuse_export and (out / "mav0").exists()): print("frames exported:", export(E, out, off))
    grid = out / "aprilgrid.json"; grid.write_text(json.dumps(dict(tagCols=6, tagRows=6, tagSize=a.tag_size, tagSpacing=0.3)))
    c1, c2 = out / "calib_cam", out / "calib_imu"; c1.mkdir(exist_ok=True); c2.mkdir(exist_ok=True)
    r = run([str(BAS / "bin/basalt_calibrate"), "--dataset-path", str(out), "--dataset-type", "euroc", "--aprilgrid", str(grid), "--result-path", str(c1) + "/",
             "--cam-types", "kb4", "kb4", "--no-gui"], c1)
    print("== basalt_calibrate exit", r.returncode); print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if any(k in l.lower() for k in ("error", "mean", "reproj", "intrinsics", "loaded", "corners", "fail")))[-2500:])
    if not (c1 / "calibration.json").exists(): print("no camera calibration -> stop"); return
    shutil.copy(c1 / "calibration.json", c2 / "calibration.json")             # calibrate_imu starts from the camera result in its result path
    r = run([str(BAS / "bin/basalt_calibrate_imu"), "--dataset-path", str(out), "--dataset-type", "euroc", "--aprilgrid", str(grid), "--result-path", str(c2) + "/",
             "--gyro-noise-std", "0.000282", "--accel-noise-std", "0.016", "--gyro-bias-std", "0.0001", "--accel-bias-std", "0.001", "--no-gui"], c2)
    print("== basalt_calibrate_imu exit", r.returncode); print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if any(k in l.lower() for k in ("error", "mean", "reproj", "offset", "fail", "bias")))[-2500:])
    res = c2 / "calibration.json"
    if not res.exists(): print("no imu calibration"); return
    v = json.loads(res.read_text())["value0"]; T = v["T_imu_cam"][0]; q = [T["qx"], T["qy"], T["qz"], T["qw"]]
    R_IC = R.from_quat(q).as_matrix(); Rci_ours = np.asarray(yaml.safe_load((ROOT / "configs/calibration/trackcam_imu_v001.yaml").read_text())["R_CI"], float)
    d_rot = np.degrees(np.linalg.norm(R.from_matrix(R_IC.T @ Rci_ours.T).as_rotvec()))
    S = dict(intrinsics=v["intrinsics"][0], resolution=v["resolution"][0], cam_in_imu_cm=[round(T[k] * 100, 2) for k in ("px", "py", "pz")],
             lever_arm_cm=round(float(np.linalg.norm([T["px"], T["py"], T["pz"]]) * 100), 2), rotation_diff_vs_ours_deg=round(float(d_rot), 2),
             cam_time_offset_ms=round(v.get("cam_time_offset_ns", 0) / 1e6, 3), applied_offset_ms=off, calib_accel_bias=v.get("calib_accel_bias")[:3], calib_gyro_bias=v.get("calib_gyro_bias")[:3])
    (out / "summary.json").write_text(json.dumps(S, indent=1)); print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
