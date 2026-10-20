"""OpenVINS STEREO-inertial on a stereo_imu_record take through ~/vio/ov_bridge.feed_stereo (2026-10-15), scored with the SAME code as
Basalt (stereo_basalt_eval.py --external-traj). Purpose: OpenVINS runs live from Python on this Mac (Basalt does not without a source
build), so if it matches Basalt here it becomes the real-time tracker backend.
  config  cam0 = left (C0), cam1 = right, equidistant KB4 from stereo_trackcam_v003 (/save_scale), T_imu_cam from stereo_trackcam_imu_v001
          (+ lever --t-ci) and T_CR_CL, use_stereo, use_mask (self-view masks), the 09-16 tuned overrides (CLAHE, chi2 x5, dt_slam_delay,
          max_clones) and NO feature-count LOST
  feed    IMU (device clock -> host + offset), then every left frame with the nearest right frame (|dt| <= 12 ms) as one stereo measurement
  output  <take>/openvins_stereo<tag>/trajectory_imu.tum (t x y z qx qy qz qw of IMU in the VIO world), ms per frame, then the scorer
    python scripts/stereo_openvins_eval.py datasets/stereo_imu/<ts>_eval --t-ci x,y,z [--tag _x] [--mask-left m.npy --mask-right m.npy]"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from handumi_collector.pose.timing import fit_device_to_host          # noqa: E402
from ego_teleop.tracking.backends.openvins import ESTIMATOR_DEFAULTS, ICM42688P_NOISE, _yaml, import_bridge   # noqa: E402
from stereo_basalt_eval import frames                                 # noqa: E402
NOISE_FILE = "configs/calibration/imu_noise_xiao_right_v001.yaml"   # 10-16 XIAO (416 Hz); Teensy era: camera_imu_right_v002.yaml
TUNED = dict(histogram_method="CLAHE", up_msckf_chi2_multipler=5, up_slam_chi2_multipler=5, dt_slam_delay=0.3, max_clones=8,
             calib_cam_extrinsics=False, calib_cam_intrinsics=False, calib_cam_timeoffset=True, init_imu_thresh=0.8, num_pts=200)

STEREO_FILE = "configs/calibration/stereo_trackcam_v004.yaml"   # 10-16 18:56 new camera mount -> v005 (live tracker sets it); old takes = v004
CAMIMU_FILE = "configs/calibration/stereo_trackcam_imu_v003.yaml"   # camera<->IMU; the live tracker sets v004 (10-16 15:40 bracket slip). Old takes = v003


def write_config(out: Path, t_ci, sc: int, overrides: dict) -> Path:
    st = yaml.safe_load((ROOT / STEREO_FILE).read_text()); ci = yaml.safe_load((ROOT / CAMIMU_FILE).read_text())
    T_C0I = np.eye(4); T_C0I[:3, :3] = np.asarray(ci["R_C0I"], float); T_C0I[:3, 3] = t_ci; T_I_C0 = np.linalg.inv(T_C0I); T_I_C1 = T_I_C0 @ np.linalg.inv(np.asarray(st["T_CR_CL"], float))
    cams = {}
    for k, (T, c) in enumerate(((T_I_C0, st["left"]), (T_I_C1, st["right"]))):
        cams[f"cam{k}"] = dict(T_imu_cam=[[float(x) for x in row] for row in T], cam_overlaps=[1 - k], camera_model="pinhole", distortion_model="equidistant",
                               distortion_coeffs=[c["k1"], c["k2"], c["k3"], c["k4"]], intrinsics=[c["fx"] / sc, c["fy"] / sc, c["cx"] / sc, c["cy"] / sc],
                               resolution=[1920 // sc, 1080 // sc], timeshift_cam_imu=0.0, rostopic=f"/cam{k}/image_raw")
    yn = yaml.safe_load((ROOT / NOISE_FILE).read_text())["imu_noise"]; noise = dict(ICM42688P_NOISE); noise.update(yn)
    eye3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    imu = {"imu0": dict(T_i_b=[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], accelerometer_noise_density=float(noise["accelerometer_noise_density"]),
                        accelerometer_random_walk=float(noise["accelerometer_random_walk"]), gyroscope_noise_density=float(noise["gyroscope_noise_density"]),
                        gyroscope_random_walk=float(noise["gyroscope_random_walk"]), rostopic="/imu0", time_offset=0.0, update_rate=float(noise.get("update_rate", 200.0)), model="kalibr",
                        Tw=eye3, R_IMUtoGYRO=eye3, Ta=eye3, R_IMUtoACC=eye3, Tg=[[0.0] * 3] * 3)}
    est = dict(ESTIMATOR_DEFAULTS); est.update(TUNED); est.update(use_stereo=True, max_cameras=2, use_mask=False)   # per-frame masks via feed_stereo apply without use_mask; est.update(overrides)
    out.mkdir(parents=True, exist_ok=True); (out / "kalibr_imucam_chain.yaml").write_text(_yaml(cams)); (out / "kalibr_imu_chain.yaml").write_text(_yaml(imu))
    p = out / "estimator_config.yaml"; p.write_text(_yaml(est)); return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--t-ci", required=True); ap.add_argument("--tag", default=""); ap.add_argument("--offset-ms", type=float, default=None)
    ap.add_argument("--mask-left", default=None); ap.add_argument("--mask-right", default=None); ap.add_argument("--override", action="append", default=[], help="key=value (yaml)"); a = ap.parse_args()
    E = Path(a.take).resolve(); sc = json.loads((E / "meta.json").read_text())["save_scale"]; out = E / f"openvins_stereo{a.tag}"
    off = a.offset_ms if a.offset_ms is not None else float(yaml.safe_load((ROOT / CAMIMU_FILE).read_text())["offset_ms"])
    t_ci = [float(v) for v in a.t_ci.split(",")]; ov = {k: yaml.safe_load(v) for k, v in (o.split("=", 1) for o in a.override)}
    cfg = write_config(out, t_ci, sc, ov); ovb = import_bridge(); br = ovb.Bridge(str(cfg), "WARNING")
    tl, tr = np.load(E / "cam_left_t_ns.npy"), np.load(E / "cam_right_t_ns.npy"); FL, FR = frames(E / "cam_left.mp4", len(tl)), frames(E / "cam_right.mp4", len(tr))
    h, w = FL[0].shape; ml = (np.load(a.mask_left) * 255).astype(np.uint8) if a.mask_left else np.zeros((h, w), np.uint8); mr = (np.load(a.mask_right) * 255).astype(np.uint8) if a.mask_right else np.zeros((h, w), np.uint8)
    z = np.load(E / "imu_right.npz"); ti = fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]).astype(np.int64) + int(off * 1e6)
    o = np.argsort(ti); ti, gy, ac = ti[o], z["gyro"][o], z["acc"][o]; j = 0; rows = []; tproc = []
    for k, t in enumerate(tl[:len(FL)]):
        i = int(np.argmin(np.abs(tr[:len(FR)] - t)))
        if abs(int(tr[i]) - int(t)) > 16e6: continue   # 10-16: 12 -> 16 ms (half a frame; recorder skew p50 12.7 ms)
        while j < len(ti) and ti[j] <= t: br.feed_imu(ti[j] * 1e-9, gy[j].astype(np.float64), ac[j].astype(np.float64)); j += 1
        t0 = time.perf_counter(); inited = br.feed_stereo(int(t) * 1e-9, FL[k], FR[i], ml, mr); tproc.append((time.perf_counter() - t0) * 1000)
        if inited:
            stt = br.state(); T = np.asarray(stt["T_G_I"]); q = R.from_matrix(T[:3, :3]).as_quat(); rows.append([int(t) * 1e-9, *T[:3, 3], *q])
    np.savetxt(out / "trajectory_imu.tum", np.array(rows), fmt="%.9f"); ms = float(np.median(tproc))
    print(f"[openvins] 포즈 {len(rows)} / 쌍 {len(tproc)}, 처리 p50 {ms:.1f} ms p95 {np.percentile(tproc, 95):.1f} ms", flush=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/stereo_basalt_eval.py"), str(E), f"--t-ci= {a.t_ci}", "--tag", f"_ov{a.tag}", "--external-traj", str(out / "trajectory_imu.tum"),
                        "--backend-label", "openvins_stereo_inertial", "--wall-ms", str(ms)], capture_output=True, text=True)
    print(r.stdout[r.stdout.find("{"):] if "{" in r.stdout else r.stdout + r.stderr[-1500:])


if __name__ == "__main__":
    main()
