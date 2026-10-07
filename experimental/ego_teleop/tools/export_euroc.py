#!/usr/bin/env python3
"""Export one HandUMI wrist stream as a EuRoC-layout mono-inertial sequence, for the VIO bake-off (MASt3R-Fusion, ORB-SLAM3, ...).

    .venv/bin/python -m ego_teleop.tools.export_euroc EPISODE --side right --out /path/seqs [--downscale 2] [--balance 0.0]

Writes  <out>/<session6>_<ep3>_<side>/mav0/cam0/data.csv, data/<t_ns>.png, sensor.yaml ; mav0/imu0/data.csv ; calib.yaml ; config.yaml
  images   fisheye (Kannala-Brandt, configs/calibration/fisheye_<side>) -> PINHOLE via cv2.fisheye.initUndistortRectifyMap (MASt3R-Fusion
           knows pinhole+radtan / omnidir only), grayscale PNG named by the head/host capture time in ns
  imu      the SAME clock model the OpenVINS pipeline uses (pose.timing: device_us -> host ns fit, + frozen camera-IMU offset), so the IMU is
           already on the camera clock and --imu_dt must be 0; columns t_ns, gx gy gz (rad/s), ax ay az (m/s^2)  -> imu_format custom_rad
  calib    Tic = T_imu_cam = inv(T_camera_imu) (our bundle stores the IMU in the camera frame; OpenVINS also received the inverse)
  config   base_euroc.yaml with our Allan noise (accel nd, gyro nd, accel rw, gyro rw) and imu_format custom_rad
Nothing in the acquisition pipeline or calibration is changed; this is a read-only exporter."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, cv2, yaml
from handumi_collector.pose.episode_io import RawEpisode
from handumi_collector.pose.timing import fit_device_to_host, imu_host_times_ns

CAL = Path(__file__).resolve().parents[2] / "configs/calibration"
BUNDLE = {"left": "wrist_bundle_left_v005.yaml", "right": "wrist_bundle_right_v005.yaml"}
FISHEYE = {"left": "fisheye_left_v002.yaml", "right": "fisheye_right_v003.yaml"}
STREAM = {"left": "left_wrist", "right": "right_wrist"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--out", required=True); ap.add_argument("--downscale", type=int, default=2); ap.add_argument("--balance", type=float, default=0.0, help="cv2.fisheye new-K balance: 0 = crop to the valid centre, 1 = keep all")
    ap.add_argument("--base-config", default=None, help="MASt3R-Fusion config/base_euroc.yaml to derive config.yaml from (else a minimal one)")
    a = ap.parse_args(argv)
    ep_dir = Path(a.episode); ep = RawEpisode.load(ep_dir); side = a.side
    fe = yaml.safe_load(open(CAL / FISHEYE[side])); bundle = yaml.safe_load(open(CAL / BUNDLE[side]))
    K = np.array(fe["K"], float); D = np.array(fe["D"], float).reshape(4, 1); W0, H0 = fe["image_size"]
    T_ci = np.array(bundle["camera_imu"]["T_camera_imu"], float); off_ms = float(bundle["camera_imu"]["time_offset_ms"]); nz = bundle["imu_noise"]
    W, H = W0 // a.downscale, H0 // a.downscale; Ks = K.copy(); Ks[:2] /= a.downscale
    Kn = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(Ks, D, (W, H), np.eye(3), balance=a.balance, new_size=(W, H))
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(Ks, D, np.eye(3), Kn, (W, H), cv2.CV_32FC1)
    name = f"{ep_dir.parent.name[-6:]}_{ep_dir.name[-3:]}_{side}"; out = Path(a.out) / name; (out / "mav0/cam0/data").mkdir(parents=True, exist_ok=True); (out / "mav0/imu0").mkdir(parents=True, exist_ok=True)
    rows = []
    for vf, fi, cap_ns, img in ep.iter_frames(STREAM[side], downscale=a.downscale, gray=True):
        und = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR); fn = f"{cap_ns}.png"; cv2.imwrite(str(out / "mav0/cam0/data" / fn), und); rows.append((cap_ns, fn))
    with open(out / "mav0/cam0/data.csv", "w", newline="") as f:
        f.write("#timestamp [ns],filename\n"); csv.writer(f).writerows(rows)
    yaml.safe_dump(dict(sensor_type="camera", comment=f"HandUMI {side} wrist fisheye undistorted to pinhole (balance {a.balance}), from {FISHEYE[side]}",
                        resolution=[int(W), int(H)], camera_model="pinhole", intrinsics=[float(Kn[0, 0]), float(Kn[1, 1]), float(Kn[0, 2]), float(Kn[1, 2])],
                        distortion_model="radial-tangential", distortion_coefficients=[0.0, 0.0, 0.0, 0.0], rate_hz=30), open(out / "mav0/cam0/sensor.yaml", "w"))
    imu = ep.imu[side]; clock = fit_device_to_host(imu.device_us, imu.host_ns); t_imu = imu_host_times_ns(imu, clock, int(off_ms * 1e6))
    order = np.argsort(t_imu, kind="stable"); t_imu, g, acc = t_imu[order], np.asarray(imu.gyro, float)[order], np.asarray(imu.accel, float)[order]
    keep = np.r_[True, np.diff(t_imu) > 0]; t_imu, g, acc = t_imu[keep], g[keep], acc[keep]          # IMUPool demands strictly increasing time
    with open(out / "mav0/imu0/data.csv", "w", newline="") as f:
        f.write("#timestamp [ns],w_x [rad/s],w_y,w_z,a_x [m/s^2],a_y,a_z\n")
        for t, gg, aa in zip(t_imu, g, acc): f.write(f"{int(t)},{gg[0]:.7f},{gg[1]:.7f},{gg[2]:.7f},{aa[0]:.6f},{aa[1]:.6f},{aa[2]:.6f}\n")
    Tic = np.linalg.inv(T_ci)
    yaml.safe_dump(dict(width=int(W), height=int(H), calibration=[float(Kn[0, 0]), float(Kn[1, 1]), float(Kn[0, 2]), float(Kn[1, 2]), 0.0, 0.0, 0.0, 0.0],
                        Tic=[[float(x) for x in r] for r in Tic]), open(out / "calib.yaml", "w"), sort_keys=False)
    noise = [float(nz["accelerometer_noise_density"]), float(nz["gyroscope_noise_density"]), float(nz["accelerometer_random_walk"]), float(nz["gyroscope_random_walk"])]
    cfg = yaml.safe_load(open(a.base_config)) if a.base_config else {}
    cfg.setdefault("ms_opt", {}); cfg["ms_opt"]["imu_noise"] = noise; cfg["ms_opt"]["imu_format"] = "custom_rad"
    cfg.setdefault("global_opt", {}); cfg["global_opt"]["imu_noise"] = noise
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)
    go = ep.protocol_events("go"); meta = dict(episode=str(ep_dir), side=side, n_frames=len(rows), t0_ns=int(rows[0][0]), t1_ns=int(rows[-1][0]), go_ns=[int(x) for x in go],
                                                 imu_rate_hz=float(1e9 / np.median(np.diff(t_imu))), imu_n=int(len(t_imu)), camera_imu_offset_ms=off_ms, K_new=Kn.tolist(), downscale=a.downscale, balance=a.balance,
                                                 clock_fit=clock.to_dict() if hasattr(clock, "to_dict") else str(clock))
    json.dump(meta, open(out / "export_meta.json", "w"), indent=1)
    print(f"{name}: {len(rows)} frames {W}x{H} fx {Kn[0,0]:.1f} | imu {len(t_imu)} @ {meta['imu_rate_hz']:.1f} Hz (+{off_ms} ms) | Tic ok -> {out}"); return 0


if __name__ == "__main__":
    sys.exit(main())
