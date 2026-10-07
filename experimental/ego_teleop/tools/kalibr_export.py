"""Raw episode -> Kalibr-ready ROS1 bag (+ target yaml) for OFFLINE camera-IMU calibration — step 3 of M1-0, the PRIMARY
camera-IMU procedure (OpenVINS online refinement only validates it afterwards).

    python -m ego_teleop.tools.kalibr_export EPISODE --side right --out calib_right.bag [--target checkerboard|circlegrid]
    # then, on a machine with Kalibr (docker is fine):
    kalibr_calibrate_cameras  --bag calib_right.bag --topics /cam0/image_raw --models pinhole-equi --target target.yaml
    kalibr_calibrate_imu_camera --bag calib_right.bag --cam camchain-calib_right.yaml --imu imu.yaml --target target.yaml
    python -m ego_teleop.tools.kalibr_import camchain-imucam-calib_right.yaml --side right --write-bundle

Targets are TAG-FREE by choice: checkerboard or asymmetric circle grid (Kalibr supports both; AprilGrid is deliberately not used
because no fiducial dependency may exist anywhere in this project's runtime). On a 160 deg fisheye the circle grid often localises
better near the rim — inspect the calibration images and pick; neither choice affects any downstream code.

Timestamps: camera frames use `capture_ns` (host monotonic); IMU samples use the device clock mapped onto the host clock with the
same robust fit the pose pipeline uses (`pose.timing.fit_device_to_host`), so the bag carries ONE clock and Kalibr's estimated
`timeshift_cam_imu` is the residual camera-IMU offset, not a clock-model artefact. IMU goes in at its full rate; never resampled."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import yaml
from handumi_collector.pose.episode_io import RawEpisode
from handumi_collector.pose.timing import fit_device_to_host, imu_host_times_ns

STREAM = {"left": "left_wrist", "right": "right_wrist"}

# Kalibr counts INTERNAL CORNERS here, not squares -- `targetCols: 6  #number of internal chessboard corners` in its
# own documentation. A 9x6-square board has 8x5 internal corners, and writing 9x6 made Kalibr extract corners from
# ZERO of 901 frames while cv2.findChessboardCornersSB(8,5) found the board in 63 % of the same images. It does not
# warn: it reports "No corners could be extracted ... Check the calibration target configuration".
TARGETS = {
    "checkerboard": dict(target_type="checkerboard", targetCols=8, targetRows=5, rowSpacingMeters=0.025, colSpacingMeters=0.025),
    "circlegrid": dict(target_type="circlegrid", targetCols=8, targetRows=5, spacingMeters=0.025, asymmetricGrid=True),
}


def write_target_yaml(path: Path, kind: str, **override) -> Path:
    if kind not in TARGETS: raise SystemExit(f"target must be one of {sorted(TARGETS)} (tag-free by design; no aprilgrid)")
    d = dict(TARGETS[kind]); d.update({k: v for k, v in override.items() if v is not None})
    Path(path).write_text(yaml.safe_dump(d, sort_keys=False)); return Path(path)


def write_imu_yaml(path: Path, noise: dict | None = None, rate_hz: float = 400.0) -> Path:
    n = dict(gyroscope_noise_density=2.0e-4, gyroscope_random_walk=2.0e-5, accelerometer_noise_density=2.0e-3, accelerometer_random_walk=3.0e-3)
    n.update({k: v for k, v in (noise or {}).items() if k in n})
    Path(path).write_text(yaml.safe_dump(dict(rostopic="/imu0", update_rate=float(rate_hz), **n), sort_keys=False)); return Path(path)


def export(ep_path: str | Path, side: str, out_bag: str | Path, *, downscale: int = 1, max_frames: int | None = None,
           stride: int = 1,
           camera_imu_offset_ns: int = 0) -> dict:
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore
    import cv2
    ts = get_typestore(Stores.ROS1_NOETIC)
    Image = ts.types["sensor_msgs/msg/Image"]; Imu = ts.types["sensor_msgs/msg/Imu"]
    Header = ts.types["std_msgs/msg/Header"]; Time = ts.types["builtin_interfaces/msg/Time"]
    Quat = ts.types["geometry_msgs/msg/Quaternion"]; Vec3 = ts.types["geometry_msgs/msg/Vector3"]
    ep = RawEpisode.load(ep_path); stream = STREAM[side]
    if stream not in ep.frames: raise SystemExit(f"{ep.path.name}: no {stream} stream")
    imu = ep.imu.get(side)
    if imu is None or len(imu) < 2: raise SystemExit(f"{ep.path.name}: no {side} IMU samples — Kalibr needs the full-rate IMU")
    clock = fit_device_to_host(imu.device_us, imu.host_ns)
    imu_t = imu_host_times_ns(imu, clock, camera_imu_offset_ns)
    out_bag = Path(out_bag); out_bag.parent.mkdir(parents=True, exist_ok=True)
    if out_bag.exists(): raise SystemExit(f"{out_bag} exists; refusing to overwrite")
    def stamp(t_ns: int): return Time(sec=int(t_ns // 1_000_000_000), nanosec=int(t_ns % 1_000_000_000))
    def header(t_ns: int, frame: str, seq: int): return Header(seq=seq, stamp=stamp(t_ns), frame_id=frame)   # ROS1 Header carries seq
    n_img = 0
    with Writer(out_bag) as w:
        c_img = w.add_connection("/cam0/image_raw", Image.__msgtype__, typestore=ts)
        c_imu = w.add_connection("/imu0", Imu.__msgtype__, typestore=ts)
        for i, (t_ns, gy, ac) in enumerate(zip(imu_t, imu.gyro, imu.accel)):
            m = Imu(header=header(int(t_ns), "imu0", i), orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=np.zeros(9), angular_velocity=Vec3(x=float(gy[0]), y=float(gy[1]), z=float(gy[2])),
                    angular_velocity_covariance=np.zeros(9), linear_acceleration=Vec3(x=float(ac[0]), y=float(ac[1]), z=float(ac[2])),
                    linear_acceleration_covariance=np.zeros(9))
            w.write(c_imu, int(t_ns), ts.serialize_ros1(m, Imu.__msgtype__))
        # `stride` thins the WHOLE take; `max_frames` truncates it. Kalibr wants around 20 Hz of images against the
        # IMU's full rate, and a 60 s recording at 30 Hz is 3.5 GB of bag for no extra information -- but taking the
        # first N frames would throw away the second half of the motion, which is where half the pose diversity is.
        kept = -1
        for vf, fi, t_cap, img in ep.iter_frames(stream, downscale=downscale, gray=True):
            kept += 1
            if stride > 1 and kept % stride:
                continue
            g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            m = Image(header=header(int(t_cap), "cam0", n_img), height=g.shape[0], width=g.shape[1], encoding="mono8",
                      is_bigendian=0, step=g.shape[1], data=np.ascontiguousarray(g).reshape(-1))
            w.write(c_img, int(t_cap), ts.serialize_ros1(m, Image.__msgtype__))
            n_img += 1
            if max_frames and n_img >= max_frames: break
    return dict(bag=str(out_bag), images=n_img, imu_samples=int(len(imu_t)), imu_rate_hz=round(len(imu_t) / max((imu_t[-1] - imu_t[0]) / 1e9, 1e-9), 1),
                clock_fit=clock.to_dict(), side=side, episode=Path(ep_path).name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--out", required=True, help="output .bag path")
    ap.add_argument("--target", default="checkerboard", choices=sorted(TARGETS)); ap.add_argument("--cols", type=int, default=None, help="INTERNAL CORNERS across, not squares (Kalibr's convention)")
    ap.add_argument("--rows", type=int, default=None, help="INTERNAL CORNERS down, not squares"); ap.add_argument("--spacing-m", type=float, default=None)
    ap.add_argument("--downscale", type=int, default=1); ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame across the whole take (IMU is never thinned)")
    a = ap.parse_args(argv)
    res = export(a.episode, a.side, a.out, downscale=a.downscale, max_frames=a.max_frames, stride=a.stride)
    out = Path(a.out).parent
    tgt = dict(targetCols=a.cols, targetRows=a.rows)
    tgt.update(dict(rowSpacingMeters=a.spacing_m, colSpacingMeters=a.spacing_m) if a.target == "checkerboard" else dict(spacingMeters=a.spacing_m))
    res["target_yaml"] = str(write_target_yaml(out / "target.yaml", a.target, **tgt))
    from ..calibration.bundle import load_bundle
    b = load_bundle(a.side)
    res["imu_yaml"] = str(write_imu_yaml(out / "imu.yaml", b.imu_noise.to_dict(), b.imu_noise.update_rate))
    print(f"{res['bag']}: {res['images']} images, {res['imu_samples']} IMU samples ({res['imu_rate_hz']} Hz), clock drift {res['clock_fit']['drift_ppm']} ppm")
    print(f"target: {res['target_yaml']} ({a.target}, tag-free)   imu: {res['imu_yaml']}")
    print("next: kalibr_calibrate_cameras --bag ... --models pinhole-equi --topics /cam0/image_raw --target target.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
