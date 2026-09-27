#!/usr/bin/env python3
"""[2026-09-21] Phase U0-B: put a HandUMI episode into the layout UMI's ORB-SLAM3 stage expects.

Writes <out>/raw_video.mp4 + <out>/imu_data.json, plus the ORB-SLAM3 settings file built from our frozen
Kalibr calibration. Nothing here is invented: the JSON schema was read out of gopro_slam.cc's LoadTelemetry,
the extrinsic convention from the exporter that MASt3R already ran on, and the clock from the same
fit_device_to_host / imu_host_times_ns the production pipeline uses -- so the camera-IMU time offset sign is
never re-decided here.

Two details that are easy to get wrong and are handled explicitly:
  * gopro_slam takes the VIDEO FRAME times from the CORI stream, not from the container. CORI therefore gets
    one sample per frame at that frame's capture time.
  * it shifts the IMU series so the first ACCL sample is t=0 but leaves CORI as it is, so both streams are
    written on one clock whose origin IS the first IMU sample.
"""
import argparse, json, pathlib, sys
import numpy as np, yaml, cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from handumi_collector.pose.episode_io import RawEpisode
from handumi_collector.pose.timing import fit_device_to_host, imu_host_times_ns

CAL = pathlib.Path(__file__).resolve().parents[1] / "configs/calibration"
FISHEYE = {"left": "fisheye_left_v002.yaml", "right": "fisheye_right_v003.yaml"}
CAMIMU = {"left": "camera_imu_left_v001.yaml", "right": "camera_imu_right_v002.yaml"}
STREAM = {"left": "left_wrist", "right": "right_wrist"}


def settings_yaml(side, W, H, K, D, T_cam_imu, nz, fps):
    """ORB-SLAM3 settings. IMU.T_b_c1 is the camera->body(IMU) transform, i.e. inv(T_camera_imu):
    our bundle stores the IMU expressed in the camera frame, which is what the validated EuRoC exporter
    inverted to get Tic. Do not re-derive this direction."""
    Tbc = np.linalg.inv(np.asarray(T_cam_imu, float))
    d = np.asarray(D, float).ravel()
    body = [
        "%YAML:1.0", "",
        f"# HandUMI {side} wrist, generated {pathlib.Path(__file__).name}",
        f"# intrinsics {FISHEYE[side]}  extrinsic/noise {CAMIMU[side]}  (frozen Kalibr, do not edit by hand)",
        'File.version: "1.0"', 'Camera.type: "KannalaBrandt8"', "",
        f"Camera1.fx: {K[0,0]:.6f}", f"Camera1.fy: {K[1,1]:.6f}",
        f"Camera1.cx: {K[0,2]:.6f}", f"Camera1.cy: {K[1,2]:.6f}", "",
        f"Camera1.k1: {d[0]:.9f}", f"Camera1.k2: {d[1]:.9f}",
        f"Camera1.k3: {d[2]:.9f}", f"Camera1.k4: {d[3]:.9f}", "",
        # ORB-SLAM3 aborts on a non-integer Camera.fps
        f"Camera.width: {W}", f"Camera.height: {H}", f"Camera.fps: {int(round(fps))}", "Camera.RGB: 1", "",
        "# camera -> body(IMU) = inv(T_camera_imu)",
        "IMU.T_b_c1: !!opencv-matrix", "    rows: 4", "    cols: 4", "    dt: f",
        "    data: [" + ", ".join(f"{v:.10f}" for v in Tbc.reshape(-1)) + "]", "",
        f"IMU.NoiseGyro: {nz['gyroscope_noise_density']:.8f}",
        f"IMU.NoiseAcc: {nz['accelerometer_noise_density']:.8f}",
        f"IMU.GyroWalk: {nz['gyroscope_random_walk']:.10f}",
        f"IMU.AccWalk: {nz['accelerometer_random_walk']:.10f}",
        f"IMU.Frequency: {nz['update_rate']:.1f}", "",
        "ORBextractor.nFeatures: 1500", "ORBextractor.scaleFactor: 1.2", "ORBextractor.nLevels: 8",
        "ORBextractor.iniThFAST: 20", "ORBextractor.minThFAST: 7", "",
        "Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9",
        "Viewer.PointSize: 2.0", "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3.0",
        "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7", "Viewer.ViewpointZ: -3.5",
        "Viewer.ViewpointF: 500.0", "",
    ]
    return "\n".join(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode"); ap.add_argument("side", choices=["left", "right"])
    ap.add_argument("out")
    ap.add_argument("--downscale", type=int, default=2)
    # [2026-09-21] Every episode opens with a 3 s HOLD STILL. That window exists so OpenVINS could do a
    # static init, but ORB-SLAM3 monocular-inertial needs MOTION to initialise the IMU, and on a loaded map
    # it reacts to an uninitialised IMU by resetting the active map -- throwing away a relocalisation it had
    # already achieved ("Relocalized!!" then "IMU is not or recently initialized. Reseting active map").
    # --start-at-go drops everything before the go_cue so the sequence begins moving.
    ap.add_argument("--start-at-go", action="store_true")
    # [2026-09-21] IMU PREROLL. UMI's ORB-SLAM3 fork splices a new sequence onto a loaded atlas by rewriting
    # the loaded keyframes' timestamps so the last one sits at -dt, then preintegrating IMU across that gap
    # (Tracking.cc, INIT_RELOCALIZE). dt there is hardcoded to 1/59.97 = 16.68 ms -- a GoPro frame -- and a
    # shadowed local `double dt` means the real mpImuPreintegrated->dT is never read. So the splice always
    # asks for ~17 ms of IMU BEFORE the first camera frame. Our exports had the opposite: the first camera
    # frame preceded the first IMU sample by 7-35 ms, leaving that window empty. Keep real IMU samples from
    # preroll ms before the first kept frame; never fabricate samples, never shift the IMU clock (that would
    # break the frozen camera-IMU calibration).
    ap.add_argument("--imu-preroll-ms", type=float, default=50.0)
    a = ap.parse_args()
    ep_dir = pathlib.Path(a.episode).expanduser(); out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ep = RawEpisode.load(ep_dir)
    fe = yaml.safe_load(open(CAL / FISHEYE[a.side])); ci = yaml.safe_load(open(CAL / CAMIMU[a.side]))
    K = np.array(fe["K"], float) / a.downscale; K[2, 2] = 1.0
    W, H = fe["image_size"][0] // a.downscale, fe["image_size"][1] // a.downscale

    # --- video and its frame times in ONE pass: RawEpisode.iter_frames yields (video_frame, index,
    # capture_ns, image), so the timestamps written into CORI are the same frames written into the mp4.
    # Reading them separately invites an off-by-one between the two.
    t_go = None
    if a.start_at_go:
        ev = json.loads((ep_dir / "events.json").read_text())
        g = [e for e in ev if e.get("kind") == "go_cue"]
        if not g:
            raise SystemExit("no go_cue in events.json; cannot --start-at-go")
        t_go = int(g[0]["t_ns"])
        print(f"  trimming everything before go_cue at t_rel {g[0].get('t_rel_s'):.2f} s")
    vw, t_cam, n_frames = None, [], 0
    for _, _, cap_ns, img in ep.iter_frames(STREAM[a.side], downscale=a.downscale):
        if t_go is not None and int(cap_ns) < t_go:
            continue
        if vw is None:
            h, w = img.shape[:2]
            assert (w, h) == (W, H), f"frame {w}x{h} != settings {W}x{H}"
            vw = cv2.VideoWriter(str(out / "raw_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
        vw.write(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        t_cam.append(float(cap_ns)); n_frames += 1
    if vw is not None:
        vw.release()
    t_cam = np.asarray(t_cam, float)
    fps = 30.0

    # --- IMU and frame times on ONE clock, via the production clock model
    imu = ep.imu[a.side]
    clock = fit_device_to_host(imu.device_us, imu.host_ns)
    off_ns = int(float(ci["time_offset_ms"]) * 1e6)
    t_imu = imu_host_times_ns(imu, clock, off_ns)
    o = np.argsort(t_imu, kind="stable")
    t_imu = t_imu[o]; acc = np.asarray(imu.accel, float)[o]; gyr = np.asarray(imu.gyro, float)[o]
    keep = np.r_[True, np.diff(t_imu) > 0]
    t_imu, acc, gyr = t_imu[keep], acc[keep], gyr[keep]
    pre_ns = int(a.imu_preroll_ms * 1e6)
    cam_start = float(t_cam[0])
    lo = cam_start - pre_ns
    if float(t_imu[0]) > lo:                      # IMU cannot reach back far enough: drop leading frames
        lo = float(t_imu[0])
        keep_cam = t_cam >= lo + pre_ns
        dropped = int((~keep_cam).sum())
        if dropped:
            print(f"  dropping {dropped} leading frames so the IMU leads the camera by {a.imu_preroll_ms:.0f} ms")
        t_cam = t_cam[keep_cam]
        n_frames = len(t_cam)
    m = t_imu >= lo
    t_imu, acc, gyr = t_imu[m], acc[m], gyr[m]
    t0 = float(t_imu[0])                       # gopro_slam re-bases the IMU to its first sample; match CORI to it
    ms = lambda t: (np.asarray(t, float) - t0) / 1e6
    tel = {"1": {"streams": {
        "ACCL": {"samples": [{"cts": float(c), "value": [float(v) for v in a3]} for c, a3 in zip(ms(t_imu), acc)]},
        "GYRO": {"samples": [{"cts": float(c), "value": [float(v) for v in g3]} for c, g3 in zip(ms(t_imu), gyr)]},
        "CORI": {"samples": [{"cts": float(c), "value": [1.0, 0.0, 0.0, 0.0]} for c in ms(t_cam)[:n_frames]]},
    }}}
    (out / "imu_data.json").write_text(json.dumps(tel))
    SPLICE_DT_MS = 1000.0 / 59.97          # what the fork's INIT_RELOCALIZE splice assumes
    lead = ms(t_cam)[0] - ms(t_imu)[0]
    print(f"  first_imu 0.00 ms | first_camera {ms(t_cam)[0]:.2f} ms | camera-imu lead {lead:.2f} ms "
          f"| splice needs >= {SPLICE_DT_MS:.2f} ms -> {'OK' if lead >= SPLICE_DT_MS else 'TOO SHORT'}")
    (out / "orbslam_setting.yaml").write_text(
        settings_yaml(a.side, W, H, K, fe["D"], ci["T_camera_imu"], ci["imu_noise"], fps))
    print(f"{ep_dir.name} {a.side}: {n_frames} frames {W}x{H} @ {fps:.1f} | imu {len(t_imu)} "
          f"@ {1e9/np.median(np.diff(t_imu)):.1f} Hz (+{ci['time_offset_ms']} ms) | cori {min(len(t_cam), n_frames)}")
    print(f"  span imu {ms(t_imu)[-1]/1000:.2f} s   cam {ms(t_cam)[:n_frames][-1]/1000:.2f} s  -> {out}")


if __name__ == "__main__":
    main()
