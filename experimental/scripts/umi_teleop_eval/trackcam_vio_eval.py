"""OpenVINS on the UMI tracking fisheye + UMI IMU, scored on the DIY-tracker protocol A-E (2026-10-15, spec "DIY Wrist 6DoF Tracker v1").

Inputs: a trackcam_imu --vio-test recording (tracking_cam.mp4 640x360 = 1080p/3, tracking_cam_t_ns.npy host clock, imu_right.npz, protocol.json),
configs/calibration/trackcam_fisheye_v001.yaml (KB4), configs/calibration/trackcam_imu_v001.yaml (R_CI, offset), IMU noise from
camera_imu_right_v002 (same Teensy board). Camera<->IMU translation unknown -> starts at 0 and OpenVINS refines extrinsic + time offset.
Self-view (UMI parts that move WITH the camera) is found as pixels that barely change over the moving part of the recording and painted
flat grey before feeding (the bridge ignores OpenVINS masks).

Scores (spec section 13/14): A stationary jitter, B 10 cm steps (scale), C fake translation under +-30 deg rotations, D stop/resume,
E cloth reach tracking ratio, plus per-frame OpenVINS CPU time (fps) and init/lost counts.
    python scripts/trackcam_vio_eval.py datasets/trackcam_imu/<ts> [--no-mask] [--extrinsic-calib/--no-extrinsic-calib]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from handumi_collector.pose.timing import fit_device_to_host           # noqa: E402
from ego_teleop.tracking.backends.openvins import OpenVinsBackend      # noqa: E402


def self_view_mask(frames, thr=4.0):
    """Pixels whose intensity barely changes across the recording (UMI body/cables in view). Returns bool mask (True = paint over)."""
    st = np.stack(frames[:: max(1, len(frames) // 300)]).astype(np.float32); sd = st.std(0)
    m = (sd < thr).astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)); m = cv2.dilate(m, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 0.01 * m.size: keep[lab == i] = 1          # only big static blobs (not one flat patch of ceiling)
    return keep.astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec"); ap.add_argument("--no-mask", action="store_true"); ap.add_argument("--no-extrinsic-calib", action="store_true")
    ap.add_argument("--override", default="{}", help="JSON OpenVINS estimator overrides"); ap.add_argument("--noise-x", type=float, default=1.0); ap.add_argument("--tag", default="")
    ap.add_argument("--segmented", action="store_true", help="09-16 segmented relative VIO: static windows -> dp=0, dR=gyro; fresh OpenVINS per moving segment")
    ap.add_argument("--t-ci", default=None, help="camera<-imu translation guess, metres 'x,y,z' (camera frame)"); a = ap.parse_args()
    E = Path(a.rec); proto = json.loads((E / "protocol.json").read_text())
    yi = yaml.safe_load((ROOT / "configs/calibration/trackcam_fisheye_v001.yaml").read_text()); ye = yaml.safe_load((ROOT / "configs/calibration/trackcam_imu_v001.yaml").read_text())
    yn = yaml.safe_load((ROOT / "configs/calibration/camera_imu_right_v002.yaml").read_text())["imu_noise"]
    intr = dict(model="kannala_brandt", K=np.asarray(yi["K"], float).reshape(3, 3), D=np.asarray(yi["D"], float), image_size=yi["image_size"])
    T_CI = np.eye(4); T_CI[:3, :3] = np.asarray(ye["R_CI"], float); off_ms = float(ye["offset_ms"])
    if a.t_ci: T_CI[:3, 3] = [float(v) for v in a.t_ci.split(",")]
    t_cam = np.load(E / "tracking_cam_t_ns.npy"); cap = cv2.VideoCapture(str(E / "tracking_cam.mp4")); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    n = min(len(frames), len(t_cam)); frames, t_cam = frames[:n], t_cam[:n]
    mask = None if a.no_mask else self_view_mask(frames)
    if mask is not None: cv2.imwrite(str(E / "self_view_mask.png"), mask.astype(np.uint8) * 255); print(f"자기 부품 가림 영역 {100 * mask.mean():.1f}%")
    z = np.load(E / "imu_right.npz"); ti = fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]).astype(np.int64) + int(off_ms * 1e6)
    gy, ac = z["gyro"], z["acc"]
    nx = a.noise_x
    noise = dict(gyroscope_noise_density=nx * yn["gyroscope_noise_density"], gyroscope_random_walk=nx * yn.get("gyroscope_random_walk", 2e-5),
                 accelerometer_noise_density=nx * yn["accelerometer_noise_density"], accelerometer_random_walk=nx * yn.get("accelerometer_random_walk", 3e-3), update_rate=200.0)
    # 09-16 lesson: OpenVINS' per-update feature count is 1-4 even when tracking well -> never call LOST from it (min_features_lost=-1)
    ov_over = {**dict(calib_cam_extrinsics=not a.no_extrinsic_calib, calib_cam_timeoffset=True, init_imu_thresh=0.8, num_pts=200), **json.loads(a.override)}
    mk = lambda: OpenVinsBackend(downscale=3, mode="production", min_features_lost=-1, min_features_degraded=-1, overrides=ov_over)
    if a.segmented:
        from handumi_collector.pose.backends.segmented import SegmentedEstimator
        seg_cfg = dict(yaml.safe_load((ROOT / "configs/handumi/pose.yaml").read_text())["segmented"]); seg_cfg["enabled"] = True
        be = SegmentedEstimator(mk, imu_t_ns=ti, gyro=gy, accel=ac, T_camera_imu=T_CI, cfg=seg_cfg)
    else: be = mk()
    be.initialize(intrinsics=intr, T_camera_imu=T_CI, imu_noise=noise)
    j = 0; rows = []; cpu = []
    for k in range(n):
        while j < len(ti) and ti[j] <= t_cam[k]: be.push_imu(int(ti[j]), gy[j], ac[j]); j += 1
        img = frames[k]
        if mask is not None: img = img.copy(); img[mask] = 128
        c0 = time.perf_counter(); est = be.push_image(int(t_cam[k]), k, img); cpu.append((time.perf_counter() - c0) * 1e3)
        T = est.T_world_camera
        rows.append(dict(t=(t_cam[k] - t_cam[0]) / 1e9, state=est.tracking_state.name, nf=est.num_features or 0,
                         p=None if T is None else T[:3, 3].tolist(), q=None if T is None else R.from_matrix(T[:3, :3]).as_quat().tolist()))
    np.save(E / "vio_cpu_ms.npy", np.array(cpu)); (E / "vio_poses.json").write_text(json.dumps(rows))
    t = np.array([r["t"] for r in rows]); P = np.array([r["p"] if r["p"] else [np.nan] * 3 for r in rows]); st = [r["state"] for r in rows]
    ok = np.all(np.isfinite(P), 1); i0 = int(np.argmax(ok)) if ok.any() else None
    seg = lambda lo, hi: (t >= lo) & (t < hi)
    phase_t = [p["t0"] for p in proto] + [t[-1]]
    S = dict(frames=n, cpu_ms_p50=round(float(np.median(cpu)), 1), cpu_ms_p95=round(float(np.percentile(cpu, 95)), 1), fps_capable=round(1000 / float(np.percentile(cpu, 95)), 1),
             init_time_s=None if i0 is None else round(float(t[i0]), 2), tracked_frac_after_init=None if i0 is None else round(float(ok[i0:].mean()), 3),
             lost_frames=int(sum(s == "LOST" for s in st)), degraded_frames=int(sum(s == "DEGRADED" for s in st)))
    # A: jitter during the 10 s still (only if initialised in time)
    mA = seg(phase_t[0] + 1, phase_t[1] - 0.5) & ok
    S["A_still_jitter_mm_p90"] = round(float(np.percentile(np.linalg.norm(P[mA] - np.median(P[mA], 0), axis=1) * 1000, 90)), 2) if mA.sum() > 20 else None
    # B: 6 x 4 s windows -> position at the end of each window (last 1 s median); 10 cm steps between consecutive windows
    ends = []
    for k in range(1, 7):
        lo, hi = phase_t[k], phase_t[k + 1]; m = seg(hi - 1.2, hi - 0.2) & ok
        ends.append(np.median(P[m], 0) if m.sum() > 5 else np.full(3, np.nan))
    st0 = np.median(P[seg(phase_t[1] - 1.2, phase_t[1] - 0.2) & ok], 0) if (seg(phase_t[1] - 1.2, phase_t[1] - 0.2) & ok).sum() > 5 else np.full(3, np.nan)
    pts = [st0] + ends; steps = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(6)]
    S["B_step_cm"] = [round(s * 100, 2) for s in steps]; good = [s for s in steps if np.isfinite(s)]
    S["B_scale_median"] = round(float(np.median(good)) / 0.10, 3) if good else None
    # C: fake translation while rotating in place
    mC = seg(phase_t[7] + 0.5, phase_t[8]) & ok
    S["C_fake_translation_cm_p90"] = round(float(np.percentile(np.linalg.norm(P[mC] - np.median(P[mC], 0), axis=1) * 100, 90)), 2) if mC.sum() > 20 else None
    # D: drift during the two 3 s stops
    for name, k in (("D_stop1", 9), ("D_stop2", 11)):
        m = seg(phase_t[k] + 0.5, phase_t[k + 1]) & ok
        S[f"{name}_drift_mm"] = round(float(np.linalg.norm(P[m][-1] - P[m][0]) * 1000), 1) if m.sum() > 10 else None
    mE = seg(phase_t[12], phase_t[13]); S["E_tracked_frac"] = round(float(ok[mE].mean()), 3) if mE.any() else None
    calib = be.calibration_estimate() if hasattr(be, "calibration_estimate") else None; S["online_extrinsic"] = None if not calib else dict(T_camera_imu_t_m=np.asarray(calib["T_camera_imu"])[:3, 3].round(4).tolist(), time_offset_ms=round(calib["time_offset_ms"], 2))
    (E / f"vio_eval{a.tag}.json").write_text(json.dumps(S, indent=1, default=float)); print(json.dumps(S, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
