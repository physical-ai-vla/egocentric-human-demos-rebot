"""OpenVINS on a trackcam board recording vs the AprilGrid ground truth (board_traj.npz from trackcam_board_lever.py) (2026-10-15).
For each camera->IMU translation candidate: run OpenVINS (09-16 tuned overrides, never LOST from feature counts), take its camera poses,
align to the board camera trajectory (SE3 and Sim3 -> scale), report ATE p50/p90, scale, 1 s relative-translation error.
    python scripts/trackcam_vio_vs_board.py datasets/trackcam_imu/<board ts> --cand name=x,y,z [--cand ...]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from handumi_collector.pose.timing import fit_device_to_host           # noqa: E402
from ego_teleop.tracking.backends.openvins import OpenVinsBackend      # noqa: E402
TUNED = dict(histogram_method="CLAHE", up_msckf_chi2_multipler=5, up_slam_chi2_multipler=5, dt_slam_delay=0.3, max_clones=8,
             calib_cam_extrinsics=False, calib_cam_timeoffset=True, init_imu_thresh=0.8, num_pts=200)


def umeyama(src, dst, scale):
    mu_s, mu_d = src.mean(0), dst.mean(0); A, B = src - mu_s, dst - mu_d; U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    d = np.sign(np.linalg.det(U @ Vt)); Dg = np.diag([1, 1, d]); Rm = U @ Dg @ Vt; s = (S * np.diag(Dg)).sum() / A.var(0).sum() if scale else 1.0
    return s, Rm, mu_d - s * Rm @ mu_s


def run_ov(E, t_ci, off_ms, frames, t_cam, imu):
    yi = yaml.safe_load((ROOT / "configs/calibration/trackcam_fisheye_v001.yaml").read_text()); ye = yaml.safe_load((ROOT / "configs/calibration/trackcam_imu_v001.yaml").read_text())
    yn = yaml.safe_load((ROOT / "configs/calibration/camera_imu_right_v002.yaml").read_text())["imu_noise"]
    intr = dict(model="kannala_brandt", K=np.asarray(yi["K"], float).reshape(3, 3), D=np.asarray(yi["D"], float), image_size=yi["image_size"])
    T = np.eye(4); T[:3, :3] = np.asarray(ye["R_CI"], float); T[:3, 3] = t_ci
    sc = 1920 // frames[0].shape[1]
    be = OpenVinsBackend(downscale=sc, mode="production", min_features_lost=-1, min_features_degraded=-1, overrides=TUNED)
    be.initialize(intrinsics=intr, T_camera_imu=T, imu_noise=dict(gyroscope_noise_density=yn["gyroscope_noise_density"], accelerometer_noise_density=yn["accelerometer_noise_density"], update_rate=200.0))
    ti, gy, ac = imu; ti = ti + int(off_ms * 1e6); j = 0; out = {}
    for k in range(len(frames)):
        while j < len(ti) and ti[j] <= t_cam[k]: be.push_imu(int(ti[j]), gy[j], ac[j]); j += 1
        e = be.push_image(int(t_cam[k]), k, frames[k])
        if e.T_world_camera is not None: out[int(t_cam[k])] = e.T_world_camera[:3, 3].copy()
    return out


def compare(est: dict, gt_t, gt_P):
    common = [i for i, t in enumerate(gt_t) if int(t) in est]
    if len(common) < 50: return dict(error=f"only {len(common)} matched poses")
    G = gt_P[common]; V = np.array([est[int(gt_t[i])] for i in common]); tt = (gt_t[common] - gt_t[0]) / 1e9
    s, Rm, t = umeyama(V, G, True); e_sim = np.linalg.norm(V @ Rm.T * s + t - G, axis=1)
    _, R1, t1 = umeyama(V, G, False); e_se3 = np.linalg.norm(V @ R1.T + t1 - G, axis=1)
    rel = []
    for i in range(len(tt)):
        j = np.searchsorted(tt, tt[i] + 1.0)
        if j < len(tt) and abs(tt[j] - tt[i] - 1.0) < 0.1:
            dg = G[j] - G[i]; dv = (V[j] - V[i]) @ R1.T
            if np.linalg.norm(dg) > 0.03: rel.append(np.linalg.norm(dv - dg) / np.linalg.norm(dg))
    return dict(matched=len(common), ate_se3_cm_p50=round(float(np.median(e_se3)) * 100, 2), ate_se3_cm_p90=round(float(np.percentile(e_se3, 90)) * 100, 2),
                sim3_scale=round(float(s), 3), ate_sim3_cm_p50=round(float(np.median(e_sim)) * 100, 2), rel_1s_err_median=round(float(np.median(rel)), 3) if rel else None,
                gt_path_m=round(float(np.sum(np.linalg.norm(np.diff(G, axis=0), axis=1))), 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec"); ap.add_argument("--cand", action="append", required=True); ap.add_argument("--offset-ms", type=float, default=None); a = ap.parse_args()
    E = Path(a.rec); gt = np.load(E / "board_traj.npz"); gt_t, gt_P = gt["t_ns"], gt["T_B_C"][:, :3, 3]
    off = a.offset_ms if a.offset_ms is not None else float(yaml.safe_load((ROOT / "configs/calibration/trackcam_imu_v001.yaml").read_text())["offset_ms"])
    t_cam = np.load(E / "tracking_cam_t_ns.npy"); cap = cv2.VideoCapture(str(E / "tracking_cam.mp4")); frames = []
    while len(frames) < len(t_cam):
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    t_cam = t_cam[:len(frames)]
    z = np.load(E / "imu_right.npz"); imu = (fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]).astype(np.int64), z["gyro"], z["acc"])
    res = {}
    for c in a.cand:
        name, xyz = c.split("="); t_ci = [float(v) for v in xyz.split(",")]
        est = run_ov(E, t_ci, off, frames, t_cam, imu); res[name] = compare(est, gt_t, gt_P); print(name, json.dumps(res[name], ensure_ascii=False), flush=True)
    (E / f"vio_vs_board_off{off:.0f}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
