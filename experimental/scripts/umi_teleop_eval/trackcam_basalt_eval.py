"""Basalt VIO (KB4 fisheye + IMU) on a trackcam_imu --vio-test recording, scored like trackcam_vio_eval.py (protocol A-E) (2026-10-15).
Basalt 0.1.7 release binary (~/vio/basalt_bin/release, sha256 checked, NOT installed into ~/.local / shell rc).

  export   EuRoC: mav0/cam0/data/<t_ns>.png (640x360 = 1080p/3) + data.csv, mav0/imu0/data.csv (IMU host clock + offset -> camera clock)
  calib    Basalt json from configs/calibration/trackcam_fisheye_v001.yaml (kb4, K/3) + trackcam_imu_v001.yaml (R_CI) + lever arm
  run      basalt_vio --dataset-type euroc --use-imu 1 --save-trajectory tum   (trajectory = IMU pose in Basalt's world)
  score    A still jitter, B 10 cm steps, C fake translation, D stop drift, E tracking, + wall time per frame
    python scripts/trackcam_basalt_eval.py datasets/trackcam_imu/<ts> --t-ci x,y,z [--config euroc_config.json]"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from handumi_collector.pose.timing import fit_device_to_host           # noqa: E402
BAS = Path.home() / "vio/basalt_bin/release"


def export(E: Path, out: Path, off_ms: float, mask=None) -> int:
    cam = out / "mav0/cam0/data"; cam.mkdir(parents=True, exist_ok=True); (out / "mav0/imu0").mkdir(parents=True, exist_ok=True)
    t_cam = np.load(E / "tracking_cam_t_ns.npy"); cap = cv2.VideoCapture(str(E / "tracking_cam.mp4")); rows = []; k = 0
    while k < len(t_cam):
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if mask is not None: g[mask] = 128
        cv2.imwrite(str(cam / f"{int(t_cam[k])}.png"), g); rows.append(f"{int(t_cam[k])},{int(t_cam[k])}.png"); k += 1
    (out / "mav0/cam0/data.csv").write_text("#timestamp [ns],filename\n" + "\n".join(rows) + "\n")
    # Basalt's EuRoC loader always reads cam0 AND cam1 and its optical flow drops a frame if any image is missing -> point cam1 at cam0
    # (the calibration has ONE camera, so only cam0 is used by the estimator)
    c1 = out / "mav0/cam1"
    if not c1.exists(): c1.symlink_to("cam0", target_is_directory=True)
    z = np.load(E / "imu_right.npz"); ti = fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]).astype(np.int64) + int(off_ms * 1e6)
    g_, a_ = z["gyro"], z["acc"]; o = np.argsort(ti); ti, g_, a_ = ti[o], g_[o], a_[o]; keep = np.r_[True, np.diff(ti) > 0]
    lines = [f"{t},{w[0]:.9f},{w[1]:.9f},{w[2]:.9f},{a[0]:.9f},{a[1]:.9f},{a[2]:.9f}" for t, w, a in zip(ti[keep], g_[keep], a_[keep])]
    (out / "mav0/imu0/data.csv").write_text("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n" + "\n".join(lines) + "\n")
    return k


def calib(out: Path, t_ci, scale: int = 3) -> Path:
    tpl = json.loads((BAS / "data/t265_kb4_calib.json").read_text()); v = tpl["value0"]
    yi = yaml.safe_load((ROOT / "configs/calibration/trackcam_fisheye_v001.yaml").read_text()); K = np.asarray(yi["K"], float).reshape(3, 3) / float(scale); D = yi["D"]
    Rci = np.asarray(yaml.safe_load((ROOT / "configs/calibration/trackcam_imu_v001.yaml").read_text())["R_CI"], float)
    T_CI = np.eye(4); T_CI[:3, :3] = Rci; T_CI[:3, 3] = t_ci; T_IC = np.linalg.inv(T_CI); q = R.from_matrix(T_IC[:3, :3]).as_quat()
    v["T_imu_cam"] = [dict(px=float(T_IC[0, 3]), py=float(T_IC[1, 3]), pz=float(T_IC[2, 3]), qx=float(q[0]), qy=float(q[1]), qz=float(q[2]), qw=float(q[3]))]
    v["intrinsics"] = [dict(camera_type="kb4", intrinsics=dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), k1=D[0], k2=D[1], k3=D[2], k4=D[3]))]
    v["resolution"] = [[1920 // scale, 1080 // scale]]; v["vignette"] = []; v["imu_update_rate"] = 200.0; v["cam_time_offset_ns"] = 0
    v["calib_accel_bias"] = [0.0] * 9; v["calib_gyro_bias"] = [0.0] * 12
    v["accel_noise_std"] = [0.016] * 3; v["gyro_noise_std"] = [0.000282] * 3; v["accel_bias_std"] = [0.001] * 3; v["gyro_bias_std"] = [0.0001] * 3
    p = out / "calib_kb4.json"; p.write_text(json.dumps(tpl, indent=1)); return p


def score(E: Path, traj: Path, wall_s: float, n_frames: int) -> dict:
    proto = json.loads((E / "protocol.json").read_text()); t_cam = np.load(E / "tracking_cam_t_ns.npy")
    A = np.loadtxt(traj)
    if A.ndim == 1 or len(A) == 0: return dict(error="empty trajectory")
    tt = A[:, 0]; tt = tt * (1e9 if tt[0] < 1e12 else 1.0); t = (tt - t_cam[0]) / 1e9; P = A[:, 1:4]
    phase_t = [p["t0"] for p in proto] + [t[-1]]; seg = lambda lo, hi: (t >= lo) & (t < hi)
    S = dict(frames=n_frames, poses=int(len(t)), wall_ms_per_frame=round(wall_s * 1000 / max(1, n_frames), 1), first_pose_s=round(float(t[0]), 2),
             coverage=round(len(t) / max(1, n_frames), 3))
    mA = seg(phase_t[0] + 1, phase_t[1] - 0.5)
    S["A_still_jitter_mm_p90"] = round(float(np.percentile(np.linalg.norm(P[mA] - np.median(P[mA], 0), axis=1) * 1000, 90)), 2) if mA.sum() > 20 else None
    pts = []
    for k in range(0, 7):
        hi = phase_t[k + 1]; m = seg(hi - 1.2, hi - 0.2); pts.append(np.median(P[m], 0) if m.sum() > 5 else np.full(3, np.nan))
    steps = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(6)]; S["B_step_cm"] = [round(s * 100, 2) for s in steps]
    good = [s for s in steps if np.isfinite(s)]; S["B_scale_median"] = round(float(np.median(good)) / 0.10, 3) if good else None
    mC = seg(phase_t[7] + 0.5, phase_t[8])
    S["C_fake_translation_cm_p90"] = round(float(np.percentile(np.linalg.norm(P[mC] - np.median(P[mC], 0), axis=1) * 100, 90)), 2) if mC.sum() > 20 else None
    for name, k in (("D_stop1", 9), ("D_stop2", 11)):
        m = seg(phase_t[k] + 0.5, phase_t[k + 1]); S[f"{name}_drift_mm"] = round(float(np.linalg.norm(P[m][-1] - P[m][0]) * 1000), 1) if m.sum() > 10 else None
    mE = seg(phase_t[12], phase_t[13]); S["E_pose_rate_hz"] = round(float(mE.sum() / max(1e-3, phase_t[13] - phase_t[12])), 1)
    sp = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3); S["jumps_gt_3m_s"] = int((sp > 3.0).sum())
    return S


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec"); ap.add_argument("--t-ci", required=True); ap.add_argument("--config", default="euroc_config.json"); ap.add_argument("--tag", default="")
    ap.add_argument("--reuse-export", action="store_true"); a = ap.parse_args()
    E = Path(a.rec).resolve(); out = E / "basalt"; out.mkdir(exist_ok=True); off = float(yaml.safe_load((ROOT / "configs/calibration/trackcam_imu_v001.yaml").read_text())["offset_ms"])
    n = len(list((out / "mav0/cam0/data").glob("*.png"))) if a.reuse_export and (out / "mav0").exists() else export(E, out, off)
    sc = json.loads((E / "meta.json").read_text())["save_scale"] if (E / "meta.json").exists() else 3
    cp = calib(out, [float(v) for v in a.t_ci.split(",")], sc)
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(BAS / "lib"))
    cmd = [str(BAS / "bin/basalt_vio"), "--show-gui", "0", "--dataset-path", str(out), "--dataset-type", "euroc", "--cam-calib", str(cp),
           "--config-path", str(BAS / "data" / a.config), "--use-imu", "1", "--save-trajectory", "tum", "--num-threads", "6"]
    t0 = time.time(); r = subprocess.run(cmd, cwd=out, env=env, capture_output=True, text=True); wall = time.time() - t0
    (out / f"basalt_stdout{a.tag}.log").write_text(r.stdout + "\n" + r.stderr)
    traj = out / "trajectory.txt"
    if r.returncode != 0 or not traj.exists(): print("basalt failed", r.returncode, r.stderr[-800:]); return
    if (E / "board_traj.npz").exists():                       # ground truth available: camera position = p_i + R_w_i t_ic, compare
        sys.path.insert(0, str(ROOT / "scripts")); import trackcam_vio_vs_board as V
        A = np.loadtxt(traj); tns = (A[:, 0] * 1e9 if A[0, 0] < 1e12 else A[:, 0]).astype(np.int64)
        T_ic = np.linalg.inv(np.r_[np.c_[np.asarray(yaml.safe_load((ROOT / "configs/calibration/trackcam_imu_v001.yaml").read_text())["R_CI"]), [float(v) for v in a.t_ci.split(",")]], [[0, 0, 0, 1]]])
        Rwi = R.from_quat(A[:, 4:8]).as_matrix(); pc = A[:, 1:4] + np.einsum("nij,j->ni", Rwi, T_ic[:3, 3])
        gt = np.load(E / "board_traj.npz"); t_cam = np.load(E / "tracking_cam_t_ns.npy")
        est = {int(t): p for t, p in zip(tns, pc)}
        # Basalt stamps = image stamps -> match exactly
        S = V.compare(est, gt["t_ns"], gt["T_B_C"][:, :3, 3]); S["wall_ms_per_frame"] = round(wall * 1000 / max(1, n), 1); S["poses"] = int(len(A))
    else: S = score(E, traj, wall, n)
    traj.rename(out / f"trajectory{a.tag}.txt")
    (out / f"basalt_eval{a.tag}.json").write_text(json.dumps(S, indent=1)); print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
