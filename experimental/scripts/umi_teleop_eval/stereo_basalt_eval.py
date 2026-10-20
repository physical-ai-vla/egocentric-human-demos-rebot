"""Basalt STEREO-inertial on a stereo_imu_record take, scored METRIC against the AprilGrid ground truth (2026-10-15, Stereo Phase S6/S8).
  export  EuRoC: cam0 = left frames (own stamps), cam1 = the right frame nearest to each left stamp (|dt| <= --max-skew-ms, else the
          pair is dropped), written under the LEFT stamp (Basalt pairs cam0/cam1 by name); IMU host clock + offset_ms -> camera clock
  calib   two KB4 cameras from stereo_trackcam_v003 (/save_scale), T_imu_cam0 = inv(T_C0I) with R_C0I from stereo_trackcam_imu_v001 and the
          lever t_C0I (--t-ci), T_imu_cam1 = T_imu_cam0 inv(T_CR_CL)
  run     basalt_vio --use-imu 1 (stereo), wall time -> output Hz / ms per frame
  score   camera C0 pose = T_w_i T_i_c0 vs board_traj_cam_left.npz (T_B_C0): SE(3)-only alignment (scale FIXED at 1 -- stereo must be metric),
          plus the S8 metrics of trackcam_m3_vs_board (10 cm steps, 30 s drift, fake translation in > 20 deg turns, per-window scale) and the
          S7 protocol steps (displacement of each +-10 cm phase, est vs ground truth), Sim(3) scale reported for information only
    python scripts/stereo_basalt_eval.py datasets/stereo_imu/<ts>_eval --t-ci x,y,z [--tag name] [--mask selfview.npy] [--config euroc_config.json]"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from handumi_collector.pose.timing import fit_device_to_host   # noqa: E402
BAS = Path.home() / "vio/basalt_bin/release"

STEREO_FILE = "configs/calibration/stereo_trackcam_v004.yaml"   # 10-16 18:56 new camera mount -> v005 (live tracker sets it); old takes = v004
CAMIMU_FILE = "configs/calibration/stereo_trackcam_imu_v003.yaml"   # camera<->IMU; the live tracker sets v004 (10-16 15:40 bracket slip). Old takes = v003


def frames(video: Path, n: int):
    cap = cv2.VideoCapture(str(video)); out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok: break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return out


def export_imu(E: Path, out: Path, off_ms: float) -> None:
    (out / "mav0/imu0").mkdir(parents=True, exist_ok=True)
    z = np.load(E / "imu_right.npz"); ti = fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]).astype(np.int64) + int(off_ms * 1e6)
    o = np.argsort(ti); ti, g_, a_ = ti[o], z["gyro"][o], z["acc"][o]; keep = np.r_[True, np.diff(ti) > 0]
    (out / "mav0/imu0/data.csv").write_text("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
                                             + "".join(f"{t},{w[0]:.9f},{w[1]:.9f},{w[2]:.9f},{x[0]:.9f},{x[1]:.9f},{x[2]:.9f}\n" for t, w, x in zip(ti[keep], g_[keep], a_[keep])))


def export(E: Path, out: Path, off_ms: float, max_skew_ms: float, mask=None, mask_r=None) -> int:
    tl, tr = np.load(E / "cam_left_t_ns.npy"), np.load(E / "cam_right_t_ns.npy"); FL, FR = frames(E / "cam_left.mp4", len(tl)), frames(E / "cam_right.mp4", len(tr))
    for c in ("cam0", "cam1"): (out / f"mav0/{c}/data").mkdir(parents=True, exist_ok=True)
    (out / "mav0/imu0").mkdir(parents=True, exist_ok=True); rows = []
    for k, t in enumerate(tl[:len(FL)]):
        j = int(np.argmin(np.abs(tr[:len(FR)] - t)))
        if abs(int(tr[j]) - int(t)) > max_skew_ms * 1e6: continue
        a, b = FL[k], FR[j]
        if mask is not None: a = a.copy(); a[mask] = 128
        if mask_r is not None: b = b.copy(); b[mask_r] = 128
        cv2.imwrite(str(out / f"mav0/cam0/data/{int(t)}.png"), a); cv2.imwrite(str(out / f"mav0/cam1/data/{int(t)}.png"), b); rows.append(int(t))
    for c in ("cam0", "cam1"): (out / f"mav0/{c}/data.csv").write_text("#timestamp [ns],filename\n" + "".join(f"{t},{t}.png\n" for t in rows))
    z = np.load(E / "imu_right.npz"); ti = fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]).astype(np.int64) + int(off_ms * 1e6)
    o = np.argsort(ti); ti, g_, a_ = ti[o], z["gyro"][o], z["acc"][o]; keep = np.r_[True, np.diff(ti) > 0]
    (out / "mav0/imu0/data.csv").write_text("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
                                             + "".join(f"{t},{w[0]:.9f},{w[1]:.9f},{w[2]:.9f},{x[0]:.9f},{x[1]:.9f},{x[2]:.9f}\n" for t, w, x in zip(ti[keep], g_[keep], a_[keep])))
    return len(rows)


def calib(out: Path, t_ci, sc: int, noise_x: float = 1.0) -> tuple[Path, np.ndarray]:
    st = yaml.safe_load((ROOT / STEREO_FILE).read_text()); ci = yaml.safe_load((ROOT / CAMIMU_FILE).read_text())
    T_C0I = np.eye(4); T_C0I[:3, :3] = np.asarray(ci["R_C0I"], float); T_C0I[:3, 3] = t_ci; T_I_C0 = np.linalg.inv(T_C0I)
    T_I_C1 = T_I_C0 @ np.linalg.inv(np.asarray(st["T_CR_CL"], float))
    tpl = json.loads((BAS / "data/t265_kb4_calib.json").read_text()); v = tpl["value0"]; Ts, intr = [], []
    for T, c in ((T_I_C0, st["left"]), (T_I_C1, st["right"])):
        q = R.from_matrix(T[:3, :3]).as_quat(); Ts.append(dict(px=float(T[0, 3]), py=float(T[1, 3]), pz=float(T[2, 3]), qx=float(q[0]), qy=float(q[1]), qz=float(q[2]), qw=float(q[3])))
        intr.append(dict(camera_type="kb4", intrinsics=dict(fx=c["fx"] / sc, fy=c["fy"] / sc, cx=c["cx"] / sc, cy=c["cy"] / sc, k1=c["k1"], k2=c["k2"], k3=c["k3"], k4=c["k4"])))
    v.update(T_imu_cam=Ts, intrinsics=intr, resolution=[[1920 // sc, 1080 // sc]] * 2, vignette=[], imu_update_rate=416.0, cam_time_offset_ns=0,
             calib_accel_bias=[0.0] * 9, calib_gyro_bias=[0.0] * 12, accel_noise_std=[0.016 * noise_x] * 3, gyro_noise_std=[0.000282 * noise_x] * 3, accel_bias_std=[0.001] * 3, gyro_bias_std=[0.0001] * 3)
    p = out / "calib_stereo_kb4.json"; p.write_text(json.dumps(tpl, indent=1)); return p, T_I_C0


def protocol_steps(E: Path, t_cam0, P, G, Rm, tt):
    """S7: displacement over each +-10 cm phase (median of the last 1.5 s of the previous phase vs of this phase), est (SE3-aligned) vs GT."""
    proto = json.loads((E / "protocol.json").read_text()); t0s = [p["t0"] for p in proto] + [1e9]; rec0 = t_cam0; out = []
    for k, p in enumerate(proto):
        if "센티" not in p["text"]: continue
        def win(lo, hi):
            m = (tt >= lo) & (tt < hi); return (np.median(P[m], 0), np.median(G[m], 0)) if m.sum() > 5 else (None, None)
        a_e, a_g = win(t0s[k] - 1.5, t0s[k]); b_e, b_g = win(t0s[k + 1] - 1.5, t0s[k + 1])
        if a_e is None or b_e is None: out.append(dict(phase=p["text"], note="no data")); continue
        de, dg = (b_e - a_e) @ Rm.T, b_g - a_g
        out.append(dict(phase=p["text"], gt_cm=round(float(np.linalg.norm(dg)) * 100, 2), est_cm=round(float(np.linalg.norm(de)) * 100, 2), err_cm=round(float(np.linalg.norm(de - dg)) * 100, 2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--t-ci", required=True); ap.add_argument("--tag", default=""); ap.add_argument("--mask", default=None)
    ap.add_argument("--offset-ms", type=float, default=None); ap.add_argument("--mask-right", default=None); ap.add_argument("--noise-x", type=float, default=1.0)
    ap.add_argument("--export-dir", default=None, help="share one image export between runs (only the IMU csv is rewritten)")
    ap.add_argument("--external-traj", default=None, help="score an existing TUM trajectory of IMU poses (e.g. OpenVINS) instead of running Basalt")
    ap.add_argument("--backend-label", default="basalt_stereo_inertial"); ap.add_argument("--wall-ms", type=float, default=None)
    ap.add_argument("--config", default="euroc_config.json"); ap.add_argument("--max-skew-ms", type=float, default=12.0); ap.add_argument("--reuse-export", action="store_true"); a = ap.parse_args()
    E = Path(a.take).resolve(); sc = json.loads((E / "meta.json").read_text())["save_scale"]; out = E / f"basalt_stereo{a.tag}"
    off = a.offset_ms if a.offset_ms is not None else float(yaml.safe_load((ROOT / CAMIMU_FILE).read_text())["offset_ms"]); t_ci = [float(v) for v in a.t_ci.split(",")]
    mask = np.load(a.mask) if a.mask else None; mask_r = np.load(a.mask_right) if a.mask_right else None
    if a.external_traj:
        out.mkdir(exist_ok=True); n = len(np.loadtxt(a.external_traj)); data_dir = None
    elif a.export_dir:
        out.mkdir(exist_ok=True); ex = Path(a.export_dir).resolve()
        if not (ex / "mav0/cam0").exists(): ex.mkdir(parents=True, exist_ok=True); export(E, ex, off, a.max_skew_ms, mask, mask_r)
        export_imu(E, ex, off); n = len(list((ex / "mav0/cam0/data").glob("*.png"))); data_dir = ex
    else:
        n = len(list((out / "mav0/cam0/data").glob("*.png"))) if a.reuse_export and (out / "mav0").exists() else export(E, out, off, a.max_skew_ms, mask, mask_r); data_dir = out
    cp, T_I_C0 = calib(out, t_ci, sc, a.noise_x); env = dict(os.environ, DYLD_LIBRARY_PATH=str(BAS / "lib"))
    cmd = [str(BAS / "bin/basalt_vio"), "--show-gui", "0", "--dataset-path", str(data_dir), "--dataset-type", "euroc", "--cam-calib", str(cp),
           "--config-path", str(BAS / "data" / a.config), "--use-imu", "1", "--save-trajectory", "tum", "--num-threads", "6"]
    if a.external_traj:
        import shutil; traj = out / "trajectory.txt"; shutil.copy(a.external_traj, traj); wall = (a.wall_ms or 0) * n / 1000.0
        r = type("R", (), dict(returncode=0, stdout="", stderr=""))()
    else:
        t0 = time.time(); r = subprocess.run(cmd, cwd=out, env=env, capture_output=True, text=True); wall = time.time() - t0
        (out / "basalt_stdout.log").write_text(r.stdout + "\n" + r.stderr); traj = out / "trajectory.txt"
    S = dict(take=E.name, backend=a.backend_label, config=a.config, t_ci=t_ci, offset_ms=off, noise_x=a.noise_x, mask=bool(a.mask or a.mask_right), frames=n, wall_ms_per_frame=round(wall * 1000 / max(1, n), 1), output_hz_offline=round(n / wall, 1))
    if r.returncode != 0 or not traj.exists():
        S["error"] = f"basalt exit {r.returncode}: {(r.stdout + r.stderr).strip().splitlines()[-3:]}"; print(json.dumps(S, indent=1, ensure_ascii=False)); return
    A = np.loadtxt(traj); tns = (A[:, 0] * 1e9 if A[0, 0] < 1e12 else A[:, 0]).astype(np.int64)
    Twi = np.tile(np.eye(4), (len(A), 1, 1)); Twi[:, :3, :3] = R.from_quat(A[:, 4:8]).as_matrix(); Twi[:, :3, 3] = A[:, 1:4]; Twc = Twi @ T_I_C0
    gt = np.load(E / "board_traj_cam_left.npz"); g_idx = {int(t): i for i, t in enumerate(gt["t_ns"])}
    m = np.array([int(t) in g_idx for t in tns]); gi = np.array([g_idx[int(t)] for t in tns[m]])
    P, Q = Twc[m, :3, 3], R.from_matrix(Twc[m, :3, :3]); G, RG = gt["T_B_C"][gi, :3, 3], R.from_matrix(gt["T_B_C"][gi, :3, :3])
    tl0 = np.load(E / "cam_left_t_ns.npy")[0]; tt = (tns[m] - tl0) / 1e9
    Rm = (RG * Q.inv()).mean().as_matrix(); tvec = G.mean(0) - (P @ Rm.T).mean(0); e = np.linalg.norm(P @ Rm.T + tvec - G, axis=1)  # SE(3) with rotation from orientations
    A_ = P @ Rm.T; a0, g0 = A_ - A_.mean(0), G - G.mean(0); s_sim = float((a0 * g0).sum() / (a0 * a0).sum())
    rot_err = np.degrees((RG * (R.from_matrix(Rm) * Q).inv()).magnitude())
    sys.argv = ["x"]; import trackcam_m3_vs_board as M
    M.FIXED_SCALE = 1.0
    S.update(poses=int(len(A)), matched_gt=int(m.sum()), ate_se3_cm_p50=round(float(np.median(e)) * 100, 2), ate_se3_cm_p90=round(float(np.percentile(e, 90)) * 100, 2),
             sim3_scale_info=round(s_sim, 3), rot_err_deg_p50=round(float(np.median(rot_err)), 2), rot_err_deg_p90=round(float(np.percentile(rot_err, 90)), 2),
             gt_path_m=round(float(np.sum(np.linalg.norm(np.diff(G, axis=0), axis=1))), 2), protocol_steps=protocol_steps(E, tl0, P, G, Rm, tt))
    w20 = tt < 20; R20 = (RG[w20] * Q[w20].inv()).mean().as_matrix(); t20 = G[w20].mean(0) - (P[w20] @ R20.T).mean(0); e20 = np.linalg.norm(P @ R20.T + t20 - G, axis=1)
    proto = json.loads((E / "protocol.json").read_text()); t0s = [p_["t0"] for p_ in proto] + [1e9]
    S["phase_err_cm_aligned_first20s"] = [dict(t0=t0s[k], phase=proto[k]["text"], med=round(float(np.median(e20[mm])) * 100, 1), max=round(float(e20[mm].max()) * 100, 1))
                                         for k in range(len(proto)) if (mm := (tt >= t0s[k]) & (tt < t0s[k + 1])).sum()]
    vv = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(np.diff(tt), 1e-3); S["jumps_gt_2m_s"] = int((vv > 2).sum())
    # S8 window metrics with scale frozen at 1 (re-uses the MASt3R/ORB evaluator logic on (frame index, pose))
    tcam = np.load(E / "cam_left_t_ns.npy"); f_of = {int(t): k for k, t in enumerate(tcam)}
    import csv as _csv
    lp = out / "basalt_as_orb3_log.csv"
    with open(lp, "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["t_ns", "tracking_state", "x", "y", "z", "qx", "qy", "qz", "qw", "num_keypoints", "num_map_points", "imu_initialized", "map_id", "n_maps", "relocalization_count", "lost_duration_ms", "processing_ms"])
        for t, T in zip(tns, Twc):
            q = R.from_matrix(T[:3, :3]).as_quat(); w.writerow([int(t), 2, *T[:3, 3], *q, 0, 0, 1, 0, 1, 0, 0, wall * 1000 / max(1, n)])
    sys.argv = ["x", str(E), str(out), "--orb3", str(lp), "--metric"]
    gt_bak = E / "board_traj.npz"; made = False
    if not gt_bak.exists(): os.symlink(E / "board_traj_cam_left.npz", gt_bak); made = True
    if not (E / "tracking_cam_t_ns.npy").exists(): os.symlink(E / "cam_left_t_ns.npy", E / "tracking_cam_t_ns.npy")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): M.main()
    w8 = json.loads((out / "orb3_eval_metric.json").read_text())
    S.update(step10_err_cm=w8.get("step10_err_cm"), drift30_cm=w8.get("drift30_cm"), fake_rot_cm=w8.get("fake_rot_cm"), scale_window_ratio=w8.get("scale_window_ratio"))
    (out / f"stereo_eval{a.tag}.json").write_text(json.dumps(S, indent=1, ensure_ascii=False)); print(json.dumps(S, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
