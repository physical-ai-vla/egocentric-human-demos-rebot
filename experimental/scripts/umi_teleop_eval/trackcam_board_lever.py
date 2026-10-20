"""Ground-truth camera trajectory from the fixed Kalibr AprilGrid + camera->IMU LEVER ARM from the accelerometer (2026-10-15).
Basalt's calibrators did not converge on this fisheye (camera init fx 117 / 51 px; imu stuck at 64 px), so:

  1. detect tags (OpenCV aruco DICT_APRILTAG_36h11, markerBorderBits=2 = Kalibr grid), board corners from the Basalt/Kalibr layout
     (id = row*cols + col, row 0 at the bottom, y up), KB4 undistortion (trackcam_fisheye_v001, scaled to the saved video), PnP
     (IPPE + LM refine) per frame -> T_B_C (camera in board frame), metric because the tag size is known.
  2. smoothing splines on the camera position (board frame) -> acceleration a_C^B.
  3. accelerometer model (IMU frame I, camera C, R_CI from trackcam_imu_v001, r = camera position in the IMU frame, omega/alpha = gyro and its
     derivative):   f_I = R_IB (a_C^B - g_B) - (alpha x r + omega x (omega x r)) + b_a        -- LINEAR in r, b_a, g_B
     solved by least squares on every frame with a good board pose; time offset re-scanned around the gyro-only value.
  Checks: |g_B| should come out ~9.81; residual RMS; r compared with the hand measurement (9 cm up, 20 cm forward).
    python scripts/trackcam_board_lever.py datasets/trackcam_imu/<board recording> [--tag-size 0.025]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.interpolate import make_smoothing_spline
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from handumi_collector.pose.timing import fit_device_to_host   # noqa: E402


def board_corners(tag_id: int, size: float, spacing: float = 0.3, cols: int = 6) -> np.ndarray:
    r, c = divmod(tag_id, cols); x0, y0 = c * size * (1 + spacing), r * size * (1 + spacing)
    # OpenCV aruco corner order: top-left, top-right, bottom-right, bottom-left of the upright marker; board y is UP
    return np.array([[x0, y0 + size, 0], [x0 + size, y0 + size, 0], [x0 + size, y0, 0], [x0, y0, 0]], float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec"); ap.add_argument("--tag-size", type=float, default=0.025); ap.add_argument("--min-tags", type=int, default=4); ap.add_argument("--border-bits", type=int, default=1)
    ap.add_argument("--stereo-cam", default=None, help="10-15 stereo takes: cam_left|cam_right -> video/stamps <cam>.mp4, intrinsics stereo_trackcam_v003, R_CI/offset from --camimu")
    ap.add_argument("--camimu", default="configs/calibration/trackcam_imu_v001.yaml"); a = ap.parse_args()
    E = Path(a.rec); meta = json.loads((E / "meta.json").read_text()); sc = meta["save_scale"]
    if a.stereo_cam:
        st = yaml.safe_load((ROOT / "configs/calibration/stereo_trackcam_v005.yaml").read_text())[a.stereo_cam.replace("cam_", "")]
        K = np.array([[st["fx"], 0, st["cx"]], [0, st["fy"], st["cy"]], [0, 0, 1.0]]) / sc; K[2, 2] = 1.0; D = np.array([st["k1"], st["k2"], st["k3"], st["k4"]]).reshape(4, 1)
        vid, tsf = f"{a.stereo_cam}.mp4", f"{a.stereo_cam}_t_ns.npy"
    else:
        yi = yaml.safe_load((ROOT / "configs/calibration/trackcam_fisheye_v001.yaml").read_text()); K = np.asarray(yi["K"], float).reshape(3, 3) / sc; D = np.asarray(yi["D"], float).reshape(4, 1)
        vid, tsf = "tracking_cam.mp4", "tracking_cam_t_ns.npy"
    ye = yaml.safe_load((ROOT / a.camimu).read_text()); Rci = np.asarray(ye.get("R_CI", ye.get("R_C0I")), float); off0 = float(ye["offset_ms"])
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), (lambda p: (setattr(p, "markerBorderBits", a.border_bits), setattr(p, "cornerRefinementMethod", cv2.aruco.CORNER_REFINE_SUBPIX), p)[-1])(cv2.aruco.DetectorParameters()))
    t_cam = np.load(E / tsf); cap = cv2.VideoCapture(str(E / vid)); T, P_ok, rep = [], [], []
    for k in range(len(t_cam)):
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); corners, ids, _ = det.detectMarkers(g)
        if ids is None or len(ids) < a.min_tags: T.append(None); continue
        obj = np.vstack([board_corners(int(i), a.tag_size) for i in ids.ravel() if i < 36]); img = np.vstack([c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if i < 36])
        und = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        ok, rv, tv = cv2.solvePnP(obj, und, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        if not ok: T.append(None); continue
        rv, tv = cv2.solvePnPRefineLM(obj, und, np.eye(3), None, rv, tv)
        proj, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rv, tv, K, D); e = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        Tcb = np.eye(4); Tcb[:3, :3] = R.from_rotvec(rv.ravel()).as_matrix(); Tcb[:3, 3] = tv.ravel(); T.append(np.linalg.inv(Tcb)); rep.append(float(np.median(e)))
    n = len(T); good = np.array([x is not None for x in T]); print(f"프레임 {n}, 보드 자세 {good.sum()} ({100 * good.mean():.0f}%), 재투영 오차 중앙 {np.median(rep):.2f}px p90 {np.percentile(rep, 90):.2f}px")
    tt = (t_cam[:n] - t_cam[0]) / 1e9; idx = np.flatnonzero(good); TB = np.array([T[i] for i in idx]); tb = tt[idx]
    np.savez(E / (f"board_traj_{a.stereo_cam}.npz" if a.stereo_cam else "board_traj.npz"), t_ns=t_cam[:n][idx], T_B_C=TB)
    # contiguous good runs only (no spline across long gaps)
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= 2: cur.append(i)
        else: runs.append(cur); cur = [i]
    runs.append(cur); runs = [r for r in runs if len(r) >= 30]
    z = np.load(E / "imu_right.npz"); ti_host = fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]).astype(float); gy, ac = z["gyro"], z["acc"]
    def solve(off_ms):
        ti = (ti_host + off_ms * 1e6 - t_cam[0]) / 1e9; A, b = [], []
        gys = np.stack([make_smoothing_spline(ti, gy[:, k], lam=1e-4) for k in range(3)]) if False else None
        for r in runs:
            t_r = tt[r]; Pb = np.array([T[i][:3, 3] for i in r]); Rbc = np.array([T[i][:3, :3] for i in r])
            acc_b = np.stack([make_smoothing_spline(t_r, Pb[:, k], lam=1e-3).derivative(2)(t_r) for k in range(3)], 1)
            m = (ti >= t_r[0]) & (ti <= t_r[-1])
            if m.sum() < 50: continue
            w_s = [make_smoothing_spline(ti[m], gy[m, k], lam=1e-5) for k in range(3)]
            w = np.stack([s(t_r) for s in w_s], 1); al = np.stack([s.derivative(1)(t_r) for s in w_s], 1)
            f = np.stack([np.interp(t_r, ti[m], ac[m, k]) for k in range(3)], 1)
            for j in range(3, len(t_r) - 3):                                      # drop spline ends
                R_IB = (Rbc[j] @ Rci).T                                            # R_BI = R_BC R_CI
                Wx = np.array([[0, -w[j, 2], w[j, 1]], [w[j, 2], 0, -w[j, 0]], [-w[j, 1], w[j, 0], 0]]); Ax = np.array([[0, -al[j, 2], al[j, 1]], [al[j, 2], 0, -al[j, 0]], [-al[j, 1], al[j, 0], 0]])
                # f - R_IB a = -R_IB g_B - (Ax + Wx Wx) r + b_a    ->  unknowns [r(3), b_a(3), g_B(3)]
                A.append(np.hstack([-(Ax + Wx @ Wx), np.eye(3), -R_IB])); b.append(f[j] - R_IB @ acc_b[j])
        A, b = np.vstack(A), np.concatenate(b); x, *_ = np.linalg.lstsq(A, b, rcond=None); res = b - A @ x
        return x, float(np.sqrt(np.mean(res ** 2))), len(b) // 3
    best = None
    for o in np.arange(off0 - 30, off0 + 30.1, 2.5):
        x, rms, nn = solve(o)
        if best is None or rms < best[1]: best = (x, rms, o, nn)
    x, rms, o, nn = best; r_I, b_a, g_B = x[:3], x[3:6], x[6:9]
    # express the lever arm in UMI terms (up = -gravity at rest in IMU, fwd = jaw axis) for comparison with the hand measurement
    sys.path.insert(0, str(ROOT)); from visual_teleop import config as vcfg
    RIH, _ = vcfg.R_I_H(vcfg.load(), "right"); up = ac[: 300].mean(0); up /= np.linalg.norm(up); fwd = RIH[:, 2] - (RIH[:, 2] @ up) * up; fwd /= np.linalg.norm(fwd); left = np.cross(up, fwd)
    S = dict(board_pose_frac=round(float(good.mean()), 3), pnp_reproj_px_median=round(float(np.median(rep)), 3), runs=len(runs), samples=nn,
             offset_ms=round(float(o), 1), residual_rms_m_s2=round(rms, 3), g_norm=round(float(np.linalg.norm(g_B)), 3),
             lever_arm_cam_in_imu_cm=np.round(r_I * 100, 2).tolist(), lever_arm_norm_cm=round(float(np.linalg.norm(r_I) * 100), 2),
             lever_in_umi_terms_cm=dict(up=round(float(r_I @ up * 100), 1), forward=round(float(r_I @ fwd * 100), 1), left=round(float(r_I @ left * 100), 1)),
             accel_bias=np.round(b_a, 3).tolist(), t_CI_m=np.round(-Rci @ r_I, 4).tolist())
    (E / (f"board_lever_{a.stereo_cam}.json" if a.stereo_cam else "board_lever.json")).write_text(json.dumps(S, indent=1)); print(json.dumps(S, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
