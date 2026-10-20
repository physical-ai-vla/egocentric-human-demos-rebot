"""Re-solve camera<->IMU from a saved trackcam_imu recording with ROTATION-ONLY visual motion (2026-10-15). The essential matrix is
degenerate for a camera rotating in place in front of a far ceiling (no baseline) -> recoverPose gave garbage (corr -0.15, 19 deg/frame
residual). Here: LK -> KB4 undistort -> unit bearings -> 3-point RANSAC Kabsch (b1 = R b0) -> camera angular rate. Then offset by
cross-correlating |w_cam| and |w_imu| (1 ms grid), then R_CI by Kabsch on rate vectors, then a joint refine of the offset.
    python scripts/trackcam_imu_offline.py datasets/trackcam_imu/<ts>"""
import json, sys
from pathlib import Path
import cv2, numpy as np, yaml
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handumi_collector.pose.timing import fit_device_to_host

def kabsch(a, b):
    U, _, Vt = np.linalg.svd(b.T @ a); d = np.sign(np.linalg.det(U @ Vt)); return U @ np.diag([1, 1, d]) @ Vt

def bearings(pts, Ks, D):
    n = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), Ks, D).reshape(-1, 2); v = np.c_[n, np.ones(len(n))]
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def rot_ransac(b0, b1, it=200, thr=np.radians(0.3), rng=np.random.default_rng(0)):
    best, bi = None, None
    for _ in range(it):
        s = rng.choice(len(b0), 3, replace=False); Rm = kabsch(b0[s], b1[s]); e = np.linalg.norm(b1 - b0 @ Rm.T, axis=1); inl = e < thr
        if bi is None or inl.sum() > bi.sum(): best, bi = Rm, inl
    return kabsch(b0[bi], b1[bi]), bi.mean()

def main():
    E = Path(sys.argv[1]); root = Path(__file__).resolve().parents[1]
    cam = sys.argv[2] if len(sys.argv) > 2 else None        # 10-15 stereo takes: cam_left / cam_right + intrinsics from stereo_trackcam_v003
    if cam:
        st = yaml.safe_load((root / "configs/calibration/stereo_trackcam_v005.yaml").read_text())[cam.replace("cam_", "")]; sc = json.loads((E / "meta.json").read_text())["save_scale"]
        Ks = np.array([[st["fx"] / sc, 0, st["cx"] / sc], [0, st["fy"] / sc, st["cy"] / sc], [0, 0, 1.0]]); D = np.array([st["k1"], st["k2"], st["k3"], st["k4"]]).reshape(4, 1)
        t_cam = np.load(E / f"{cam}_t_ns.npy"); cap = cv2.VideoCapture(str(E / f"{cam}.mp4")); frames = []
    else:
        y = yaml.safe_load((root / "configs/calibration/trackcam_fisheye_v001.yaml").read_text()); K = np.asarray(y["K"], float).reshape(3, 3); D = np.asarray(y["D"], float).reshape(4, 1)
        Ks = K.copy(); Ks[:2] /= 3.0
        t_cam = np.load(E / "tracking_cam_t_ns.npy"); cap = cv2.VideoCapture(str(E / "tracking_cam.mp4")); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    n = min(len(frames), len(t_cam)); tc, w_cam, inl_r = [], [], []
    for i in range(n - 1):
        p0 = cv2.goodFeaturesToTrack(frames[i], 400, 0.01, 8)
        if p0 is None or len(p0) < 20: continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(frames[i], frames[i + 1], p0, None); m = st.ravel() == 1
        if m.sum() < 20: continue
        fl = np.linalg.norm((p1 - p0).reshape(-1, 2), axis=1)
        if np.median(fl[m]) > 3.0: m = m & (fl > 0.6)          # 10-15: UMI parts/cables in view move WITH the camera -> zero flow
        if m.sum() < 20: continue
        Rm, fr = rot_ransac(bearings(p0[m], Ks, D), bearings(p1[m], Ks, D)); dt = (t_cam[i + 1] - t_cam[i]) / 1e9
        # b1 = Rm b0 for points fixed in the world means the CAMERA rotated by Rm^T: camera-frame angular rate = -log(Rm)/dt
        w_cam.append(-R.from_matrix(Rm).as_rotvec() / dt); tc.append((t_cam[i] + t_cam[i + 1]) / 2); inl_r.append(fr)
    tc, w_cam = np.array(tc), np.array(w_cam)
    z = np.load(E / "imu_right.npz"); clk = fit_device_to_host(z["device_us"], z["host_ns"]); ti = clk.to_host_ns(z["device_us"]).astype(np.float64); gy = z["gyro"]
    def imu_at(t, off_ms):                                # gyro (IMU frame) averaged over each camera interval centred at t, shifted by the offset
        tt = ti + off_ms * 1e6; return np.stack([np.interp(t, tt, gy[:, k]) for k in range(3)], 1)
    a_cam = np.linalg.norm(w_cam, axis=1); offs = np.arange(-200, 200.5, 1.0); cs = []
    for o in offs:
        a_imu = np.linalg.norm(imu_at(tc, o), axis=1); cs.append(np.corrcoef(a_cam, a_imu)[0, 1])
    o0 = float(offs[int(np.argmax(cs))]); Rci = kabsch(imu_at(tc, o0), w_cam)
    best = None
    for o in np.arange(o0 - 10, o0 + 10.01, 0.25):        # joint refine: vector residual with R_CI
        wi = imu_at(tc, o); Rr = kabsch(wi, w_cam); r = np.sqrt(np.mean(np.sum((w_cam - wi @ Rr.T) ** 2, 1)))
        if best is None or r < best[0]: best = (r, o, Rr)
    r, o, Rci = best; wi = imu_at(tc, o); res = w_cam - wi @ Rci.T
    rep = dict(n_pairs=int(len(tc)), inlier_ratio_median=round(float(np.median(inl_r)), 3), offset_ms=round(float(o), 2), mag_corr_peak=round(float(max(cs)), 3),
               vec_corr=[round(float(np.corrcoef(w_cam[:, k], (wi @ Rci.T)[:, k])[0, 1]), 3) for k in range(3)],
               residual_rate_rms_deg_s=round(float(np.degrees(r)), 2), rate_rms_deg_s=round(float(np.degrees(np.sqrt(np.mean(np.sum(w_cam ** 2, 1))))), 1),
               R_CI=Rci.round(6).tolist(), R_CI_rotvec_deg=np.degrees(R.from_matrix(Rci).as_rotvec()).round(2).tolist(),
               note="t_imu_host + offset_ms = camera time; w_cam = R_CI w_imu")
    (E / (f"camimu_rotonly_{cam}.json" if cam else "camimu_rotonly.json")).write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))

if __name__ == "__main__": main()
