"""[2026-10-06] robot RIGHT wrist cam: intrinsics + hand-eye (camera in TCP) from hcNN captures (board fixed, arm moved).
TCP = the same Pink/URDF TCP the UI uses (dataset frame). Joint order of /observe joints_rad: left 7 (6 + jaw), right 7."""
import json, glob, sys, pathlib, numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); sys.path.insert(0, str(H / "holobrain-mac-model"))
import eef_kin, pink_ik
kin = eef_kin.Kin({"max_joint_delta": None}); names = [kin._names[j] for j in kin._arm["left"]] + [kin._names[j] for j in kin._arm["right"]]
P = pink_ik.PinkIK(names)
obj = np.zeros((40, 3), np.float32); obj[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * 0.025
ims, cs, Ts, nm = [], [], [], []
for f in sorted(glob.glob("hc*_right.jpg")):
    n = f.split("_")[0]; g = cv2.imread(f, 0)
    if n in ("hc05", "hc14"): continue          # refused moves: same robot pose as the previous view
    ok, c = cv2.findChessboardCorners(g, (8, 5), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok: continue
    c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (3, 30, 1e-3))
    j = np.array(json.load(open(n + ".json"))["joints_rad"]); q = np.zeros(12); q[:6] = j[0:6]; q[6:] = j[7:13]
    Ts.append(P.fk(q)[1]); cs.append(c); nm.append(n)
print("views", len(cs), nm)
for n, T in zip(nm, Ts):
    a = T[:3, 0]; print(n, "FK tcp mm", np.round(T[:3, 3] * 1000, 1), "pitch", round(float(np.degrees(np.arcsin(-a[2]))), 1))
rms, K, D, rv, tv = cv2.calibrateCamera([obj] * len(cs), cs, (640, 480), np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.0]]), np.zeros(5),
                                        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
print("intrinsics rms %.3f px  f %.1f  c (%.1f, %.1f)  dist %s" % (rms, K[0, 0], K[0, 2], K[1, 2], np.round(D.ravel(), 3)))
Rg = [T[:3, :3] for T in Ts]; tg = [T[:3, 3] for T in Ts]; Rt = [cv2.Rodrigues(r)[0] for r in rv]; tt = [t.ravel() for t in tv]
res = {}
for name, m in (("tsai", cv2.CALIB_HAND_EYE_TSAI), ("park", cv2.CALIB_HAND_EYE_PARK), ("horaud", cv2.CALIB_HAND_EYE_HORAUD), ("daniilidis", cv2.CALIB_HAND_EYE_DANIILIDIS)):
    R, t = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=m); X = np.eye(4); X[:3, :3] = R; X[:3, 3] = t.ravel()
    # consistency: board in base from each view should be one fixed pose
    B = [T @ X @ np.vstack([np.hstack([r, t_[:, None]]), [0, 0, 0, 1]]) for T, r, t_ in zip(Ts, Rt, tt)]
    pb = np.array([b[:3, 3] for b in B]); rot_sp = [np.degrees(np.linalg.norm(Rot.from_matrix(b[:3, :3].T @ B[0][:3, :3]).as_rotvec())) for b in B]
    z_cam_in_tcp = X[:3, 2]; ang_x = np.degrees(np.arccos(np.clip(z_cam_in_tcp @ np.array([1, 0, 0.]), -1, 1)))
    res[name] = X
    print(f"{name:10s} cam pos in TCP mm {np.round(X[:3,3]*1000,1)}  optical axis in TCP {np.round(z_cam_in_tcp,3)} (vs TCP x {ang_x:.1f} deg)  "
          f"board-in-base spread pos {np.round(pb.std(0)*1000,1)} mm, rot p50 {np.median(rot_sp):.2f} deg")
json.dump(dict(K=K.tolist(), dist=D.ravel().tolist(), rms=rms, views=nm, X_tcp_cam={k: v.tolist() for k, v in res.items()}), open("robot_right_wrist_handeye.json", "w"), indent=1)
