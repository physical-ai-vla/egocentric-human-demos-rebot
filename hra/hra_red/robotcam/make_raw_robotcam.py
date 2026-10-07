"""[2026-10-06 user] HRA raw tree -> ROBOT-TCP labels: T_robot = T_umi @ inv(M), M = robot TCP -> HandUMI TCP measured by hand-eye
(robot right wrist cam X_tcp_cam (Park) @ HandUMI X_cam_tcp_full). With these labels and the exact-robot-camera wrist images the
policy runs on the robot with NO inference-side convention (V4_UMI_TCP_PITCH_DEG=0, V4_UMI_TCP_OFFSET_MM=0,0,0).
Copies every episode dir of ~/c8/hra_red/raw (incl. _scale_qc, export logs) and rewrites only right_position / right_quaternion."""
import json, pathlib, shutil, numpy as np
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); SRC = H / "c8/hra_red/raw"; DST = H / "c8/hra_red/raw_robotcam"
Xr = np.array(json.load(open(H / "c8/hra_red/handeye/robot_right_wrist_handeye.json"))["X_tcp_cam"]["park"])
Xu = np.load(H / "c8/hra_red/base_frame/X_cam_tcp_full.npy"); M = Xr @ Xu; Mi = np.linalg.inv(M)
assert not DST.exists(), DST
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("raw_episode.npz"))
n = 0
for d in sorted(SRC.iterdir()):
    f = d / "raw_episode.npz"
    if not f.exists(): continue
    z = dict(np.load(f)); p = z["right_position"]; q = z["right_quaternion"]          # quaternion order as stored (wxyz, pose_to_T)
    from ego_cart20.geometry.transforms import pose_to_T
    T = pose_to_T(p, q) @ Mi
    # back to the stored quaternion convention: verify by round trip
    from ego_cart20.geometry.rotation6d import matrix_to_quaternion
    q2 = matrix_to_quaternion(T[:, :3, :3]); p2 = T[:, :3, 3]
    assert np.allclose(pose_to_T(p2, q2), T, atol=1e-9)
    z["right_position"] = p2; z["right_quaternion"] = q2
    np.savez(DST / d.name / "raw_episode.npz", **z); n += 1
json.dump(dict(M_robot_tcp_to_umi_tcp=M.tolist(), source=str(SRC), episodes=n,
               note="T_robot = T_umi @ inv(M); M = handeye X_tcp_cam(park) @ X_cam_tcp_full"), open(DST / "ROBOTCAM.json", "w"), indent=1)
print("episodes", n, "M rotvec deg", np.round(np.degrees(Rot.from_matrix(M[:3, :3]).as_rotvec()), 2), "t mm", np.round(M[:3, 3] * 1000, 1))
