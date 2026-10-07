"""[2026-10-06] robot wrist view at pose A vs human HRA wrist start frames (undistorted to the robot camera's pinhole f/size)."""
import json, sys, pathlib, numpy as np, cv2
H = pathlib.Path.home(); sys.path.insert(0, str(H / "holobrain-mac-model")); sys.path.insert(0, str(H / "ego_cart20"))
import eef_kin, pink_ik
from ego_cart20 import cube_pnp as CP
kin = eef_kin.Kin({"max_joint_delta": None}); names = [kin._names[j] for j in kin._arm["left"]] + [kin._names[j] for j in kin._arm["right"]]
P = pink_ik.PinkIK(names); HE = json.load(open("robot_right_wrist_handeye.json")); Xr = np.array(HE["X_tcp_cam"]["park"]); Kr = np.array(HE["K"])
j = np.array(json.load(open("viewA.json"))["joints_rad"]); q = np.zeros(12); q[:6] = j[:6]; q[6:] = j[7:13]; C = P.fk(q)[1] @ Xr
a = C[:3, 2]; print("robot cam at A: pos mm", np.round(C[:3, 3] * 1000, 1), "above table", round(C[2, 3] * 1000 + 54.3), "pitch", round(float(np.degrees(np.arcsin(-a[2]))), 1))
B = json.load(open(H / "c8/hra_red/base_frame/base_start.json")); Xu = np.load(H / "c8/hra_red/base_frame/X_cam_tcp_full.npy")
cams = []
for b in B:
    c = np.array(b["T_start"]) @ np.linalg.inv(Xu); cams.append((np.linalg.norm(c[:3, 3] - np.array([0.217, -0.346, 0.153])), b["ep"], c))
cams.sort(key=lambda x: x[0]); tiles = [cv2.putText(cv2.imread("viewA_right.jpg"), "ROBOT pose A  cam %d mm above table, pitch %.0f" % (C[2, 3] * 1000 + 54.3, np.degrees(np.arcsin(-a[2]))), (8, 22), 0, 0.55, (0, 255, 255), 2)]
EXP = H / "c8/robotlike/export"
for _, ep, c in cams[:3]:
    ex = EXP / f"red_20261003_152502_{ep.split('_')[-1]}_right"
    Kw, Dw, size = CP.read_setting(ex / "orbslam_setting.yaml")
    cap = cv2.VideoCapture(str(ex / "raw_video.mp4")); ok, img = cap.read(); cap.release()
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(Kw, Dw, np.eye(3), Kr, (640, 480), cv2.CV_32FC1) if len(Dw) == 4 else cv2.initUndistortRectifyMap(Kw, Dw, np.eye(3), Kr, (640, 480), cv2.CV_32FC1)
    u = cv2.remap(img, m1, m2, cv2.INTER_LINEAR); ac = c[:3, 2]
    tiles.append(cv2.putText(u, "HUMAN %s  cam %d mm, pitch %.0f" % (ep[-4:], c[2, 3] * 1000 + 54.3, np.degrees(np.arcsin(-ac[2]))), (8, 22), 0, 0.55, (0, 255, 255), 2))
cv2.imwrite("compare_A.jpg", np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])); print("dist model", "fisheye" if len(Dw) == 4 else "pinhole", size)
