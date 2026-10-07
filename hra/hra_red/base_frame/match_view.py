"""[2026-10-06] camera-to-camera start matching: human UMI wrist-cam pose in base (T_start @ inv(X_cam_tcp_full)) vs the robot
right wrist-cam (hand-eye, Park) -> the robot TCP pose that puts the robot camera where the human camera was; offline Pink IK."""
import json, sys, pathlib, numpy as np
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); sys.path.insert(0, str(H / "holobrain-mac-model"))
import eef_kin, pink_ik
B = json.load(open("base_start.json")); Xu = np.load("X_cam_tcp_full.npy")           # UMI: TCP in camera
HE = json.load(open(H / "c8/hra_red/handeye/robot_right_wrist_handeye.json")); Xr = np.array(HE["X_tcp_cam"]["park"])   # robot: camera in TCP
def desc(T):
    a = T[:3, 2]; return dict(pos_mm=np.round(T[:3, 3] * 1000, 1).tolist(), pitch=round(float(np.degrees(np.arcsin(-a[2]))), 1),
                              heading=round(float(np.degrees(np.arctan2(a[1], a[0]))), 1), roll=round(float(np.degrees(np.arcsin(np.clip(T[2, 0], -1, 1)))), 1))
Cam = [np.array(b["T_start"]) @ np.linalg.inv(Xu) for b in B]
D = [desc(c) for c in Cam]
pos = np.median([d["pos_mm"] for d in D], 0); print("HUMAN wrist cam (n=%d) pos p50 %s  p10 %s  p90 %s" % (len(D), np.round(pos), np.round(np.percentile([d['pos_mm'] for d in D], 10, 0)), np.round(np.percentile([d['pos_mm'] for d in D], 90, 0))))
for k in ("pitch", "heading", "roll"): print(f"  {k}: p10/50/90", np.percentile([d[k] for d in D], [10, 50, 90]).round(1))
# median camera rotation (chordal mean)
Rm = Rot.from_matrix(np.stack([c[:3, :3] for c in Cam])).mean().as_matrix(); Tc = np.eye(4); Tc[:3, :3] = Rm; Tc[:3, 3] = pos / 1000
print("median human cam", desc(Tc))
kin = eef_kin.Kin({"max_joint_delta": None}); names = [kin._names[j] for j in kin._arm["left"]] + [kin._names[j] for j in kin._arm["right"]]
P = pink_ik.PinkIK(names); lo, hi = P.model.lowerPositionLimit, P.model.upperPositionLimit
S = json.load(open(H / "c8/hra_red/umi_start_pose.json")); q0 = np.zeros(12); q0[6:] = np.radians(S["right_joints_deg"])
def reach(Tt):
    q, _, _ = P.solve(q0, [P.fk(q0)[0], Tt], iters=500); f = P.fk(q)[1]
    return np.linalg.norm(f[:3, 3] - Tt[:3, 3]) * 1000, np.degrees(np.linalg.norm(Rot.from_matrix(f[:3, :3].T @ Tt[:3, :3]).as_rotvec())), np.degrees(np.minimum(q - lo, hi - q))[6:].min(), np.degrees(q[6:])
# robot camera at the current RUN start (TCP pitch 25, rest heading)
T0 = np.array(S["T_base_tcp"]); up = np.array([0, 0, 1.0]); h = T0[:3, 0] - (T0[:3, 0] @ up) * up; h /= np.linalg.norm(h)
def R_of(p):
    pr = np.radians(p); a = np.cos(pr) * h - np.sin(pr) * up; y = np.cross(up, a); y /= np.linalg.norm(y)
    if y @ T0[:3, 1] < 0: y = -y
    return np.stack([a, y, np.cross(a, y)], 1)
Tr = np.eye(4); Tr[:3, :3] = R_of(25); Tr[:3, 3] = [0.295, -0.309, 0.084]
print("ROBOT cam at run start (TCP pitch 25):", desc(Tr @ Xr))
Tt = Tc @ np.linalg.inv(Xr); pe, re, mg, qd = reach(Tt)
print("ROBOT TCP that puts its cam on the median human cam:", desc(Tt) | {"tcp_x_pitch": round(float(np.degrees(np.arcsin(-Tt[2, 0]))), 1)},
      f"IK pos {pe:.1f} mm rot {re:.2f} deg margin {mg:.1f} deg q {np.round(qd, 1)}")
json.dump(dict(human_cam_median=Tc.tolist(), robot_tcp_for_matched_view=Tt.tolist(), ik=dict(pos_mm=pe, rot_deg=re, margin_deg=mg, q_deg=qd.tolist())), open("matched_view.json", "w"), indent=1)
# feasible search: camera POSITION = human median exactly, TCP pitch swept (camera pitch = TCP pitch + ~33), heading = human cam heading
print("\nTCP pitch sweep with the camera placed on the median human camera position:")
hc = Tc[:3, 2] - (Tc[:3, 2] @ up) * up; hc /= np.linalg.norm(hc); best = []
for hd in (hc, h):
    for p in range(0, 45, 3):
        pr = np.radians(p); a = np.cos(pr) * hd - np.sin(pr) * up; y = np.cross(up, a); y /= np.linalg.norm(y)
        if y @ T0[:3, 1] < 0: y = -y
        R = np.stack([a, y, np.cross(a, y)], 1); T = np.eye(4); T[:3, :3] = R
        T[:3, 3] = Tc[:3, 3] - R @ Xr[:3, 3]
        pe, re, mg, qd = reach(T); c = desc(T @ Xr)
        ok = pe < 2 and re < 1 and mg > 5
        print(f"  heading {'human' if hd is hc else 'robot'} TCP pitch {p:2d}: TCP {np.round(T[:3,3]*1000)} (table+{T[2,3]*1000+54.3:.0f})  cam pitch {c['pitch']:5.1f}  IK {'OK' if ok else 'NO'} pos {pe:5.1f} margin {mg:5.1f}")
        if ok: best.append((abs(c["pitch"] - 51.0), p, T, qd))
if best:
    d, p, T, qd = min(best, key=lambda b: b[0]); print("BEST feasible: TCP pitch", p, "TCP mm", np.round(T[:3, 3] * 1000, 1), "cam pitch err", round(d, 1), "q", np.round(qd, 1))
    json.dump(dict(T_base_tcp=T.tolist(), tcp_mm=np.round(T[:3, 3] * 1000, 1).tolist(), tcp_pitch=p, cam_pitch_err_deg=d, q_deg=qd.tolist(),
                   note="2026-10-06 robot TCP putting the robot wrist cam on the median human HRA wrist-cam position (hand-eye Park); offline IK only"),
              open("matched_view_feasible.json", "w"), indent=1)
