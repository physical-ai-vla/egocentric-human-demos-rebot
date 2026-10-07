"""[2026-10-06] candidate right-arm poses whose TCP approach axis points at the checkerboard centre (heading kept = current
robot heading, as /hra_start_pose keeps it); offline Pink IK from the CURRENT measured joints, chained pose to pose."""
import json, sys, pathlib, itertools, numpy as np
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); sys.path.insert(0, str(H / "holobrain-mac-model"))
import eef_kin, pink_ik
kin = eef_kin.Kin({"max_joint_delta": None}); names = [kin._names[j] for j in kin._arm["left"]] + [kin._names[j] for j in kin._arm["right"]]
P = pink_ik.PinkIK(names); lo, hi = P.model.lowerPositionLimit, P.model.upperPositionLimit
S = json.load(open(H / "c8/hra_red/umi_start_pose.json")); q0 = np.zeros(12); q0[6:] = np.radians(S["right_joints_deg"])
T0 = P.fk(q0)[1]; up = np.array([0, 0, 1.0]); h = T0[:3, 0] - (T0[:3, 0] @ up) * up; h /= np.linalg.norm(h)
ctr = (np.load("T_base_board_now.npy") @ np.array([0.0875, 0.05, 0, 1]))[:3]
def R_of(pitch, roll):
    pr = np.radians(pitch); a = np.cos(pr) * h - np.sin(pr) * up; y = np.cross(up, a); y /= np.linalg.norm(y)
    if y @ T0[:3, 1] < 0: y = -y
    rr = np.radians(roll); y = np.cos(rr) * y - np.sin(rr) * np.cross(a, y)
    return np.stack([a, y, np.cross(a, y)], 1)
out = []
for pitch, d, dy, roll in itertools.product((40, 50, 60, 70), (0.14, 0.20), (-0.12, -0.06, 0.0), (-15, 0, 15)):
    R = R_of(pitch, roll); p = ctr - d * R[:, 0] + np.array([0, dy, 0])
    if p[2] < -0.054 + 0.06: continue
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    q, _, _ = P.solve(q0, [P.fk(q0)[0], T], iters=400)
    if not np.isfinite(q).all(): continue
    f = P.fk(q)[1]; pe = np.linalg.norm(f[:3, 3] - p) * 1000; re = np.degrees(np.linalg.norm(Rot.from_matrix(f[:3, :3].T @ R).as_rotvec()))
    marg = np.degrees(np.minimum(q - lo, hi - q))[6:].min()
    if pe < 2 and re < 1 and marg > 8 and np.abs(np.degrees(q[6:])).max() < 120:
        out.append(dict(pitch=pitch, roll=roll, d=d, dy=dy, tcp_mm=np.round(p * 1000, 1).tolist(), margin=round(float(marg), 1), q=np.round(np.degrees(q[6:]), 1).tolist()))
print(len(out), "feasible"); [print(o) for o in out[:60]]
json.dump(out, open("feasible.json", "w"), indent=1)
