"""[2026-10-06 user step 1] OFFLINE reachability of candidate robot start poses near the human HRA start median (base frame).
Same Pink IK + URDF the UI uses (IK free, posture 1e-3); seeded at the saved rest pose (umi_start_pose.json right joints). No robot motion."""
import json, pathlib, sys, itertools
import numpy as np
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); sys.path.insert(0, str(H / "holobrain-mac-model"))
import eef_kin, pink_ik
kin = eef_kin.Kin({"max_joint_delta": None})
names = [kin._names[j] for j in kin._arm["left"]] + [kin._names[j] for j in kin._arm["right"]]
P = pink_ik.PinkIK(names)
S = json.load(open(H / "c8/hra_red/umi_start_pose.json")); Tsaved = np.array(S["T_base_tcp"])
q0 = np.zeros(12); q0[6:] = np.radians(S["right_joints_deg"])
fk0 = P.fk(q0)[1]
print("FK check of saved rest: FK", np.round(fk0[:3, 3] * 1000, 1), "saved", S["tcp_mm"], " rot err deg",
      round(float(np.degrees(np.linalg.norm(Rot.from_matrix(fk0[:3, :3].T @ Tsaved[:3, :3]).as_rotvec()))), 2))
lo = P.model.lowerPositionLimit; hi = P.model.upperPositionLimit
B = json.load(open(H / "c8/hra_red/base_frame/base_start.json"))
med = np.median([b["start_tcp_mm"] for b in B], 0)
TABLE_Z = -54.3
up = np.array([0, 0, 1.0]); h = Tsaved[:3, 0] - (Tsaved[:3, 0] @ up) * up; h /= np.linalg.norm(h)


def R_of(pitch, heading_from=h):
    pr = np.radians(pitch); a = np.cos(pr) * heading_from - np.sin(pr) * up
    y = np.cross(up, a); y /= np.linalg.norm(y)
    if y @ Tsaved[:3, 1] < 0: y = -y
    return np.stack([a, y, np.cross(a, y)], 1)


pts = {"human_median": med, "mid(265,-345,40)": np.array([265, -345, 40.0]), "run_start(295,-309,84)": np.array([295, -309, 84.0])}
for k, (dx, dy) in {"p10x": (184 - med[0], 0), "p90x": (286 - med[0], 0), "p10y": (0, -383 - med[1]), "p90y": (0, -322 - med[1])}.items():
    pts["human_" + k] = med + np.array([dx, dy, 0])
rows = []
for (name, p), pitch in itertools.product(pts.items(), (25, 35, 45, 53)):
    T = np.eye(4); T[:3, :3] = R_of(pitch); T[:3, 3] = p / 1000
    q, perr, it = P.solve(q0, [P.fk(q0)[0], T], iters=400)
    if not np.isfinite(q).all():
        rows.append((name, pitch, "NaN")); continue
    f = P.fk(q)[1]; pe = np.linalg.norm(f[:3, 3] - T[:3, 3]) * 1000
    re = np.degrees(np.linalg.norm(Rot.from_matrix(f[:3, :3].T @ T[:3, :3]).as_rotvec()))
    marg = np.degrees(np.minimum(q - lo, hi - q))[6:].min()
    ok = pe < 3 and re < 2 and marg > 3
    rows.append(dict(pt=name, xyz_mm=np.round(p, 0).tolist(), above_table_mm=round(p[2] - TABLE_Z), pitch=pitch, ok=bool(ok), pos_err_mm=round(pe, 1),
                     rot_err_deg=round(re, 2), min_limit_margin_deg=round(float(marg), 1), right_q_deg=np.round(np.degrees(q[6:]), 1).tolist()))
for r in rows:
    print(r if isinstance(r, tuple) else f"{'OK ' if r['ok'] else 'NO '} {r['pt']:24s} {str(r['xyz_mm']):22s} tbl+{r['above_table_mm']:4d}  pitch {r['pitch']:2d}  "
          f"pos {r['pos_err_mm']:6.1f} mm  rot {r['rot_err_deg']:5.2f}  margin {r['min_limit_margin_deg']:6.1f}  q {r['right_q_deg']}")
json.dump(rows, open(H / "c8/hra_red/base_frame/start_reach_check.json", "w"), indent=1, default=str)
