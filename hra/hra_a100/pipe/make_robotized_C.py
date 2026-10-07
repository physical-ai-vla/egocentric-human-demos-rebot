"""[2026-10-07 user "5개 빼고 C 만들어서 학습"] C = ROBOTIZED trajectories in the shared frame: every trained segment (after the AUTO go) of
the robot-TCP label trajectory, in the robot base (T_b = F @ T_F), is warped so it STARTS at the robot start pose T_S (umi_start_pose.json =
HRA_A100 human-median start) and ENDS at the human endpoint unchanged: T_c(t) = E(w(s)) @ T_b(t), E = T_S @ inv(T_b(go)),
E(w) = [Rot(w * rotvec(E)) | w * t_E], w(s) = 1 - smoothstep(s), s = 0 at go .. 1 at the last row. Stored back in F (B convention;
convert_origin.py -> anchor = identity -> state = absolute pose in F; inference = the B fixed-anchor mode).
Excludes the B-replay episodes whose jaw tip goes > 10 mm below the table (scale / SLAM errors; table = X plate - 35 mm).
usage: make_robotized_C.py <raw_v2_origin> <out>"""
import json, pathlib, shutil, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); sys.path.insert(0, str(H / "ego_cart20"))
from ego_cart20.geometry.transforms import pose_to_T
from ego_cart20.geometry.rotation6d import matrix_to_quaternion
A = H / "c8/hra_a100"; SRC, OUT = map(pathlib.Path, sys.argv[1:3])
F = np.array(json.load(open(A / "B_anchor_F.json"))["T_base_F"]); Fi = np.linalg.inv(F)
TS = np.array(json.load(open(H / "c8/hra_red/umi_start_pose.json"))["T_base_tcp"])
ik = json.load(open(A / "ik_check_a93.json")); bad = {k for k, v in ik.items() if v["B"]["min_tip_above_plate_mm"] + 35.0 < -10.0}
print("excluded (B replay > 10 mm below the table):", sorted(bad))
assert not OUT.exists(); OUT.mkdir(parents=True); rep = {}
for d in sorted(p for p in SRC.iterdir() if (p / "raw_episode.npz").exists()):
    if d.name in bad: continue
    shutil.copytree(d, OUT / d.name, ignore=shutil.ignore_patterns("raw_episode.npz"))
    z = dict(np.load(d / "raw_episode.npz", allow_pickle=True)); v = z["right_valid"].astype(bool); idx = np.flatnonzero(v)
    TF = pose_to_T(z["right_position"], z["right_quaternion"]); Tb = F[None] @ TF
    E = TS @ np.linalg.inv(Tb[idx[0]]); rv = Rot.from_matrix(E[:3, :3]).as_rotvec(); tE = E[:3, 3]
    s = np.clip((np.arange(len(Tb)) - idx[0]) / max(idx[-1] - idx[0], 1), 0, 1); w = 1 - (3 * s ** 2 - 2 * s ** 3)
    Tc = Tb.copy()
    for i in idx:
        Ew = np.eye(4); Ew[:3, :3] = Rot.from_rotvec(w[i] * rv).as_matrix(); Ew[:3, 3] = w[i] * tE; Tc[i] = Ew @ Tb[i]
    TcF = Fi[None] @ Tc; z["right_position"] = TcF[:, :3, 3]; z["right_quaternion"] = matrix_to_quaternion(TcF[:, :3, :3])
    np.savez(OUT / d.name / "raw_episode.npz", **z)
    rep[d.name] = dict(start_shift_mm=round(float(np.linalg.norm(tE)) * 1000, 1), start_rot_deg=round(float(np.degrees(np.linalg.norm(rv))), 1),
                       start_err_mm=round(float(np.linalg.norm(Tc[idx[0]][:3, 3] - TS[:3, 3]) * 1000), 3),
                       end_moved_mm=round(float(np.linalg.norm(Tc[idx[-1]][:3, 3] - Tb[idx[-1]][:3, 3]) * 1000), 3))
for f in ("ORIGIN_FRAME.json",):
    if (SRC / f).exists(): shutil.copy(SRC / f, OUT / f)
json.dump(dict(excluded=sorted(bad), episodes=rep), open(OUT / "ROBOTIZED_C.json", "w"), indent=1)
sh = np.array([r["start_shift_mm"] for r in rep.values()]); ro = np.array([r["start_rot_deg"] for r in rep.values()])
print(f"episodes {len(rep)} | start shift mm p10/50/90 {np.percentile(sh, [10, 50, 90]).round(0)} | start rotation deg p10/50/90 {np.percentile(ro, [10, 50, 90]).round(1)} | "
      f"start err max {max(r['start_err_mm'] for r in rep.values())} mm | end moved max {max(r['end_moved_mm'] for r in rep.values())} mm")
