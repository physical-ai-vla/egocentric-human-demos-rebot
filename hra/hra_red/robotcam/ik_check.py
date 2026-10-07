"""[2026-10-06] offline reachability QC of the robotcam labels (CHECK ONLY, labels untouched): every episode's ROBOT-TCP trajectory
re-anchored at pose A (T_A @ inv(T_0) @ T_t), Pink IK solved in time order seeded with the previous q (first seed = A joints).
Per episode: pos/rot residual, min joint-limit margin, min TCP height above the table (base z -54.3 mm), frames unreachable
(pos > 5 mm or rot > 3 deg or margin < 2 deg). Output ik_check.json + a summary."""
import json, sys, pathlib, numpy as np
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); sys.path.insert(0, str(H / "holobrain-mac-model")); sys.path.insert(0, str(H / "ego_cart20"))
import eef_kin, pink_ik
from ego_cart20.geometry.transforms import pose_to_T
kin = eef_kin.Kin({"max_joint_delta": None}); names = [kin._names[j] for j in kin._arm["left"]] + [kin._names[j] for j in kin._arm["right"]]
P = pink_ik.PinkIK(names); lo, hi = P.model.lowerPositionLimit, P.model.upperPositionLimit
S = json.load(open(H / "c8/hra_red/umi_start_pose.json")); TA = np.array(S["T_base_tcp"]); qA = np.zeros(12); qA[6:] = np.radians(S["right_joints_deg"])
TABLE_Z = -0.0543; R = H / "c8/hra_red/raw_robotcam"; out = []
eps = [json.loads(l)["episode_id"] for s in ("train", "val") for l in open(H / "c8/hra_red/processed_robotcam" / f"{s}_manifest.jsonl")]
for ep in eps:
    z = np.load(R / ep / "raw_episode.npz"); v = z["right_valid"].astype(bool); T = pose_to_T(z["right_position"], z["right_quaternion"])[v][::3]
    tgt = TA @ np.linalg.inv(T[0]) @ T; q = qA.copy(); L = P.fk(qA)[0]; pe, re, mg, hz, bad = [], [], [], [], 0
    for Tt in tgt:
        q, _, _ = P.solve(q, [L, Tt], iters=60); f = P.fk(q)[1]
        e_p = np.linalg.norm(f[:3, 3] - Tt[:3, 3]) * 1000; e_r = np.degrees(np.linalg.norm(Rot.from_matrix(f[:3, :3].T @ Tt[:3, :3]).as_rotvec()))
        m = float(np.degrees(np.minimum(q - lo, hi - q))[6:].min()); pe.append(e_p); re.append(e_r); mg.append(m); hz.append((Tt[2, 3] - TABLE_Z) * 1000)
        bad += (e_p > 5 or e_r > 3 or m < 2)
    out.append(dict(ep=ep, n=len(tgt), unreachable_frac=round(bad / len(tgt), 3), pos_err_p95_mm=round(float(np.percentile(pe, 95)), 1),
                    rot_err_p95_deg=round(float(np.percentile(re, 95)), 2), min_margin_deg=round(min(mg), 1),
                    min_target_above_table_mm=round(min(hz), 1), below_table_frac=round(float(np.mean(np.array(hz) < 0)), 3)))
json.dump(out, open(H / "c8/hra_red/robotcam/ik_check.json", "w"), indent=1)
u = np.array([o["unreachable_frac"] for o in out]); bt = np.array([o["min_target_above_table_mm"] for o in out])
print(f"episodes {len(out)}  unreachable frames: mean {u.mean()*100:.1f}%  eps with any {np.mean(u>0)*100:.0f}%  eps >20% {np.sum(u>0.2)}")
print("pos err p95 (median over eps) mm", np.median([o['pos_err_p95_mm'] for o in out]), " rot p95 deg", np.median([o['rot_err_p95_deg'] for o in out]))
print("min target height above table mm: p10/50/90", np.percentile(bt, [10, 50, 90]).round(0), " eps whose target goes below the table:", int(np.sum(bt < 0)))
print("min joint margin deg p10/50", np.percentile([o['min_margin_deg'] for o in out], [10, 50]).round(1))
