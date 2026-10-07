"""[2026-10-07] offline reachability QC of the 93 HRA_A100 training episodes (CHECK ONLY): the trained part (after the AUTO go) of every
robot-TCP label trajectory, replayed two ways, sequential Pink IK seeded with the previous q (first seed = IK of the start):
  A  relative: T_S @ inv(T_go) @ T(t), T_S = the robot start pose (umi_start_pose.json = HRA_A100 human-median start)
  B  absolute: F @ T_F(t), F = B_anchor_F.json (what the origin-anchored model would reproduce in the robot base)
Per episode: frames unreachable (pos > 5 mm or rot > 3 deg or joint margin < 2 deg), min joint margin, min PHYSICAL jaw-tip height
relative to the X plate surface (z 29.1 mm; the table is lower by the plate thickness)."""
import json, pathlib, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); sys.path[:0] = [str(H / "holobrain-mac-model"), str(H / "ego_cart20")]
import eef_kin, pink_ik
from ego_cart20.geometry.transforms import pose_to_T
A = H / "c8/hra_a100"
kin = eef_kin.Kin({"max_joint_delta": None}); names = [kin._names[j] for j in kin._arm["left"]] + [kin._names[j] for j in kin._arm["right"]]
P = pink_ik.PinkIK(names); lo, hi = P.model.lowerPositionLimit, P.model.upperPositionLimit
TS = np.array(json.load(open(H / "c8/hra_red/umi_start_pose.json"))["T_base_tcp"]); F = np.array(json.load(open(A / "B_anchor_F.json"))["T_base_F"])
PLATE_Z = 0.0291; TIP = np.array([-0.082, 0, 0, 1.0])
ids = sorted(p.name for p in (A / "processed_v2_origin/episodes").iterdir())
q_start, _, _ = P.solve(np.zeros(12), [P.fk(np.zeros(12))[0], TS], iters=600); L = P.fk(q_start)[0]
out = {}
for eid in ids:
    zr = np.load(A / "raw_v2_robotcam" / eid / "raw_episode.npz"); zo = np.load(A / "raw_v2_origin" / eid / "raw_episode.npz")
    v = zr["right_valid"].astype(bool); Tr = pose_to_T(zr["right_position"], zr["right_quaternion"])[v][::2]; To = pose_to_T(zo["right_position"], zo["right_quaternion"])[v][::2]
    res = {}
    for mode, tgt in (("A", TS @ np.linalg.inv(Tr[0]) @ Tr), ("B", F[None] @ To)):
        q, _, _ = P.solve(q_start, [L, tgt[0]], iters=400); bad = 0; mg = []; tip = []; pe_l = []
        for Tt in tgt:
            q, _, _ = P.solve(q, [L, Tt], iters=60); f = P.fk(q)[1]
            pe = np.linalg.norm(f[:3, 3] - Tt[:3, 3]) * 1000; re = np.degrees(np.linalg.norm(Rot.from_matrix(f[:3, :3].T @ Tt[:3, :3]).as_rotvec()))
            m = float(np.degrees(np.minimum(q - lo, hi - q))[6:].min()); mg.append(m); pe_l.append(pe); tip.append(((Tt @ TIP)[2] - PLATE_Z) * 1000)
            bad += (pe > 5 or re > 3 or m < 2)
        res[mode] = dict(unreach=round(bad / len(tgt), 3), min_margin=round(min(mg), 1), min_tip_above_plate_mm=round(min(tip), 1), pos_err_p95=round(float(np.percentile(pe_l, 95)), 1))
    out[eid] = res
json.dump(out, open(A / "ik_check_a93.json", "w"), indent=1)
for mode in ("A", "B"):
    u = np.array([r[mode]["unreach"] for r in out.values()]); tip = np.array([r[mode]["min_tip_above_plate_mm"] for r in out.values()]); mg = np.array([r[mode]["min_margin"] for r in out.values()])
    print(f"[{mode}] episodes {len(u)} | unreachable frames mean {u.mean()*100:.1f}%  eps with any {np.mean(u > 0)*100:.0f}%  eps >20% {int(np.sum(u > 0.2))} | "
          f"min joint margin p10/50 {np.percentile(mg, [10, 50]).round(1)} | min jaw-tip height vs plate mm p10/50/90 {np.percentile(tip, [10, 50, 90]).round(0)}  eps below plate {int(np.sum(tip < 0))}")
