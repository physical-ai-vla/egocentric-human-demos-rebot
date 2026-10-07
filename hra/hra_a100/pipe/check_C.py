"""[2026-10-07] post-warp checks of C vs the unwarped B trajectories (both in the robot base, trained segment, every 2nd row for IK):
max frame-to-frame translation / rotation jump, Pink IK residual + joint margin replaying C from the robot start, physical jaw-tip clearance
above the table (table z = X plate 29.1 mm - 35 mm), endpoint error."""
import json, pathlib, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
H = pathlib.Path.home(); sys.path[:0] = [str(H / "holobrain-mac-model"), str(H / "ego_cart20")]
import eef_kin, pink_ik
from ego_cart20.geometry.transforms import pose_to_T
A = H / "c8/hra_a100"; F = np.array(json.load(open(A / "B_anchor_F.json"))["T_base_F"]); TS = np.array(json.load(open(H / "c8/hra_red/umi_start_pose.json"))["T_base_tcp"])
TABLE = 0.0291 - 0.035; TIP = np.array([-0.082, 0, 0, 1.0])
kin = eef_kin.Kin({"max_joint_delta": None}); names = [kin._names[j] for j in kin._arm["left"]] + [kin._names[j] for j in kin._arm["right"]]
P = pink_ik.PinkIK(names); lo, hi = P.model.lowerPositionLimit, P.model.upperPositionLimit
q0, _, _ = P.solve(np.zeros(12), [P.fk(np.zeros(12))[0], TS], iters=600); L = P.fk(q0)[0]
def jumps(T):
    dp = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) * 1000
    dr = np.degrees([np.linalg.norm(Rot.from_matrix(T[i, :3, :3].T @ T[i + 1, :3, :3]).as_rotvec()) for i in range(len(T) - 1)])
    return float(dp.max()), float(np.max(dr))
rows = {}
for d in sorted(p for p in (A / "raw_v2_C").iterdir() if (p / "raw_episode.npz").exists()):
    zc = np.load(d / "raw_episode.npz"); zb = np.load(A / "raw_v2_origin" / d.name / "raw_episode.npz"); v = zc["right_valid"].astype(bool)
    Tc = F[None] @ pose_to_T(zc["right_position"], zc["right_quaternion"])[v]; Tb = F[None] @ pose_to_T(zb["right_position"], zb["right_quaternion"])[v]
    jc, jb = jumps(Tc), jumps(Tb)
    q = q0.copy(); pe = []; mg = []
    for Tt in Tc[::2]:
        q, _, _ = P.solve(q, [L, Tt], iters=60); f = P.fk(q)[1]; pe.append(np.linalg.norm(f[:3, 3] - Tt[:3, 3]) * 1000); mg.append(np.degrees(np.minimum(q - lo, hi - q))[6:].min())
    tip = min(((Tt @ TIP)[2] - TABLE) * 1000 for Tt in Tc)
    rows[d.name] = dict(jump_mm_C=round(jc[0], 2), jump_mm_B=round(jb[0], 2), jump_deg_C=round(jc[1], 2), jump_deg_B=round(jb[1], 2), ik_res_p95_mm=round(float(np.percentile(pe, 95)), 2),
                        min_margin_deg=round(float(min(mg)), 1), min_tip_above_table_mm=round(float(tip), 1),
                        start_err_mm=round(float(np.linalg.norm(Tc[0, :3, 3] - TS[:3, 3]) * 1000), 3), end_err_mm=round(float(np.linalg.norm(Tc[-1, :3, 3] - Tb[-1, :3, 3]) * 1000), 3))
json.dump(rows, open(A / "check_C.json", "w"), indent=1)
g = lambda k: np.array([r[k] for r in rows.values()])
print(f"episodes {len(rows)}")
print(f"max frame jump mm  C p50/max {np.median(g('jump_mm_C')):.1f}/{g('jump_mm_C').max():.1f}   (B {np.median(g('jump_mm_B')):.1f}/{g('jump_mm_B').max():.1f})")
print(f"max frame jump deg C p50/max {np.median(g('jump_deg_C')):.2f}/{g('jump_deg_C').max():.2f}   (B {np.median(g('jump_deg_B')):.2f}/{g('jump_deg_B').max():.2f})")
print(f"IK residual p95 mm max {g('ik_res_p95_mm').max():.2f} | min joint margin deg p10/min {np.percentile(g('min_margin_deg'), 10):.1f}/{g('min_margin_deg').min():.1f}")
t = g('min_tip_above_table_mm'); print(f"min jaw-tip height above table mm p10/50/90 {np.percentile(t, [10, 50, 90]).round(0)}  below table {int((t < 0).sum())}  below -10 {int((t < -10).sum())}")
print(f"start err max {g('start_err_mm').max()} mm | endpoint err max {g('end_err_mm').max()} mm")
