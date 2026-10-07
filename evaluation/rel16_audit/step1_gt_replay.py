"""[2026-09-28] REL16 audit step 1 -- GT through the deployment command path, no model.

For a recorded frame i with joints q_i (source R150/R30 headview, 30 Hz, degrees):
    T_now  = infer_core_v4._tcp_mat(q_i)                       FK + V4_FRAME_FIX, dataset frame
    target = T_now @ A_k(GT from r180_umi76_rel16_v1)           what the deploy loop asks for
    q_cmd  = solve_waypoint path (frame fix -> IK -> DQ_MAX clip -> LO/HI)
    got    = _tcp_mat(q_cmd)
and compares `got` against the independent reference FK(q_{i+m}) of the RECORDED future joints, m = 1.5015*(k+1)
source frames (k+1 = 2,4,8,16 -> m = 3,6,12,24, off by <= 0.024 frames). Also:
    label check   A_k vs inv(tcp_state_i) tcp_state_{i+m}        is the r180 label the recorded motion?
    FK check      _tcp_mat(q_i) vs tcp_state_i                   is the deployed T_now the training T_t?
    infer() path  solve_chunk on dataset-frame quats (no frame fix), as infer() computes valid_step
r180 row f of an episode = source frame 2f+2 (converter: q_local = arange(0,n,2), first kept t >= 50.05 ms).
"""
import glob, json, math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
os.environ.setdefault("V4_STATE_MODE", "umi76"); os.environ.setdefault("V4_ACTION_MODE", "umi")
import infer_core_v4 as IC
from scipy.spatial.transform import Rotation as Rot
from umi.common.pose_util import pose10d_to_mat

LR = os.path.expanduser("~/holobrain-data/lerobot")
OUT = os.path.expanduser("~/umi_bridge/rel16_audit/step1_gt_replay.json")
N = int(os.environ.get("N", "50")); SEED = int(os.environ.get("SEED", "0"))
KS = [1, 3, 7, 15]                                    # 0-based k; exec_k=8 -> index 7
M = {1: 3, 3: 6, 7: 12, 15: 24}


def load(p):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(p, recursive=True))], ignore_index=True)


r180 = load(f"{LR}/r180_umi76_rel16_v1/data/chunk-000/*.parquet")
src = {0: load(f"{LR}/src_rebot_3stack_R150_headview/data/**/*.parquet"),
       1: load(f"{LR}/src_rebot_3stack_R30_day4_headview/data/**/*.parquet")}
src_idx = {p: {e: g.sort_values("frame_index") for e, g in d.groupby("episode_index")} for p, d in src.items()}


def src_ep(e):
    return src_idx[0][e] if e < 150 else src_idx[1][e - 150]


def joints14(qdeg):
    q = np.asarray(qdeg, np.float64).copy()
    q[IC.ARM_IDX] = np.radians(q[IC.ARM_IDX])             # arms deg -> rad; gripper stays a raw count
    return q


def tcp16_mat(v):
    out = np.zeros((2, 4, 4))
    for r in (0, 1):
        o = r * 8
        out[r][:3, :3] = Rot.from_quat(v[o + 3:o + 7]).as_matrix(); out[r][:3, 3] = v[o:o + 3]; out[r][3, 3] = 1
    return out


def err(a, b):
    d = np.linalg.inv(a) @ b
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000), float(np.degrees(np.linalg.norm(Rot.from_matrix(d[:3, :3]).as_rotvec())))


class K:          # the pieces of V4Inferencer that need no model
    kin = IC.eef_kin.Kin()
    _tcp_mat = IC.V4Inferencer._tcp_mat


k_ = K()
rng = np.random.default_rng(SEED)
rows = r180.sample(n=N, random_state=SEED).sort_values(["episode_index", "frame_index"])
res = []
for _, row in rows.iterrows():
    e, f = int(row.episode_index), int(row.frame_index)
    s = src_ep(e); i = 2 * f + 2
    if i + 24 >= len(s):
        continue
    q = [np.asarray(s["observation.state"].iloc[j]) for j in range(i, i + 25)]
    tcp = [tcp16_mat(np.asarray(s["observation.ee.tcp_state"].iloc[j])) for j in range(i, i + 25)]
    A = np.asarray(np.stack(row.action), np.float64).reshape(16, 20)
    T_now, _ = k_._tcp_mat(joints14(q[0]))
    rec = {"ep": e, "f": f, "src_i": i, "task": int(row.task_index)}
    rec["fk_vs_tcp_state"] = [err(T_now[r], tcp[0][r]) for r in (0, 1)]
    for kk in KS:
        m = M[kk]
        T_fut, _ = k_._tcp_mat(joints14(q[m]))
        tgt = np.stack([T_now[r] @ pose10d_to_mat(A[kk, r * 10:r * 10 + 9][None])[0] for r in (0, 1)])
        lab = [err(np.linalg.inv(tcp[0][r]) @ tcp[m][r], pose10d_to_mat(A[kk, r * 10:r * 10 + 9][None])[0]) for r in (0, 1)]
        pos = np.stack([tgt[r][:3, 3] for r in (0, 1)]); quat = np.stack([Rot.from_matrix(tgt[r][:3, :3]).as_quat() for r in (0, 1)])
        # deployment path: solve_waypoint (frame fix inside), seeded from the measured joints
        cmd, ikerr, ok, clipped = IC.V4Inferencer.solve_waypoint(k_, joints14(q[0]), pos, quat)
        cmd = np.asarray(cmd); cmd[IC.FLIP_IDX] *= -1.0              # leader -> follower frame for FK
        got, _ = k_._tcp_mat(cmd)
        # infer() path: solve_chunk on the dataset-frame quaternions, which is what sets valid_step
        _, fk_nofix, ok_nofix = k_.kin.solve_chunk(joints14(q[0])[IC.ARM_IDX], pos[None], quat[None])
        rec[f"k{kk+1}"] = {
            "gt_motion_mm": [round(float(np.linalg.norm(T_fut[r][:3, 3] - T_now[r][:3, 3]) * 1000), 2) for r in (0, 1)],
            "label_vs_recorded": lab,
            "target_vs_recFK": [err(tgt[r], T_fut[r]) for r in (0, 1)],
            "deployed_vs_recFK": [err(got[r], T_fut[r]) for r in (0, 1)],
            "deployed_vs_target": [err(got[r], tgt[r]) for r in (0, 1)],
            "ik_fk_err_mm": ikerr, "ik_ok": ok, "dq_clipped": clipped,
            "infer_path_fkerr_mm": [float(x) for x in fk_nofix[0]], "infer_path_valid": bool(ok_nofix.all() and fk_nofix.max() <= IC.FK_MAX_MM)}
    res.append(rec)
    print(e, f, {kk: [round(x[0], 1) for x in rec[f'k{kk}']['deployed_vs_recFK']] for kk in (2, 8, 16)}, flush=True)

json.dump(res, open(OUT, "w"), indent=1)


def summ(key, idx):
    a = np.array([[x[key][r][idx] for r in (0, 1)] for x in res]).ravel()
    return f"med {np.median(a):.2f} p90 {np.percentile(a, 90):.2f} max {a.max():.2f}"


print(f"\nn={len(res)}  FRAME_FIX={IC.USE_FRAME_FIX}")
fk = np.array([[x["fk_vs_tcp_state"][r] for r in (0, 1)] for x in res]).reshape(-1, 2)
print(f"FK(q_i) vs tcp_state_i: pos med {np.median(fk[:,0]):.2f} max {fk[:,0].max():.2f} mm | rot med {np.median(fk[:,1]):.2f} max {fk[:,1].max():.2f} deg")
for kk in KS:
    sub = [x[f"k{kk+1}"] for x in res]
    for key in ("label_vs_recorded", "target_vs_recFK", "deployed_vs_target", "deployed_vs_recFK"):
        a = np.array([[s[key][r] for r in (0, 1)] for s in sub]).reshape(-1, 2)
        print(f"k={kk+1:2d} {key:20s} pos med {np.median(a[:,0]):6.2f} p90 {np.percentile(a[:,0],90):6.2f} max {a[:,0].max():6.2f} mm | "
              f"rot med {np.median(a[:,1]):5.2f} p90 {np.percentile(a[:,1],90):5.2f} max {a[:,1].max():5.2f} deg")
    gm = np.array([s["gt_motion_mm"] for s in sub]).ravel()
    inv = np.array([s["infer_path_fkerr_mm"] for s in sub]).ravel()
    print(f"      gt motion med {np.median(gm):.1f} mm | infer()-path fkerr med {np.median(inv):.1f} max {inv.max():.1f} mm, "
          f"invalid {sum(not s['infer_path_valid'] for s in sub)}/{len(sub)} | dq_clipped {sum(s['dq_clipped']>0 for s in sub)}")
