"""[2026-09-28] Pre-training audit of r380_umi94_rel16_v2B (B; A differs only by state[:76]) against the raw teleop.
Check only -- nothing is modified. Items (user 2026-09-28):
 1 source mapping: 380 episodes 1:1 to source teleop episodes, no duplicates, task string == source task, order balance
 2 REL16 arm label vs recorded teleop: T_now = FK(q_t)+frame fix, T_now @ A_k vs FK(q_{t+m}) (k = 2/4/8/16), per pool
 3 binary gripper: independent recompute from the leader, OPEN->CLOSE / CLOSE->OPEN onset vs follower jaw response
   (lag in source frames, per pool), hold consistency
 4 state94: TCP18 vs FK(q_t)+frame fix per pool/arm; HEAD vs FRONT TCP distribution
 5 order/layout shortcut: cube pick positions from close events -> can the order be told from the colored layout
   alone, and is the first pick predictable from the uncolored layout by a fixed spatial rule
 6 gripper loss contribution: stats, frames whose chunk contains a gripper transition
"""
import glob, json, os, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model")); sys.path.insert(0, os.path.expanduser("~/umi_bridge/umi76"))
os.environ.setdefault("V4_STATE_MODE", "umi94")
import infer_core_v4 as IC
from umi.common.pose_util import pose10d_to_mat
from scipy.spatial.transform import Rotation as Rot
import derive_v2 as DV

L = os.path.expanduser("~/holobrain-data/lerobot"); N = "/home/bh-aiteam/holobrain-data/lerobot"
DS = f"{L}/r380_umi94_rel16_v2B"
REMAP = {f"{N}/rebot_3stack_R150_headview": f"{L}/src_rebot_3stack_R150_headview",
         f"{N}/rebot_3stack_R30_day4_headview": f"{L}/src_rebot_3stack_R30_day4_headview",
         f"{N}/rebot_3stack_center675_s96": f"{L}/src_rebot_3stack_center675_s96"}
DV.REMAP.update(REMAP)
rng = np.random.default_rng(0)
OUT = {}


def col(t, c):
    a = t.column(c).combine_chunks(); return a.storage if isinstance(a, pa.ExtensionArray) else a


tabs = [pq.read_table(f) for f in sorted(glob.glob(f"{DS}/data/chunk-*/*.parquet"))]
D = {c: np.concatenate([np.asarray(col(t, c).to_pylist()) for t in tabs]) for c in
     ("observation.state", "action", "episode_index", "frame_index", "task_index")}
S, A, EP, FR, TK = (D[c] for c in ("observation.state", "action", "episode_index", "frame_index", "task_index"))
tasks = pd.read_parquet(f"{DS}/meta/tasks.parquet"); TS = {int(v): k for k, v in tasks["task_index"].items()}
src_of = DV.row_sources(DS, os.path.expanduser("~/umi_bridge/rel16_audit/r380/r675_rbp.json"))
pool = {e: ("HEAD" if "headview" in r else "FRONT") for e, (r, *_ ) in src_of.items()}


def load_src(root):
    root = REMAP.get(root, root)
    fs = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    t = [pq.read_table(f, columns=["observation.state", "action", "episode_index", "frame_index", "task_index"]) for f in fs]
    d = {c: np.concatenate([np.asarray(col(x, c).to_pylist()) for x in t]) for c in ("observation.state", "action", "episode_index", "frame_index", "task_index")}
    ts = pd.read_parquet(f"{root}/meta/tasks.parquet"); tsm = {int(v): k for k, v in ts["task_index"].items()}
    by = {}
    for e in np.unique(d["episode_index"]):
        m = np.flatnonzero(d["episode_index"] == e); m = m[np.argsort(d["frame_index"][m])]
        by[int(e)] = dict(q=d["observation.state"][m].astype(np.float64), lead=d["action"][m].astype(np.float64),
                          task=tsm[int(d["task_index"][m[0]])])
    return by


SRC = {r: load_src(r) for r in sorted({v[0] for v in src_of.values()})}
print("loaded", {os.path.basename(k): len(v) for k, v in SRC.items()}, flush=True)

# ---------------- 1 mapping
keys = [(v[0], v[1]) for v in src_of.values()]
dup = len(keys) - len(set(keys))
bad_task = [e for e in src_of if TS[int(TK[EP == e][0])] != SRC[src_of[e][0]][src_of[e][1]]["task"]]
len_bad = [e for e in src_of if (FR[EP == e].max() + 1) != len(np.arange(0, len(SRC[src_of[e][0]][src_of[e][1]]["q"]), 2)[
    (np.arange(0, len(SRC[src_of[e][0]][src_of[e][1]]["q"]), 2) / 30 - 0.05005 >= -1e-9) &
    (np.arange(0, len(SRC[src_of[e][0]][src_of[e][1]]["q"]), 2) / 30 + 16 * 0.05005 <= (len(SRC[src_of[e][0]][src_of[e][1]]["q"]) - 1) / 30 + 1e-9)])]
cnt = pd.crosstab(pd.Series([pool[e] for e in src_of]), pd.Series([TS[int(TK[EP == e][0])].split(",")[0][10:].replace(" cube on the bottom", "") for e in src_of]))
print(f"\n[1] episodes {len(src_of)}  duplicate source eps {dup}  task mismatches {len(bad_task)}  length mismatches {len(len_bad)}")
print("    bottom-cube counts per pool:\n" + cnt.to_string())
OUT["1"] = dict(n=len(src_of), dup=dup, task_mismatch=len(bad_task), len_mismatch=len(len_bad))


class K:
    kin = IC.eef_kin.Kin(); _tcp_mat = IC.V4Inferencer._tcp_mat


kk = K()


def j14(qdeg):
    q = np.asarray(qdeg, np.float64).copy(); q[IC.ARM_IDX] = np.radians(q[IC.ARM_IDX]); return q


def err(a, b):
    d = np.linalg.inv(a) @ b
    return np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000, np.degrees(np.linalg.norm(Rot.from_matrix(d[:3, :3]).as_rotvec()))


# ---------------- 2 + 4 geometry, per pool: 60 random + 40 large-motion rows
M = {1: 3, 3: 6, 7: 12, 15: 24}
g16 = np.linalg.norm(A[:, 15, :3], axis=1) + np.linalg.norm(A[:, 15, 10:13], axis=1)
res2 = {p: {k: [] for k in M} for p in ("HEAD", "FRONT")}; res4 = {p: [] for p in ("HEAD", "FRONT")}
for p in ("HEAD", "FRONT"):
    rows = np.flatnonzero(np.array([pool[e] for e in EP]) == p)
    pick = np.r_[rng.choice(rows, 60, replace=False), rows[np.argsort(-g16[rows])[:400]][rng.choice(400, 40, replace=False)]]
    for i in pick:
        s = SRC[src_of[int(EP[i])][0]][src_of[int(EP[i])][1]]; i0 = 2 * int(FR[i]) + 2
        Tn, _ = kk._tcp_mat(j14(s["q"][i0]))
        for r in (0, 1):
            T18 = pose10d_to_mat(S[i, 76 + 9 * r:85 + 9 * r][None].astype(np.float64))[0]
            res4[p].append(err(T18, Tn[r]))
        for k, m in M.items():
            if i0 + m >= len(s["q"]):
                continue
            Tf, _ = kk._tcp_mat(j14(s["q"][i0 + m]))
            for r in (0, 1):
                res2[p][k].append(err(Tn[r] @ pose10d_to_mat(A[i, k, r * 10:r * 10 + 9][None].astype(np.float64))[0], Tf[r]))
print("\n[2] T_now(FK) @ A_k vs FK(q_{t+m}) recorded     [4] TCP18 vs FK(q_t)+frame fix")
for p in ("HEAD", "FRONT"):
    for k in M:
        a = np.array(res2[p][k]); print(f"    {p:5s} k={k+1:2d}: pos med {np.median(a[:,0]):.3f} p99 {np.percentile(a[:,0],99):.3f} max {a[:,0].max():.3f} mm | rot max {a[:,1].max():.3f} deg")
    b = np.array(res4[p]); print(f"    {p:5s} TCP18: pos max {b[:,0].max():.4f} mm, rot max {b[:,1].max():.4f} deg")
OUT["2"] = {p: {k + 1: float(np.max(np.array(res2[p][k])[:, 0])) for k in M} for p in res2}
OUT["4"] = {p: float(np.max(np.array(res4[p])[:, 0])) for p in res4}
for p in ("HEAD", "FRONT"):
    m = np.array([pool[e] for e in EP]) == p
    print(f"    {p:5s} TCP18 pos mean L {S[m,76:79].mean(0).round(3)} R {S[m,85:88].mean(0).round(3)}  std L {S[m,76:79].std(0).round(3)}")

# ---------------- 3 gripper
rec_ok, lag = [], {p: {"close": [], "open": []} for p in ("HEAD", "FRONT")}
for e in src_of:
    s = SRC[src_of[e][0]][src_of[e][1]]; m = np.flatnonzero(EP == e); m = m[np.argsort(FR[m])]
    i0 = 2 * FR[m] + 2
    for r, gi in ((0, 6), (1, 13)):
        v = np.interp(i0 + 1.5015, np.arange(len(s["lead"])), s["lead"][:, gi])
        rec_ok.append(np.mean((v >= 27).astype(float) == A[m, 0, r * 10 + 9]))
        lab = A[m, 0, r * 10 + 9]; w = S[m, 37 if r == 0 else 75]
        for j in np.flatnonzero(np.diff(lab) != 0):
            kind = "close" if lab[j] == 1 else "open"
            # follower response: first row after the switch where the width moved >= 5 mm in the commanded direction
            w0 = w[j + 1]; fut = w[j + 1:j + 31] - w0
            hit = np.flatnonzero(fut <= -0.005) if kind == "close" else np.flatnonzero(fut >= 0.005)
            if len(hit):
                lag[pool[e]][kind].append(hit[0])
print(f"\n[3] label == independent leader recompute: min {min(rec_ok)*100:.3f} % of rows per episode-arm")
for p in lag:
    for kind in ("close", "open"):
        a = np.array(lag[p][kind]) * (1 / 15)
        print(f"    {p:5s} {kind:5s} n {len(a):4d}  follower responds after label switch: med {np.median(a)*1000:4.0f} ms  p10 {np.percentile(a,10)*1000:4.0f}  p90 {np.percentile(a,90)*1000:4.0f}")
hold = (S[:, 37] > 0.02) & (S[:, 37] < 0.06) & (np.abs(S[:, 37] - S[:, 36]) < 0.0005)
print(f"    static jaw at cube width (20-60 mm): labelled CLOSE {100*np.mean(A[hold,0,9]==0):.1f} % (L)")
OUT["3"] = dict(recompute_min=float(min(rec_ok)), lag={p: {k: float(np.median(v)) / 15 for k, v in lag[p].items()} for p in lag})

# ---------------- 5 layout shortcut: pick = CLOSE onset whose jaw stalls at 25-95 mm within 1 s, position = TCP18
COL = {"red": 0, "blue": 1, "purple": 2}
lay = []
for e in src_of:
    m = np.flatnonzero(EP == e); m = m[np.argsort(FR[m])]
    t = TS[int(TK[m[0]])]; import re
    order = re.findall(r"(\w+) cube", t)
    picks = []
    for r in (0, 1):
        lab = A[m, 0, r * 10 + 9]; w = S[m, 37 if r == 0 else 75]
        for j in np.flatnonzero((lab[:-1] == 1) & (lab[1:] == 0)):
            ws = w[min(j + 15, len(w) - 1)]
            if 0.025 < ws < 0.095:
                picks.append((j, r, S[m[min(j + 15, len(m) - 1)], 76 + 9 * r:79 + 9 * r]))
    picks.sort(key=lambda x: x[0])
    if len(picks) >= 3 and len(order) == 3:
        lay.append(dict(e=e, pool=pool[e], order=tuple(order), pos=np.stack([p[2][:2] for p in picks[:3]]),
                        arms=[p[1] for p in picks[:3]], npick=len(picks)))
print(f"\n[5] episodes with >= 3 detected picks: {len(lay)}/{len(src_of)} (exactly 3: {sum(l['npick']==3 for l in lay)})")
# colored layout: position of red / blue / purple (pick i is the order's i-th color)
X, Y, F = [], [], []
for l in lay:
    cp = np.zeros((3, 2))
    for i, c in enumerate(l["order"]):
        cp[COL[c]] = l["pos"][i]
    X.append(cp.ravel()); Y.append("".join(c[0].upper() for c in l["order"])); F.append(l["pos"])
X = np.array(X); Y = np.array(Y)
Z = (X - X.mean(0)) / X.std(0)
d = np.linalg.norm(Z[:, None] - Z[None], axis=-1); np.fill_diagonal(d, np.inf)
knn = np.mean(Y[np.argmin(d, 1)] == Y)
maj = pd.Series(Y).value_counts(normalize=True).iloc[0]
print(f"    order from COLORED layout alone, 1-NN leave-one-out: {100*knn:.1f} %  (majority {100*maj:.1f} %, chance {100/6:.1f} %)")
EE = np.array([l["e"] for l in lay]); d2 = d.copy(); d2[np.abs(EE[:, None] - EE[None]) <= 10] = np.inf
knn2 = np.mean(Y[np.argmin(d2, 1)] == Y)
print(f"    same, excluding neighbours within +-10 episodes (same-session near-duplicates): {100*knn2:.1f} %")
for pl in ("HEAD", "FRONT"):
    mm = np.array([l["pool"] == pl for l in lay]); dd = d2[np.ix_(mm, mm)]
    print(f"      {pl}: {100*np.mean(Y[mm][np.argmin(dd, 1)] == Y[mm]):.1f} % (n {mm.sum()})")
OUT_KNN2 = float(knn2)
for c, ci in COL.items():
    print(f"    {c:6s} cube position mean {X[:, 2*ci:2*ci+2].mean(0).round(3)} std {X[:, 2*ci:2*ci+2].std(0).round(3)} m")
rules = {}
for nm, key in (("max y (image-left arm side)", lambda p: -p[:, 1]), ("min y", lambda p: p[:, 1]),
                ("min x (nearest base)", lambda p: p[:, 0]), ("max x", lambda p: -p[:, 0])):
    rules[nm] = np.mean([np.argmin(key(p)) == 0 for p in F])
for nm, v in rules.items():
    print(f"    first pick predicted by uncolored rule '{nm}': {100*v:.1f} %  (chance 33.3 %)")
pairs = sum(1 for i in range(len(X)) for j in range(i + 1, len(X)) if np.abs(X[i] - X[j]).max() < 0.04 and Y[i] != Y[j])
print(f"    near-identical colored layouts (all cubes within 4 cm) with DIFFERENT orders: {pairs} pairs")
OUT["5"] = dict(n=len(lay), knn=float(knn), knn_excl_session=OUT_KNN2, majority=float(maj), rules={k: float(v) for k, v in rules.items()}, same_layout_diff_order=pairs)

# ---------------- 6 loss contribution
st = json.load(open(f"{DS}/meta/stats.json"))["action"]
g = A[:, :, [9, 19]]
trans = (np.abs(np.diff(g, axis=1)).sum(axis=(1, 2)) > 0).mean()
print(f"\n[6] gripper mean/std L {st['mean'][9]:.3f}/{st['std'][9]:.3f} R {st['mean'][19]:.3f}/{st['std'][19]:.3f}; "
      f"frames whose 16-step chunk contains a gripper switch: {100*trans:.1f} %")
OUT["6"] = dict(transition_frames=float(trans))
json.dump(OUT, open(os.path.expanduser("~/umi_bridge/rel16_audit/r380/r380_train_audit.json"), "w"), indent=1)
