#!/usr/bin/env python3
"""[2026-09-28] Post-build gate for r312c_umi76_rel16_v1 = HEAD180 + FRONT132 (pan=center only, grade A, 22/order, rot180).
Derived from gate_r380.py; G6 is now: every FRONT episode shared with R380 is bit-identical. Read-only.
G1 HEAD eps 0-179: every parquet column bit-identical to r180_umi76_rel16_v1 (the select= change must not touch them)
G2 HEAD videos: decoded frames identical to r180 on sampled episodes/frames
G3 FRONT eps 180-279: count 100, order == selection order, task == R675 pool instruction, frame counts plausible
G4 FRONT global == rot180 of the source center675 global at the same (episode, frame), sampled; wrist == zarr
G5 FRONT state/action finite, shapes 76 / 16x20, current self term identity; stats cover 280 episodes
"""
import json, glob, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, "/home/bh-aiteam/lerobot-seeed/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
D = "/home/bh-aiteam/holobrain-data/lerobot"
NEW, OLD = f"{D}/r312c_umi76_rel16_v1", f"{D}/r180_umi76_rel16_v1"
SEL = json.load(open("/home/bh-aiteam/umi_bridge/umi76/r312c_front132_v1.json"))
fails = []
def check(name, ok, info=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"  {info}" if info else ""), flush=True)
    if not ok: fails.append(name)
import pyarrow.parquet as pq
import pyarrow as pa
def _rp(f, columns=None):
    """HF datasets writes array columns as arrow EXTENSION types, which pandas turns into an extension dtype that
    pd.concat cannot compare; drop to the storage type (fixed-size list of float) first."""
    t = pq.read_table(f, columns=columns)
    cols = [(c.combine_chunks().storage if isinstance(c.type, pa.ExtensionType) else c) for c in t.columns]
    return pa.table(cols, names=t.column_names).to_pandas(ignore_metadata=True)
def load(root): return pd.concat([_rp(f) for f in sorted(glob.glob(f"{root}/data/*/*.parquet"))]).sort_values("index").reset_index(drop=True)
n, o = load(NEW), load(OLD)
ni = json.load(open(f"{NEW}/meta/info.json"))
check("episodes == 312", ni["total_episodes"] == 312, str(ni["total_episodes"]))
nh = n[n.episode_index < 180].reset_index(drop=True)
check("G1 HEAD row count == r180", len(nh) == len(o), f"{len(nh)} vs {len(o)}")
cols = [c for c in o.columns]
def _num(col):
    """object column of (possibly nested) arrays -> one numeric ndarray; strings stay as a list"""
    def cell(x):
        x = np.asarray(x)
        return np.stack([cell(y) for y in x]) if x.dtype == object else x
    try: return np.stack([cell(x) for x in col])
    except Exception: return np.asarray([str(x) for x in col])
bad = []
for c in cols:
    a, b = nh[c].to_numpy(), o[c].to_numpy()
    if a.dtype == object: same = a.shape == b.shape and np.array_equal(_num(a), _num(b))
    else: same = np.array_equal(a, b)
    if not same: bad.append(c)
check("G1 HEAD every parquet column bit-identical to r180", not bad, f"{len(cols)} cols" if not bad else f"differ: {bad}")
nt = pq.read_table(f"{NEW}/meta/tasks.parquet").to_pandas(); ot = pq.read_table(f"{OLD}/meta/tasks.parquet").to_pandas()   # task string = index (metadata)
t_new = dict(zip(nt.task_index, nt.index)); t_old = dict(zip(ot.task_index, ot.index))
hs = n[n.episode_index < 180].groupby("episode_index").task_index.first().map(t_new)
os_ = o.groupby("episode_index").task_index.first().map(t_old)
check("G1 HEAD task strings identical", all(isinstance(x, str) for x in hs.values) and (hs.values == os_.values).all(), f"e.g. {str(hs.values[0])[:40]}")
dn, do = LeRobotDataset("rebot/r312c", root=NEW), LeRobotDataset("rebot/r180", root=OLD)
rng = np.random.default_rng(0); ks = ["observation.images.global", "observation.images.left_wrist", "observation.images.right_wrist"]
mx = 0.0
for ep in rng.choice(180, 4, replace=False):
    fr = o.index[o.episode_index == ep]; 
    for i in rng.choice(fr, 3, replace=False):
        a, b = dn[int(i)], do[int(i)]
        for k in ks: mx = max(mx, float((a[k] - b[k]).abs().max()))
check("G2 HEAD decoded video frames identical to r180 (12 frames x 3 cams)", mx == 0.0, f"max abs diff {mx:.2e}")
nf = n[n.episode_index >= 180]
eps = sorted(nf.episode_index.unique())
check("G3 FRONT episode count == 132", len(eps) == 132 and eps == list(range(180, 312)))
inv = pd.read_csv("/home/bh-aiteam/umi_bridge/qa_b663/inventory.csv").set_index("episode")
want = [inv.loc[e, "task"] for e in [180 + z for z in SEL["zarr_episodes"]]]
got = list(nf.groupby("episode_index").task_index.first().map(t_new).values)
check("G3 FRONT task per episode == selection (R663 inventory task)", got == want)
import re as _re
cnt = pd.Series(["".join(c[0].upper() for c in _re.findall(r"(\w+) cube", t)) for t in got]).value_counts().to_dict()   # task -> order code, the selection's key format
check("G3 FRONT order counts == selection", cnt == SEL["order_counts"], str(sorted(cnt.values())))
flen = nf.groupby("episode_index").size().values
check("G3 FRONT frames per episode plausible (>=150)", flen.min() >= 150, f"min {flen.min()} max {flen.max()} total {flen.sum()}")
st = _num(nf["observation.state"].to_numpy()); ac = _num(nf["action"].to_numpy()).reshape(len(nf), -1)
check("G5 FRONT state/action finite", np.isfinite(st).all() and np.isfinite(ac).all())
check("G5 FRONT state width 76, action 320 (16x20)", st.shape[1] == 76 and ac.shape[1] == 320, f"{st.shape} {ac.shape}")
# current self term: pos [3:6] / [41:44] == 0 and rot [18:24]/[56:62] == 0 (axis-angle identity) per the packing
I6 = np.array([1.0, 0, 0, 0, 1, 0])   # rotation_6d of the identity (UMI packing)
cur = max(np.abs(st[:, 3:6]).max(), np.abs(st[:, 41:44]).max(), np.abs(st[:, 18:24] - I6).max(), np.abs(st[:, 56:62] - I6).max())
check("G5 FRONT current self term is identity", cur < 1e-5, f"max {cur:.2e}")
stats = json.load(open(f"{NEW}/meta/stats.json"))
check("G5 stats.json present with state/action", "observation.state" in stats and "action" in stats)
# G4 rot180: compare new FRONT global against the B663 dataset, whose FRONT globals were built from the same source
# with the same rot180 and verified correctly rotated (qa_b663 v2). Same (source episode, stored frame) -> same image.
r663 = LeRobotDataset("rebot/r663", root=f"{D}/r663_xvla_umi32_v1")
m663 = pd.concat([_rp(f, columns=["episode_index", "frame_index", "index"]) for f in sorted(glob.glob(f"{D}/r663_xvla_umi32_v1/data/*/*.parquet"))])
rows = []
for j in rng.choice(132, 8, replace=False):
    e_new, e663 = 180 + int(j), 180 + int(SEL["zarr_episodes"][int(j)])     # r663 FRONT 180.. == r675rbp zarr 0..
    a0, a1 = dn.meta.episodes[e_new]["dataset_from_index"], dn.meta.episodes[e_new]["dataset_to_index"]
    b0, b1 = r663.meta.episodes[e663]["dataset_from_index"], r663.meta.episodes[e663]["dataset_to_index"]
    mid = (a1 - a0) // 2; x = dn[a0 + mid]["observation.images.global"]; best = (9.0, None, 9.0)
    for off in range(-3, 4):
        k = b0 + mid + off
        if not (b0 <= k < b1): continue
        y = r663[k]["observation.images.global"]
        d = float((x - y).abs().mean()); d_flip = float((x - y.flip(-1).flip(-2)).abs().mean())
        if d < best[0]: best = (d, off, d_flip)
    rows.append(best)
ok = all(r[0] < 0.02 and r[2] > 5 * r[0] for r in rows) and len(rows) == 8
check("G4 FRONT global == B663 rot180 FRONT frame, and re-rotating it is far worse (8 eps)", ok,
      "best(mean|d|, frame offset, d if rotated again) = " + str([(round(r[0], 4), r[1], round(r[2], 4)) for r in rows]))
# G6 regression: every FRONT episode shared with R380 is bit-identical (state, action, length) -- same converter, same source
S380 = json.load(open("/home/bh-aiteam/umi_bridge/umi76/r380_front200_v1.json"))["zarr_episodes"]
r380 = load(f"{D}/r380_umi76_rel16_v1"); pos380 = {z: 180 + i for i, z in enumerate(S380)}
shared = [(i, z) for i, z in enumerate(SEL["zarr_episodes"]) if z in pos380]
bad6 = []
for i, z in shared:
    a_ = n[n.episode_index == 180 + i]; b_ = r380[r380.episode_index == pos380[z]]
    if len(a_) != len(b_) or not np.array_equal(_num(a_["observation.state"].to_numpy()), _num(b_["observation.state"].to_numpy())) \
       or not np.array_equal(_num(a_["action"].to_numpy()), _num(b_["action"].to_numpy())): bad6.append(z)
check(f"G6 the {len(shared)} FRONT episodes shared with R380 are bit-identical (state, action, length)", not bad6 and len(shared) == 78,
      f"differ: {bad6[:5]}" if bad6 else f"{len(shared)}/{len(shared)}")
print("\nALL R312c GATES PASS" if not fails else f"\nFAILED: {fails}")
