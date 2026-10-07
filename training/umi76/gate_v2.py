"""[2026-09-28] Gates for derive_v2.py outputs. usage: gate_v2.py <parent> <A dir> [<B dir>]

G1 every column except action[..., 9|19] bit-identical to the parent (B: state[:76] too); arm action dims identical
G2 videos hard-linked (same inode) -> bit-identical
G3 stats: unchanged dims bit-identical to the parent; A and B share identical action stats
G4 gripper label is binary {0,1}; open fraction; per-episode open->close transitions; agreement with the
   follower-open rule (raw <= -135, measured through state76's current width <= 135*0.05/118 m) at k=1
G5 (B) TCP18 consistent with state76: inv(T_other) @ T_self rebuilt from TCP18 == state76 cross-arm current term
G6 LeRobotDataset loads; sample shapes; A and B identical except state tail
"""
import glob, json, os, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.environ.get("UMI_ROOT", os.path.expanduser("~/holobrain-mac-model/umi_pkg")))
from umi.common.pose_util import pose10d_to_mat

P, A = sys.argv[1], sys.argv[2]; B = sys.argv[3] if len(sys.argv) > 3 else None
res = []


def gate(name, ok, msg):
    res.append(ok); print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}", flush=True)


def col(t, c):
    a = t.column(c).combine_chunks()
    return a.storage if isinstance(a, pa.ExtensionArray) else a


def load(d):
    ts = [pq.read_table(f) for f in sorted(glob.glob(f"{d}/data/chunk-*/*.parquet"))]
    return {c: np.concatenate([np.asarray(col(t, c).to_pylist()) for t in ts]) for c in ts[0].column_names}


p, a = load(P), load(A); b = load(B) if B else None
mask = np.ones(20, bool); mask[[9, 19]] = False
for nm, x in (("A", a),) + ((("B", b),) if b else ()):
    same = all(np.array_equal(p[c], x[c]) for c in p if c not in ("action", "observation.state"))
    gate(f"G1 {nm} other columns", same, "episode/frame/index/task/timestamp/prev_valid identical")
    gate(f"G1 {nm} arm action dims", np.array_equal(p["action"][..., mask], x["action"][..., mask]), "18/20 action dims identical")
    st = x["observation.state"]
    gate(f"G1 {nm} state76", np.array_equal(p["observation.state"], st[:, :76]), f"state width {st.shape[1]}")

for nm, d in (("A", A),) + ((("B", B),) if B else ()):
    pv = sorted(glob.glob(f"{P}/videos/**/*.mp4", recursive=True))
    ok = all(os.stat(v).st_ino == os.stat(v.replace(P, d, 1)).st_ino for v in pv)
    gate(f"G2 {nm} videos", ok and len(pv) > 0, f"{len(pv)} mp4 hard-linked")

sp = json.load(open(f"{P}/meta/stats.json")); sa = json.load(open(f"{A}/meta/stats.json"))
okc = all(np.array_equal(np.asarray(sp["action"][k])[mask], np.asarray(sa["action"][k])[mask]) for k in ("mean", "std", "min", "max"))
gate("G3 A action stats arm dims", okc, f"gripper mean L/R {sa['action']['mean'][9]:.3f}/{sa['action']['mean'][19]:.3f} std {sa['action']['std'][9]:.3f}/{sa['action']['std'][19]:.3f}")
gate("G3 A state stats", sp["observation.state"] == sa["observation.state"], "identical")
if B:
    sb = json.load(open(f"{B}/meta/stats.json"))
    gate("G3 B action stats == A", sb["action"] == sa["action"], "identical")
    gate("G3 B state stats[:76] == parent", all(np.array_equal(np.asarray(sp["observation.state"][k]), np.asarray(sb["observation.state"][k])[:76])
                                              for k in ("mean", "std", "min", "max")), "identical; 18 appended")

g = a["action"][..., [9, 19]]
CONT = os.environ.get("GRIP", "binary") == "continuous"   # [2026-09-29] variant C: continuous leader command in [0, 1]
if CONT:
    gate("G4 continuous range", float(g.min()) >= 0.0 and float(g.max()) <= 1.0,
         f"min {g.min():.3f} max {g.max():.3f}; exactly 0: {100*np.mean(g == 0):.1f} %, exactly 1: {100*np.mean(g == 1):.1f} %, "
         f"strictly between: {100*np.mean((g > 0) & (g < 1)):.1f} %")
    gb = (g >= 0.5).astype(np.float32)
else:
    gate("G4 binary", set(np.unique(g).tolist()) <= {0.0, 1.0}, f"open frac L {g[..., 0].mean():.3f} R {g[..., 1].mean():.3f}")
    gb = g
ep = a["episode_index"]; tr = []
for e in np.unique(ep):
    for r in (0, 1):
        s_ = gb[ep == e, 0, r]; tr.append(int(((s_[:-1] == 1) & (s_[1:] == 0)).sum()))
tr = np.array(tr)
gate("G4 transitions", (tr >= 1).mean() > 0.5, f"open->close (at g = 0.5) per episode-arm: median {np.median(tr):.0f}, max {tr.max()}, "
     f"episode-arms with >= 1 close {100*(tr>=1).mean():.0f} %")
W_OPEN = 135 * 0.05 / 118
fol = np.stack([p["observation.state"][:, 37] >= W_OPEN, p["observation.state"][:, 75] >= W_OPEN], 1)
agree = (gb[:, 0, :] == fol).mean()
gate("G4 vs follower rule", agree > 0.85, f"label k=1 (>= 0.5) vs follower width >= {W_OPEN*1000:.0f} mm at t: {100*agree:.1f} %")

if b is not None:
    st = b["observation.state"]; errs = []
    idx = np.random.default_rng(0).choice(len(st), 2000, replace=False)
    for i in idx:
        T = [pose10d_to_mat(st[i, 76 + 9 * r:76 + 9 * r + 9][None].astype(np.float64))[0] for r in (0, 1)]
        for r, off in ((0, 6), (1, 44)):
            o = 1 - r; rel = np.linalg.inv(T[o]) @ T[r]
            errs.append(np.linalg.norm(rel[:3, 3] - st[i, off + 3:off + 6]) * 1000)
    errs = np.array(errs)
    gate("G5 B TCP18 vs state76 cross-arm", errs.max() < 0.5, f"|inv(T_o)T_r pos - state cross_c| med {np.median(errs):.4f} max {errs.max():.4f} mm")

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    for nm, d, w in (("A", A, 76),) + ((("B", B, 94),) if B else ()):
        ds = LeRobotDataset(f"rebot/{os.path.basename(d)}", root=d, video_backend="pyav")
        x = ds[1234]
        gate(f"G6 {nm} load", tuple(x["observation.state"].shape) == (w,) and tuple(x["action"].shape) == (16, 20),
             f"{len(ds)} frames, state {tuple(x['observation.state'].shape)}, stats state {len(ds.meta.stats['observation.state']['mean'])}")
except Exception as e:  # noqa: BLE001
    gate("G6 load", False, f"{type(e).__name__}: {e}")
print(f"\n{sum(res)}/{len(res)} PASS")
