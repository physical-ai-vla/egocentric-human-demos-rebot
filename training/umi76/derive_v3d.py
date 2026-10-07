"""[2026-09-29] REL16-v3 variant D: REL16 + continuous gripper (parent v3c) + Δq16 joint auxiliary + current q for FK consistency.

    action (16, 32) = [ REL16 arm+grip 20 (bit-identical to the parent) | Δq 12 ]
        Δq_k = q(t + (k+1)·3/59.94 s) − q(t), 12 arm joints [L j1..j6, R j1..j6] in rad, FOLLOWER observation.state
        (the same measured source the REL16 TCP labels come from: dataset TCP == rebot_fk_torch.tcp(q_follower), 0.000 mm /
        0.00 deg, checked 2026-09-29), linear interpolation on the source 30 Hz grid exactly like the arm targets.
    aux.q_t (12,) = q(t) in rad -- NOT a model input (state stays state76); the loss uses it for FK(q_t + Δq̂).
Everything else (state76, videos hard-linked, prompts, episode order) is bit-identical to the parent.
Gate built in: FK(q_t + Δq_k) expressed in the T_t frame == REL16 A_k position (the FK-consistency loss is ~0 at GT).
usage: derive_v3d.py --parent <v3c dir> --out <dir>      (node, holobrain env, PYTHONPATH lerobot-seeed + UMI)
"""
import argparse, glob, json, os, pathlib, shutil, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derive_v2 as DV
sys.path.insert(0, "/home/bh-aiteam/c8old/xvla"); import rebot_fk_torch
from umi.common.pose_util import pose10d_to_mat

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
ap = argparse.ArgumentParser(); ap.add_argument("--parent", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); parent, out = pathlib.Path(a.parent), pathlib.Path(a.out)
if out.exists():
    raise SystemExit(f"{out} exists, refusing to overwrite")
files, tabs = DV.read_parent(parent)
src_of = DV.row_sources(parent)
src = {r: DV.load_src(r, ["observation.state"]) for r in sorted({v[0] for v in src_of.values()})}
fk = rebot_fk_torch.ReBotFKTorch(dtype=torch.float64)

out.mkdir(parents=True)
for p in parent.rglob("*"):
    rel = p.relative_to(parent); q = out / rel
    if p.is_dir():
        q.mkdir(parents=True, exist_ok=True); os.chmod(q, 0o755)       # the parent is read-only; the copy must be writable
    elif rel.parts[0] == "videos":
        os.link(p, q)
    elif rel.parts[0] != "data":
        shutil.copy2(p, q); os.chmod(q, 0o644)
(out / "data").mkdir(exist_ok=True)
allA, allQ, fk_err, per_ep = [], [], [], {}
rng = np.random.default_rng(0)
for f, t in zip(files, tabs):
    ep = np.asarray(DV._col(t, "episode_index").to_numpy()); fr = np.asarray(DV._col(t, "frame_index").to_numpy())
    A = np.asarray(DV._col(t, "action").to_pylist(), np.float32)                       # (n, 16, 20)
    DQ = np.zeros((len(ep), 16, 12), np.float32); Q = np.zeros((len(ep), 12), np.float32)
    for i in range(len(ep)):
        root, se, *_ = src_of[int(ep[i])]
        qs = src[root][se]["observation.state"]; n = len(qs); i0 = 2 * int(fr[i]) + 2
        tk = i0 + DV.UMI_DT_FRAMES * (np.arange(16) + 1)
        assert tk[-1] <= n - 1 + 1e-6
        q0 = np.radians(qs[i0, ARM])
        qk = np.stack([np.interp(tk, np.arange(n), np.radians(qs[:, j])) for j in ARM], -1)      # (16, 12)
        DQ[i] = (qk - q0[None]).astype(np.float32); Q[i] = q0.astype(np.float32)
    # built-in GT gate on a sample of rows: FK(q_t + Δq_k) in the T_t frame vs the REL16 position label
    for i in rng.choice(len(ep), min(64, len(ep)), replace=False):
        qt = torch.tensor(Q[i], dtype=torch.float64)[None]; qk = qt + torch.tensor(DQ[i], dtype=torch.float64)
        Tt = fk.tcp(qt).numpy()[0]; Tk = fk.tcp(qk).numpy()                                    # (2,4,4), (16,2,4,4)
        for r in (0, 1):
            p_fk = np.einsum("ij,kj->ki", np.linalg.inv(Tt[r])[:3, :3], Tk[:, r, :3, 3] - Tt[r][:3, 3])
            p_lab = A[i, :, r * 10:r * 10 + 3]
            fk_err.append(np.linalg.norm(p_fk - p_lab, axis=1).max() * 1000)
    A32 = np.concatenate([A, DQ], -1)
    allA.append(A32); allQ.append(Q)
    for e in np.unique(ep):
        m = ep == e; per_ep[int(e)] = (A32[m], Q[m])
    cols, fields = [], []
    for name in t.column_names:
        fld = t.schema.field(name)
        if name == "action":
            typ = pa.list_(pa.list_(pa.float32(), 32), 16)
            cols.append(pa.array(A32.tolist(), type=typ)); fields.append(pa.field("action", typ))
        else:
            cols.append(t.column(name)); fields.append(fld)
    qtyp = pa.list_(pa.float32(), 12)
    cols.append(pa.array(Q.tolist(), type=qtyp)); fields.append(pa.field("aux.q_t", qtyp))
    md = dict(t.schema.metadata or {})
    if b"huggingface" in md:
        h = json.loads(md[b"huggingface"])
        h["info"]["features"]["action"] = {"shape": [16, 32], "dtype": "float32", "_type": "Array2D"}
        h["info"]["features"]["aux.q_t"] = {"feature": {"dtype": "float32", "_type": "Value"}, "length": 12, "_type": "List"}
        md[b"huggingface"] = json.dumps(h).encode()
    pq.write_table(pa.Table.from_arrays(cols, schema=pa.schema(fields, metadata=md)), out / pathlib.Path(f).relative_to(parent))
fk_err = np.array(fk_err)
print(f"GT FK-consistency gate: |FK(q_t+dq) in T_t frame - REL16 pos| over 16 steps: med {np.median(fk_err):.4f} max {fk_err.max():.4f} mm")
assert fk_err.max() < 1.0, "GT FK consistency > 1 mm -- the joint aux and the REL16 label disagree"
A = np.concatenate(allA); Q = np.concatenate(allQ)
st = json.load(open(parent / "meta/stats.json"))
new = DV.stats(A.reshape(-1, 32).astype(np.float64)); qs_ = DV.stats(Q.astype(np.float64))
old = st["action"]
st["action"] = {k: (list(old[k]) if k == "count" else list(np.concatenate([np.asarray(old[k], np.float64), new[k][20:]])))
                for k in DV.STAT_KEYS}
st["aux.q_t"] = {k: (list(qs_[k]) if k != "count" else [int(len(Q))]) for k in DV.STAT_KEYS}
json.dump(st, open(out / "meta/stats.json", "w"), indent=4)
info = json.load(open(parent / "meta/info.json"))
info["features"]["action"]["shape"] = [16, 32]
info["features"]["aux.q_t"] = {"dtype": "float32", "shape": [12], "names": None}
json.dump(info, open(out / "meta/info.json", "w"), indent=4)
for f in sorted(glob.glob(str(out / "meta/episodes/chunk-*/*.parquet"))):
    t = pq.read_table(f); d = {c: t.column(c) for c in t.column_names}
    eps = np.asarray(DV._col(t, "episode_index").to_numpy())
    for k in DV.STAT_KEYS:
        olds = [DV._col(t, f"stats/action/{k}")[j].as_py() for j in range(len(eps))]
        vals = []
        for j, e in enumerate(eps):
            s_ = DV.stats(per_ep[int(e)][0].reshape(-1, 32).astype(np.float64))
            vals.append(list(olds[j]) if k == "count" else list(np.concatenate([np.asarray(olds[j], np.float64), s_[k][20:]])))
        d[f"stats/action/{k}"] = pa.array(vals, type=t.schema.field(f"stats/action/{k}").type)
        qv = [(list(DV.stats(per_ep[int(e)][1].astype(np.float64))[k]) if k != "count" else [len(per_ep[int(e)][1])]) for e in eps]
        d[f"stats/aux.q_t/{k}"] = pa.array(qv, type=pa.list_(pa.float64()) if k != "count" else pa.list_(pa.int64()))
    pq.write_table(pa.table(d).replace_schema_metadata(t.schema.metadata), f)
prov = json.load(open(parent / "v4_provenance.json"))
prov.update(parent=str(parent), derived_by="derive_v3d.py", variant="D",
            action_layout="[REL16 L pos3 rot6d grip | R pos3 rot6d grip] (20) + dq12 [L j1..6, R j1..6] rad (follower, q(t+(k+1)dt)-q(t))",
            aux="aux.q_t (12) = follower q(t) rad, loss-only (FK consistency), not a model input",
            gt_fk_consistency_mm=dict(median=float(np.median(fk_err)), max=float(fk_err.max())))
json.dump(prov, open(out / "v4_provenance.json", "w"), indent=2)
print(f"done {out}: rows {len(A)} action (16,32), dq |p99| {np.percentile(np.abs(A[..., 20:]), 99):.3f} rad")
