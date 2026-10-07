"""[2026-09-29] CART20 state ablation of REL16-v3d: the ONLY change is observation.state (76 -> 20).

    state (20) = [ L TCP pos3 (m) | L rot6d | R TCP pos3 (m) | R rot6d | g_L | g_R ]
        TCP = rebot_fk_torch.tcp(aux.q_t) in the robot BASE frame (== dataset TCP, 0.000 mm / 0.00 deg, 2026-09-29),
        rot6d = first two ROWS of R (umi pose_util.mat_to_rot6d, the same convention as the REL16 action),
        g = follower jaw openness at t = width / W_OPEN, W_OPEN = 270 * 0.05/118 m (raw -270 = fully open), clip [0, 1];
        0 = CLOSED, 1 = OPEN (same semantics as the action g). Width = state76 dims 37 (L) / 75 (R), the current follower width.
    No history, no L<->R relative pose, no joints, no velocity. Current time only.
Everything else -- action (16, 32) REL16 + continuous grip + dq12, aux.q_t, videos (hard-linked), prompts, episode order,
timestamps -- is bit-identical to the parent. Base frame = dataset TCP frame (reBot-only ablation; no canonical workspace frame).
Gates: (1) FK(aux.q_t) re-expressed in the T_t frame reproduces the parent's state76 cross-arm term inv(T_R) T_L (pos < 1 mm),
i.e. the absolute poses are the ones the parent state was built from; (2) g in [0, 1]; (3) action/aux bytes unchanged.
usage: derive_cart20.py --parent <v3d dir> --out <dir>      (node, holobrain env)
"""
import argparse, glob, json, os, pathlib, shutil, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derive_v2 as DV
sys.path.insert(0, "/home/bh-aiteam/c8old/xvla"); import rebot_fk_torch

W_OPEN = 270.0 * 0.05 / 118.0
ap = argparse.ArgumentParser(); ap.add_argument("--parent", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); parent, out = pathlib.Path(a.parent), pathlib.Path(a.out)
if out.exists():
    raise SystemExit(f"{out} exists, refusing to overwrite")
files, tabs = DV.read_parent(parent)
fk = rebot_fk_torch.ReBotFKTorch(dtype=torch.float64)

out.mkdir(parents=True)
for p in parent.rglob("*"):
    rel = p.relative_to(parent); q = out / rel
    if p.is_dir():
        q.mkdir(parents=True, exist_ok=True); os.chmod(q, 0o755)
    elif rel.parts[0] == "videos":
        os.link(p, q)
    elif rel.parts[0] != "data":
        shutil.copy2(p, q); os.chmod(q, 0o644)
(out / "data").mkdir(exist_ok=True)
allS, per_ep, cross_err = [], {}, []
for f, t in zip(files, tabs):
    ep = np.asarray(DV._col(t, "episode_index").to_numpy())
    S76 = np.asarray(DV._col(t, "observation.state").to_pylist(), np.float64)
    Q = torch.tensor(np.asarray(DV._col(t, "aux.q_t").to_pylist(), np.float64))
    with torch.no_grad():
        T = fk.tcp(Q).numpy()                                                              # (n, 2, 4, 4) base frame
    S = np.zeros((len(ep), 20), np.float32)
    for r in (0, 1):
        S[:, r * 9:r * 9 + 3] = T[:, r, :3, 3]
        S[:, r * 9 + 3:r * 9 + 9] = T[:, r, :2, :3].reshape(-1, 6)                          # first two ROWS
    w = S76[:, [37, 75]]
    S[:, 18:20] = np.clip(w / W_OPEN, 0.0, 1.0)
    # gate 1: parent's current cross term r0 pos_wrt (dims 9..11) = pos of inv(T_R) T_L
    c = np.einsum("nji,nj->ni", T[:, 1, :3, :3], T[:, 0, :3, 3] - T[:, 1, :3, 3])
    cross_err.append(np.linalg.norm(c - S76[:, 9:12], axis=1) * 1000)
    allS.append(S)
    for e in np.unique(ep):
        per_ep[int(e)] = S[ep == e]
    cols, fields = [], []
    for name in t.column_names:
        if name == "observation.state":
            typ = pa.list_(pa.float32(), 20)
            cols.append(pa.array(S.tolist(), type=typ)); fields.append(pa.field(name, typ))
        else:
            cols.append(t.column(name)); fields.append(t.schema.field(name))
    md = dict(t.schema.metadata or {})
    if b"huggingface" in md:
        h = json.loads(md[b"huggingface"])
        h["info"]["features"]["observation.state"] = {"feature": {"dtype": "float32", "_type": "Value"}, "length": 20, "_type": "List"}
        md[b"huggingface"] = json.dumps(h).encode()
    pq.write_table(pa.Table.from_arrays(cols, schema=pa.schema(fields, metadata=md)), out / pathlib.Path(f).relative_to(parent))
cross_err = np.concatenate(cross_err); S = np.concatenate(allS)
print(f"gate 1 (FK(aux.q_t) vs parent cross term): med {np.median(cross_err):.4f} max {cross_err.max():.4f} mm")
assert cross_err.max() < 1.0, "absolute TCP from aux.q_t does not reproduce the parent state76 -- wrong FK/frame"
assert S[:, 18:].min() >= 0 and S[:, 18:].max() <= 1, "gripper openness out of [0, 1]"
print(f"g open fraction: L mean {S[:, 18].mean():.3f} R mean {S[:, 19].mean():.3f}; share clipped at 1: {(S[:, 18:] >= 1).mean():.4f}")
# gate 3: every column other than observation.state is byte-identical to the parent
for f in files:
    pt = pq.read_table(f); ot = pq.read_table(out / pathlib.Path(f).relative_to(parent))
    for name in pt.column_names:
        if name != "observation.state":
            assert pt.column(name).equals(ot.column(name)), f"{name} changed in {f}"
st = json.load(open(parent / "meta/stats.json"))
ns = DV.stats(S.astype(np.float64))
st["observation.state"] = {k: (list(ns[k]) if k != "count" else [int(len(S))]) for k in DV.STAT_KEYS}
st["observation.state"] = {k: [float(x) if k != "count" else int(x) for x in v] for k, v in st["observation.state"].items()}
json.dump(st, open(out / "meta/stats.json", "w"), indent=4)
info = json.load(open(parent / "meta/info.json"))
info["features"]["observation.state"] = {"dtype": "float32", "shape": [20], "names": None}
json.dump(info, open(out / "meta/info.json", "w"), indent=4)
for f in sorted(glob.glob(str(out / "meta/episodes/chunk-*/*.parquet"))):
    t = pq.read_table(f); d = {c: t.column(c) for c in t.column_names}
    eps = np.asarray(DV._col(t, "episode_index").to_numpy())
    for k in DV.STAT_KEYS:
        vals = [(list(DV.stats(per_ep[int(e)].astype(np.float64))[k]) if k != "count" else [len(per_ep[int(e)])]) for e in eps]
        d[f"stats/observation.state/{k}"] = pa.array(vals, type=t.schema.field(f"stats/observation.state/{k}").type)
    pq.write_table(pa.table(d).replace_schema_metadata(t.schema.metadata), f)
prov = json.load(open(parent / "v4_provenance.json"))
prov.update(parent=str(parent), derived_by="derive_cart20.py", variant="CART20",
            state_layout="[L pos3 rot6d(rows) | R pos3 rot6d(rows) | g_L g_R] base frame, current t only; g = follower width / 0.11441 m",
            gate_cross_term_mm=dict(median=float(np.median(cross_err)), max=float(cross_err.max())))
json.dump(prov, open(out / "v4_provenance.json", "w"), indent=2)
print(f"done {out}: rows {len(S)} state (20,)")
