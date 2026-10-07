"""[2026-09-29] RELCART20 state (primary ego->robot state candidate) from REL16-v3d: the ONLY change is observation.state (76 -> 20).

    state (20) = [ L_rel pos3 (m) | L_rel rot6d | R_rel pos3 (m) | R_rel rot6d | g_L | g_R ]
        T_rel(t) = inv(T_anchor) @ T(t), per arm, T = rebot_fk_torch.tcp(aux.q_t) in the robot base frame (== dataset TCP)
        anchor   = the episode's FIRST ROW (frame_index 0). Segment = one LeRobot episode (one teleop recording of the whole
                   3-stack task); v3d has no finer segmentation and none is added here.
        rot6d    = first two ROWS of R (umi pose_util.mat_to_rot6d, the REL16 action convention); identity = [1,0,0,0,1,0]
        g        = follower jaw openness at t = width / W_OPEN, W_OPEN = 270*0.05/118 m, clip [0,1]; 0 CLOSED, 1 OPEN
                   (the AbsCart20 converter, unchanged)
Action (16, 32) REL16 + continuous grip + dq12, aux.q_t, videos (hard-linked), prompts, episode order: bit-identical to the parent.
Gates: QA1 anchor rows == identity (pos < 1e-6 m, rot < 1e-4 rad); QA2 = relcart20_qa.py (independent recomputation, Mac deploy FK);
QA3 every non-state column byte-identical to the parent.
usage: derive_relcart20.py --parent <v3d dir> --out <dir>      (node, holobrain env)
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

# pass 1: absolute TCP of every row, and the anchor (frame_index 0) of every episode -- episodes may span parquet files
absT, rows = {}, []
for fi, t in enumerate(tabs):
    ep = np.asarray(DV._col(t, "episode_index").to_numpy()); fr = np.asarray(DV._col(t, "frame_index").to_numpy())
    with torch.no_grad():
        T = fk.tcp(torch.tensor(np.asarray(DV._col(t, "aux.q_t").to_pylist(), np.float64))).numpy()   # (n, 2, 4, 4)
    absT[fi] = T; rows.append((ep, fr))
anchor = {}
for fi, (ep, fr) in enumerate(rows):
    for j in np.flatnonzero(fr == 0):
        assert int(ep[j]) not in anchor, f"episode {ep[j]} has two frame_index-0 rows"
        anchor[int(ep[j])] = absT[fi][j]
assert len(anchor) == len({int(e) for ep, _ in rows for e in ep}), "an episode has no frame_index-0 row"

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
allS, per_ep, qa1_pos, qa1_rot = [], {}, [], []
for fi, (f, t) in enumerate(zip(files, tabs)):
    ep, fr = rows[fi]; T = absT[fi]
    S76 = np.asarray(DV._col(t, "observation.state").to_pylist(), np.float64)
    A0 = np.stack([anchor[int(e)] for e in ep])                                        # (n, 2, 4, 4)
    Trel = np.einsum("naij,najk->naik", np.linalg.inv(A0), T)
    S = np.zeros((len(ep), 20), np.float32)
    for r in (0, 1):
        S[:, r * 9:r * 9 + 3] = Trel[:, r, :3, 3]
        S[:, r * 9 + 3:r * 9 + 9] = Trel[:, r, :2, :3].reshape(-1, 6)
    S[:, 18:20] = np.clip(S76[:, [37, 75]] / W_OPEN, 0.0, 1.0)
    z = fr == 0
    if z.any():
        qa1_pos.append(np.abs(Trel[z][..., :3, 3]).max())
        qa1_rot.append(np.abs(Trel[z][..., :3, :3] - np.eye(3)).max())
    allS.append(S)
    for e in np.unique(ep):
        per_ep.setdefault(int(e), []).append(S[ep == e])
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
per_ep = {e: np.concatenate(v) for e, v in per_ep.items()}
S = np.concatenate(allS)
print(f"QA1 anchor identity over {len(anchor)} episodes: max |pos| {max(qa1_pos):.2e} m, max |R - I| {max(qa1_rot):.2e}")
assert max(qa1_pos) < 1e-6 and max(qa1_rot) < 1e-4, "QA1 FAILED"
for f in files:                                                                          # QA3 (node side)
    pt = pq.read_table(f); ot = pq.read_table(out / pathlib.Path(f).relative_to(parent))
    for name in pt.column_names:
        if name != "observation.state":
            assert pt.column(name).equals(ot.column(name)), f"QA3: {name} changed in {f}"
print("QA3 non-state columns byte-identical to the parent: PASS")
st = json.load(open(parent / "meta/stats.json"))
ns = DV.stats(S.astype(np.float64))
st["observation.state"] = {k: ([float(x) for x in ns[k]] if k != "count" else [int(len(S))]) for k in DV.STAT_KEYS}
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
prov.update(parent=str(parent), derived_by="derive_relcart20.py", variant="RELCART20",
            state_layout="[L_rel pos3 rot6d(rows) | R_rel pos3 rot6d(rows) | g_L g_R], T_rel = inv(T_anchor) T_t, anchor = episode frame_index 0",
            anchor="first row (frame_index 0) of each LeRobot episode; segment = episode", gripper="follower width / 0.11441 m, clip [0,1], 1 OPEN",
            qa1=dict(max_pos_m=float(max(qa1_pos)), max_rot=float(max(qa1_rot))))
json.dump(prov, open(out / "v4_provenance.json", "w"), indent=2)
print(f"done {out}: rows {len(S)} state (20,)")
