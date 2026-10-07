"""[2026-09-29] REL16-v3 variant L (INTENT labels): the arm action is the LEADER command's future TCP seen from the current follower.

    A_k = inv(T_f(t)) @ T_l(t + (k+1)*dt), k = 0..15, dt = 3/59.94 s (1.5015 source frames), linear interpolation on the 30 Hz source
        T_f = rebot_fk_torch.tcp(follower observation.state q)   (== dataset TCP, the v3d anchor)
        T_l = rebot_fk_torch.tcp(leader `action` q)              (the source action is already in the follower joint frame: per-joint
                                                                 fit slope 0.88..1.00, offset < 1 deg, R150 + R30, 2026-09-29)
    dq_k = q_l(t + (k+1)dt) - q_f(t)    so FK(q_t + dq_k) reproduces A_k exactly (the FK-consistency loss stays ~0 at GT)
Why: v3d labels are the follower's measured future, which lags the leader by 400-550 ms and under-tracks it (|REL| leader/follower
14x at k1, 1.6x at k16; rest_go k1 0.1-0.3 mm vs 2-6 mm; leader_vs_follower_rel.py). C-old learns the leader command.
Unchanged (bit-identical to the parent v3d): state76, gripper dims 9/19 (already leader-cmd continuous g), aux.q_t (follower q_t),
videos (hard-linked), prompts, episode order, timestamps.
Gates: per-source leader/follower frame check (slope > 0.5 on every arm joint); FK(q_t + dq) vs A_k < 1 mm; non-action columns and
action dims 9/19 byte-identical.
usage: derive_v3L.py --parent <v3d dir> --out <dir>      (node, holobrain env)
"""
import argparse, glob, json, os, pathlib, shutil, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derive_v2 as DV
sys.path.insert(0, "/home/bh-aiteam/c8old/xvla"); import rebot_fk_torch
from umi.common.pose_util import mat_to_pose10d

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
ap = argparse.ArgumentParser(); ap.add_argument("--parent", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); parent, out = pathlib.Path(a.parent), pathlib.Path(a.out)
if out.exists():
    raise SystemExit(f"{out} exists, refusing to overwrite")
files, tabs = DV.read_parent(parent)
src_of = DV.row_sources(parent)
src = {r: DV.load_src(r, ["observation.state", "action"]) for r in sorted({v[0] for v in src_of.values()})}
for r, eps in src.items():                                                                  # frame gate
    S = np.concatenate([eps[e]["observation.state"] for e in eps]); L = np.concatenate([eps[e]["action"] for e in eps])
    sl = [np.polyfit(S[:, j], L[:, j], 1)[0] for j in ARM]
    print(f"frame gate {os.path.basename(r)}: leader-vs-follower slope min {min(sl):.2f} max {max(sl):.2f}")
    assert min(sl) > 0.5, f"{r}: a leader joint is not in the follower frame (slope {min(sl):.2f})"
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
allA, fk_err, per_ep, mag = [], [], {}, []
for f, t in zip(files, tabs):
    ep = np.asarray(DV._col(t, "episode_index").to_numpy()); fr = np.asarray(DV._col(t, "frame_index").to_numpy())
    A0 = np.asarray(DV._col(t, "action").to_pylist(), np.float32)                             # (n, 16, 32) parent v3d
    Qt = np.asarray(DV._col(t, "aux.q_t").to_pylist(), np.float64)
    n = len(ep); QL = np.zeros((n, 16, 12))
    for i in range(n):
        root, se, *_ = src_of[int(ep[i])]
        ql = src[root][se]["action"]; m = len(ql); i0 = 2 * int(fr[i]) + 2
        tk = i0 + DV.UMI_DT_FRAMES * (np.arange(16) + 1)
        assert tk[-1] <= m - 1 + 1e-6
        QL[i] = np.radians(np.stack([np.interp(tk, np.arange(m), ql[:, j]) for j in ARM], -1))
    with torch.no_grad():
        Tt = fk.tcp(torch.tensor(Qt)).numpy()                                                   # (n,2,4,4) follower at t
        Tl = fk.tcp(torch.tensor(QL.reshape(-1, 12))).numpy().reshape(n, 16, 2, 4, 4)
    Arel = np.einsum("naij,nkajl->nkail", np.linalg.inv(Tt), Tl)                              # (n,16,2,4,4)
    A = A0.copy()
    for r in (0, 1):
        A[:, :, r * 10:r * 10 + 9] = mat_to_pose10d(Arel[:, :, r].reshape(-1, 4, 4)).reshape(n, 16, 9)
    A[:, :, 20:32] = (QL - Qt[:, None]).astype(np.float32)
    assert np.array_equal(A[:, :, [9, 19]], A0[:, :, [9, 19]]), "gripper dims changed"
    # FK gate on the stored float32 labels: FK(q_t + dq) in the T_t frame == REL position
    idx = np.random.default_rng(0).choice(n, min(64, n), replace=False)
    with torch.no_grad():
        Tk = fk.tcp(torch.tensor(Qt[idx][:, None] + A[idx][:, :, 20:32].astype(np.float64)).reshape(-1, 12)).numpy().reshape(len(idx), 16, 2, 4, 4)
    for j, i in enumerate(idx):
        for r in (0, 1):
            p_fk = np.einsum("ij,kj->ki", np.linalg.inv(Tt[i, r])[:3, :3], Tk[j, :, r, :3, 3] - Tt[i, r, :3, 3])
            fk_err.append(np.linalg.norm(p_fk - A[i, :, r * 10:r * 10 + 3], axis=1).max() * 1000)
    mag.append(np.stack([np.linalg.norm(A[:, :, o:o + 3], axis=-1) for o in (0, 10)], -1) / np.maximum(
        np.stack([np.linalg.norm(A0[:, :, o:o + 3], axis=-1) for o in (0, 10)], -1), 1e-4))
    allA.append(A)
    for e in np.unique(ep):
        per_ep.setdefault(int(e), []).append(A[ep == e])
    cols, fields = [], []
    for name in t.column_names:
        if name == "action":
            typ = pa.list_(pa.list_(pa.float32(), 32), 16)
            cols.append(pa.array(A.tolist(), type=typ)); fields.append(pa.field("action", typ))
        else:
            cols.append(t.column(name)); fields.append(t.schema.field(name))
    pq.write_table(pa.Table.from_arrays(cols, schema=pa.schema(fields, metadata=t.schema.metadata)), out / pathlib.Path(f).relative_to(parent))
fk_err = np.array(fk_err); mag = np.concatenate([m.reshape(-1, 16, 2) for m in mag])
print(f"FK gate: |FK(q_t+dq) - REL pos| med {np.median(fk_err):.4f} max {fk_err.max():.4f} mm")
assert fk_err.max() < 1.0, "FK consistency gate FAILED"
print("|REL| intent / follower (median over rows) at k1/k4/k8/k16:", [round(float(np.median(mag[:, k])), 2) for k in (0, 3, 7, 15)])
for f in files:
    pt = pq.read_table(f); ot = pq.read_table(out / pathlib.Path(f).relative_to(parent))
    for name in pt.column_names:
        if name != "action":
            assert pt.column(name).equals(ot.column(name)), f"{name} changed in {f}"
print("non-action columns byte-identical: PASS")
A = np.concatenate(allA)
st = json.load(open(parent / "meta/stats.json"))
new = DV.stats(A.reshape(-1, 32).astype(np.float64)); old = st["action"]
keep = {9, 19}
st["action"] = {k: (list(old[k]) if k == "count" else [float(old[k][d]) if d in keep else float(new[k][d]) for d in range(32)]) for k in DV.STAT_KEYS}
json.dump(st, open(out / "meta/stats.json", "w"), indent=4)
per_ep = {e: np.concatenate(v) for e, v in per_ep.items()}
for f in sorted(glob.glob(str(out / "meta/episodes/chunk-*/*.parquet"))):
    t = pq.read_table(f); d = {c: t.column(c) for c in t.column_names}
    eps = np.asarray(DV._col(t, "episode_index").to_numpy())
    for k in DV.STAT_KEYS:
        olds = [DV._col(t, f"stats/action/{k}")[j].as_py() for j in range(len(eps))]
        vals = []
        for j, e in enumerate(eps):
            s_ = DV.stats(per_ep[int(e)].reshape(-1, 32).astype(np.float64))
            vals.append(list(olds[j]) if k == "count" else [float(olds[j][dd]) if dd in keep else float(s_[k][dd]) for dd in range(32)])
        d[f"stats/action/{k}"] = pa.array(vals, type=t.schema.field(f"stats/action/{k}").type)
    pq.write_table(pa.table(d).replace_schema_metadata(t.schema.metadata), f)
prov = json.load(open(parent / "v4_provenance.json"))
prov.update(parent=str(parent), derived_by="derive_v3L.py", variant="L (intent)",
            action_layout="[REL16 L pos3 rot6d grip | R ...] (20) + dq12; pose = inv(T_follower(t)) T_leader(t+(k+1)dt); dq = q_leader(t+k) - q_follower(t); grip unchanged (leader cmd)",
            gt_fk_consistency_mm=dict(median=float(np.median(fk_err)), max=float(fk_err.max())))
json.dump(prov, open(out / "v4_provenance.json", "w"), indent=2)
print(f"done {out}: rows {len(A)}")
