"""[2026-09-28] REL16-v2 datasets derived from an existing umi76 REL16 dataset by rewriting parquet columns only.

Why not reconvert: the twin requirement is that arm REL16 labels, state76, video, prompts and episode order are
BIT-IDENTICAL to the parent. Rewriting two action channels (and, for B, appending 18 state dims) guarantees that
by construction and skips the multi-hour video encode. Videos are hard-linked, never re-encoded.

    A  --variant A   action[:, :, 9] / [:, :, 19] := binary LEADER gripper intent, state76 unchanged
    C  --variant C   [2026-09-29] action[:, :, 9|19] := CONTINUOUS leader command g = clip(cmd / 45, 0, 1), 0 = fully CLOSE,
                     1 = fully OPEN (45 = the measured full jaw travel, GRIPPER_COMMAND_SCALE.md; above it the follower is
                     saturated at raw -270). state76 unchanged. Inference binarises (g >= threshold -> OPEN) -- the model
                     learns the common ego/robot semantic "how open", the robot interface alone is binary.
    B  --variant B   A + observation.state 76 -> 94: per arm base-frame TCP [pos3, rot6d] at t, L then R
                     (DIAGNOSTIC ONLY: ego has no robot-base absolute pose; never a shared representation)

Gripper label contract (v2): 1 = OPEN, 0 = CLOSE, from the LEADER command at the same physical times as the arm
targets, t + (k+1) * 3/59.94 s (source 30 Hz, linear interp, then threshold). The follower width used in v1 stalls
at cube contact while the leader squeezes (R150 hold: follower raw -79..-118, leader 0-7), so it carried no grasp
intent. The actuator side is separate: the deploy adapter maps 1 -> cmd OPEN, 0 -> cmd CLOSED from
gripper_contract_v2.json (gripper_smoke_test.py --cube). Leader: 0 = closed (rest, frame 0), open ~50-80, same
sign in R150/R30/center675 (checked 2026-09-28: frame-0 leader 0 / follower 0, corr(follower, leader) -0.8).

Row -> source mapping (converter umi76_to_lerobot.py): sources in provenance order, episodes in select order
(or 0..n-1), src_ep via episodes_json; stored row f of an episode = source frame 2f + 2 (first kept t >= 50.05 ms).

usage: derive_v2.py --parent <dir> --out <dir> --variant A|B [--source-root <lerobot dir>] [--rbp <json>]
       derive_v2.py --parent <dir> --selftest       recompute the parent's stats and compare (no writes)
"""
import argparse, glob, json, os, pathlib, shutil, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

UMI_DT_FRAMES = 30.0 * 3.0 / 59.94          # 1.5015 source frames
OPEN_THRESH = 27.0                          # leader cmd >= 27 = opening (humanik rule, video-verified on R150)
CMD_FULL_OPEN = 45.0                        # variant C: measured full jaw travel cmd 0..45
GRIP_IDX = {0: 6, 1: 13}
REMAP = {}
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def _col(t, c):
    a = t.column(c).combine_chunks()
    return a.storage if isinstance(a, pa.ExtensionArray) else a


def load_src(root, cols):
    out = {c: [] for c in cols}; ep = []; fr = []
    for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True)):
        t = pq.read_table(f, columns=list(cols) + ["episode_index", "frame_index"])
        for c in cols:
            out[c].append(np.asarray(_col(t, c).to_pylist(), np.float64))
        ep.append(np.asarray(_col(t, "episode_index").to_numpy())); fr.append(np.asarray(_col(t, "frame_index").to_numpy()))
    ep = np.concatenate(ep); fr = np.concatenate(fr)
    arr = {c: np.concatenate(v) for c, v in out.items()}
    by = {}
    for e in np.unique(ep):
        m = np.flatnonzero(ep == e); m = m[np.argsort(fr[m])]
        assert (fr[m] == np.arange(len(m))).all(), f"{root} ep {e}: frame_index not contiguous"
        by[int(e)] = {c: arr[c][m] for c in cols}
    return by


def stats(x):
    """LeRobot-style stats over axis 0 of a (N, D) array (action: frames*16 rows)."""
    q = np.quantile(x, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return dict(min=x.min(0), max=x.max(0), mean=x.mean(0), std=x.std(0), count=np.array([len(x)]),
                q01=q[0], q10=q[1], q50=q[2], q90=q[3], q99=q[4])


def read_parent(parent):
    files = sorted(glob.glob(f"{parent}/data/chunk-*/*.parquet"))
    tabs = [pq.read_table(f) for f in files]
    return files, tabs


def row_sources(parent, rbp_override=None):
    """(episode_index -> (source lerobot root, source episode)) from the parent's provenance."""
    prov = json.load(open(f"{parent}/v4_provenance.json"))
    mapping, e = {}, 0
    for s in prov["sources"]:
        root = s["lerobot"]
        sel = s.get("select_zarr_episodes")
        if sel is None:
            n = json.load(open(f"{REMAP.get(root, root)}/meta/info.json"))["total_episodes"]
            zeps = list(range(n)); epmap = None
        else:
            zeps = [int(x) for x in sel]
            epmap = json.load(open(rbp_override or s.get("episodes_json") or
                                   "/home/bh-aiteam/umi_bridge/r675_rbp.json"))["episodes"]
        for z in zeps:
            mapping[e] = (root, z if epmap is None else int(epmap[z]), s["zarr"], z); e += 1
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True); ap.add_argument("--out")
    ap.add_argument("--variant", choices=["A", "B", "C"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--remap-root", action="append", default=[], help="old=new, to run off-node with local copies")
    ap.add_argument("--rbp", default=None)
    a = ap.parse_args()
    parent = pathlib.Path(a.parent)
    files, tabs = read_parent(parent)
    stored = json.load(open(parent / "meta/stats.json"))

    if a.selftest:
        A = np.concatenate([np.asarray(_col(t, "action").to_pylist(), np.float64) for t in tabs])
        S = np.concatenate([np.asarray(_col(t, "observation.state").to_pylist(), np.float64) for t in tabs])
        for name, x in (("action", A.reshape(-1, A.shape[-1])), ("observation.state", S)):
            mine = stats(x)
            for k in ("min", "max", "mean", "std", "q50"):
                d = np.abs(np.asarray(stored[name][k]) - mine[k]).max()
                print(f"selftest {name:18s} {k:5s} max|stored-recomputed| {d:.3e}")
            print(f"selftest {name} count stored {stored[name]['count']} recomputed {mine['count']}")
        return

    remap = dict(x.split("=", 1) for x in a.remap_root); REMAP.update(remap)
    src_of = row_sources(parent, a.rbp)
    roots = sorted({v[0] for v in src_of.values()})
    need = ["action"]
    src = {r: load_src(remap.get(r, r), need) for r in roots}
    if a.variant == "B":
        sys.path.insert(0, os.environ.get("UMI_ROOT", "/home/bh-aiteam/universal_manipulation_interface"))
        from umi.common.pose_util import mat_to_pose10d
        from scipy.spatial.transform import Rotation as Rot
        import zarr
        # TCP18 comes from the SAME zarr eef poses state76 was computed from (converter _traj/_sample; the query
        # rows sit on the stored grid, so the pose at t is the raw zarr sample, no interpolation). center675_s96
        # carries no observation.ee.tcp_state, and one source for every pool keeps B consistent by construction.
        ZP = {}
        for zp in sorted({v[2] for v in src_of.values()}):
            zz = zarr.open(remap.get(zp, zp), "r")
            ends = np.asarray(zz["meta"]["episode_ends"][:], int)
            ZP[zp] = dict(starts=np.concatenate([[0], ends[:-1]]),
                          pos=[np.asarray(zz["data"][f"robot{r}_eef_pos"][:], np.float64) for r in (0, 1)],
                          rv=[np.asarray(zz["data"][f"robot{r}_eef_rot_axis_angle"][:], np.float64) for r in (0, 1)])

    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists, refusing to overwrite")
    out.mkdir(parents=True)
    # videos: hard links (same inode, zero bytes, bit-identical by construction); meta copied then rewritten
    for p in parent.rglob("*"):
        rel = p.relative_to(parent); q = out / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        elif rel.parts[0] == "videos":
            os.link(p, q)
        elif rel.parts[0] != "data":
            shutil.copy2(p, q)
    (out / "data").mkdir(exist_ok=True)

    allA, allS, rep = [], [], dict(open_frac=[], transitions=[], band=[])
    per_ep = {}
    for f, t in zip(files, tabs):
        ep = np.asarray(_col(t, "episode_index").to_numpy()); fr = np.asarray(_col(t, "frame_index").to_numpy())
        A = np.asarray(_col(t, "action").to_pylist(), np.float32)          # (n,16,20)
        S = np.asarray(_col(t, "observation.state").to_pylist(), np.float32)
        newS = []
        for i in range(len(ep)):
            root, se, zp, zep = src_of[int(ep[i])]
            s = src[root][se]; lead = s["action"]; n = len(lead)
            i0 = 2 * int(fr[i]) + 2
            tk = i0 + UMI_DT_FRAMES * (np.arange(16) + 1)
            assert tk[-1] <= n - 1 + 1e-6, f"ep {ep[i]} f {fr[i]}: target beyond source episode ({tk[-1]:.2f} > {n-1})"
            for r, gi in GRIP_IDX.items():
                v = np.interp(tk, np.arange(n), lead[:, gi])
                A[i, :, r * 10 + 9] = (np.clip(v / CMD_FULL_OPEN, 0.0, 1.0) if a.variant == "C" else (v >= OPEN_THRESH)).astype(np.float32)
                rep["band"].append(float(((v > 10) & (v < 40)).mean()))
            if a.variant == "B":
                zi = int(ZP[zp]["starts"][zep]) + i0
                ext = []
                for r in (0, 1):
                    m = np.eye(4)
                    m[:3, :3] = Rot.from_rotvec(ZP[zp]["rv"][r][zi]).as_matrix(); m[:3, 3] = ZP[zp]["pos"][r][zi]
                    ext.append(mat_to_pose10d(m[None])[0])
                newS.append(np.concatenate([S[i], *ext]).astype(np.float32))
        if a.variant == "B":
            S = np.stack(newS)
        allA.append(A); allS.append(S)
        for e in np.unique(ep):
            m = ep == e
            per_ep[int(e)] = (A[m], S[m])
        # write back with the parent's schema; only the replaced columns change type/length
        cols, fields = [], []
        for name in t.column_names:
            fld = t.schema.field(name)
            if name == "action":
                arr = pa.array(A.tolist(), type=_col(t, "action").type)
            elif name == "observation.state" and a.variant == "B":
                fld = pa.field(name, pa.list_(pa.float32(), 94)); arr = pa.array(S.tolist(), type=fld.type)
            else:
                cols.append(t.column(name)); fields.append(fld); continue
            cols.append(arr); fields.append(fld)
        md = dict(t.schema.metadata or {})
        if a.variant == "B" and b"huggingface" in md:
            h = json.loads(md[b"huggingface"]); h["info"]["features"]["observation.state"]["length"] = 94
            md[b"huggingface"] = json.dumps(h).encode()
        pq.write_table(pa.Table.from_arrays(cols, schema=pa.schema(fields, metadata=md)),
                       out / pathlib.Path(f).relative_to(parent))

    A = np.concatenate(allA); S = np.concatenate(allS)
    # stats: the parent's stored values are kept BIT-FOR-BIT for every unchanged dim (arm action dims, state76);
    # only gripper dims 9/19 (and B's 18 appended state dims) are computed here, and `count` stays the parent's.
    def splice(old, x, dims=None, append=None):
        mine = stats(x.astype(np.float64)); outd = {}
        for k in STAT_KEYS:
            if k == "count":
                outd[k] = list(old[k]); continue
            v = np.asarray(old[k], np.float64).copy()
            if dims is not None:
                v[dims] = mine[k][dims]
            if append is not None:
                v = np.concatenate([v, mine[k][append:]])
            outd[k] = v.tolist()
        return outd
    st = json.load(open(parent / "meta/stats.json"))
    st["action"] = splice(st["action"], A.reshape(-1, 20), dims=[9, 19])
    if a.variant == "B":
        st["observation.state"] = splice(st["observation.state"], S, append=76)
    json.dump(st, open(out / "meta/stats.json", "w"), indent=4)
    info = json.load(open(parent / "meta/info.json"))
    if a.variant == "B":
        info["features"]["observation.state"]["shape"] = [94]
    json.dump(info, open(out / "meta/info.json", "w"), indent=4)
    for f in sorted(glob.glob(str(out / "meta/episodes/chunk-*/*.parquet"))):
        t = pq.read_table(f); d = {c: t.column(c) for c in t.column_names}
        eps = np.asarray(_col(t, "episode_index").to_numpy())
        for key in ("action",) + (("observation.state",) if a.variant == "B" else ()):
            olds = [{k: _col(t, f"stats/{key}/{k}")[j].as_py() for k in STAT_KEYS} for j in range(len(eps))]
            per = [splice(olds[j], per_ep[int(e)][0].reshape(-1, 20), dims=[9, 19]) if key == "action"
                   else splice(olds[j], per_ep[int(e)][1], append=76) for j, e in enumerate(eps)]
            for k in STAT_KEYS:
                d[f"stats/{key}/{k}"] = pa.array([p[k] for p in per], type=t.schema.field(f"stats/{key}/{k}").type)
        pq.write_table(pa.table(d).replace_schema_metadata(t.schema.metadata), f)
    prov = json.load(open(parent / "v4_provenance.json"))
    prov.update(parent=str(parent), derived_by="derive_v2.py", variant=a.variant,
                gripper_label=("continuous leader command g = clip(cmd/45, 0, 1), 0 = CLOSE, 1 = OPEN, at t+(k+1)*50.05ms"
                               if a.variant == "C" else "binary leader intent, 1 = OPEN, 0 = CLOSE, leader cmd >= 27 at t+(k+1)*50.05ms"),
                state_dim=94 if a.variant == "B" else 76,
                state_extra=("[L pos3 rot6d, R pos3 rot6d] base-frame TCP at t (DIAGNOSTIC)" if a.variant == "B" else None))
    json.dump(prov, open(out / "v4_provenance.json", "w"), indent=2)
    of = A[:, :, [9, 19]]
    print(f"done {out}: rows {len(A)}  open frac L {of[..., 0].mean():.3f} R {of[..., 1].mean():.3f}  "
          f"leader in ambiguous 10-40 band {np.mean(rep['band']):.4f}")


if __name__ == "__main__":
    main()
