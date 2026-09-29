#!/usr/bin/env python3
"""[2026-09-28] Append-only invariant for the final C-old v2 TR-only set (old259 + HRL80).

D_final = D_old259 (FINAL-FROZEN snapshot ~/c8/c8oldv2_data, MANIFEST_tr.json)  U  D_HRL80 (new, appended).
Every one of the 689 old259 segments must be bit-identical in the combined dataset:
  I0  the frozen snapshot itself is untouched (meta/data sha == MANIFEST_tr.dataset_meta_sha256)
  I1  same segment IDs, same source_start_frame, same train/val assignment
  I2  append-only order: in each split the old segments are LeRobot episodes 0..n_old-1 in the SAME order; new ones after
  I3  same rows: observation.state, action, observation.ee.tcp_tgt, timestamp, frame_index byte-identical; same task string
  I4  same video frames: every decoded frame (rgb24, native resolution) of all 3 cameras of every old segment identical
  I5  (with <master_combined>) every old259 master file (JSON/NPZ) byte-identical to ~/c8/robotlike/master_old259_frozen
Scope: only old-episode PAYLOAD (rows / task / video / master entries) must be identical. Global LeRobot metadata
(info.json totals, stats.json, episode meta offsets of new episodes) is allowed to change in the combined set.
Usage: c8oldv2_append_invariant.py <combined_root> <train_split> <val_split> <combined_manifest.json> [<master_combined>]
Env: ~/xvla-mac/bin/python.  Writes <combined_root>/APPEND_INVARIANT.json.
"""
import glob, hashlib, json, pathlib, sys
from collections import defaultdict
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, av
pd_concat = pd.concat
H = pathlib.Path.home(); OLD = H / "c8/c8oldv2_data"; OLD_SPL = ("c8oldv2_tr_train", "c8oldv2_tr_val")
NEW = pathlib.Path(sys.argv[1]); NEW_SPL = (sys.argv[2], sys.argv[3]); NEW_MAN = json.load(open(sys.argv[4]))
OMAN = json.load(open(OLD / "MANIFEST_tr.json"))
CAMS = ("observation.images.global", "observation.images.left_wrist", "observation.images.right_wrist")
ROWCOLS = ("observation.state", "action", "observation.ee.tcp_tgt", "timestamp", "frame_index")
fails = []; rep = {}
def check(n, ok, info=""):
    print(("PASS  " if ok else "FAIL  ") + n + (f"  {info}" if info else ""), flush=True); rep[n] = dict(ok=bool(ok), info=info); (fails.append(n) if not ok else None)
def rd(f):
    t = pq.read_table(f); cols = [(c.combine_chunks().storage if isinstance(c.type, pa.ExtensionType) else c) for c in t.columns]
    return pa.table(cols, names=t.column_names).to_pandas(ignore_metadata=True)
def hdir(d): hh = hashlib.sha256(); [hh.update(p.read_bytes()) for p in sorted(pathlib.Path(d).rglob("*")) if p.is_file() and p.suffix in (".parquet", ".json", ".jsonl")]; return hh.hexdigest()
# I0
cur = {s: hdir(OLD / s) for s in OLD_SPL}
check("I0 frozen old259 snapshot untouched", cur == OMAN["dataset_meta_sha256"])
# I1 / I2 (role = train / val, split dir names may differ)
role = lambda spl, pair: "train" if spl == pair[0] else "val"
om = {m["segment_id"]: (role(m["split"], OLD_SPL), m["source_start_frame"], m["lerobot_episode_index"]) for m in OMAN["mapping"]}
nm = {m["segment_id"]: (role(m["split"], NEW_SPL), m["source_start_frame"], m["lerobot_episode_index"]) for m in NEW_MAN["mapping"]}
missing = [s for s in om if s not in nm]
moved = [s for s in om if s in nm and nm[s][:2] != om[s][:2]]
check("I1 all 689 old segment IDs present, same source_start_frame and train/val role", len(om) == 689 and not missing and not moved,
      f"old {len(om)}  missing {len(missing)}  changed start/split {len(moved)}")
reidx = [s for s in om if s in nm and nm[s][2] != om[s][2]]
n_old = {r: sum(v[0] == r for v in om.values()) for r in ("train", "val")}
tail_ok = all(v[2] >= n_old[v[0]] for s, v in nm.items() if s not in om)
check("I2 append-only: old segments keep LeRobot episode index (prefix), every new segment after them", not reidx and tail_ok,
      f"old train {n_old['train']} / val {n_old['val']}; reindexed {len(reidx)}; new segments {len(nm) - len(om)}")
# I3 rows + task
def load(root, spl):
    d = pd_concat([rd(f) for f in sorted(glob.glob(str(root / spl / "data/*/*.parquet")))])
    tk = pq.read_table(root / spl / "meta/tasks.parquet").to_pandas()          # task string lives in the pandas index
    tmap = {int(r): str(t) for t, r in zip(tk.index, tk["task_index"])}
    eps = pd_concat([rd(f) for f in sorted(glob.glob(str(root / spl / "meta/episodes/*/*.parquet")))]).set_index("episode_index")
    return {e: g.sort_values("frame_index") for e, g in d.groupby("episode_index")}, tmap, eps
bad_rows = 0; bad_task = 0; worst = {}
for k in (0, 1):
    Do, To, Eo = load(OLD, OLD_SPL[k]); Dn, Tn, En = load(NEW, NEW_SPL[k])
    for e in range(n_old[("train", "val")[k]]):
        go, gn = Do[e], Dn.get(e)
        if gn is None or len(gn) != len(go): bad_rows += 1; continue
        for c in ROWCOLS:
            a, b = np.stack(go[c].to_numpy()) if go[c].dtype == object else go[c].to_numpy(), np.stack(gn[c].to_numpy()) if gn[c].dtype == object else gn[c].to_numpy()
            if a.dtype != b.dtype or a.tobytes() != b.tobytes(): bad_rows += 1; worst[c] = worst.get(c, 0) + 1; break
        if [Tn[int(t)] for t in gn["task_index"]] != [To[int(t)] for t in go["task_index"]]: bad_task += 1
    rep[f"_eps_{k}"] = (Eo, En)
check("I3 old segments: state / action / tcp_tgt / timestamp / frame_index byte-identical", bad_rows == 0, f"{sum(n_old.values())} episodes, {bad_rows} mismatched {worst}")
check("I3 old segments: same task string per row", bad_task == 0, f"{bad_task} mismatched")
# I4 video: decode each mp4 once, hash every frame, collect per-episode frame hash lists (native res, rgb24)
def frame_hashes(root, spl, eps, n_ep):
    by_file = defaultdict(list)
    for e in range(n_ep):
        r = eps.loc[e]
        for c in CAMS: by_file[(c, int(r[f"videos/{c}/chunk_index"]), int(r[f"videos/{c}/file_index"]))].append((e, float(r[f"videos/{c}/from_timestamp"]), int(r["length"])))
    out = {}
    for (c, ch, fi), lst in by_file.items():
        p = root / spl / "videos" / c / f"chunk-{ch:03d}" / f"file-{fi:03d}.mp4"; H_ = {}
        with av.open(str(p)) as ct:
            s = ct.streams.video[0]; fps = float(s.average_rate); last = max(t0 * fps + n for _, t0, n in lst)
            for fr in ct.decode(s):
                j = int(round(float(fr.pts * s.time_base) * fps)); H_[j] = hashlib.sha1(fr.to_ndarray(format="rgb24").tobytes()).hexdigest()
                if j > last + 2: break
        for e, t0, n in lst:
            j0 = int(round(t0 * fps)); out[(c, e)] = [H_.get(j0 + i) for i in range(n)]
    return out
bad_v = 0; nfr = 0
for k in (0, 1):
    Eo, En = rep.pop(f"_eps_{k}"); n = n_old[("train", "val")[k]]
    ho, hn = frame_hashes(OLD, OLD_SPL[k], Eo, n), frame_hashes(NEW, NEW_SPL[k], En, n)
    for key, lo in ho.items():
        nfr += len(lo); bad_v += int(None in lo or lo != hn.get(key))
    print(f"      video split {k}: {len(ho)} (camera, episode) streams hashed", flush=True)
check("I4 old segments: every decoded video frame identical (3 cams, rgb24 native res)", bad_v == 0, f"{nfr} frames, {bad_v} (camera, episode) streams differ")
if len(sys.argv) > 5:
    MF = H / "c8/robotlike/master_old259_frozen"; MC = pathlib.Path(sys.argv[5]); fm = json.load(open(MF / "MASTER.json"))["files_sha256"]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    badm = [f for f, h in fm.items() if f != "MASTER.json" and not (sha(MF / f) == h == sha(MC / f))]
    check("I5 old259 master entries byte-identical in the combined master (frozen copy re-verified too)", not badm, f"{len(fm) - 1} files, {len(badm)} differ")
res = dict(schema="c_old_v2_append_invariant/v1", frozen_snapshot=str(OLD / "MANIFEST_tr.json"),
           frozen_snapshot_sha256=hashlib.sha256((OLD / "MANIFEST_tr.json").read_bytes()).hexdigest(), combined=str(NEW), checks=rep, failed=fails)
(NEW / "APPEND_INVARIANT.json").write_text(json.dumps(res, indent=1)) if NEW != OLD else None
print("\nAPPEND-ONLY INVARIANT PASS (old259 part bit-identical)" if not fails else f"\nINVARIANT FAILED: {fails}")
sys.exit(1 if fails else 0)
