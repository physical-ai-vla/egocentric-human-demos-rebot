#!/usr/bin/env python3
"""[2026-09-28] Final C-old v2 TR-only = frozen old259 snapshot (~/c8/c8oldv2_data, never rewritten) || HRL80 TR part
(~/c8/c8oldv2_hrl80_data, validated standalone first). Per split: lerobot aggregate_datasets(roots=[old snapshot split, HRL80 split])
-> old episodes keep indices 0..n_old-1, HRL80 appended after; videos are remuxed/concatenated (no re-encode). Then
write_stage.verify_root + write_stage.spot_check_frames (aggregated vs source part, pixel-identical), and a combined rows view
(~/c8/c8oldv2_final_rows: symlinks to the two frozen rows caches + merged index.json) for the validator / append invariant.
Env: ~/xvla-mac/bin/python.  Output ~/c8/c8oldv2_final_data/{c8oldv2_final_train, c8oldv2_final_val}"""
import json, pathlib, shutil, sys, time
H = pathlib.Path.home(); C8 = H / "c8"; sys.path.insert(0, str(C8))
import write_stage as W
from lerobot.datasets.aggregate import aggregate_datasets
OLD, NEW, OUT, RV = C8 / "c8oldv2_data", C8 / "c8oldv2_hrl80_data", C8 / "c8oldv2_final_data", C8 / "c8oldv2_final_rows"
PARTS = {"train": ("c8oldv2_tr_train", "c8oldv2_hrl80_train", "c8oldv2_final_train"), "val": ("c8oldv2_tr_val", "c8oldv2_hrl80_val", "c8oldv2_final_val")}
for f in (OLD / "MANIFEST_tr.json", NEW / "MANIFEST_hrl80.json"):
    m = json.load(open(f)); assert not m["checks_failed"], f"{f}: {m['checks_failed']}"
OUT.mkdir(exist_ok=True); rep = {}
for role, (o, n, fin) in PARTS.items():
    no, nn = (json.load(open(r / "meta/info.json"))["total_episodes"] for r in (OLD / o, NEW / n))
    tmp = OUT / f"{fin}.tmp"; shutil.rmtree(tmp, ignore_errors=True)
    aggregate_datasets(repo_ids=[f"local/{o}", f"local/{n}"], aggr_repo_id=f"local/{fin}", roots=[OLD / o, NEW / n], aggr_root=tmp)
    why = W.verify_root(tmp, no + nn) or W.spot_check_frames(tmp, fin, [(o, OLD / o, no), (n, NEW / n, nn)], n=24)
    if why: raise SystemExit(f"FINAL_APPEND_VERIFY_FAIL {role}: {why}")
    shutil.rmtree(OUT / fin, ignore_errors=True); tmp.rename(OUT / fin)
    rep[role] = dict(old_episodes=no, hrl80_episodes=nn, total=no + nn, rows=65 * (no + nn), chunk_starts=31 * (no + nn)); print(role, rep[role], flush=True)
RV.mkdir(exist_ok=True)
for src in (C8 / "c8oldv2_tr_rows", C8 / "c8oldv2_tr_hrl80_rows"):
    for f in sorted(src.glob("2*.npz")):
        l = RV / f.name; assert not l.exists() or l.resolve() == f.resolve(), f"row name collision {f.name}"
        if not l.exists(): l.symlink_to(f)
i_old, i_new = json.load(open(C8 / "c8oldv2_tr_rows/index.json")), json.load(open(C8 / "c8oldv2_tr_hrl80_rows/index.json"))
json.dump(dict(parts=["c8oldv2_tr_rows (old259 TR, frozen)", "c8oldv2_tr_hrl80_rows (HRL80 TR)"], val_episodes=sorted(i_old["val_episodes"] + i_new["val_episodes"]),
               segments=i_old["segments"] + i_new["segments"], episodes=i_old["episodes"] + i_new["episodes"]), open(RV / "index.json", "w"), indent=1)
json.dump(dict(schema="c_old_v2_final_append/v1", finished=time.strftime("%F %T"), report=rep, old=str(OLD), hrl80=str(NEW),
               method="aggregate_datasets([old snapshot, HRL80 part]) per split; no re-encode; verify_root + 24 pixel-exact spot checks"),
          open(OUT / "append_report.json", "w"), indent=1)
print("FINAL APPEND DONE", rep)
