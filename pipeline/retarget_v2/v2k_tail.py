#!/usr/bin/env python3
"""[2026-09-28] Where do v2-K's rescued segments come from, and are they rescued by odd postures?

Per segment: v1 first failure (workspace / IK / collision), and the R150-manifold distance of the chosen v2 trajectory
(median over its frames of the 12-D NN distance to R150, rad). Groups: segments usable in both, v2-only (rescued),
v1-only. Also: what share of the v2 tail (frames with NN > v1 p90) comes from rescued vs both segments, and how the
v2-Kr rescoring S + lambda * d_R150 would pick among the kept variants (<= 3 stored per segment) for a few lambdas.
usage: v2k_tail.py <v2k out dir>
"""
import collections, glob, json, pathlib, sys
import numpy as np
from scipy.spatial import cKDTree

H = pathlib.Path.home(); D = pathlib.Path(sys.argv[1])
d = np.load(H / "umi_bridge/track_c/data/r150_q_tcp.npz"); R150 = np.radians(np.c_[d["J"][:, :6], d["J"][:, 7:13]]); TREE = cKDTree(R150[::3])
rows = []
for p in sorted(glob.glob(str(D / "*.json"))):
    r = json.load(open(p)); f = D / f"{r['episode']}.npz"
    if not f.exists(): continue
    z = np.load(f)
    for s in r.get("segments", []):
        if "v1" not in s: continue
        v1f = "none" if s["v1_pass"] else ("workspace" if not s["v1"]["ws_ok"] else "IK" if not s["v1"]["ik_ok"] else "collision")
        nn1 = float(np.median(TREE.query(z[f"{s['start']}_v1"][::3])[0]))
        var = []
        for j, k in enumerate(s["v2"]["kept"]):
            q = z[f"{s['start']}_v2_{j}"]; var.append((k["score"], float(np.median(TREE.query(q[::3])[0]))))
        rows.append(dict(v1=s["v1_pass"], v2=s["v2"]["pass_"], v1_fail=v1f, nn_v1=nn1, var=var))
grp = lambda r: "both" if r["v1"] and r["v2"] else "v2-only" if r["v2"] else "v1-only" if r["v1"] else "neither"
print("segments", len(rows), dict(collections.Counter(grp(r) for r in rows)))
print("v2-only (rescued) by v1 failure:", dict(collections.Counter(r["v1_fail"] for r in rows if grp(r) == "v2-only")))
for g in ("both", "v2-only"):
    x = [r["var"][0][1] for r in rows if grp(r) == g]; y = [r["nn_v1"] for r in rows if grp(r) == g]
    if x: print(f"{g:8s} v2 top-1 NN-to-R150 p50 {np.median(x):.3f} p90 {np.percentile(x, 90):.3f} | same segs v1 NN p50 {np.median(y):.3f} p90 {np.percentile(y, 90):.3f}")
v1p90 = np.percentile([r["nn_v1"] for r in rows if r["v1"]], 90) if any(r["v1"] for r in rows) else np.inf
tail = [grp(r) for r in rows if r["v2"] and r["var"][0][1] > v1p90]
print(f"v2 top-1 segments beyond v1 p90 ({v1p90:.3f}): {len(tail)}  from {dict(collections.Counter(tail))}")
for lam in (0.0, 0.5, 1.0, 2.0):
    pick = [min(r["var"], key=lambda v: v[0] + lam * v[1])[1] for r in rows if r["v2"]]
    print(f"v2-Kr lambda {lam}: chosen NN p50 {np.median(pick):.3f} p90 {np.percentile(pick, 90):.3f}  (among <= 3 stored variants)")
