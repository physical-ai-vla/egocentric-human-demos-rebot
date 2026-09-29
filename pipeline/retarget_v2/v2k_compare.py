#!/usr/bin/env python3
"""[2026-09-28] v1 (single anchor) vs v2-K (kinematic multi-seed) on the same human segments -> one table.

Posture-manifold metrics use R150 as the reference (all 173,818 frames, 12 arm joints):
  joint range coverage  per joint, overlap of pseudo-q [p1, p99] with R150 [p1, p99], / R150 width; mean over 12 joints
  cluster coverage      k-means (k=200, seed 0) on R150 q12; share of clusters that receive >= 1 pseudo-q frame
  NN distance           12-D joint distance (rad) from each pseudo-q frame (every 5th) to the nearest R150 frame
  start diversity       mean pairwise 12-D distance between segment-start postures; distinct start clusters
v1 parity: v1 gate flags must equal robotlike_offline_check's for the same segment (old C8 only).
usage: v2k_compare.py --group NAME=<v2k out dir> [...] [--parity-offline <baseline_offline dir>] --out <json>
"""
import argparse, glob, json, pathlib
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

H = pathlib.Path.home()
import sys as _sys
d = np.load(H / "umi_bridge/track_c/data/r150_q_tcp.npz")
_ref = _sys.argv[_sys.argv.index("--eval-ref") + 1] if "--eval-ref" in _sys.argv else "r150full"
if _ref == "heldout120":        # R150 minus the R30 calibration episodes: postures the R30 contract never saw
    _m = ~np.isin(d["ep"], json.load(open(H / "umi_bridge/track_c/v2k/r30_episodes.json"))["episodes"])
else:
    _m = np.ones(len(d["ep"]), bool)
R150 = np.radians(np.c_[d["J"][_m, :6], d["J"][_m, 7:13]])
LO, HI = np.percentile(R150, 1, axis=0), np.percentile(R150, 99, axis=0)
TREE = cKDTree(R150[::3])
from scipy.stats import wasserstein_distance
def _ep_diffs(mask):
    """|dq|, |ddq| at 30 fps pooled over the episodes selected by mask (never across an episode boundary)"""
    J = np.radians(np.c_[d["J"][:, :6], d["J"][:, 7:13]]); dv, da = [], []
    for e in np.unique(d["ep"][mask]):
        q = J[d["ep"] == e]; dv.append(np.abs(np.diff(q, axis=0))); da.append(np.abs(np.diff(q, 2, axis=0)))
    return np.concatenate(dv), np.concatenate(da)
_r30m = np.isin(d["ep"], json.load(open(H / "umi_bridge/track_c/v2k/r30_episodes.json"))["episodes"])
DYN_REF = {"R30": _ep_diffs(_r30m), "heldout120": _ep_diffs(~_r30m)}
np.random.seed(0); CENT, _ = kmeans2(R150[::10], 200, seed=0, minit="++"); CTREE = cKDTree(CENT)


def manifold(Q, starts):
    if not len(Q): return {}
    Qraw = Q; Q = np.concatenate(Q); lo, hi = np.percentile(Q, 1, axis=0), np.percentile(Q, 99, axis=0)
    cov = np.clip(np.minimum(hi, HI) - np.maximum(lo, LO), 0, None) / (HI - LO)
    nn, _ = TREE.query(Q[::5]); _, cl = CTREE.query(Q)
    S = np.array(starts); pair = np.linalg.norm(S[:, None] - S[None], axis=-1)[np.triu_indices(len(S), 1)] if len(S) > 1 else np.array([0.0])
    _, scl = CTREE.query(S)
    dv = np.concatenate([np.abs(np.diff(x, axis=0)) for x in Qraw]); da = np.concatenate([np.abs(np.diff(x, 2, axis=0)) for x in Qraw])
    dyn = {}
    for rn, (rv, ra) in DYN_REF.items():
        dyn[f"W1_dq_vs_{rn}"] = round(float(np.mean([wasserstein_distance(dv[::3, j], rv[::5, j]) for j in range(12)])), 5)
        dyn[f"W1_ddq_vs_{rn}"] = round(float(np.mean([wasserstein_distance(da[::3, j], ra[::5, j]) for j in range(12)])), 5)
    dyn.update(dq_p50=round(float(np.median(dv)), 5), dq_p95=round(float(np.percentile(dv, 95)), 5),
               ddq_p50=round(float(np.median(da)), 5), ddq_p95=round(float(np.percentile(da, 95)), 5))
    return dict(**dyn, joint_range_coverage=round(float(cov.mean()), 3), joint_range_coverage_min=round(float(cov.min()), 3),
                cluster_coverage=round(len(set(cl.tolist())) / len(CENT), 3), nn_dist_rad_p50=round(float(np.median(nn)), 3),
                nn_dist_rad_p90=round(float(np.percentile(nn, 90)), 3), start_pairwise_rad_mean=round(float(pair.mean()), 3),
                start_distinct_clusters=int(len(set(scl.tolist()))))


def v2block(key, segs, recs, dirp):
    ok = [s for _, s in segs if s[key]["pass_"]]; n = len(segs); vp = len(ok)
    Q, S = [], []
    for r in recs:
        f = pathlib.Path(dirp) / f"{r['episode']}.npz"
        if not f.exists(): continue
        z = np.load(f)
        for s in r.get("segments", []):
            if key in s and s[key]["pass_"]:
                q = z[f"{s['start']}_{key}_0"]; Q.append(q); S.append(q[0])
    var_d = [x for _, s in segs for x in s[key]["variant_dist_rad"]]
    return dict(usable=vp, rate=round(vp / n, 3) if n else None,
                both=sum(s["v1_pass"] and s[key]["pass_"] for _, s in segs), only_v2=sum((not s["v1_pass"]) and s[key]["pass_"] for _, s in segs),
                only_v1=sum(s["v1_pass"] and not s[key]["pass_"] for _, s in segs),
                ws_feasible_anchor_rate=round(float(np.mean([s[key]["n_ws_candidates"] > 0 for _, s in segs])), 3),
                feasible_seeds_per_seg_p50=float(np.median([s[key]["n_feasible"] for _, s in segs])),
                segs_with_2plus_variants=int(sum(1 for s in ok if len(s[key]["kept"]) >= 2)),
                variant_dist_rad_p50=round(float(np.median(var_d)), 3) if var_d else None,
                fk_pos_mm_p50=round(float(np.median([s[key]["kept"][0]["pos_err_p50"] for s in ok])), 3) if ok else None,
                fk_pos_mm_p90=round(float(np.median([s[key]["kept"][0]["pos_err_p90"] for s in ok])), 3) if ok else None,
                dmin_m_p50=round(float(np.median([s[key]["kept"][0]["dmin"] for s in ok])), 3) if ok else None,
                manifold=manifold(Q, S))


def summarize(dirp, parity_dir=None):
    recs = [json.load(open(p)) for p in sorted(glob.glob(f"{dirp}/*.json"))]
    segs = [(r["episode"], s) for r in recs for s in r.get("segments", []) if "v1" in s]
    n = len(segs); v1p = sum(s["v1_pass"] for _, s in segs); v2p = sum(s["v2"]["pass_"] for _, s in segs)
    both = sum(s["v1_pass"] and s["v2"]["pass_"] for _, s in segs); only2 = sum((not s["v1_pass"]) and s["v2"]["pass_"] for _, s in segs)
    only1 = sum(s["v1_pass"] and not s["v2"]["pass_"] for _, s in segs)
    g = lambda k: round(float(np.mean([bool(s["v1"][k]) for _, s in segs])), 3) if segs else None
    Qv1, Qv2, S1, S2 = [], [], [], []; nvar = []
    for r in recs:
        f = pathlib.Path(dirp) / f"{r['episode']}.npz"
        if not f.exists(): continue
        z = np.load(f)
        for s in r.get("segments", []):
            if "v1" not in s: continue
            if s["v1_pass"]: Qv1.append(z[f"{s['start']}_v1"]); S1.append(z[f"{s['start']}_v1"][0])
            if s["v2"]["pass_"]: Qv2.append(z[f"{s['start']}_v2_0"]); S2.append(z[f"{s['start']}_v2_0"][0]); nvar.append(len(s["v2"]["kept"]))
    var_d = [x for _, s in segs for x in s["v2"]["variant_dist_rad"]]
    out = dict(episodes=len(recs), errors=sum(1 for r in recs if "error" in r), segments=n,
               v1=dict(usable=v1p, rate=round(v1p / n, 3) if n else None, workspace=g("ws_ok"), ik=g("ik_ok"), collision=g("collision_ok"),
                       fk_pos_mm_p50=round(float(np.median([s["v1"]["pos_err_p50"] for _, s in segs])), 3) if segs else None,
                       fk_pos_mm_p90=round(float(np.median([s["v1"]["pos_err_p90"] for _, s in segs])), 3) if segs else None,
                       manifold=manifold(Qv1, S1)),
               **{kk: v2block(kk, segs, recs, dirp) for kk in ("v2", "v2r") if segs and kk in segs[0][1]})
    if parity_dir:
        mism = tot = 0
        for e, s in segs:
            f = pathlib.Path(parity_dir) / f"{e}.json"
            if not f.exists(): continue
            ref = {x["start"]: x for x in json.load(open(f)).get("segments", []) if "ws_pass" in x}
            if s["start"] not in ref: continue
            x = ref[s["start"]]; tot += 1
            mism += int((x["first_fail"] is None) != s["v1_pass"])
        out["v1_parity_vs_offline_check"] = dict(segments=tot, mismatches=mism)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--group", action="append", required=True); ap.add_argument("--parity-offline"); ap.add_argument("--out", required=True)
    ap.add_argument("--eval-ref", default="r150full", choices=["r150full", "heldout120"], help="read at import time (reference arrays)")
    a = ap.parse_args(); res = {}
    for gspec in a.group:
        name, dirp = gspec.split("=", 1); res[name] = summarize(dirp, a.parity_offline if name.startswith("old") else None)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
