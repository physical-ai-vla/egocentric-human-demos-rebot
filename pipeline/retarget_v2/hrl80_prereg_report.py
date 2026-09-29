#!/usr/bin/env python3
"""[2026-09-28] HRL80 vs old259 report exactly as pre-registered (hrl80_interpretation_prereg.json + amendment_1). No new metric.
  size-matched: old259 TR segments subsampled to N_HRL80_TR (numpy default_rng(seed) seeds 0..19, without replacement), the SAME
                v2k_compare.manifold() on each subsample -> median / min-max; HRL80 read by its position in that distribution
  T0: exact reproduction of the old259 -0.009 (live over_frac_ang_w vs v1 offline-check full-pass fraction, argsort ranks)
  T1: U_e = TR segments / candidate (v1-record) segments per episode in the master; scipy spearmanr (average ties)
Usage: ~/xvla-mac/bin/python hrl80_prereg_report.py --eval-ref heldout120"""
import glob, json, pathlib, sys
import numpy as np
from scipy.stats import spearmanr
H = pathlib.Path.home(); V2 = H / "umi_bridge/track_c/v2k"; RL = H / "c8/robotlike"
sys.path.insert(0, str(V2)); import v2k_compare as C                     # reads --eval-ref heldout120 at import
RAW = H / "ego_collector/datasets/human_handumi_raw/HRL80/HRL80_20260928_101010"


def tr_segments(master):
    Q, S = [], []
    for p in sorted(master.glob("2*.json")):
        r = json.load(open(p)); f = master / f"{r['episode']}.npz"
        if not f.exists(): continue
        z = np.load(f)
        for s in r["segments"]:
            if isinstance(s.get("v2"), dict) and s["v2"]["retarget_method"] == "TR": q = z[f"{s['start']}_v2_0"]; Q.append(q); S.append(q[0])
    return Q, S


def u_e(master):
    out = {}
    for p in sorted(master.glob("2*.json")):
        r = json.load(open(p)); c = [s for s in r["segments"] if "v1" in s]
        if c: out[r["episode"]] = sum(s["v2"]["retarget_method"] == "TR" for s in c) / len(c)
    return out


def live(d):
    o = {}
    for p in glob.glob(f"{d}/*.json"):
        s = json.load(open(p)); sess, ep = pathlib.Path(p).stem.split("__"); eid = f"{sess.split('_', 1)[1]}_{ep.split('_', 1)[1]}"
        o[eid] = dict(w=max(s["sides"][x]["over_frac_ang_w"] for x in ("left", "right")), a=max(s["sides"][x]["over_frac_ang_a"] for x in ("left", "right")), g=s["grasps_total"])
    return o


def argsort_rank_corr(x, y): return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


rep = {}
# ---- posture / dynamics, size-matched
Qo, So = tr_segments(RL / "master_old259_frozen"); Qn, Sn = tr_segments(RL / "hybrid_hrl80"); N = len(Qn)
keys = ["cluster_coverage", "start_distinct_clusters", "start_pairwise_rad_mean", "nn_dist_rad_p50", "nn_dist_rad_p90",
        "W1_dq_vs_heldout120", "W1_ddq_vs_heldout120", "W1_dq_vs_R30", "W1_ddq_vs_R30", "dq_p95"]
hr = C.manifold(Qn, Sn); full_old = C.manifold(Qo, So); subs = []
for seed in range(20):
    idx = np.random.default_rng(seed).choice(len(Qo), N, replace=False); subs.append(C.manifold([Qo[i] for i in idx], [So[i] for i in idx]))
rep["size_matched"] = dict(N_HRL80_TR=N, N_old259_TR=len(Qo), rows={k: dict(HRL80=hr[k], old259_full=full_old[k], old259_N_matched_median=float(np.median([s[k] for s in subs])),
                        old259_N_matched_min=float(min(s[k] for s in subs)), old259_N_matched_max=float(max(s[k] for s in subs)),
                        HRL80_position=("above max" if hr[k] > max(s[k] for s in subs) else "below min" if hr[k] < min(s[k] for s in subs) else "inside range")) for k in keys})
# ---- transfer
Lo, Ln = live(RL / "baseline_live"), live(RL / "hrl80_live")
t0 = {}
for name, L, get in (("old259", Lo, lambda e: RL / "baseline_offline" / f"{e}.json"),
                     ("HRL80", Ln, lambda e: RAW / f"episode_{e.rsplit('_', 1)[1]}" / "derived/robot_like/offline_check.json")):
    xs, ys = [], []
    for e, v in L.items():
        f = get(e)
        if not f.exists(): continue
        segs = [g for g in json.load(open(f)).get("segments", []) if "ws_pass" in g]
        if segs: xs.append(v["w"]); ys.append(np.mean([g["first_fail"] is None for g in segs]))
    t0[name] = dict(n=len(xs), argsort_rank_corr=round(argsort_rank_corr(np.array(xs), np.array(ys)), 3), spearman_avg_ties=round(float(spearmanr(xs, ys).statistic), 3))
rep["T0_v1_offline_fullpass_fraction"] = t0
Uo, Un = u_e(RL / "master_old259_frozen"), u_e(RL / "hybrid_hrl80"); t1 = {}
for name, L, U in (("old259", Lo, Uo), ("HRL80", Ln, Un), ("pooled", {**Lo, **Ln}, {**Uo, **Un})):
    ee = [e for e in U if e in L]; row = dict(n=len(ee), U_mean=round(float(np.mean([U[e] for e in ee])), 3))
    for k, lab in (("w", "live_rot_speed_over_p95 (primary)"), ("a", "live_rot_acc_over_p95"), ("g", "grasps_total")):
        r = spearmanr([L[e][k] for e in ee], [U[e] for e in ee]); row[lab] = dict(rho=round(float(r.statistic), 3), p=round(float(r.pvalue), 3))
    row["live_rot_speed_median"] = round(float(np.median([L[e]["w"] for e in ee])), 3); t1[name] = row
rep["T1_TR_usable_fraction_U_e"] = t1
(RL / "hrl80_prereg_report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
