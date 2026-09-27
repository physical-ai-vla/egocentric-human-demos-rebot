#!/usr/bin/env python3
"""Summarize prompt_swap_<tag>.npz.

Per (sample, arm, horizon k), in FK TCP position space (mm):
  prompt_spread : mean pairwise distance between the 6 order-instruction predictions (same noise seed)
  noise_spread  : mean pairwise distance between the 3 true-instruction predictions (seeds n, n+10000, n+20000)
  err_true      : |pred(true instruction) - measured target|
  err_wrong     : mean over the 5 wrong instructions of |pred - measured target|
  rank_true     : rank (1 = best) of the true instruction among the 6 by error
MOVING = measured target TCP moves >= 20 mm by k30 (same gate as motion_tcp in c8old_mac_eval.py).
Also checks that the true-instruction / seed-n prediction reproduces the reference eval (<ref npz> k*_fk).
Usage: summarize_prompt_swap.py <prompt_swap npz> <reference eval npz> <out json>
"""
import itertools, json, sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True); ref = np.load(sys.argv[2]); OUT = sys.argv[3]
PP, GT, P0, TRUE = z["PP"], z["GT"], z["P0"], z["TRUE"]; N, C = PP.shape[:2]; NT = 6; KS = (1, 4, 8, 16, 30)
def pair_mean(X):   # X [m, 3] -> mean pairwise distance
    return np.mean([np.linalg.norm(X[i] - X[j]) for i, j in itertools.combinations(range(len(X)), 2)])
moving = np.linalg.norm(GT[:, -1] - P0, axis=-1) >= 0.020          # [N, 2]
res = dict(npz=sys.argv[1].rsplit("/", 1)[-1], checkpoint="pretrain300k/" + str(z["checkpoint"]).split("pretrain300k/")[-1], samples=int(N), arm_samples=int(2 * N), moving_arm_samples=int(moving.sum()),
           tasks=[str(t) for t in z["tasks"]])
repro = []
for k in KS:
    i = k - 1; recs = []
    for n in range(N):
        t = TRUE[n]
        for a in (0, 1):
            P = PP[n, :, i, a]; g = GT[n, i, a]
            errs = np.linalg.norm(P[:NT] - g, axis=-1) * 1e3
            recs.append(dict(ps=pair_mean(P[:NT]) * 1e3, ns=pair_mean(P[[t, NT, NT + 1]]) * 1e3, et=errs[t],
                             ew=np.delete(errs, t).mean(), rank=1 + int((errs < errs[t]).sum()), disp=np.linalg.norm(P[t] - P0[n, a]) * 1e3,
                             mv=bool(moving[n, a])))
            repro.append(abs(errs[t] - ref[f"k{k}_fk"][2 * n + a]))
    def agg(rs):
        if not rs: return {}
        f = lambda key: np.array([r[key] for r in rs])
        ps, ns, et, ew, rk, dp = f("ps"), f("ns"), f("et"), f("ew"), f("rank"), f("disp")
        return dict(n=len(rs), prompt_spread_mm_p50=float(np.median(ps)), noise_spread_mm_p50=float(np.median(ns)),
                    prompt_over_noise_p50=float(np.median(ps / np.maximum(ns, 1e-9))), frac_prompt_gt_noise=float(np.mean(ps > ns)),
                    pred_disp_mm_p50=float(np.median(dp)), err_true_mm_p50=float(np.median(et)), err_wrong_mm_p50=float(np.median(ew)),
                    frac_true_better_than_wrong_mean=float(np.mean(et < ew)), rank_true_mean=float(rk.mean()), frac_rank1=float(np.mean(rk == 1)),
                    chance_rank_mean=3.5, chance_frac_rank1=1 / 6)
    res[f"k{k}"] = dict(all=agg(recs), moving=agg([r for r in recs if r["mv"]]))
res["repro_max_abs_diff_mm_vs_reference_eval"] = float(np.max(repro))
json.dump(res, open(OUT, "w"), indent=1)
for k in KS:
    for s in ("all", "moving"):
        a = res[f"k{k}"][s]; print(f"k{k:>2} {s:6} n={a['n']:4} spread prompt {a['prompt_spread_mm_p50']:6.2f} noise {a['noise_spread_mm_p50']:6.2f} "
                                   f"ratio {a['prompt_over_noise_p50']:5.2f} | disp {a['pred_disp_mm_p50']:6.1f} | err true {a['err_true_mm_p50']:6.2f} wrong {a['err_wrong_mm_p50']:6.2f} "
                                   f"| rank {a['rank_true_mean']:.2f} r1 {a['frac_rank1']:.2f}")
print("repro max |diff| mm:", res["repro_max_abs_diff_mm_vs_reference_eval"])
