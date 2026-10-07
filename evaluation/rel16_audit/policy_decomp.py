#!/usr/bin/env python3
"""[2026-09-29] Per-checkpoint policy-error decomposition on a v2_ckpt_eval npz (teacher-forced, "real" condition), L/R separate.
Position error of the target the controller would get: |T_t Â_k − T_t A_k| (translation) = |p̂_k − p_k| (R_t is orthonormal),
so no FK is needed.
  err(k)        single-draw error, k = 1..16, p50/p90                            (all samples, and arm-moving |GT k16| > 20 mm)
  bias(k16)     error of the draw MEAN (systematic part)
  disp(k16)     median over draw pairs ||p̂_i − p̂_j|| (sampling part)  -- D_draw
Appends one row per call to <csv>. Thresholds fixed here: moving = own-arm |GT A16 pos| > 20 mm.
usage: policy_decomp.py <v2eval npz> <run label> <step> <csv>
"""
import csv, os, sys
import numpy as np

npz, run, step, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
d = np.load(npz, allow_pickle=True); n = int(d["done"])
gt, pr = d["gt"][:n], d["pred_real"][:n]                          # (n,16,20), (n,K,16,20)
row = dict(run=run, step=step, n=n, draws=pr.shape[1])
print(f"[policy decomp] {run} {step}  n {n} x {pr.shape[1]} draws  (mm; moving = own arm |GT A16| > 20 mm)")
for ai, arm in ((0, "L"), (1, "R")):
    o = ai * 10
    g = gt[:, :, o:o + 3] * 1000; p = pr[:, :, :, o:o + 3] * 1000
    e = np.linalg.norm(p - g[:, None], axis=-1)                    # (n,K,16)
    mv = np.linalg.norm(g[:, 15], axis=-1) > 20
    eb = np.linalg.norm(p.mean(1) - g, axis=-1)                    # (n,16)
    K = p.shape[1]
    disp = np.array([np.median([np.linalg.norm(p[i, a, 15] - p[i, b, 15]) for a in range(K) for b in range(a + 1, K)]) for i in range(n)]) if K > 1 else np.zeros(n)
    curve = [float(np.median(e[:, :, k])) for k in range(16)]
    # [2026-09-29] direction cosine of the predicted vs GT displacement at k4 / k8 / k16, own-arm moving samples (|GT k| > 5 mm)
    for kk in (3, 7, 15):
        gk = g[:, kk]; pk = p[:, :, kk]; mk = np.linalg.norm(gk, axis=-1) > 5
        cs = np.sum(pk * gk[:, None], -1) / np.maximum(np.linalg.norm(pk, axis=-1) * np.linalg.norm(gk, axis=-1)[:, None], 1e-9)
        row[f"{arm}_cos_k{kk + 1}_moving"] = float(np.median(cs[mk])) if mk.any() else float("nan")
    curve_mv = [float(np.median(e[mv][:, :, k])) if mv.any() else float("nan") for k in range(16)]
    row.update({f"{arm}_k16_p50": np.median(e[:, :, 15]), f"{arm}_k16_p90": np.percentile(e[:, :, 15], 90),
                f"{arm}_k16_p50_moving": np.median(e[mv][:, :, 15]) if mv.any() else np.nan, f"{arm}_n_moving": int(mv.sum()),
                f"{arm}_bias_k16_p50": np.median(eb[:, 15]), f"{arm}_bias_k16_p90": np.percentile(eb[:, 15], 90),
                f"{arm}_disp_k16_med": np.median(disp), f"{arm}_disp_k16_med_moving": np.median(disp[mv]) if mv.any() else np.nan,
                f"{arm}_err_k_p50": " ".join(f"{x:.1f}" for x in curve)})
    print(f"  {arm}: k16 p50 {row[f'{arm}_k16_p50']:.1f} p90 {row[f'{arm}_k16_p90']:.1f} | moving (n {mv.sum()}) p50 {row[f'{arm}_k16_p50_moving']:.1f} | "
          f"bias(draw-mean) p50 {row[f'{arm}_bias_k16_p50']:.1f} p90 {row[f'{arm}_bias_k16_p90']:.1f} | D_draw med {row[f'{arm}_disp_k16_med']:.1f} "
          f"(moving {row[f'{arm}_disp_k16_med_moving']:.1f})")
    print(f"     cos(pred, GT) moving: k4 {row[f'{arm}_cos_k4_moving']:.2f} k8 {row[f'{arm}_cos_k8_moving']:.2f} k16 {row[f'{arm}_cos_k16_moving']:.2f}")
    print(f"     err(k) p50 k1..16: " + " ".join(f"{x:.0f}" for x in curve))
    print(f"     err(k) p50 moving: " + " ".join(f"{x:.0f}" for x in curve_mv))
new = not os.path.exists(out)
out = out.replace(".csv", "_v2.csv")   # [2026-09-29] new columns (cos k4/k8/k16) -> new file; the old CSV keeps its header
new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader()
    w.writerow({k: (round(float(v), 2) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
