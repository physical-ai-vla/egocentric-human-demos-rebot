#!/usr/bin/env python3
"""R30 three-arm fine-tuning comparison (training loss only): scratch vs ego ROBOT100-pretrain 100k / 300k initializations.

Reads results/v4/loss_curves/R30_*.csv (extracted from the Ray job logs with analysis/v3/loss_curves_from_logs.py),
writes results/v4/r30_ablation_summary.json and report/figures/fig8_r30_ablation.pdf.
"""
import csv, json, pathlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]
C = R / "results/v4/loss_curves"
ARMS = {"scratch": "R30_scratch", "ego100k": "R30_init_robot100pt100k", "ego300k": "R30_init_robot100pt300k"}
cur = {a: {int(r["step"]): float(r["loss"]) for r in csv.DictReader(open(C / f"{n}.csv"))} for a, n in ARMS.items()}
common = sorted(set.intersection(*(set(c) for c in cur.values())))
WIN = [(1, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300)]
out = {"note": "training loss (200-step running mean, 3 decimals), one seed per arm, R30 = R312c eps 150-179 (17,600 frames)",
       "last_step": {a: max(c) for a, c in cur.items()}, "common_last_step": common[-1],
       "points": {str(s): {a: cur[a][s] for a in cur} for s in [200, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 300000]},
       "window_mean": {}, "ratio_vs_scratch": {}}
for lo, hi in WIN:
    ss = [s for s in common if lo * 1000 < s <= hi * 1000]
    m = {a: sum(cur[a][s] for s in ss) / len(ss) for a in cur}
    k = f"{lo}k-{hi}k"; out["window_mean"][k] = {a: round(v, 5) for a, v in m.items()}
    out["ratio_vs_scratch"][k] = {a: round(m[a] / m["scratch"], 3) for a in ("ego100k", "ego300k")}
json.dump(out, open(R / "results/v4/r30_ablation_summary.json", "w"), indent=1)

def smooth(y, k=9):
    return [sum(y[max(0, i - k // 2):i + k // 2 + 1]) / len(y[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(y))]
col = {"scratch": "#6b6b6b", "ego100k": "#1f77b4", "ego300k": "#d62728"}
lab = {"scratch": "scratch (base)", "ego100k": "ego pretrain 100k init", "ego300k": "ego pretrain 300k init"}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
for a in cur:
    xs = [x for x in sorted(cur[a]) if x <= 310000]; ax[0].plot([x / 1000 for x in xs], smooth([cur[a][x] for x in xs]), color=col[a], lw=1, label=lab[a])
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("step (k)", fontsize=8); ax[0].set_ylabel("training loss", fontsize=8); ax[0].legend(fontsize=6, frameon=False)
ax[0].set_title("(a) R30 fine-tuning, training loss", fontsize=8)
xs = [s for s in common if s <= 150000]  # beyond ~150k the 3-decimal losses (0.001-0.004) make ratios meaningless
for a in ("ego100k", "ego300k"):
    ax[1].plot([x / 1000 for x in xs], smooth([cur[a][s] / cur["scratch"][s] for s in xs], 25), color=col[a], lw=1, label=lab[a])
ax[1].axhline(1, color="k", lw=0.6, ls="--"); ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.4); ax[1].set_xlabel("step (k)", fontsize=8)
ax[1].set_ylabel("loss ratio vs scratch", fontsize=8); ax[1].set_title("(b) ratio (<1 favours ego init)", fontsize=8)
for x in ax: x.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(R / "report/figures/fig8_r30_ablation.pdf", bbox_inches="tight")
print(json.dumps(out["ratio_vs_scratch"]), out["last_step"])
