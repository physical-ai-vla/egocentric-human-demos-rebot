#!/usr/bin/env python3
"""Figures for the technical report. All numbers are read from results/ JSON (or the analysis outputs), never typed in,
except the collection-time order counts / funnel which are cited from results/phase3_census.json + episode metadata."""
import json, pathlib, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.image import imread
R = pathlib.Path(__file__).resolve().parents[2]; F = pathlib.Path(__file__).resolve().parent
SEL = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else json.load(open(R / "results/selection.json"))
PS = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42})
# ---------------- Fig. 2: dataset design ----------------
fig = plt.figure(figsize=(7.16, 3.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.2], hspace=0.55, wspace=0.38)
axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
for a, t in zip(axs, (1, 8, 22)):
    a.imshow(imread(F / f"headcam_BPR_t{t}s.jpg")); a.set_axis_off(); a.set_title(f"t = {t} s", fontsize=7, color=INK2, pad=2)
axs[0].text(-0.02, 1.18, "(a) head-camera view, order BPR (held-out episode)", transform=axs[0].transAxes, fontsize=7.5, color=INK)
orders = ["RBP", "RPB", "BRP", "BPR", "PRB", "PBR"]
main = dict(RBP=51, RPB=50, BRP=50, BPR=48, PRB=48, PBR=47); early = dict(RBP=13, RPB=12, BRP=11, BPR=8, PRB=7, PBR=4)
used = dict(PBR=36, BPR=38, RBP=48, PRB=43, RPB=46, BRP=48); val = dict(PBR=5, BPR=3, RBP=5, PRB=3, RPB=6, BRP=4)
ax = fig.add_subplot(gs[1, 0:2]); x = np.arange(6); w = 0.38
ax.bar(x - w / 2 - 0.01, [main[o] for o in orders], w, color=BLUE, label="collected, main (294)")
ax.bar(x - w / 2 - 0.01, [early[o] for o in orders], w, bottom=[main[o] for o in orders], color="#86b6ef", label="collected, early (55)")
ax.bar(x + w / 2 + 0.01, [used[o] - val[o] for o in orders], w, color=ORANGE, label="used, train (233)")
ax.bar(x + w / 2 + 0.01, [val[o] for o in orders], w, bottom=[used[o] - val[o] for o in orders], color="#f4a582", label="used, val (26)")
ax.set_xticks(x, orders); ax.set_ylabel("episodes"); ax.set_xlabel("stacking order (bottom $\\to$ top; R=red, B=blue, P=purple)")
ax.legend(ncol=4, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.42), handlelength=1.2, columnspacing=0.8); ax.set_ylim(0, 70)
ax.text(-0.07, 1.12, "(b) stacking orders", transform=ax.transAxes, fontsize=7.5)
ax = fig.add_subplot(gs[1, 2])
st = [("episodes", 349), ("sync $\\leq$20 ms", 284), ("$\\geq$1 bimanual\nsegment", 259), ("train split", 233)]
ax.barh(range(4)[::-1], [s[1] for s in st], color=[INK2, INK2, BLUE, ORANGE], height=0.62)
for i, (n, v) in enumerate(st): ax.text(v + 6, 3 - i, str(v), va="center", fontsize=7, color=INK)
ax.set_yticks(range(4)[::-1], [s[0] for s in st], fontsize=6.5); ax.set_xlim(0, 420); ax.set_xlabel("episodes"); ax.grid(axis="y", visible=False)
ax.text(-0.62, 1.12, "(c) QA funnel", transform=ax.transAxes, fontsize=7.5)
fig.savefig(F / "fig2_dataset.pdf", bbox_inches="tight"); plt.close(fig)
# ---------------- Fig. 4: evaluation / failure analysis ----------------
T = SEL["table"]; steps = np.array(sorted(int(s) for s in T)); g = lambda key: np.array([T[f"{s:06d}"][key] for s in steps])
ncol = 3 if PS else 2
fig, axs = plt.subplots(1, ncol, figsize=(7.16, 1.75), gridspec_kw=dict(wspace=0.42))
ax = axs[0]; k = steps / 1e3
ax.plot(k, g("geo"), color=BLUE, lw=1.5, label="all samples (pre-registered)")
ax.plot(k, g("mgeo_tcp"), color=ORANGE, lw=1.5, label="moving samples ($\\geq$20 mm)")
ax.axvline(160, color=INK2, lw=0.6, ls=":"); ax.text(163, 26, "selected\n(160k)", fontsize=6, color=INK2, va="top")
ax.set_xlabel("pretraining step (k)"); ax.set_ylabel("geo score: mean FK TCP err. (mm)"); ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
ax.set_title("(a) held-out error vs. training", fontsize=7.5, loc="left")
ax = axs[1]
ax.plot(k, g("k30_mae_tcp"), color=ORANGE, lw=1.5, label="policy, moving")
ax.plot(k, g("k30_zero_tcp"), color=INK2, lw=1.2, ls="--", label="zero-motion baseline, moving")
ax.set_xlabel("pretraining step (k)"); ax.set_ylabel("joint MAE at k=30 (deg)"); ax.set_ylim(0, 10); ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
ax.set_title("(b) 1-s horizon vs. no-motion", fontsize=7.5, loc="left")
if PS:
    ax = axs[2]; KS = (1, 4, 8, 16, 30); kk = np.array(KS); M = lambda key: [PS[f"k{q}"]["moving"][key] for q in KS]
    ax.plot(kk, M("pred_disp_mm_p50"), color=INK2, lw=1.2, marker="s", ms=3, label="predicted displacement")
    ax.plot(kk, M("prompt_spread_mm_p50"), color=ORANGE, lw=1.5, marker="o", ms=3.5, label="spread over 6 order prompts")
    ax.plot(kk, M("noise_spread_mm_p50"), color=BLUE, lw=1.2, ls="--", marker="o", ms=3, label="spread over 3 noise seeds")
    ax.set_yscale("log"); ax.set_xlabel("horizon k (steps of 1/30 s)"); ax.set_ylabel("median, moving samples (mm)")
    ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=1); ax.set_title("(c) prompt swap", fontsize=7.5, loc="left")
fig.savefig(F / "fig4_eval.pdf", bbox_inches="tight"); plt.close(fig)
print("ok")
