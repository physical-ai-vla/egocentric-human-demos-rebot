#!/usr/bin/env python3
"""Figures for the v3 sections of the report (Cartesian ego pretraining, transfer, HRA_red). All values are read from
results/v3/ files. Run from anywhere: python3 report/figures/make_figures_v3.py"""
import csv, glob, json, pathlib
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parents[2]; F = pathlib.Path(__file__).resolve().parent; V = R / "results/v3"
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42})


def curve(name):
    r = list(csv.DictReader(open(V / "loss_curves" / f"{name}.csv")))
    return np.array([int(x["step"]) for x in r]), np.array([float(x["loss"]) for x in r])


def smooth(y, k=5):
    return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


# ---------------- Fig. 5: ego initialisation vs scratch on R312c (training loss) ----------------
sa, la = curve("A60_scratch_R312c_150kdecay"); sb, lb = curve("B60_egoinit_R312c_150kdecay")
n = min(len(sa), len(sb)); sa, la, sb, lb = sa[:n], la[:n], sb[:n], lb[:n]
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.0), gridspec_kw={"wspace": 0.3})
ax[0].plot(sa / 1e3, smooth(la), color=INK2, lw=1, label="scratch (xvla-base)")
ax[0].plot(sb / 1e3, smooth(lb), color=BLUE, lw=1, label="ego-pretrained init (50k)")
ax[0].set_yscale("log"); ax[0].set_xlabel("fine-tuning step (k)"); ax[0].set_ylabel("training loss (REL, 5-pt mean)")
ax[0].legend(fontsize=6.5); ax[0].set_title("(a) R312c fine-tuning loss, same data and schedule", fontsize=7.5, loc="left")
ax[1].plot(sa / 1e3, smooth(lb / la, 15), color=BLUE, lw=1); ax[1].axhline(1, color=INK2, lw=0.6, ls="--")
ax[1].set_xlabel("fine-tuning step (k)"); ax[1].set_ylabel("loss ratio ego-init / scratch (15-pt mean)"); ax[1].set_ylim(0, 1.2)
ax[1].set_title("(b) the head start shrinks to a tie by ~60k", fontsize=7.5, loc="left")
fig.savefig(F / "fig5_ego_init_loss.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig. 6: metric scale from a cube of known size ----------------
hrl = [r for r in json.load(open(V / "hra_red/hrl_val_1face.json")) if r["valid"]]
hra = [json.load(open(p)) for p in sorted(glob.glob(str(V / "hra_red/scale_qc_per_episode/*.json")))]
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw={"wspace": 0.3})
x = np.array([r["s_imu"] for r in hrl]); y = np.array([r["s_pnp"] for r in hrl])
lim = [0, 0.8]; inside = (x <= lim[1]) & (y <= lim[1]); ratio = np.median(y / x)
ax[0].scatter(x[inside], y[inside], s=7, color=BLUE); ax[0].plot(lim, lim, color=INK2, lw=0.6, ls="--")
if (~inside).any(): ax[0].text(0.03, 0.92, f"{(~inside).sum()} off-scale (PnP > 0.8)", transform=ax[0].transAxes, fontsize=6.5, color=INK2)
ax[0].set_xlim(lim); ax[0].set_ylim(lim); ax[0].set_xlabel("IMU visual-inertial scale"); ax[0].set_ylabel("cube-PnP scale")
ax[0].set_title(f"(a) HRL80 takes, {len(hrl)} PnP-valid: median PnP/IMU {ratio:.2f}", fontsize=7.5, loc="left")
v = [r for r in hra if r.get("valid")]
ax[1].scatter([r["s_imu"] for r in v], [r["s_pnp"] for r in v], s=6, color=ORANGE)
ax[1].axvline(0, color=INK2, lw=0.6, ls="--"); ax[1].set_xlabel("IMU visual-inertial scale"); ax[1].set_ylabel("cube-PnP scale")
ax[1].set_title(f"(b) HRA_red slow approaches ({len(v)} PnP-valid): IMU scale unobservable", fontsize=7.5, loc="left")
fig.savefig(F / "fig6_cube_scale.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote fig5_ego_init_loss.pdf, fig6_cube_scale.pdf")
