#!/usr/bin/env python3
"""[2026-10-12] Figures of today's UMI pose work -> ~/Desktop/Hfold_review_20261012/viz/ (PNG).
1 MASt3R validity timelines   2 Orbbec laundry_v1 tracker coverage   3 frame-level MASt3R x Orbbec overlap (152646 ep2)
4 IMU rotation-bridge error vs gap length   5 Orbbec tracks (top view + height above table)."""
import csv, json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcap.reader import make_reader

H = pathlib.Path.home(); RAW = H / "ego_collector/datasets/human_handumi_raw"; RUNS = H / "c8/robotlike/runs"
OUT = H / "Desktop/Hfold_review_20261012/viz"; OUT.mkdir(parents=True, exist_ok=True)
C_M3, C_OB, C_EI = "#2a78d6", "#eb6834", "#1baf7a"          # reference categorical slots 1-3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
EPS = {"143136": RAW / "Hfold/Hfold_20261012_143136/episode_000001", "152646": RAW / "Hfold/Hfold_20261012_152646/episode_000002",
       "163312": RAW / "Hfold/Hfold_20261012_163312/episode_000001", "163800": RAW / "Hfold/Hfold_20261012_163800/episode_000001",
       "170552": RAW / "UMIpilot/UMIpilot_20261012_170552/episode_000001"}
TAG = {"143136": "20261012_143136_000001", "152646": "20261012_152646_000002", "170552": "20261012_170552_000001"}


def m3_valid(tag, side):
    r = list(csv.DictReader(open(RUNS / f"{tag}_{side}" / "ss1.csv"))); return np.array([x["is_lost"] == "false" for x in r])


def frame_meta(ep):
    fm = {}
    with open(ep / "sensors.mcap", "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages():
            if ch.topic.endswith("/frame_meta"):
                d = json.loads(msg.data); fm.setdefault(ch.topic.split("/")[1], []).append((d["video_frame"], d["capture_ns"]))
    return {k: np.array(sorted(v), np.int64) for k, v in fm.items()}


def spans(ax, y, t, v, color, h=0.6):
    v = np.asarray(v, bool); d = np.diff(np.r_[0, v.astype(int), 0]); st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
    dt = np.median(np.diff(t)) if len(t) > 1 else 1 / 30
    ax.broken_barh([(t[s], t[e - 1] - t[s] + dt) for s, e in zip(st, en)], (y - h / 2, h), facecolors=color)


def longest_gap(v, dt):
    v = np.asarray(v, bool); best = cur = 0
    for x in v: cur = 0 if x else cur + 1; best = max(best, cur)
    return best * dt


# 1 -------------------------------------------------------------- MASt3R validity
fig, ax = plt.subplots(figsize=(11, 3.6)); rows = []
for k, tag in TAG.items():
    for side in ("left", "right"):
        v = m3_valid(tag, side); t = np.arange(len(v)) / 30.0; y = len(rows); rows.append(f"{k} {side}")
        ax.broken_barh([(0, t[-1])], (y - 0.3, 0.6), facecolors=GRID); spans(ax, y, t, v, C_M3)
        ax.text(t[-1] + 1, y, f"{100 * v.mean():.0f}%  (valid 0–{np.flatnonzero(v)[-1] / 30:.1f} s, never recovers)", va="center", color=INK2, fontsize=9)
ax.set_yticks(range(len(rows)), rows); ax.set_xlabel("time in episode (s)"); ax.set_xlim(0, 160); ax.grid(axis="y", visible=False)
ax.set_title("MASt3R tracking validity (blue = tracked). Every side tracks from the start, loses once, and stays in RELOC", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "1_mast3r_validity.png", dpi=150); plt.close(fig)

# 2 -------------------------------------------------------------- Orbbec tracker coverage
fig, ax = plt.subplots(figsize=(11, 3.6)); rows = []
for k in ("152646", "163312", "163800"):
    z = np.load(EPS[k] / "derived/body_track/track.npz", allow_pickle=True); t = (z["t_ns"] - z["t_ns"][0]) / 1e9; dt = float(np.median(np.diff(t)))
    for side in ("left", "right"):
        v = z[f"valid_{side}"]; y = len(rows); rows.append(f"{k} {side}")
        ax.broken_barh([(0, t[-1])], (y - 0.3, 0.6), facecolors=GRID); spans(ax, y, t, v, C_OB)
        ax.text(t[-1] + 1, y, f"{100 * v.mean():.0f}%   longest gap {longest_gap(v, dt):.1f} s", va="center", color=INK2, fontsize=9)
ax.set_yticks(range(len(rows)), rows); ax.set_xlabel("time in episode (s)"); ax.set_xlim(0, 160); ax.grid(axis="y", visible=False)
ax.set_title("Orbbec laundry_v1 tracker: frames with a unit detected (orange)  — tuning episodes, held-out pending", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "2_orbbec_coverage.png", dpi=150); plt.close(fig)

# 3 -------------------------------------------------------------- frame-level overlap on the wrist clock
ep = EPS["152646"]; fm = frame_meta(ep); z = np.load(ep / "derived/body_track/track.npz", allow_pickle=True)
fig, ax = plt.subplots(figsize=(11, 3.2)); rows = []; summ = []
for side in ("left", "right"):
    m3 = m3_valid(TAG["152646"], side); wf = fm[f"{side}_wrist"]; n = min(len(m3), len(wf)); tw = wf[:n, 1]
    j = np.clip(np.searchsorted(z["t_ns"], tw), 1, len(z["t_ns"]) - 1); j = np.where(np.abs(z["t_ns"][j - 1] - tw) < np.abs(z["t_ns"][j] - tw), j - 1, j)
    ob = z[f"valid_{side}"][j] & (np.abs(z["t_ns"][j] - tw) < 50e6); m = m3[:n]; t = (tw - tw[0]) / 1e9
    for lab, v, c in ((f"{side}  MASt3R", m, C_M3), (f"{side}  Orbbec", ob, C_OB), (f"{side}  either", m | ob, C_EI)):
        y = len(rows); rows.append(lab); ax.broken_barh([(0, t[-1])], (y - 0.3, 0.6), facecolors=GRID); spans(ax, y, t, v, c)
        ax.text(t[-1] + 1, y, f"{100 * v.mean():.0f}%  gap {longest_gap(v, 1 / 30):.1f} s", va="center", color=INK2, fontsize=9)
    summ.append(f"{side}: both {100 * (m & ob).mean():.0f}%")
ax.set_yticks(range(len(rows)), rows); ax.set_xlabel("time on the wrist-camera clock (s)"); ax.set_xlim(0, 150); ax.grid(axis="y", visible=False)
ax.set_title("Hfold 15:26 ep2: per-frame union of MASt3R and Orbbec position\n(rotation in Orbbec-only frames would come from the IMU bridge)   " + ", ".join(summ), loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "3_overlap_152646.png", dpi=150); plt.close(fig)

# 4 -------------------------------------------------------------- IMU bridge
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for side, c in (("left", C_M3), ("right", C_OB)):
    d = json.load(open(RUNS / f"{TAG['143136']}_{side}" / "imu_bridge.json"))["dt"]; dts = [float(k) for k in d]
    p50 = [d[k]["kalibr"]["p50"] for k in d]; p90 = [d[k]["kalibr"]["p90"] for k in d]
    ax.plot(dts, p50, "-o", color=c, lw=2, ms=6, label=f"{side} p50"); ax.plot(dts, p90, "--o", color=c, lw=2, ms=6, mfc=SURF, label=f"{side} p90")
ax.axhspan(0, 5, color="#0ca30c", alpha=0.06); ax.axhspan(5, 10, color="#fab219", alpha=0.08)
ax.axvline(0.5, color=INK2, lw=1, ls=":"); ax.axvline(3.0, color=INK2, lw=1, ls=":")
ax.text(0.52, 9.3, "PASS ≤ 0.5 s", color=INK2, fontsize=9); ax.text(3.02, 9.3, "> 3 s invalid", color=INK2, fontsize=9)
ax.set_xscale("log"); ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 3], ["0.1", "0.25", "0.5", "1", "2", "3"]); ax.set_ylim(0, 10)
ax.set_xlabel("gyro-only bridge length Δt (s)"); ax.set_ylabel("rotation error vs MASt3R (deg)"); ax.legend(frameon=False, ncol=2, fontsize=9)
ax.set_title("IMU rotation bridge: error grows with time from the last MASt3R anchor (14:31 ep)", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "4_imu_bridge.png", dpi=150); plt.close(fig)

# 5 -------------------------------------------------------------- Orbbec tracks
z = np.load(EPS["163800"] / "derived/body_track/track.npz", allow_pickle=True); t = (z["t_ns"] - z["t_ns"][0]) / 1e9
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1, 1.4]))
for side, c in (("left", C_M3), ("right", C_OB)):
    v = z[f"valid_{side}"]; p = np.where(v[:, None], z[f"xyz_{side}"], np.nan)
    a1.plot(p[:, 0], p[:, 1], "-", color=c, lw=1.2, label=side); a2.plot(t, np.where(v, z[f"height_{side}"], np.nan) * 100, "-", color=c, lw=1.5, label=side)
a1.set_aspect("equal"); a1.set_xlabel("camera x (m)"); a1.set_ylabel("camera y (m)"); a1.invert_yaxis(); a1.legend(frameon=False)
a1.set_title("top view of the two unit tracks (16:38 ep)", loc="left", fontsize=11)
a2.set_xlabel("time (s)"); a2.set_ylabel("height above table (cm)"); a2.legend(frameon=False); a2.set_title("unit height above the table plane", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "5_orbbec_tracks_163800.png", dpi=150); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))

# 6 -------------------------------------------------------------- restart_v1 vs old runner
RR = H / "c8/robotlike/runs_restart"; tags = sorted(p.name for p in RR.iterdir() if (p / "ss1.csv").exists())
fig, ax = plt.subplots(figsize=(12, 0.42 * len(tags) + 1.4)); rows = []
for tag in tags:
    r = list(csv.DictReader(open(RR / tag / "ss1.csv"))); v = np.array([x["is_lost"] == "false" for x in r]); sg = np.array([int(x["segment"]) for x in r])
    t = np.arange(len(v)) / 30.0; y = len(rows); rows.append(tag.replace("20261012_", "").replace("_00000", " ep"))
    ax.broken_barh([(0, t[-1])], (y - 0.3, 0.6), facecolors=GRID); spans(ax, y, t, v, C_M3)
    for b in np.flatnonzero(np.diff(sg) != 0): ax.plot([t[b + 1]] * 2, [y - 0.36, y + 0.36], color=INK, lw=0.8)
    old = RUNS / tag / "ss1.csv"; o = np.mean([x["is_lost"] == "false" for x in csv.DictReader(open(old))]) if old.exists() else None
    ax.text(t[-1] + 1, y, (f"{100 * o:.0f}% → " if o is not None else "new → ") + f"{100 * v.mean():.0f}%   {len(set(sg) - {-1})} segments", va="center", color=INK2, fontsize=9)
ax.set_yticks(range(len(rows)), rows); ax.set_xlabel("time in episode (s)"); ax.set_xlim(0, 160); ax.grid(axis="y", visible=False)
ax.set_title("MASt3R restart_v1: tracked frames (blue), segment restarts (black ticks), old runner → restart coverage. Longest gap 0.6 s everywhere", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "6_mast3r_restart_v1.png", dpi=150); plt.close(fig)
print("wrote 6")
