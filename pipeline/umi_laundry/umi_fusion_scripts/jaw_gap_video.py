#!/usr/bin/env python3
"""[2026-10-13] Geometric jaw opening from the wrist video (no learning): in a band of rows near the bottom of the frame,
find the two dark jaw silhouettes either side of the centre and measure the gap between their INNER edges (px). The white tape
on the inner faces keeps the two jaws separable when they touch (closed gap = tape width, small and constant).
Validates against the servo (calibration-direction-aware g, binary open = g >= 0.25).
usage: jaw_gap_video.py <episode> [<episode> ...] [--plot out.png]
"""
import argparse, json, pathlib, sys
import cv2, numpy as np
sys.path.insert(0, str(pathlib.Path.home() / "ego_cart20"))
from ego_cart20.sources.hfold_export import gripper_binary, read_mcap  # noqa: E402

DARK = 70


def jaw_gap(img):
    """-> gap px (at 1920 width) or nan. Rows 80-97 % height; per row, dark runs within the centre 40 %; inner edges of the
    nearest dark run left and right of the image centre."""
    h, w = img.shape[:2]; s = 1920.0 / w
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaps = []
    for r in np.linspace(0.80 * h, 0.97 * h, 8).astype(int):
        row = g[r, int(0.30 * w):int(0.70 * w)] < DARK; c = len(row) // 2
        left = np.flatnonzero(row[:c]); right = np.flatnonzero(row[c:])
        if len(left) < 5 or len(right) < 5: continue
        gaps.append((c + right[0]) - left[-1])                       # inner edge of left jaw .. inner edge of right jaw
    return float(np.median(gaps)) * s if len(gaps) >= 3 else np.nan


def run(ep, side):
    fm, gr = read_mcap(ep); vf = fm[f"{side}_wrist"]; b, gn, info = gripper_binary(gr[side], side, vf[:, 1], ep=ep)
    cap = cv2.VideoCapture(str(ep / f"{side}_wrist.mp4")); gap = []; i = 0
    while True:
        ok, img = cap.read()
        if not ok or i >= len(gn): break
        gap.append(jaw_gap(img)); i += 1
    gap = np.array(gap); gn = gn[:len(gap)]; b = b[:len(gap)]; ok = np.isfinite(gap)
    r = float(np.corrcoef(gap[ok], gn[ok])[0, 1]) if ok.sum() > 20 else np.nan
    # threshold: the gap value that best separates servo open/closed (reported, plus a fixed rule for transfer)
    best = max(((float((((gap[ok] > th) == (b[ok] > 0.5))).mean()), th) for th in np.percentile(gap[ok], np.arange(5, 96, 1))), default=(np.nan, np.nan))
    return dict(episode=f"{ep.parent.name}/{ep.name}", side=side, detect=round(float(ok.mean()), 3), corr_gap_vs_servo=round(r, 3),
                best_acc=round(best[0], 3), best_thr_px=round(best[1], 1), gap_closed_p50=round(float(np.nanmedian(gap[ok & (b < 0.5)])), 1),
                gap_open_p50=round(float(np.nanmedian(gap[ok & (b > 0.5)])), 1) if (ok & (b > 0.5)).any() else None), gap, gn


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("episodes", nargs="+"); ap.add_argument("--plot"); a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figs = []
    for e in a.episodes:
        ep = pathlib.Path(e)
        for side in ("left", "right"):
            r, gap, gn = run(ep, side); print(json.dumps(r)); figs.append((r, gap, gn))
    if a.plot:
        fig, axs = plt.subplots(len(figs), 1, figsize=(11, 2.2 * len(figs)), sharex=False)
        for ax, (r, gap, gn) in zip(np.atleast_1d(axs), figs):
            t = np.arange(len(gap)) / 30; ax.plot(t, gn, color="#2a78d6", lw=1.2, label="servo g (0 closed, 1 open)")
            ax2 = ax.twinx(); ax2.plot(t, gap, color="#eb6834", lw=1, label="video jaw gap (px)")
            ax.set_title(f"{r['episode']} {r['side']}: corr {r['corr_gap_vs_servo']}, best acc {r['best_acc']}", loc="left", fontsize=9)
            ax.set_ylabel("servo g"); ax2.set_ylabel("gap px")
        fig.tight_layout(); fig.savefig(a.plot, dpi=120); print("plot", a.plot)
