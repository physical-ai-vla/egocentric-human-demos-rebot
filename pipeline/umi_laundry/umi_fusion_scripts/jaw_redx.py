#!/usr/bin/env python3
"""[2026-10-13] Jaw opening from the RED X marks on the white tape of each jaw's inner face (wrist video, no learning).
Red pixels (HSV hue <10 or >170, S > 90, V > 60) in the lower 55 % / centre 60 % of the frame -> connected blobs ->
the largest blob left of centre and right of centre = the two jaws -> centroid distance (px at 1920 width) = opening.
Compared with the servo (calibration-direction-aware g). usage: jaw_redx.py <episode> [--plot out.png] [--debug out.jpg]"""
import argparse, json, pathlib, sys
import cv2, numpy as np
sys.path.insert(0, str(pathlib.Path.home() / "ego_cart20"))
from ego_cart20.sources.hfold_export import gripper_binary, read_mcap  # noqa: E402


def red_mask(img):
    """[2026-10-13 v3] redness = R - max(G, B): thin X strokes on white tape come out pale/pinkish after compression, so a
    hue+saturation gate misses them; redness > 35 with R > 90 keeps them and still rejects white tape, table, black cloth."""
    b, g, r = (img[..., k].astype(np.int16) for k in range(3))
    m = ((r - np.maximum(g, b)) > 35) & (r > 90)
    m = m.astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def jaw_redx(img):
    """[2026-10-13 v4, solid red tape on both jaw tips] Closed jaws merge the two red patches into ONE blob, so blobs are not
    paired any more: inside the jaw ROI (rows 25-100 %, centre 50 %), keep the red components that are large (the tape, not
    stray orange objects), split the red pixels at the image centre column, and return the distance between the left-half
    and right-half red centroids (px at 1920 width). Closed -> about half a tape width, open -> large."""
    h, w = img.shape[:2]; y0, x0, x1 = int(0.25 * h), int(0.25 * w), int(0.75 * w); s = 1920.0 / w
    m = red_mask(img[y0:, x0:x1]); n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    keep = [k for k in range(1, n) if st[k, 4] >= 0.0008 * h * w]
    if not keep: return np.nan, None
    sel = np.isin(lab, keep); ys, xs = np.nonzero(sel); cx = (x1 - x0) / 2
    L, R = xs < cx, xs >= cx
    if L.sum() < 0.0003 * h * w or R.sum() < 0.0003 * h * w: return np.nan, None
    pl = np.array([xs[L].mean(), ys[L].mean()]); pr = np.array([xs[R].mean(), ys[R].mean()])
    return float(np.linalg.norm(pr - pl)) * s, (pl + [x0, y0], pr + [x0, y0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--plot"); ap.add_argument("--debug"); a = ap.parse_args()
    ep = pathlib.Path(a.episode); fm, gr = read_mcap(ep); out = {}; dbg = []
    for side in ("left", "right"):
        b, gn, _ = gripper_binary(gr[side], side, fm[f"{side}_wrist"][:, 1], ep=ep)
        cap = cv2.VideoCapture(str(ep / f"{side}_wrist.mp4")); d = []; i = 0
        while True:
            ok, img = cap.read()
            if not ok or i >= len(gn): break
            v, pts = jaw_redx(img); d.append(v)
            if a.debug and i in (int(np.argmin(gn[:len(gn) // 2])), int(np.argmax(gn))):
                im = img.copy()
                if pts: [cv2.circle(im, tuple(map(int, p)), 18, (0, 255, 0), 4) for p in pts]
                cv2.putText(im, f"{side} f{i} servo g={gn[i]:.2f} redX d={v:.0f}px", (30, 70), 0, 2, (0, 255, 255), 4); dbg.append(cv2.resize(im, (640, 360)))
            i += 1
        d = np.array(d); gn = gn[:len(d)]; b = b[:len(d)]; ok = np.isfinite(d)
        thr = sorted(((float(((d[ok] > th) == (b[ok] > 0.5)).mean()), th) for th in np.percentile(d[ok], np.arange(5, 96))), reverse=True)[0] if ok.sum() > 20 else (np.nan, np.nan)
        out[side] = dict(detect=round(float(ok.mean()), 3), corr=round(float(np.corrcoef(d[ok], gn[ok])[0, 1]), 3) if ok.sum() > 20 else None,
                         best_acc=round(thr[0], 3), thr_px=round(thr[1], 1), d_closed_p50=round(float(np.nanmedian(d[ok & (b < 0.5)])), 1) if (ok & (b < 0.5)).any() else None,
                         d_open_p50=round(float(np.nanmedian(d[ok & (b > 0.5)])), 1) if (ok & (b > 0.5)).any() else None)
        out[side + "_series"] = (d, gn)
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_series")}))
    if a.debug and dbg: cv2.imwrite(a.debug, np.vstack([np.hstack(dbg[i:i + 2]) for i in range(0, len(dbg) - 1, 2)]))
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(11, 4.6))
        for ax, side in zip(axs, ("left", "right")):
            d, gn = out[side + "_series"]; t = np.arange(len(d)) / 30
            ax.plot(t, gn, color="#2a78d6", lw=1.2, label="servo g"); ax2 = ax.twinx(); ax2.plot(t, d, color="#eb6834", lw=1, label="red-X distance px")
            ax.set_title(f"{side}: corr {out[side]['corr']}, open/closed acc {out[side]['best_acc']}, detect {out[side]['detect']}", loc="left", fontsize=9)
            ax.set_ylabel("servo g (blue)"); ax2.set_ylabel("red-X px (orange)")
        fig.tight_layout(); fig.savefig(a.plot, dpi=120)
