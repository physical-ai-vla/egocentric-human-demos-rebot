#!/usr/bin/env python3
"""[2026-10-14] Per-frame visual jaw opening for every wrist frame (servos removed 10-13) -> <episode>/derived/jaw/jaw_{side}.npz

v2 (part selection). v1 (= jaw_redx v4: all red pixels split at the centre column) picked up the OTHER unit's orange
tape, stray red specks and, when one tape was hidden, a few specks on the empty half (160900 ep16: 535-629 px while closed).
  1. red mask (jaw_redx.red_mask) in rows 10-100 %, cols 20-80 %; every connected component (>= MIN_FRAC of the frame) is
     split at the centre column into a left part and a right part (closed jaws merge into one blob);
  2. per episode-side reference from 'clean' frames (exactly one part per half): median tip height y_ref and area a_ref;
  3. per frame and half, candidates with |y - y_ref| <= BAND_PX and area in [AREA_LO, AREA_HI] * a_ref; of several, the
     one nearest the previous pick among the tape-sized ones (>= BIG * a_ref) when any exist, the largest when there
     is no previous pick; both halves -> d_px (px at 1920 width).
  The band is deliberately loose: the wrist camera is NOT rigid to the jaws (mount flexes: 112920 ep33 left tips at y 437
  and 635 in one take), so a tight band / x range / y(x) arc dropped 10-60 % of left-wrist frames. The band only has to
  reject the other unit's tape (~430 px above the tips); continuity rejects desk specks (144936 ep3 f965: 584 -> 119 px).
Works for the solid tape (UMIalign 15:xx on) and the older red X marks (144219/144936) alike.
Saves the candidate parts too (re-select with --reselect, no video decode). Binary open/closed: ego_cart20/sources/jaw_visual.py.
usage: jaw_track.py <episode|session> ... [-j 6] [--force | --reselect]"""
import argparse, json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

import cv2, numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); sys.path.insert(0, str(pathlib.Path.home() / "ego_cart20"))
from jaw_redx import red_mask                                                        # noqa: E402
from ego_cart20.sources.hfold_export import gripper_binary, read_mcap                # noqa: E402

ROI = (0.10, 0.20, 0.80)          # y0, x0, x1 as fractions of the frame
MIN_FRAC = 1e-4                   # smallest part kept (a red X is ~3e-4 of a 1080p frame, a tape ~2e-3)
BAND_PX, AREA_LO, AREA_HI, BIG = 320.0, 0.1, 4.0, 0.5


def parts(img):
    """-> list of (half 0/1, cx, cy, area) in 1920-width px, one per component half."""
    h, w = img.shape[:2]; y0, x0, x1 = int(ROI[0] * h), int(ROI[1] * w), int(ROI[2] * w); s = 1920.0 / w
    m = red_mask(img[y0:, x0:x1]); n, lab, st, _ = cv2.connectedComponentsWithStats(m); out = []; cx = w / 2 - x0
    for k in range(1, n):
        if st[k, 4] < MIN_FRAC * h * w: continue
        ys, xs = np.nonzero(lab[st[k, 1]:st[k, 1] + st[k, 3], st[k, 0]:st[k, 0] + st[k, 2]] == k); xs = xs + st[k, 0]; ys = ys + st[k, 1]
        for half, sel in ((0, xs < cx), (1, xs >= cx)):
            if sel.sum() >= 0.3 * MIN_FRAC * h * w:
                out.append((half, (xs[sel].mean() + x0) * s, (ys[sel].mean() + y0) * s, float(sel.sum()) * s * s))
    return out


def select(P, n):
    """P: (k, 5) rows (frame, half, cx, cy, area) -> d_px (n,), ref dict."""
    d = np.full(n, np.nan); ref = {}
    if not len(P): return d, ref
    f, hf = P[:, 0].astype(int), P[:, 1].astype(int)
    cnt = np.zeros((n, 2), int); np.add.at(cnt, (f, hf), 1); clean = (cnt[:, 0] == 1) & (cnt[:, 1] == 1)
    for half in (0, 1):
        m = (hf == half) & clean[f]
        if m.sum() < 30:                                                  # other unit's tape in view most of the take (144936 ep12):
            big = {}                                                      # reference = the largest part per frame (own tape is nearer)
            for i in np.flatnonzero(hf == half):
                if f[i] not in big or P[i, 4] > P[big[f[i]], 4]: big[f[i]] = i
            m = np.zeros(len(P), bool); m[list(big.values())] = True
            if m.sum() < 30: return d, ref
        ref[half] = (float(np.median(P[m, 3])), float(np.median(P[m, 4])))
    best = np.full((n, 2, 2), np.nan)                                     # chosen (cx, cy) per frame & half
    for half in (0, 1):
        yr, ar = ref[half]
        ok = (hf == half) & (np.abs(P[:, 3] - yr) <= BAND_PX) & (P[:, 4] >= AREA_LO * ar) & (P[:, 4] <= AREA_HI * ar)
        cand = {}
        for i in np.flatnonzero(ok): cand.setdefault(f[i], []).append(P[i, 2:5])
        prev = None                                                       # temporal continuity: nearest to the last pick
        for k in range(n):
            c = cand.get(k)
            if not c: continue
            c = np.array(c)
            if (c[:, 2] >= BIG * ar).any(): c = c[c[:, 2] >= BIG * ar]   # [10-15] a tape-sized part beats a speck: continuity alone kept
            j = int(np.argmin(np.linalg.norm(c[:, :2] - prev, axis=1))) if prev is not None and len(c) > 1 else int(np.argmax(c[:, 2]))  # a speck at the closed spot (112743 ep11 R f535 open read 59 px)
            best[k, half] = prev = c[j, :2]
    d = np.linalg.norm(best[:, 1] - best[:, 0], axis=1)
    return d, {("left_half", "right_half")[k]: dict(y_ref=round(v[0], 1), area_ref=round(v[1], 1)) for k, v in ref.items()}


def one(job):
    ep, side, force, reselect = job; ep = pathlib.Path(ep); out = ep / "derived/jaw" / f"jaw_{side}.npz"
    if reselect:
        z = dict(np.load(out, allow_pickle=True)); d, ref = select(z["parts"], len(z["t_ns"]))
        z.update(d_px=d, ref=json.dumps(ref)); np.savez_compressed(out, **z)
        return f"resel {ep.parent.name}/{ep.name} {side}: detect {np.isfinite(d).mean():.3f} {ref}"
    if out.exists() and not force and "parts" in np.load(out).files: return f"skip {ep.parent.name}/{ep.name} {side}"
    fm, gr = read_mcap(ep); vf = fm[f"{side}_wrist"]; cap = dict(zip(vf[:, 0].tolist(), vf[:, 1].tolist()))
    vc = cv2.VideoCapture(str(ep / f"{side}_wrist.mp4")); fr, P, i = [], [], 0
    while True:
        ok, img = vc.read()
        if not ok: break
        if i in cap:
            k = len(fr); fr.append(i); P += [(k, *p) for p in parts(img)]
        i += 1
    fr = np.array(fr, np.int64); t = np.array([cap[k] for k in fr], np.int64); P = np.array(P, float).reshape(-1, 5)
    d, ref = select(P, len(fr))
    g = np.zeros(0)
    if len(gr[side]):
        try: g = gripper_binary(gr[side], side, t, ep=ep)[1]
        except AssertionError: g = np.zeros(0)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, video_frame=fr, t_ns=t, d_px=d, parts=P, ref=json.dumps(ref), servo_g=g, n_video_frames=i,
                        tool="jaw_track v2 tip-band (ego_collector/scripts/umi_fusion/jaw_track.py)")
    return f"ok   {ep.parent.name}/{ep.name} {side}: {len(d)} frames, detect {np.isfinite(d).mean():.3f} {ref}" + (", servo" if len(g) else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("-j", type=int, default=6)
    ap.add_argument("--force", action="store_true"); ap.add_argument("--reselect", action="store_true"); a = ap.parse_args(); eps = []
    for p in map(pathlib.Path, a.paths):
        for ep in ([p] if (p / "episode_meta.json").exists() else sorted(p.glob("episode_*"))):
            if not (ep / ".complete").exists(): continue
            if json.loads((ep / "episode_meta.json").read_text()).get("exclude_from_training"): continue
            eps += [(str(ep), s, a.force, a.reselect) for s in ("left", "right")]
    print(f"{len(eps)} sides", flush=True)
    with ProcessPoolExecutor(a.j) as ex:
        for r in ex.map(one, eps): print(r, flush=True)
