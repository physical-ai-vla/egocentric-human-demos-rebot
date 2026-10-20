#!/usr/bin/env python3
"""[2026-10-13] Can the jaw state be read from the WRIST VIDEO instead of the Feetech servo (whose shaft keeps breaking)?
Ground truth = servo ticks re-normalised per episode (p2 = closed, calibrated arc), binary open = g >= 0.25 (2 cm of 8 cm),
same rule as the training labels. Input = bottom-centre jaw crop of the wrist frame (grey, 64x32). Models: PCA + logistic
regression (open/closed) and PCA + ridge (g). Evaluation is LEAVE-ONE-SESSION-OUT so no episode of the test session is seen.
Excluded / rotated-camera episodes are skipped; one side can be dropped per episode via --skip ep:side.
usage: jaw_from_video_eval.py --sessions S1 S2 ... [--step 3]
"""
import argparse, json, pathlib, sys
import cv2, numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path.home() / "ego_cart20"))
from ego_cart20.sources.hfold_export import gripper_binary, read_mcap  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--sessions", nargs="+", required=True); ap.add_argument("--step", type=int, default=3)
ap.add_argument("--skip", nargs="*", default=[]); a = ap.parse_args()


def crop(img):
    h, w = img.shape[:2]; c = img[int(0.55 * h):h, int(0.25 * w):int(0.75 * w)]
    g = cv2.cvtColor(cv2.resize(c, (64, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return g.ravel()


X, yb, yg, grp, who = [], [], [], [], []
for S in a.sessions:
    for ep in sorted(pathlib.Path(S).glob("episode_*")):
        if not (ep / ".complete").exists(): continue
        m = json.loads((ep / "episode_meta.json").read_text())
        if m.get("exclude_from_training"): continue
        fm, gr = read_mcap(ep)
        for side in ("left", "right"):
            if f"{ep.parent.name}/{ep.name}:{side}" in a.skip: continue
            vf = fm[f"{side}_wrist"]
            try: b, gn, _ = gripper_binary(gr[side], side, vf[:, 1], ep=ep)
            except AssertionError: continue
            cap = cv2.VideoCapture(str(ep / f"{side}_wrist.mp4")); i = 0
            while True:
                ok = cap.grab()
                if not ok or i >= len(b): break
                if i % a.step == 0:
                    _, img = cap.retrieve(); X.append(crop(img)); yb.append(b[i]); yg.append(gn[i]); grp.append(ep.parent.name); who.append(side)
                i += 1
X = np.array(X); yb = np.array(yb).astype(int); yg = np.array(yg); grp = np.array(grp); who = np.array(who)
print(f"samples {len(X)}  open fraction {yb.mean():.2f}  sessions {sorted(set(grp))}")
res = {}
PER_SIDE = True                                                  # [2026-10-13] separate model per wrist (different cameras, mirrored jaws)
for side_m in (("left", "right") if PER_SIDE else (None,)):
  for test in sorted(set(grp)):
    sm = (who == side_m) if side_m else np.ones(len(grp), bool)
    tr, te = (grp != test) & sm, (grp == test) & sm
    if te.sum() == 0: continue
    if yb[tr].min() == yb[tr].max(): continue
    test_key = f"{side_m}:{test}"
    clf = make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")).fit(X[tr], yb[tr])
    reg = make_pipeline(StandardScaler(), PCA(64, random_state=0), Ridge(alpha=10.0)).fit(X[tr], yg[tr])
    pb = clf.predict(X[te]); pg = np.clip(reg.predict(X[te]), 0, 1)
    r = dict(n=int(te.sum()), acc=round(float((pb == yb[te]).mean()), 3), bal_acc=round(float(balanced_accuracy_score(yb[te], pb)), 3),
             f1_open=round(float(f1_score(yb[te], pb)), 3), g_mae=round(float(np.abs(pg - yg[te]).mean()), 3),
             by_side={s: round(float((pb[who[te] == s] == yb[te][who[te] == s]).mean()), 3) for s in ("left", "right") if (who[te] == s).any()})
    res[test_key] = r; print(f"{side_m:5s} test {test}: acc {r['acc']}  bal_acc {r['bal_acc']}  f1_open {r['f1_open']}  g_mae {r['g_mae']}  n {r['n']}")
json.dump(res, open("/tmp/jaw_from_video_eval.json", "w"), indent=1)
