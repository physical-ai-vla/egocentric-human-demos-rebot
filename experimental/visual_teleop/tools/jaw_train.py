"""Train the jaw OPEN / CLOSED classifier from jaw_capture recordings (2026-10-16) -> configs/calibration/jaw_cls_v001.npz.
Features: both ROI crops resized to 70 x 34, only the pixels with the lowest within-class std (the jaws are rigid to the cameras, the
room behind them is not; --mask-q), standardised, L2 logistic regression. The first --skip-s of every phase is dropped (the operator
reacts 1-3 s after the voice cue; first capture: 0.55 -> 0.96 held-out accuracy). Leave-one-cycle-out accuracy is printed before saving.
    .venv/bin/python -m visual_teleop.tools.jaw_train datasets/jaw_cls/<ts>.npz [more.npz] [--out configs/calibration/jaw_cls_v001.npz]"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIZE = (70, 34)


def features(l: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.concatenate([cv2.resize(v, SIZE, interpolation=cv2.INTER_AREA).ravel() for v in (l, r)]).astype(np.float32) / 255


def load(paths, skip_s: float, hz: float = 8.0):
    X, Y, G = [], [], []
    for gi, p in enumerate(paths):
        z = np.load(p); y = z["label"]; n = len(y); per = int(round((len(y) / (2 * int(z["cycles"])))))   # labelled frames per phase
        k_in = np.arange(n) % per; ph = np.arange(n) // per
        keep = k_in >= int(skip_s * hz)
        ok = np.array([np.std(l) > 8 for l in z["left"]])                      # covered / black camera frames are not labelled
        X += [features(l, r) for l, r, k in zip(z["left"], z["right"], keep & ok) if k]; Y += list(y[keep & ok]); G += list(gi * 100 + ph[keep & ok] // 2)
    return np.array(X), np.array(Y), np.array(G)


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+"); ap.add_argument("--out", default=str(ROOT / "configs/calibration/jaw_cls_v001.npz"))
    ap.add_argument("--skip-s", type=float, default=2.0); ap.add_argument("--mask-q", type=float, default=0.25); ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--drop", nargs="*", default=[], help="capture_index:cycle to exclude (operator did not follow the cue; check the montage)"); a = ap.parse_args()
    X, Y, G = load(a.captures, a.skip_s)
    if a.drop:
        bad = np.isin(G, [int(d.split(":")[0]) * 100 + int(d.split(":")[1]) for d in a.drop]); X, Y, G = X[~bad], Y[~bad], G[~bad]
    print(f"[jaw-train] 프레임 {len(Y)} (열림 {int(Y.sum())}), 사이클 {len(set(G))}")
    wstd = np.sqrt(0.5 * X[Y == 1].var(0) + 0.5 * X[Y == 0].var(0)); M = wstd <= np.quantile(wstd, a.mask_q)
    acc = []
    for g in sorted(set(G)):
        tr, te = G != g, G == g; sc = StandardScaler().fit(X[tr][:, M]); m = LogisticRegression(C=a.C, max_iter=5000).fit(sc.transform(X[tr][:, M]), Y[tr])
        acc.append(float((m.predict(sc.transform(X[te][:, M])) == Y[te]).mean()))
    print(f"[jaw-train] 사이클 홀드아웃 정확도 {np.round(acc, 3).tolist()} 평균 {np.mean(acc):.3f}")
    sc = StandardScaler().fit(X[:, M]); m = LogisticRegression(C=a.C, max_iter=5000).fit(sc.transform(X[:, M]), Y)
    z0 = np.load(a.captures[0])
    np.savez(a.out, mask=M, mean=sc.mean_, scale=sc.scale_, coef=m.coef_[0], intercept=m.intercept_[0], size=np.array(SIZE),
             roi=z0["roi"], ds=z0["ds"], cv_acc=np.array(acc), captures=np.array([str(p) for p in a.captures]))
    print(f"[jaw-train] 저장 {a.out}")


if __name__ == "__main__":
    main()
