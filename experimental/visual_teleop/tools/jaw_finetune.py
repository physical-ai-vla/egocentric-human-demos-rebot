"""Fine-tune ImageNet ResNet18 as the jaw OPEN / CLOSED classifier (2026-10-16). Pixel / HOG / frozen-ResNet linear probes stayed at
0.64 live / 0.90 cycle-held-out (desk / floor scenes failed); the jaws are rigid to the cameras (after the 15:40 bracket re-seat), so
a small fine-tuned CNN on the fixed ROI of BOTH views generalises across backgrounds. Input: left crop on top of right crop (each
224 x 112, gray -> 3 ch), augmentation = brightness / contrast / gamma / small shift / blur. Phases: first --skip-s dropped.
    .venv/bin/python -m visual_teleop.tools.jaw_finetune cap1.npz cap2.npz ... [--drop 1:2] [--val 0:3 2:6] [--epochs 6]
    -> configs/calibration/jaw_resnet18_v001.pt (TorchScript, input 1x3x224x224 float, output logit open)"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None], np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


_MASKS: dict = {}


def roi_masks(shape) -> tuple:
    """10-16 jaw-sweep self-masks (configs/calibration/stereo_selfmask_{left,right}_960.npy, the jaws open AND closed) mapped into the
    jaw_capture ROI crop: pixels outside are set to grey so the classifier only sees the jaws and the gap between them (whole-ROI models
    flipped with the background: 0.80-0.85 live on the new mount)."""
    if shape not in _MASKS:
        from visual_teleop.tools.jaw_capture import ROI, DS
        out = []
        for side in ("left", "right"):
            p = ROOT / f"configs/calibration/stereo_selfmask_{side}_960.npy"
            if not p.exists(): out.append(None); continue
            m = cv2.resize(np.load(p).astype(np.uint8), (800, 450), interpolation=cv2.INTER_NEAREST)[ROI]
            out.append(cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool))
        _MASKS[shape] = tuple(out)
    return _MASKS[shape]


def to_input(l: np.ndarray, r: np.ndarray) -> np.ndarray:
    """ROI crops (gray uint8) -> jaw-sweep mask (outside = 128) -> 3 x 224 x 224 normalised."""
    ml, mr = roi_masks(l.shape)
    if ml is not None: l = np.where(ml, l, 128).astype(np.uint8)
    if mr is not None: r = np.where(mr, r, 128).astype(np.uint8)
    x = np.vstack([cv2.resize(l, (224, 112), interpolation=cv2.INTER_AREA), cv2.resize(r, (224, 112), interpolation=cv2.INTER_AREA)]).astype(np.float32) / 255
    return (np.repeat(x[None], 3, 0) - MEAN) / STD


def augment(l, r, rng):
    out = []
    for v in (l, r):
        v = v.astype(np.float32); a = rng.uniform(0.6, 1.4); b = rng.uniform(-40, 40); v = np.clip(v * a + b, 0, 255)
        v = 255 * (v / 255) ** rng.uniform(0.7, 1.4)
        dx, dy = rng.integers(-6, 7, 2); v = cv2.warpAffine(v, np.float32([[1, 0, dx], [0, 1, dy]]), (v.shape[1], v.shape[0]), borderMode=cv2.BORDER_REFLECT)
        if rng.random() < 0.3: v = cv2.GaussianBlur(v, (5, 5), rng.uniform(0.5, 1.5))
        out.append(v.astype(np.uint8))
    return out


def load(paths, skip_s, drop, hz=8.0):
    L, R, Y, G = [], [], [], []
    for gi, p in enumerate(paths):
        z = np.load(p); y = z["label"]; n = len(y); per = n // (2 * int(z["cycles"])); k = np.arange(n) % per; cyc = (np.arange(n) // per) // 2
        keep = (k >= int(skip_s * hz)) & np.array([np.std(l) > 8 for l in z["left"]]) & ~np.isin(cyc, [c for g, c in drop if g == gi])
        L += list(z["left"][keep]); R += list(z["right"][keep]); Y += list(y[keep]); G += list(gi * 100 + cyc[keep])
    return L, R, np.array(Y), np.array(G)


def main() -> None:
    import torchvision
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+"); ap.add_argument("--drop", nargs="*", default=[]); ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--skip-s", type=float, default=2.0)
    ap.add_argument("--out", default=str(ROOT / "configs/calibration/jaw_resnet18_v001.pt")); a = ap.parse_args()
    pair = lambda s: (int(s.split(":")[0]), int(s.split(":")[1]))
    L, R, Y, G = load(a.captures, a.skip_s, {pair(d) for d in a.drop}); val = np.isin(G, [g * 100 + c for g, c in map(pair, a.val)])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"; rng = np.random.default_rng(0); torch.manual_seed(0)
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1); net.fc = torch.nn.Linear(512, 1); net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); tr = np.where(~val)[0]; lossf = torch.nn.BCEWithLogitsLoss()
    Xv = torch.tensor(np.stack([to_input(L[i], R[i]) for i in np.where(val)[0]])) if val.any() else None
    print(f"[jaw-ft] 학습 {len(tr)} / 검증 {int(val.sum())} 프레임, {dev}", flush=True)
    for ep in range(a.epochs):
        net.train(); rng.shuffle(tr); tot = 0.0
        for b in range(0, len(tr), 32):
            idx = tr[b:b + 32]; x = torch.tensor(np.stack([to_input(*augment(L[i], R[i], rng)) for i in idx])).to(dev); y = torch.tensor(Y[idx], dtype=torch.float32).to(dev)
            loss = lossf(net(x).squeeze(1), y); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        msg = f"[jaw-ft] epoch {ep + 1}: loss {tot / len(tr):.3f}"
        if Xv is not None:
            net.eval()
            with torch.no_grad(): p = torch.cat([net(Xv[i:i + 64].to(dev)).squeeze(1).cpu() for i in range(0, len(Xv), 64)]).numpy() > 0
            yv, gv = Y[val], G[val]; msg += f" | 검증 정확도 {np.mean(p == yv):.3f} " + str({int(g): round(float(np.mean(p[gv == g] == yv[gv == g])), 2) for g in sorted(set(gv))})
        print(msg, flush=True)
    net.eval().cpu(); ts = torch.jit.trace(net, torch.zeros(1, 3, 224, 224)); ts.save(a.out); print(f"[jaw-ft] 저장 {a.out}")


if __name__ == "__main__":
    main()
