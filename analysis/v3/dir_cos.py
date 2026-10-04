"""[2026-10-02] direction test: per-arm cosine between predicted and GT action translation (CURRENT-TCP body frame, k4/k8/k16)
on R312c HEAD180 frames (teacher-forced, same frames + noise for every checkpoint). usage: dir_cos.py <ckpt> [<ckpt> ...]"""
import os, sys, time
import numpy as np, torch
ckpts = sys.argv[1:]
os.environ.setdefault("N_FRAMES", "120"); os.environ.setdefault("V4_DTYPE", "bf16"); os.environ.setdefault("V4_DENOISE_STEPS", "5")
src = open(os.path.expanduser("~/umi_bridge/rel16_audit/compare_ckpt_offline.py")).read()
src = src[:src.index('print(f"{N} HEAD180 frames')]
sys.argv = [sys.argv[0], ckpts[0]]; g = {"__name__": "x"}; exec(compile(src, "cco", "exec"), g)
GT = g["GT"]
for ck in ckpts:
    P = g["run"](ck); name = os.path.basename(ck.rstrip("/"))[8:60]
    out = []
    for arm, o in (("L", 0), ("R", 10)):
        for k in (4, 8, 16):
            p = P[:, k - 1, o:o + 3]; t = GT[:, k - 1, o:o + 3]
            n_t = np.linalg.norm(t, axis=1); m = n_t > 0.01                     # GT moves > 10 mm
            c = np.sum(p[m] * t[m], 1) / (np.linalg.norm(p[m], axis=1) * n_t[m] + 1e-9)
            out.append(f"{arm}k{k} cos med {np.median(c):+.2f} mean {c.mean():+.2f} neg {np.mean(c<0)*100:.0f}% (n {m.sum()})")
        z = np.sign(P[:, 7, o + 2]) == np.sign(GT[:, 7, o + 2]); out.append(f"{arm} k8 z-sign agree {z.mean()*100:.0f}%")
    print(f"## {name}"); [print("   " + x) for x in out]
