import os, sys, numpy as np
os.environ.setdefault("N_FRAMES", "120"); os.environ.setdefault("V4_DTYPE", "bf16"); os.environ.setdefault("V4_DENOISE_STEPS", "5")
ck = sys.argv[1]
src = open(os.path.expanduser("~/umi_bridge/rel16_audit/compare_ckpt_offline.py")).read(); src = src[:src.index('print(f"{N} HEAD180 frames')]
sys.argv = [sys.argv[0], ck]; g = {"__name__": "x"}; exec(compile(src, "cco", "exec"), g)
GT = g["GT"]; P = g["run"](ck)
def cos(p, t):
    n = np.linalg.norm(t, axis=1); m = n > 0.01
    return np.median(np.sum(p[m] * t[m], 1) / (np.linalg.norm(p[m], axis=1) * n[m] + 1e-9))
k = 8
for swap in (False, True):
    for flip in ([1,1,1], [1,-1,1], [-1,1,1], [1,1,-1], [-1,-1,1], [-1,-1,-1]):
        f = np.array(flip, float)
        pl = P[:, k-1, 10:13] if swap else P[:, k-1, 0:3]; pr = P[:, k-1, 0:3] if swap else P[:, k-1, 10:13]
        cl = cos(pl * f, GT[:, k-1, 0:3]); cr = cos(pr * f, GT[:, k-1, 10:13])
        print(f"swap L<->R {swap!s:5} flip {flip}: cos L {cl:+.2f}  R {cr:+.2f}")
