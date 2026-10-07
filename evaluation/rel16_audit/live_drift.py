"""[2026-10-01] Live closed-loop drift per run (user): first-N-cycle signed drift and how fast each arm reaches the edge of its
training support. Runs = contiguous cycles of one ckpt in ~/v4_cycles.jsonl (a cycle counter reset starts a new run).
Support edge (HEAD180 r180_relcart20_rel16_v3d, FK(aux.q_t), 1st/99th pct): R x >= 172 mm, L x >= 184 mm, L y >= -97 mm, R y <= +94 mm.
usage: live_drift.py [N=10] [ckpt substring ...]     (default: last 6 runs)"""
import json, sys, os, numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10; subs = sys.argv[2:]
L = [json.loads(l) for l in open(os.path.expanduser("~/v4_cycles.jsonl")) if l.strip()]
runs, cur = [], []
for x in L:
    if cur and (x["cycle"] <= cur[-1]["cycle"] or x["ckpt"] != cur[-1]["ckpt"]): runs.append(cur); cur = []
    cur.append(x)
runs.append(cur)
runs = [r for r in runs if len(r) >= 3 and (not subs or any(s in r[0]["ckpt"] for s in subs))][-6 if not subs else None:]
EDGE = {("R", 0): (172, +1), ("L", 0): (184, +1), ("L", 1): (-97, +1), ("R", 1): (94, -1)}
print(f"first {N} cycles: mean per-cycle Δ (mm, measured tcp_after - tcp_now), net, cycle of first support exit")
for r in runs:
    a = r[:N]; now = np.array([c["tcp_now_mm"] for c in a]); aft = np.array([c["tcp_after_mm"] or c["tcp_now_mm"] for c in a])
    d = aft - now; tag = r[0]["ckpt"].replace("ckpt_UI_", "")[-48:]
    exits = []
    for (arm, ax), (lim, sgn) in EDGE.items():
        ai = 0 if arm == "L" else 1
        pos = np.array([c["tcp_after_mm"][ai][ax] for c in r if c["tcp_after_mm"]])
        out = np.flatnonzero(sgn * (pos - lim) < 0); exits.append(f"{arm}{'xy'[ax]}@{out[0]}" if len(out) else f"{arm}{'xy'[ax]}:-")
    print(f"{tag:48s} k{r[0]['exec_k']} n{len(r):3d} | L dx{d[:,0,0].mean():+6.1f} dy{d[:,0,1].mean():+6.1f} dz{d[:,0,2].mean():+6.1f} | "
          f"R dx{d[:,1,0].mean():+6.1f} dy{d[:,1,1].mean():+6.1f} dz{d[:,1,2].mean():+6.1f} | exit {' '.join(exits)}")
