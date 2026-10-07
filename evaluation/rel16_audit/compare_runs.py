"""[2026-09-30 user] compare the latest real-robot run of each checkpoint over its FIRST N cycles (default 10), same conditions:
per arm per-cycle policy step |target - now| p50/max, first-approach direction (sum of the first 3 steps), net x/y/z travel
(TCP now at cycle N minus cycle 0), tracking |after - target| p50, reached rate. Reads ~/v4_cycles.jsonl.
usage: compare_runs.py [N] [ckpt substring ...]     e.g. compare_runs.py 10 V3_155k V3_160k V3_130k
"""
import json, os, sys
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
keys = sys.argv[2:] or ["V3_155k", "V3_160k"]
rows = [json.loads(l) for l in open(os.path.expanduser("~/v4_cycles.jsonl")) if '"tcp_target_mm"' in l]
runs, cur = [], []
for r in rows:
    if cur and (r["cycle"] < cur[-1]["cycle"] or r["ckpt"] != cur[-1]["ckpt"]):
        runs.append(cur); cur = []
    cur.append(r)
if cur:
    runs.append(cur)
import time
for key in keys:
    rr = [x for x in runs if key in x[0]["ckpt"] and len(x) >= 3]
    if not rr:
        print(f"{key}: no run"); continue
    run = rr[-1][:N]; a = run[0]
    print(f"== {key}  {time.strftime('%H:%M:%S', time.localtime(a['t']))}  exec_k {a.get('exec_k')}  IK {a.get('ik_backend_executed')}  "
          f"first {len(run)} cycles (run has {len(rr[-1])})")
    for r_, nm in ((0, "L"), (1, "R")):
        st = np.array([np.subtract(x["tcp_target_mm"][r_], x["tcp_now_mm"][r_]) for x in run])
        tr = np.array([np.subtract((x["tcp_after_mm"] or x["tcp_now_mm"])[r_], x["tcp_target_mm"][r_]) for x in run])
        net = np.subtract((run[-1]["tcp_after_mm"] or run[-1]["tcp_now_mm"])[r_], run[0]["tcp_now_mm"][r_])
        first = st[:3].sum(0)
        print(f"   {nm}: step |d| p50 {np.median(np.linalg.norm(st, axis=1)):5.1f} max {np.linalg.norm(st, axis=1).max():5.1f} mm | "
              f"first-3 dir xyz {np.round(first).astype(int).tolist()} | net xyz {np.round(net).astype(int).tolist()} | "
              f"track p50 {np.median(np.linalg.norm(tr, axis=1)):4.1f} mm")
    print(f"   reached {sum(1 for x in run if x.get('reached'))}/{len(run)}")
