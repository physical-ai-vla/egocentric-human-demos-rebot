#!/usr/bin/env python3
"""Count control cycles that each v3 checkpoint executed on the real reBot (from the deployment UI's per-cycle log).

The log has no episode, instruction or outcome field: it shows THAT a checkpoint ran on hardware and how well the IK
tracked its own waypoints, not whether a task succeeded. `starts` = cycles with index 0 (a UI run or step start, not a trial).
usage: hardware_cycles_by_ckpt.py <v4_cycles.jsonl> <out.csv>   (only checkpoints of the v3 runs are kept)"""
import csv, json, re, statistics as st, sys, time
from collections import defaultdict

KEEP = re.compile(r"(R312C-RELCART20-RELONLY-D20-B8|R384-RELCART20-RELONLY-D20-B8|EGO-CART20V2-RELONLY-D20|FT-R312C-FROM-EGOV2BB850K|COTRAIN-R312C-EGOROBOT100|HRA-RIGHTONLY)")
g = defaultdict(list)
for line in open(sys.argv[1]):
    try: r = json.loads(line)
    except ValueError: continue
    c = r.get("ckpt") or ""
    if KEEP.search(c): g[re.sub(r"^ckpt_(UI_)?", "", c)].append(r)
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["checkpoint", "cycles", "starts", "first_local", "last_local", "tcp_err_mm_p50", "ik_residual_mm_p50"])
    for k in sorted(g):
        rs = g[k]; ts = [r["t"] for r in rs]
        def fl(key):
            out = []
            for r in rs:
                v = r.get(key); v = v if isinstance(v, list) else [v]
                out += [x for x in v if isinstance(x, (int, float))]
            return out
        te, ik = fl("tcp_err_mm"), fl("ik_residual_mm")
        w.writerow([k, len(rs), sum(r.get("cycle") == 0 for r in rs), time.strftime("%Y-%m-%d %H:%M", time.localtime(min(ts))),
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(max(ts))), round(st.median(te), 1) if te else "", round(st.median(ik), 1) if ik else ""])
