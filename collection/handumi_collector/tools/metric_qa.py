#!/usr/bin/env python3
"""Metric sanity of a SLAM trajectory. Coverage does not imply the poses are metric or in one frame."""
import csv, sys, numpy as np
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r["is_lost"]=="false"]
if not rows: print("  TRACKED NOTHING"); sys.exit()
p=np.array([[float(r["x"]),float(r["y"]),float(r["z"])] for r in rows])
d=np.linalg.norm(np.diff(p,axis=0),axis=1); n=np.linalg.norm(p,axis=1)
ext=p.max(0)-p.min(0)
# a wrist over a ~0.6 m table: extent <1.5 m, per-frame step <0.12 m (=3.6 m/s at 30 Hz)
ok = n.max()<3.0 and np.percentile(d,99)<0.12 and ext.max()<2.0
print(f"  {'PASS' if ok else 'FAIL':<5} n={len(rows):>5}  |p| p50 {np.percentile(n,50):6.2f} max {n.max():9.2f} m"
      f"  extent {ext[0]:.2f}x{ext[1]:.2f}x{ext[2]:.2f} m"
      f"  step p50 {np.percentile(d,50)*1000:6.1f} p99 {np.percentile(d,99)*1000:8.1f} max {d.max()*1000:9.1f} mm")
