#!/usr/bin/env python3
"""[2026-10-14] End-trim points for takes stopped with SPACE (finish_early): the hand that reaches for the key is cut off.
usage: space_trim_points.py <session> ... -> ./space_trim_points.json (writes nothing into the dataset; trim.json sidecars are written from it)"""
import json, sys, pathlib, numpy as np
from mcap.reader import make_reader
def gyro(ep, side):
    t, w = [], []
    with open(ep/"sensors.mcap","rb") as f:
        for _, ch, m in make_reader(f).iter_messages(topics=[f"/{side}/imu"]):
            d = json.loads(m.data); t.append(d["host_receive_ns"]); w.append([d["gx"],d["gy"],d["gz"]])
    o = np.argsort(t); return np.array(t)[o], np.degrees(np.linalg.norm(np.array(w)[o],axis=1))
res = {}
for S in map(pathlib.Path, sys.argv[1:]):
  for ep in sorted(S.glob("episode_*")):
    if not (ep/".complete").exists(): continue
    m = json.load(open(ep/"episode_meta.json"))
    if m.get("exclude_from_training"): continue
    ev = json.load(open(ep/"events.json")); fe = [e["t_ns"] for e in ev if e.get("detail",{}).get("cue")=="finish_early"]
    t0, t1 = m["t_start_monotonic_ns"], m["t_stop_monotonic_ns"]
    if not fe: print(S.name, ep.name, "no finish_early (ran full time / manual stop)", round((t1-t0)/1e9,1)); continue
    ts = fe[0]; ks = np.arange(-3.5, 0.001, 0.125); win = (ks >= -1.75) & (ks <= -0.25); best = None
    for side in ("left", "right"):
        t, g = gyro(ep, side); prof = []
        for k in ks:
            sel = (t >= ts+(k-0.25)*1e9) & (t < ts+k*1e9); prof.append(np.percentile(g[sel],90) if sel.sum()>5 else np.nan)
        prof = np.array(prof); ip = np.where(win)[0][np.nanargmax(prof[win])]
        if best is None or prof[ip] > best[2]: best = (side, prof, prof[ip], ip)
    side, prof, peak, ip = best
    j = ip
    while j > 0 and ks[j-1] >= -3.0 and not (prof[j] < 0.6*peak and prof[j] <= prof[j-1]): j -= 1
    onset = ks[j]                    # window END time of the quiet bin = start of rise
    cut_rel = min(onset - 0.25, -1.0)
    cut_ns = int(ts + cut_rel*1e9)
    res[f"{S.name}/{ep.name}"] = dict(reach_side=side, space_rel_s=round((ts-t0)/1e9,3), cut_before_space_s=round(-cut_rel,3), peak_dps=round(float(peak)), peak_at_s=float(ks[ip]),
                                      onset_s=float(onset), cut_t_ns=cut_ns, removed_s=round((t1-cut_ns)/1e9,2), kept_s=round((cut_ns-t0)/1e9,2))
    r = res[f"{S.name}/{ep.name}"]; print(S.name[-15:], ep.name[-2:], side[0], f"space@{r['space_rel_s']:.1f}s peak {r['peak_dps']}dps@{r['peak_at_s']} onset {r['onset_s']} -> cut {r['cut_before_space_s']:.2f}s before space, remove {r['removed_s']}s, keep {r['kept_s']}s")
json.dump(res, open("space_trim_points.json","w"), indent=1)
