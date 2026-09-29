#!/usr/bin/env python3
"""[2026-09-28] Build the hybrid master (hybrid_frozen.json): TR top-1 if TR has a feasible rollout, else Kr top-1.

Reads ~/c8/robotlike/tr_old (TR, key v2) and ~/c8/robotlike/r30_old (v1 + Kr, key v2r) -- the same 1,229 segments.
Writes ~/c8/robotlike/hybrid_master/<episode>.json + .npz in the v2k_retarget layout (key "v2" = the hybrid choice, "v1"
kept for comparison), so v2k_compare.py reads it unchanged. No rollout is recomputed.
"""
import collections, glob, json, pathlib, sys
import numpy as np

H = pathlib.Path.home(); R = H / "c8/robotlike"
# paths only (rule unchanged): HM_TR / HM_KR / HM_OUT, defaults = the old259 inputs / master
import os; TR, KR, OUT = (pathlib.Path(os.environ.get(k, R / d)) for k, d in (("HM_TR", "tr_old"), ("HM_KR", "r30_old"), ("HM_OUT", "hybrid_master")))
OUT.mkdir(exist_ok=True); cnt = collections.Counter(); reasons = collections.Counter()
for p in sorted(glob.glob(str(TR / "*.json"))):
    e = pathlib.Path(p).stem; t = json.load(open(p)); k = json.load(open(KR / f"{e}.json"))
    zt = np.load(TR / f"{e}.npz") if (TR / f"{e}.npz").exists() else None; zk = np.load(KR / f"{e}.npz") if (KR / f"{e}.npz").exists() else None
    ks = {s["start"]: s for s in k.get("segments", [])}; rec = dict(episode=e, first_fail=t.get("first_fail"), segments=[], master="hybrid_v1"); q = {}
    for s in t.get("segments", []):
        if "v1" not in s: rec["segments"].append(s); continue
        sk = ks[s["start"]]                                                           # same segment in both runs
        trv, krv = s["v2"], sk["v2r"]
        if trv["pass_"]:
            method, why, src, z, key = "TR", None, trv, zt, "v2"
        elif krv["pass_"]:
            method, why, src, z, key = "Kr_fallback", ("no_workspace_window" if trv["n_ws_candidates"] == 0 else "no_feasible_rollout"), krv, zk, "v2r"
        else:
            method, why, src, z, key = "none", ("no_workspace_window" if trv["n_ws_candidates"] == 0 else "no_feasible_rollout"), trv, None, None
        cnt[method] += 1
        if why: reasons[(method, why)] += 1
        top = src["kept"][:1] if method != "none" else []
        rec["segments"].append(dict(start=s["start"], v1=sk["v1"], v1_pass=sk["v1_pass"],
            v2=dict(pass_=method != "none", kept=top, n_ws_candidates=src["n_ws_candidates"], n_rolled=src["n_rolled"], n_feasible=src["n_feasible"],
                    variant_dist_rad=[], retarget_method=method, fallback_reason=why,
                    tr_window_id=(trv["kept"][0]["seed"] if trv["pass_"] else None), kr_seed_id=(krv["kept"][0]["seed"] if krv["pass_"] else None),
                    tr_feasible=trv["n_feasible"], kr_feasible=krv["n_feasible"])))
        q[f"{s['start']}_v1"] = zk[f"{s['start']}_v1"]
        if method != "none": q[f"{s['start']}_v2_0"] = z[f"{s['start']}_{key}_0"]
    (OUT / f"{e}.json").write_text(json.dumps(rec, indent=1, default=float))
    if q: np.savez_compressed(OUT / f"{e}.npz", **q)
n = sum(cnt.values())
print("segments", n, dict(cnt), "| usable", cnt["TR"] + cnt["Kr_fallback"], f"({(cnt['TR'] + cnt['Kr_fallback']) / n:.3f})",
      "| TR share of usable", f"{cnt['TR'] / max(1, cnt['TR'] + cnt['Kr_fallback']):.3f}")
print("fallback / none reasons:", {f"{a}:{b}": c for (a, b), c in reasons.items()})
