#!/usr/bin/env python3
"""[2026-09-26] C-old checkpoint selection, TWO selectors side by side (contract §24(3) + 2026-09-26 POST-HOC addendum).
  best_geo_pre_registered   : min geo_score (all samples)            — the pre-registered rule, unchanged
  best_motion_gated_posthoc : min motion_tcp.geo_score (moving subset, geometry target TCP moves >= 20 mm at k30) — POST-HOC,
                              defined 2026-09-26 after the 10k-60k results; tie-break higher motion_tcp k30 dir_cos
Guards reported for both: collapse_std_ratio_k30 must not drop below 0.5; joint MAE vs zero-action = auxiliary.
Reads pretrain300k/eval_gated/*.json (retro, 10k-60k) and pretrain300k/eval/*.json (70k+, evaluator with gating). Writes
c8old_runs/<run>/selection.json. Usage: c8old_selection.py [pretrain300k|r150ft600k]"""
import json, pathlib, sys
R = pathlib.Path.home() / "c8/c8old_runs" / (sys.argv[1] if len(sys.argv) > 1 else "pretrain300k"); rows = {}
for d in ("eval", "eval_gated"):
    for f in sorted((R / d).glob("*.json")):
        j = json.load(open(f))
        if "motion_tcp" in j or f.stem not in rows: rows[f.stem] = j
print(f"{'ckpt':>7} {'geo':>6} {'mgeo_tcp':>9} {'mgeo_jnt':>9} {'k30 cos tcp':>11} {'k30 fk50 tcp':>12} {'k30 mae/zero tcp':>17} {'collapse':>8}")
tab = {}
for s, j in sorted(rows.items()):
    mt, mj = j.get("motion_tcp"), j.get("motion_joint")
    tab[s] = dict(geo=j["geo_score"], collapse=j["collapse_std_ratio_k30"], mgeo_tcp=mt and mt["geo_score"], mgeo_joint=mj and mj["geo_score"],
                  k30_cos_tcp=mt and mt["k30"]["dir_cos_p50"], k30_fk50_tcp=mt and mt["k30"]["fk_mm_p50"],
                  k30_mae_tcp=mt and mt["k30"]["joint_mae_deg"], k30_zero_tcp=mt and mt["k30"]["zero_action_mae_deg"])
    t = tab[s]; f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"
    print(f"{s:>7} {f(t['geo']):>6} {f(t['mgeo_tcp']):>9} {f(t['mgeo_joint']):>9} {f(t['k30_cos_tcp'],3):>11} {f(t['k30_fk50_tcp'],1):>12} {f(t['k30_mae_tcp']):>8}/{f(t['k30_zero_tcp']):<8} {f(t['collapse'],3):>8}")
ok = {s: t for s, t in tab.items() if t["collapse"] >= 0.5}
bg = min(ok, key=lambda s: ok[s]["geo"]) if ok else None
okm = {s: t for s, t in ok.items() if t["mgeo_tcp"] is not None}
bm = min(okm, key=lambda s: (round(okm[s]["mgeo_tcp"], 2), -okm[s]["k30_cos_tcp"])) if okm else None
out = dict(best_geo_pre_registered=bg, best_motion_gated_posthoc=bm, rule_pre_registered="min geo_score (all samples)",
           rule_posthoc="min motion_tcp.geo_score (TCP >= 20 mm at k30), tie-break max k30 dir_cos; POST-HOC 2026-09-26", guard="collapse_std_ratio_k30 >= 0.5", table=tab)
json.dump(out, open(R / "selection.json", "w"), indent=1)
print(f"best_geo_pre_registered = {bg}   |   best_motion_gated_posthoc = {bm}")
