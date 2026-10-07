#!/usr/bin/env python3
"""Recompute the Part II (single-arm approach) numbers from results/v4/hra/ and write results/v4/hra/summary.json.
Also draws report/figures/fig9_hra_val.pdf (held-out vs training-subset loss per checkpoint)."""
import json, pathlib, statistics as st
from collections import Counter
R = pathlib.Path(__file__).resolve().parents[1]; H = R / "results/v4/hra"
J = lambda p: json.load(open(H / p))
out = {}
def val(name):
    d = J(f"val_eval/{name}")
    return {int(k) // 1000: {"val": round(v["val_mean"], 3), "train": round(v["train_seed0"]["mean_batch_loss"], 3),
            "n_val": v["val_seed0"]["n_samples"]} for k, v in sorted(d.items()) if "val_mean" in v}
out["val_v1"] = val("v1_results.json"); out["val_a93_start"] = val("a93_start_results.json"); out["val_a93_origin"] = val("a93_origin_results.json")
ta = {}
for c in ["C", "C4", "C5", "C6", "C7"]:
    d = J(f"table_aware/table_aware_{c}_report.json")
    stt = Counter(v.get("status") for v in d.values())
    acc = [v for v in d.values() if v.get("status") in ("PASS", "PASS-CORRECTED", "PASS_CORRECTED")]
    clr = [v["corrected_min_clear_mm"] for v in acc if isinstance(v.get("corrected_min_clear_mm"), (int, float))]
    hr = Counter(v.get("reason") for v in d.values() if v.get("status") == "HARD REJECT")
    ta[c] = {"episodes": len(d), "status": dict(stt), "accepted": len(acc),
             "clear_p50_mm": round(st.median(clr), 1) if clr else None, "clear_min_mm": round(min(clr), 1) if clr else None,
             "hard_reject_reasons": dict(hr)}
out["table_aware"] = ta
g = J("calib/G_frame_calib.json")
out["G"] = {k: g[k] for k in ("origin_base_mm", "x_heading_base_deg", "baseline_mm", "collinearity_resid_mm") if k in g}
out["G"]["table_z_mm"] = round(1000 * g["table_z_m"], 1); out["G"]["plate_thickness_mm"] = round(1000 * g["plate_thickness_m"], 1)
he = J("calib/robot_right_wrist_handeye.json"); out["handeye_keys"] = list(he)[:30]
out["handeye"] = {k: he[k] for k in he if isinstance(he[k], (int, float, str)) and k not in ("note",)}
json.dump(out, open(H / "summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False)[:5000])

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3))
for key, lab, c in [("val_v1", "HRA_red v1 (15 val eps)", "#6b6b6b")]:
    d = out[key]; xs = sorted(d)
    ax[0].plot(xs, [d[x]["val"] for x in xs], "o-", color=c, ms=3, lw=1, label="held-out")
    ax[0].plot(xs, [d[x]["train"] for x in xs], "s--", color=c, ms=3, lw=1, alpha=.6, label="training subset")
ax[0].set_title("(a) HRA_red v1", fontsize=8)
for key, lab, c in [("val_a93_start", "A: start-anchored", "#1f77b4"), ("val_a93_origin", "B: origin-anchored", "#d62728")]:
    d = out[key]; xs = sorted(d)
    ax[1].plot(xs, [d[x]["val"] for x in xs], "o-", color=c, ms=3, lw=1, label=lab + ", held-out")
    ax[1].plot(xs, [d[x]["train"] for x in xs], "s--", color=c, ms=3, lw=1, alpha=.5, label=lab + ", train")
ax[1].set_title("(b) HRA_A100, 93 episodes (11 val)", fontsize=8)
for a in ax:
    a.set_xlabel("checkpoint (k steps)", fontsize=8); a.set_ylabel("loss", fontsize=8); a.tick_params(labelsize=7); a.legend(fontsize=6, frameon=False); a.set_ylim(0, None)
fig.tight_layout(); fig.savefig(R / "report/figures/fig9_hra_val.pdf", bbox_inches="tight")
