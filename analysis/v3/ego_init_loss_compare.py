#!/usr/bin/env python3
"""Matched comparison of R312c training loss: scratch (A60) vs ego-pretrained init (B60).

Both runs: same R312c REL-only data, same recipe (batch 8, AdamW8bit, bf16, domain slot 20), same seed, same 150k cosine
schedule; both were stopped by us near 60k. Only the initialization differs (and the GPU: 4090 GPU0 vs GPU1).
This is TRAINING loss on the fine-tuning data (one seed per arm). It measures optimisation speed, not generalisation.
writes analysis/out/v3_ego_init_loss.json
"""
import csv, json, pathlib

R = pathlib.Path(__file__).resolve().parents[2]
D = R / "results/v3/loss_curves"
rd = lambda n: {int(r["step"]): float(r["loss"]) for r in csv.DictReader(open(D / f"{n}.csv"))}
A, B = rd("A60_scratch_R312c_150kdecay"), rd("B60_egoinit_R312c_150kdecay")
common = sorted(set(A) & set(B))
out = {"A": "A60_scratch_R312c_150kdecay", "B": "B60_egoinit_R312c_150kdecay", "matched_points": len(common),
       "last_matched_step": common[-1], "step200": {"A": A[200], "B": B[200]}, "windows": []}
for lo, hi in ((0, 5000), (5000, 20000), (20000, 40000), (40000, 60000), (55000, 60000)):
    s = [k for k in common if lo < k <= hi]
    a = sum(A[k] for k in s) / len(s); b = sum(B[k] for k in s) / len(s)
    out["windows"].append({"window": f"{lo // 1000}k-{hi // 1000}k", "n": len(s), "A_mean": round(a, 4), "B_mean": round(b, 4),
                           "B_over_A": round(b / a, 3), "B_lower_points": sum(B[k] < A[k] for k in s)})
(R / "analysis/out").mkdir(exist_ok=True)
json.dump(out, open(R / "analysis/out/v3_ego_init_loss.json", "w"), indent=1)
print(json.dumps(out, indent=1))
