# r60scratch600k — 2026-09-29 09:02:38

Ray job `c8old-r60scratch-09290144` on 4090 GPU 0: **RUNNING**  
current step None · R120 = r150_nested_subset_v1.json sha256 8996a20fc88a… (sets 0-19) · init lerobot/xvla-base md5 0bed9714… (scratch, B1-old recipe) — R60 · terminal ckpt 300k  
Eval: probe val-10 (all10) — seen8 = in R120 train, heldout2 = episodes 66, 77 (in R150, not R120). Transfer diagnostic, n tiny for heldout2.

| step | all10 geo | all10 motion geo | k30 FK p50 | motion k30 dir cos | motion k30 act cos | motion k30 MAE (zero) | seen8 motion geo | heldout2 geo | heldout2 motion geo | collapse |
|---|---|---|---|---|---|---|---|---|---|---|
