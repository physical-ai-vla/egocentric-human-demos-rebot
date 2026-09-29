# r90ft600k — 2026-09-28 17:42:43

Ray job `c8old-r90ft-09281448` on 4090 GPU 1: **RUNNING**  
current step 36000 · R120 = r150_nested_subset_v1.json sha256 8996a20fc88a… (sets 0-19) · init pretrain 300k-final md5 004974bb… — R90 (fresh, 2026-09-28) · terminal ckpt 300k  
Eval: probe val-10 (all10) — seen8 = in R120 train, heldout2 = episodes 66, 77 (in R150, not R120). Transfer diagnostic, n tiny for heldout2.

| step | all10 geo | all10 motion geo | k30 FK p50 | motion k30 dir cos | motion k30 act cos | motion k30 MAE (zero) | seen8 motion geo | heldout2 geo | heldout2 motion geo | collapse |
|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 15.87 | 54.30 | 21.1 | 0.909 | 0.849 | 7.52 (10.94) | 47.66 | 34.79 | 93.40 | 0.80 |
| 20k | 11.13 | 39.68 | 16.3 | 0.960 | 0.909 | 6.05 (10.94) | 33.61 | 23.57 | 86.43 | 0.90 |
