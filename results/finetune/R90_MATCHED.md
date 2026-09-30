# R90 matched-step comparison — 2026-09-30 08:52:21

scratch = xvla-base init · C-old = legacy-v1 ego pretrain 300k-final init · everything else identical (code hash, R90 list, seed, recipe, first batch).
Eval = probe val-10 (8 seen in R90 train + 2 held-out, eps 66/77). Lower is better except cosines. `Δ` = (C-old − scratch)/scratch.

Evaluated: scratch 110k · C-old 20k · matched steps: 2

## Curve (scratch / C-old / Δ)

| step | motion geo | geo (all) | motion k30 FK p90 mm | motion k30 dir cos | motion k30 MAE deg | seen8 motion geo | heldout2 motion geo (n=2) |
|---|---|---|---|---|---|---|---|
| 10k | 53.43 / **54.30** / +2% | 23.52 / **15.87** / -33% | 239.7 / **221.6** / -8% | 0.915 / **0.909** / -1% | 7.16 / **7.52** / +5% | 48.87 / **47.66** / -2% | 94.73 / **93.40** / -1% |
| 20k | 35.26 / **39.68** / +13% | 12.44 / **11.13** / -11% | 178.5 / **166.9** / -6% | 0.955 / **0.960** / +0% | 6.48 / **6.05** / -7% | 31.55 / **33.61** / +7% | 75.47 / **86.43** / +15% |

## Per-step detail

### 10k

| metric | scratch | C-old | Δ | better |
|---|---|---|---|---|
| geo (all) | 23.52 | 15.87 | -32.5% | C-old |
| motion geo | 53.43 | 54.30 | +1.6% | scratch |
| k30 FK p50 mm | 28.8 | 21.1 | -26.7% | C-old |
| k30 FK p90 mm | 162.6 | 172.9 | +6.3% | scratch |
| motion k30 FK p50 mm | 66.6 | 64.4 | -3.3% | C-old |
| motion k30 FK p90 mm | 239.7 | 221.6 | -7.6% | C-old |
| motion k30 dir cos | 0.915 | 0.909 | -0.6% | scratch |
| motion k30 action cos | 0.855 | 0.849 | -0.7% | scratch |
| motion k30 MAE deg | 7.16 | 7.52 | +4.9% | scratch |
| collapse | 0.86 | 0.80 | -7.0% |  |
| seen8 geo | 21.08 | 14.19 | -32.7% | C-old |
| seen8 motion geo | 48.87 | 47.66 | -2.5% | C-old |
| heldout2 geo (n=2) | 38.07 | 34.79 | -8.6% | C-old |
| heldout2 motion geo (n=2) | 94.73 | 93.40 | -1.4% | C-old |

### 20k

| metric | scratch | C-old | Δ | better |
|---|---|---|---|---|
| geo (all) | 12.44 | 11.13 | -10.5% | C-old |
| motion geo | 35.26 | 39.68 | +12.5% | scratch |
| k30 FK p50 mm | 17.0 | 16.3 | -4.6% | C-old |
| k30 FK p90 mm | 122.9 | 119.1 | -3.1% | C-old |
| motion k30 FK p50 mm | 46.4 | 51.8 | +11.7% | scratch |
| motion k30 FK p90 mm | 178.5 | 166.9 | -6.5% | C-old |
| motion k30 dir cos | 0.955 | 0.960 | +0.5% | C-old |
| motion k30 action cos | 0.891 | 0.909 | +2.1% | C-old |
| motion k30 MAE deg | 6.48 | 6.05 | -6.7% | C-old |
| collapse | 0.85 | 0.90 | +5.0% |  |
| seen8 geo | 10.73 | 10.29 | -4.1% | C-old |
| seen8 motion geo | 31.55 | 33.61 | +6.5% | scratch |
| heldout2 geo (n=2) | 23.34 | 23.57 | +1.0% | scratch |
| heldout2 motion geo (n=2) | 75.47 | 86.43 | +14.5% | scratch |

