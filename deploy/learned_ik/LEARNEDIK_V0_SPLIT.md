# LearnedIK-v0 tuples and split (frozen: LEARNEDIK_V0_SPLIT.json, read-only, sha256 prefix 3c35ff684431)

Source: r180_umi76_rel16_v3d (read-only). 1 tuple = 1 query row = (q_t, REL16×18, Δq16×12).

**Total: 102,523 tuples / 180 episodes.**
- R150 (eps 0–149): 84,923 tuples.
- R30 (eps 150–179): 17,600 tuples.
- Episode length: min 444, median 555, max 866 frames.
- 6 orders × 30 episodes.

Consecutive rows overlap 15/16 of their horizon, so the effective sample count is far below the tuple count.
Stride-1 rows from the 30 Hz source would double the count; they are not used in v0 ("existing data only").

**There is no pre-existing HEAD180 train/val split.** HEAD180-REL16V3 trains on all 180 episodes, so the split below
is new and specific to LearnedIK.

## Selection rule (uses aux.q_t only, no model output, fixed before training)

1. Each joint is binned at the global quartiles, giving each arm a 4⁶ cell id.
2. Greedily pick the episode that adds the most new L+R cells. Ties go to the lowest episode index.
3. P12 caps each order at 2 episodes.

| split | episodes | tuples | q-space cell coverage L / R (of 1098 / 981) |
|---|---|---|---|
| **P12 train (primary)**: 17,148,44,85,132,158,53,67,12,166,21,63 (2 per order; 2 of them R30) | 12 | **7,494** | 421 / 409 |
| P10 train (alternative): 17,148,44,85,132,158,53,179,24,20 (orders unbalanced) | 10 | 6,084 | 403 / 356 |
| reference: first 10 episodes 0..9 | 10 | 6,044 | 259 / 232 |
| val = held-out with ep % 10 == 9 (P12) | 18 | 10,307 | — |
| test = the other held-out (P12) | 150 | 84,722 | — |

Planned ablation: the train set grows 12 → 30 by continuing the same greedy rule. Val and test stay fixed.
