# Stage 3 same-layout / conflicting-prompt fixtures (HEAD, frozen 2026-09-28)

Clean HEAD layouts (exactly 3 picks, cubes >= 10 cm apart), ranked by NN margin = distance to the nearest training layout with a DIFFERENT order. Base frame metres, x forward, y left. Positions +-~5 cm.

| layout_id | R(x,y) | B(x,y) | P(x,y) | recorded_order | recorded_first | prompts to run (same layout) | NN margin other-order cm | min cube sep cm | arms |
|---|---|---|---|---|---|---|---|---|---|
| ep074 | (0.15, -0.01) | (0.32, +0.17) | (0.43, +0.09) | BRP | B | BRP / RBP / PBR | 16.3 | 13 | LRL |
| ep153 | (0.17, +0.07) | (0.17, -0.24) | (0.42, +0.02) | BPR | B | BPR / RBP / PBR | 15.9 | 25 | RLR |
| ep017 | (0.30, +0.19) | (0.19, -0.10) | (0.51, +0.10) | PBR | P | PBR / RBP / BPR | 13.6 | 23 | LRL |
| ep026 | (0.26, -0.14) | (0.42, +0.04) | (0.32, +0.17) | BRP | B | BRP / RBP / PBR | 13.2 | 17 | LRL |
| ep037 | (0.22, -0.15) | (0.31, -0.03) | (0.27, +0.17) | RPB | R | RPB / BPR / PBR | 11.8 | 15 | RLR |
| ep090 | (0.24, -0.15) | (0.34, +0.05) | (0.24, +0.09) | RBP | R | RBP / BPR / PBR | 11.8 | 11 | RLR |
| ep091 | (0.19, -0.13) | (0.19, +0.09) | (0.37, +0.03) | RPB | R | RPB / BPR / PBR | 11.2 | 18 | RLR |
| ep028 | (0.38, +0.14) | (0.36, -0.05) | (0.29, +0.19) | PRB | P | PRB / RBP / BPR | 10.8 | 10 | LRL |

Run all three prompts on the SAME placement. Pass = first reach goes to each prompt's first cube; a model that copies the layout keeps going to `recorded_first`.

## Protocol (frozen 2026-09-28)

- 8 layouts x 3 prompts = 24 trials; schedule in `stage3_trials.csv` (seed 20260928). Prompt order inside each layout is
  permuted so the recorded prompt is not always first: recorded prompt by slot 1/2/3 = [3, 2, 3].
- Place the cubes at the table positions, run ONE prompt, then **reset the cubes to the same positions** before the next.
- **Primary endpoint = first-target selection** (which cube the first reach/grasp goes to). The trial may end after
  `correct cube approach -> grasp/lift`; a full stack is not required.
- Report **conflicting (16 trials) first-target accuracy** as the headline; recorded (8 trials) separately -- recorded-only
  success is explainable by layout memorization.
- Also log `observed_first == layout_shortcut_first` on conflicting trials: that is the shortcut rate.
- Only checkpoints that passed Stage 1 (HEAD prompt/draw > 1, confirmed on HEAD n >= 30) and Stage 2 run this fixture.
