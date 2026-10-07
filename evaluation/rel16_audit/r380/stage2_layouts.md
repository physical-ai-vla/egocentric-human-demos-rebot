# Stage 2 ordinary-layout fixture (HEAD, prompt = recorded order, frozen 2026-09-28)

Runtime: REL16-v2B, V4_STATE_MODE=umi94, V4_GRIPPER=binary, exec_k 16, n_action 1, dwell 1.5 s, rot180/mirror off.
Base frame metres, x forward, y left, +-~5 cm.

| layout_id | order (= prompt) | R(x,y) | B(x,y) | P(x,y) | expected first | clean (3 picks) |
|---|---|---|---|---|---|---|
| ep093 | BPR | (0.23, -0.06) | (0.26, +0.18) | (0.34, -0.01) | B | True |
| ep110 | BRP | (0.26, +0.14) | (0.35, -0.01) | (0.32, +0.27) | B | False |
| ep113 | PBR | (0.22, +0.10) | (0.39, +0.04) | (0.22, -0.15) | P | True |
| ep172 | PRB | (0.17, -0.04) | (0.41, -0.10) | (0.31, +0.14) | P | False |
| ep150 | RBP | (0.22, -0.16) | (0.28, +0.13) | (0.35, -0.07) | R | True |
| ep127 | RPB | (0.23, +0.11) | (0.37, +0.04) | (0.22, -0.17) | R | True |

Per trial record (stage2_trials.csv): reach, grasp, lift, stack_attempt, stack_success, wrong_target (first reach to a
cube other than the prompt's first), execution_timeout. 2 repeats per layout, trial order shuffled (seed 20260928).
Reset cubes to the table positions between trials. Stage 2 pass = the stacking chain on ordinary layouts; the headline
is stack_success count and the furthest stage reached per trial. Layouts are fixed: do not swap in easier ones.
