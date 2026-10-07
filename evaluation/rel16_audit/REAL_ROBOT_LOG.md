# REL16 real-robot log (execution-contract A/B, 2026-09-29)

Fixed unless stated: REL16 umi, chunk 16, n_action 1, dwell 1.5 s, clamp off, grip threshold 0.6 (cmd 42/0), ROT180 0,
mirror 0, IK solver clip OFF ("IK-unclipped"), DQ_MAX 0.6 + LO/HI. IK per-waypoint log: ~/v4_ik_waypoints.jsonl.

| time (09-29) | ckpt | state | exec_k | observation (user) | IK log (live) |
|---|---|---|---|---|---|
| ~13:3x–13:52 | v3 50k | state76 | 16 | inaccurate approach / overshoot ("너무 안 된다") | n 18: IK FK err p50 33 / p95 61 mm, LO/HI hits 7/18 (L j2, j3) |
| 13:52–14:01 | v3 50k | state76 | 4 | "굉장히 느리는 한데 overshoot은 없어진거 같아. 정확히 cube로 가기는 해." | n 97: IK FK err p50 5.6 / p95 10.9 mm, max|dq| p50 0.019 rad, LO/HI 6/97 (L j3) |
| 14:01– | v3 50k | state76 | 8 | no overshoot; approaches a cube stably; WRONG-colour target; grasp NOT achieved (user: "grasp도 못해요"; miss = BOTH closed too early AND laterally off, "a b 둘다야"); lift NOT evaluable (grasp first). /status: L closed to 0.0 mm on air while TCP z rose 69->109 mm | n 69: IK FK err p50 12.6 / p95 26.2 mm, max|dq| p50 0.049 rad, LO/HI 40/69 (L j3, R j2/j3 at the HI=0 stop, magnitude unlogged), DQ_MAX 0 |
| 14:3x– | v3 50k | state76 | 8 | same settings, logging only: ~/v4_cycles.jsonl (g[1:16], g_sel, grip sent, TCP now/target/after, IK residual, q now/cmd, limit excess) | |

Offline (50k, moving, policy-only p50 L/R): k4 16/13, k8 26/26, k12 43/37, k16 57/73 mm.
Next: cart20 at the same exec_k (UI :8026, follower EXEC_K=4); endpoint-loss ablation ON HOLD until k8 result.

## 14:13–14:20 v3 50k k8, cycle log (~/v4_cycles.jsonl, 3 episodes, 150 cycles)
User: "a 하고 엉뚱한대로 가서 입을 벌리고 있어. 큐브도 없고 아무것도 없는곳에서."
Grip command per cycle (o open / C close):
  ep0 L oCooo...oC ooo   close at z 143 mm (air), reopened next cycle (g8 0.23 -> 0.87 flip)
  ep0 R oooCCCC...C      close at z 194 mm (idle arm, air), never reopened (48 cycles)
  ep1 L CCCC...C (74)    started closed, never opened
  ep1 R CCCCCooo...      opened, descended to z 24 mm with an open jaw where there was no cube
  ep2 L CooCCC...C       close at z 90 mm
Findings: (1) early close in the air (a); (2) once closed, stays closed -> suspect copying the current jaw width in the state;
(3) open-jaw descent to an empty spot (target selection / language). limit_excess 0 throughout (the 58 % LO/HI in the k8 row was
the HI=0 boundary, not a real excess).
Offline next (no model/deploy change): grip-state flip test (current width open<->closed, does g follow?), language path trace.

| 15:2x– | v3 60k | state76 | 8 | (pending) -- cycle log ~/v4_cycles.jsonl + frames ~/v4_cycle_frames/ on | |

## 17:1x RELCART20 20k + LEARNED IK (:8027) -- FIRST GRASP
User: "learned ik로 하니깐 처음으로 집었어!"
- Setup: exec_k 8, dwell 1.5, grip 0.6, IK_BACKEND=learned (LIK0-P12 best.pt); stop-on-fail, no fallback.
- Cycle log: 48 cycles, all executed with learned, 0 stops.
- L jaw: 107 mm open, then closed and HELD at 36 mm for 18 cycles (object between fingers; an empty close reads 0 mm).
- Shadow numerical on the same targets (n 49):
  - equal FK error (p50 4.9 mm both);
  - joint solutions differ by p50 3.5 deg, p90 17.3 deg (max over joints).
  - Same Cartesian accuracy, different posture.
- REL input diff (re-anchored vs policy) p50 2.3 mm.
- Next: repeat under the same condition; :8026 (same ckpt, numerical) from GO TO TRAIN START as the paired control.
