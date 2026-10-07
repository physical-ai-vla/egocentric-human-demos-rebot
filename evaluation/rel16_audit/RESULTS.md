# REL16 400k audit results (2026-09-28)

Scripts: step1_gt_replay.py, step2_predict.py (+QUOTA grip_go), analyze.py, promptswap_joint.py, cycle_logger.py.
Data: training frames of r180_umi76_rel16_v1 (all 180 eps trained, no val). Provenance: PROVENANCE.md.

## 1. Deployment-path GT replay (n=50) -- PASS
FK(q)+FRAME_FIX vs tcp_state 0.00 mm/0.00 deg; GT label vs recorded motion k16 med 0.01 max 0.44 mm.
IK single-solve tail on large moves: k8 p90 12 mm/13.6 deg, k16 p90 19.7 mm/26 deg, max 58 mm/63 deg.

## 2. Motion (active arm, real state, K=2 fixed noise), pred/GT at k16
moving 1.00 (cos 1.00) | rest_go 0.95 (cos 0.98) | ep_start: model moves 28 mm where GT moves 1 mm.
rest->go GT is an ease-in: k1 0.4, k8 8.6, k16 56.8 mm -> stop-and-go + exec_k=8 executes only the slow head.

## 3. History ablation -- copycat REJECTED
identity / reversed history: moving k16 |x|/|real| 0.96/0.90, cos 1.00/0.99; rest_go unchanged (1.00).

## 2g. Gripper onset (grip_go n=20)
cur 114 mm, GT k16 80 mm (closes 33), pred 91 (closes 23): closing fraction med 0.64 (IQR 0.51-0.86); same
under ident/rev history. Label flaw independent of this: target = follower width, which stalls at the cube
(R150 hold raw -79..-118) while the leader commands 0-7 (squeeze).

## 4. Prompt counterfactual, A16 / 0.8 s endpoint, prompt pairwise vs same-prompt draw spread (mm)
| model | moving | rest_go |
|---|---|---|
| REL16 400k | 3.4 vs 12.6 (0.27x) | 4.5 vs 21.9 (0.21x) |
| B1-old 250k | 8.1 vs 2.6 (3.1x) | 11.6 vs 14.7 (0.8x) |
| C-old R150ft 250k | 15.0 vs 1.8 (8.3x) | 21.2 vs 7.5 (2.8x) |
Sensitivity only (not correctness of the chosen cube). REL16: smallest prompt effect AND 5-7x larger sampler
spread than C-old -> the chunk is multimodal and the prompt does not pick the mode.

## Real robot (8025, REL16 400k, n_action=1)
exec_k 8 / dwell 0.6: 246 cyc, L 1346 mm, 5.5 mm/cyc, 2.8 mm/s
exec_k 16 / dwell 0.6: 123 cyc, L 1736 mm, 14.1 mm/cyc, ~6 mm/s (first ~51 cycles MPS-contended by the audit)
exec_k 16 / dwell 1.5: 43 cyc, 24.4 mm/cyc, 9.2 mm/s, exec ratio med 0.70, 8/42 timeouts (6 IK_RESIDUAL); still
wrong cube, no grasp. Uncontended infer ~1.04 s/cycle. Demo speed ~110 mm/s.
