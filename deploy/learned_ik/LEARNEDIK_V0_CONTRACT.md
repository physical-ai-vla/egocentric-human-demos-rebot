# LearnedIK-v0 tensor contract (frozen before implementation, 2026-09-29)

No vision, no language, no gripper. Deterministic MLP regressor (no flow/generative). Separate code/folder:
`~/umi_bridge/learned_ik/`. Outputs go to new run names `LIK0-*`. HEAD180-REL16V3 and v3d are never written.

## Input  x = [q_t (12) | rel (16×18)] → flat 300

| field | shape | unit | source (v3d) | convention |
|---|---|---|---|---|
| q_t | 12 | rad | `aux.q_t` | follower/observation frame (NOT FLIP/leader), order L j1..j6, R j1..j6 = joints14[ARM_IDX] |
| rel_k, k=0..15 | 16×18 | m / — | `action[:, k, 0:9]` ⊕ `action[:, k, 10:19]` | per arm pos3 (m) + rot6d (first two ROWS of R) of A_k = inv(T_t)·T_{t+(k+1)dt}, dataset TCP frame (= rebot_fk_torch.tcp) |

- Gripper dims 9 and 19 are excluded. The Δq dims 20:32 are the target, never an input.
- Horizon encoding: implicit by position. A flat MLP assigns slot k to its own input and output columns, so no extra channel is needed.
- At runtime, rel is recomputed from the measured q at waypoint time: rel_k = inv(FK(q_meas))·T_target,k. When the robot is static (stop-and-go) this equals the inference-time REL.

## Output  Δq̂ (16×12) rad; q_cmd,k = q_t + Δq̂_k

Target: `action[:, k, 20:32]` = q(t+(k+1)dt) − q(t), in the same frame and order as q_t.

The GT is self-consistent:
- FK(q_t + Δq_gt) reproduces REL pos to a median of 0.0004 mm (max 0.16 mm).
- The HEAD180 GT has |Δq| max 1.52 rad, p95 at k16 up to 0.70 rad (L j2).

## Normalization (computed on the TRAIN episodes only, frozen to `lik0_stats.json`)

- q_t: per-joint mean/std.
- rel: per (k, dim) mean/std, with std floor 1e-4.
- Δq: per (k, joint) mean/std, with std floor 1e-4. Pooling over k is not used because the scale grows ~16× from k1 to k16.
- Denormalization before FK: Δq = ŷ·std + mean (rad).

## Model / loss

- MLP 300 → 1024 → 1024 → 1024 → 192, SiLU, LayerNorm. AdamW lr 1e-3 with cosine decay. Batch 1024, fp32.
- L = L_dq + λ_fk · L_fk_pos
  - L_dq = MSE(normalized Δq̂, normalized Δq_gt)
  - L_fk_pos = mean_{k,arm} ‖ R_tᵀ(FK(q_t+Δq̂_k).p − FK(q_t).p) − p_rel,k ‖², in **cm²**
  - FK = rebot_fk_torch.tcp, the same function and frame as rel16_aux.py, so no frame fix is needed.
  - Position only.
- λ_fk = 1.0 (a 1 cm FK error costs as much as 1 σ of Δq error). This is fixed now.
  **Smoke result (2026-09-29): init ratio 26.7 → failed → λ_fk = 0.1 (ratio 2.7). This is the one allowed adjustment; λ_fk is final from here.** At smoke it is only checked
  that both terms are within 10× of each other at init; it is not tuned on validation results.
- Diagnostics only, never in the loss: orientation error, smoothness (second difference over k), spike rate.
- Selection: last checkpoint, plus best validation L_fk_pos on the val episodes. Test episodes are touched only for the final report.

## Offline evaluation (declared now)

Backends:
- numerical-deployed (primary baseline)
- numerical-unclamped (diagnostic)
- LIK0

Inputs:
- (a) GT REL on the test episodes
- (b) HEAD180-REL16V3 predicted REL on the same frames

Note on (b): those frames were in the VLA's training set. This is fine for testing IK, but it is not a VLA
generalization number.

Metrics, per arm and per k with k16 as the headline:
- FK position error p50/p90/p95/max
- FK orientation error
- Δq MAE per joint L/R
- step jump, second difference and spike rate
- LO/HI violations
- failure rate
- latency (Mac MPS/CPU, batch 1)
