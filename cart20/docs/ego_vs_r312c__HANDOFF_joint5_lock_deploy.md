# Handoff to the robot deployment owner: V4_PINK_LOCK=joint5 cannot reproduce R312c's own teleop poses (2026-10-01)

**This is a deployment-stack issue, not an ego-data issue. No dataset or policy changes are needed for it.**

## Finding
I ran the current deployment IK (`holobrain-mac-model/pink_ik.PinkIK`) with the same settings as `load_relonly_d6_ckpt_to_ui.sh`:
`IK_BACKEND=pink`, `V4_PINK_LOCK=joint5`, ori full, posture 1e-3 toward the seed, quadprog, ≤ 200 iterations.

- The IK was run on **R312c's own teleop trajectories**. Target = FK(q_t), solved continuously and seeded from q0. 60 episodes, at most 250 rows each.
- Success means position error < 2 mm and rotation error < 2°.

| Setting | L | R |
|---|---:|---:|
| Full pose, **joint5 locked (current deploy)** | **0.452** | **0.522** |
| Full pose, joint5 unlocked | 1.000 | 1.000 |
| Position only, joint5 locked | 1.000 | 1.000 |
| ori=yaw, joint5 locked | 0.360 | 0.500 |
| Pose error p95 under the lock (all solves) | 40.5 mm / 14.4° | 33.6 mm / 9.6° |

- Real teleop moves joint5, but deployment IK holds it at the measured value every cycle. So even when the policy predicts training-like targets, the IK cannot reproduce about half of them in full pose.
- Relative targets 0.8 s ahead, solved from q_t under the lock, succeed on R312c at k1 0.94/0.96, k8 0.76/0.84 and k16 0.66/0.74. Error builds up as the horizon gets longer.
- Consistency check: FK(q) reproduces the stored R312c state and CART20 to 0.000 / 0.05 mm, and Pink FK equals rebot_fk_torch. The FK and frame setup is not the cause.

## Proposed A/B (real robot)
- **A:** current UI (joint5 locked).
- **B:** same checkpoint and settings, only `V4_PINK_LOCK=` empty (joint5 unlocked).
- Same scene, same prompts.
- Compare:
  - target pose error
  - actual TCP tracking error
  - wrist orientation error
  - grasp alignment
  - collision and joint-limit behaviour
- Caution: the lock was originally added to stop yaw drift (pink_ik.py, 09-30: "yaw still moves"). Check B for wrist-yaw oscillation and drift.

## Reproduce
`~/ego_cart20/ego_cart20/scripts/kinematic_compat.py --ego <ego lerobot> --robot ~/c8/compare_ref/r312c_relcart20_rel16_v4 --output <dir>`
The R312c reference rows are in the `robot_*` entries of `kinematic_summary.json` → `ik_state_feasibility`.
