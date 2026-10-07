# Current numerical IK runtime path (as deployed by load_v3_ckpt_to_ui.sh, 2026-09-29)

```
mac_v4_smoke_ui._one_press()
 ├─ _observe() ×2 (OBS_GAP_S apart) → joints14 (follower frame, rad; jaw raw 0..-270)
 ├─ INF.infer([h0,h1])                                    infer_core_v4.py:388
 │   ├─ build_state_umi76()  FK(q)→T (eef_kin + V4_FRAME_FIX R_y(90°) → dataset TCP frame) → state76
 │   ├─ X-VLA predict_action_chunk → post → act (16,32)   ONLY dims 0:9, 9, 10:19, 19 are read (Δq 20:32 ignored)
 │   ├─ T_target,k = T_now · pose10d_to_mat(act_k)        current-anchor, no chaining, clamp off
 │   ├─ kin.solve_chunk(q_now, tgt_pos, tgt_quat)         VALIDITY ONLY (valid_step, FK guard 60 mm)
 │   │     ⚠ quaternions passed in DATASET frame, no R_DS_TO_FK here (ori_weight tiny, so position check ~ok)
 │   └─ returns cmd[:n_action=1] (unused joints; the UI re-solves below)
 ├─ for k in [EXEC_K−1] = [15]   (exec_k 16, n_action 1 → exactly ONE waypoint per cycle = A16, 0.8 s ahead)
 │   ├─ _observe() → q_meas
 │   ├─ INF.solve_waypoint(q_meas, tgt_pos[15], tgt_quat[15], widths[15])     infer_core_v4.py:470
 │   │   ├─ quat · R_DS_TO_FK (dataset → FK frame)
 │   │   ├─ eef_kin.Kin.solve_chunk (T=1) → HandUMI solver.ik (handumi-sw/src/handumi/robots/kinematics.py)
 │   │   │    jaxls LM, ONE solve seeded at q_meas:
 │   │   │      cost = pose(pos_weight 125, ori_weight 0.6) + rest_cost(toward q_meas, weight 20) + joint limits
 │   │   │    then limit_joint_delta(q_meas, q_sol, max_joint_delta = 0.35 rad)   ← per-joint elementwise clip
 │   │   ├─ dq = q_sol − q_meas ; q_cmd = clip(q_meas + clip(dq, ±DQ_MAX 0.6), LO, HI)   (LO/HI = URDF ∓ 5°, j2/j3 ≤ 0)
 │   │   ├─ grip: width→g→binary @ V4_GRIP_THRESH 0.6 → cmd 42 / 0
 │   │   └─ cmd[FLIP_IDX [0,1,5,7,8,12]] *= −1   (follower → leader frame for /execute_step)
 │   └─ _goto(cmd): slew-ramped sub-steps at CMD_HZ, then dwell ≤ 1.5 s until joint_tol & TCP tol (tol ≥ IK residual + 2 mm)
 └─ next cycle (stop-and-go: observe → infer ≈1 s MPS → execute)
```

## What the HEAD180 GT says about this path (measured on v3d, GT Δq = q(t+(k+1)dt) − q(t))

| hidden limit | GT rows that exceed it at k16 (the executed waypoint) |
|---|---|
| solver `max_joint_delta` 0.35 rad (per joint) | **33.2 %**. Worst joints: L j2, R j2/j3/j4. Per k: 0 % at k1, 5.4 % at k6, 33 % at k16 |
| runtime `DQ_MAX` 0.6 rad | 16.1 %, but never binding in practice because 0.35 clips first |
| runtime LO/HI | 0.21 % (mostly R j4 > 1.48 rad) |

The numerical backend therefore cannot reproduce one third of the teleop A16 motions in a single waypoint. This is
by construction, not a model error. The measured single-solve undershoot (k16 p90 19.7 mm / 26°, max 58 mm / 63°) is
consistent with this, plus two cost terms:
- rest_cost 20 toward the seed shrinks large moves;
- ori_weight 0.6 against pos 125 means orientation is nearly unconstrained.

Also, the elementwise clip does *not* preserve direction, despite its docstring.

Not changed: this is the numerical baseline exactly as deployed. For the Phase A offline comparison I propose:
- **numerical-deployed**: this path, the primary baseline;
- **numerical-unclamped**: max_joint_delta = None, a diagnostic reference only, never deployed without a separate decision.

Both variants are declared before any learned-IK number exists.

Both backends share everything downstream of the joint solution: DQ_MAX, LO/HI, FLIP, gripper, _goto.
With IK_BACKEND=learned, only the `solver.ik` box is replaced.
