# Ego CART20 (v2 train) vs R312c RELCART20: interpretation (2026-10-01)

Read-only diagnostic.
- Numbers: `summary.md` / `summary.json` / `*.csv` / `plots/`.
- Kinematic (IK) section: `kinematic_summary.md`, appended when that run finishes.
- State-jump root cause: `STATE_JUMP_ROOT_CAUSE.md`.

## 0. Bottom line
- **Contract parity: PASS.**
  - UMI_DT 50.05 ms, rot6d rows, `[L9|R9|gL gR]` state, LEFT first, gripper 0 = closed.
  - FK recomputation reproduces R312c state/CART20 to 0.000 / 0.05 mm (p99).
  - **Tool frame:** gravity in the TCP frame at task start is ego [0.38, −0.06, 0.90] vs robot [0.37–0.40, −0.09, 0.89]. Same axis convention, no 90° error.
- **Action (CART20) magnitudes nearly overlap.** At k4/k8 the translation p95 differs by 2–10% and the rotation p95 by 20–40%.
- **The state (task-relative workspace) does not overlap.** The robot spreads about twice as wide. This is structural: the two datasets start from different poses.
- **Gripper:** both use the same continuous 0..1 scale, but the dynamics differ in shape.
  - Human: fewer opens/closes per episode; the gripper stays at a held value.
  - Robot: the state is the follower signal and the action is the leader signal, so the two differ even within the same dataset.
- **Wrist cameras: large visual gap.** Ego = wide fisheye, sharpness 1,500–1,900. Robot = narrow close-up, 58–62.

## 1. Summary table (guide §28)

| Feature | Ego | R312c | Gap |
|---|---:|---:|---|
| L state workspace p95 (task-rel, m) | 0.221 | 0.395 | robot ×1.8 |
| R state workspace p95 (m) | 0.214 | 0.369 | robot ×1.7 |
| k4 translation p95, L / R (mm) | 52.4 / 55.9 | 47.5 / 52.7 | ego +6–10% |
| k8 translation p95, L / R (mm) | 93.1 / 99.2 | 91.0 / 99.9 | ≈ equal |
| k16 translation p95, L / R (mm) | 142.3 / 150.9 | 165.5 / 171.5 | robot +12–16% |
| k16 rotation p95, L / R (°) | 47.8 / 56.2 | 35.4 / 46.5 | ego +21–35% |
| Gripper mean, L / R (state) | 0.259 / 0.238 | 0.236 / 0.246 | ≈ equal (shape differs, §4) |
| Both arms stationary at k8 (< 2 mm) | 19.5% | 2.4% | ego ×8 |
| R312c workspace 2 cm bins covered by ego (min 1) | L 0.199 / R 0.255 | – | robot SAMPLES in ego-covered bins: 0.60 / 0.65 |
| R312c CART20 k8 NN distance p95: robot→ego / robot→robot baseline | 4.41 | 2.29 | ×1.9; 78% of robot rows are within the robot baseline p95 |

## 2. Similarities (supported by the numbers)
1. **Action translation magnitude and how it grows with horizon match.**
   - At k1/k4/k8, p50/p95 agree within 2–10%.
   - The R/L ratio is about 0.6 in both (right arm p50 smaller).
   - Up to the 0.8 s horizon, the "how far to move" prior is shared.
2. **Action occupancy:** 85–98% of robot action samples fall in dxyz bins that ego also visits (k4 0.97, k8 0.93, k16 0.85). Only rare robot motions (the tails) are missing.
3. **Directions:** at k4–k16, JS divergence is 0.07–0.11 bits and the robot has no direction bins that ego lacks. Ego covers every direction the robot uses.
4. **Standardized PCA:** both datasets share the same cross-shaped structure. The robot only extends further along the main axes.
5. **Head (global) view:** brightness and contrast are similar, and the top-down plate + cube layout matches.
6. **Order composition:** both cover all 6 orders. R312c has 52 per order; ego has 42–55.

## 3. Differences (supported by the numbers)
1. **Task-relative workspace.**
   - Robot |xyz| p95 is 0.37–0.40 m vs ego 0.21–0.22 m.
   - On TCP z (which points down for both, per the gravity check), the robot reaches p95 +0.16–0.21 m and ego only +0.05 m. Ego has −0.12 m (upward) at p05.
   - Interpretation: **the robot starts from a raised rest pose and descends to the table; the human starts with the hand already near the workspace.** Because the anchor is the task start, the absolute meaning of the state differs.
   - Robot L y p95 is 0.345 m vs ego 0.135 m, so the robot makes large lateral moves.
   - Only 20–25% of robot 2 cm bins are covered by ego, though by sample count 60–65% are.
   - State NN: robot→ego p95 5.29 vs baseline 3.21; 75% of robot rows are within the baseline.
2. **Rotation:** ego rotates more over short horizons.
   - k4 rotation p95: ego 14.6/16.6° vs robot 10.6/13.8°.
   - k16: 47.8/56.2° vs 35.4/46.5°.
   - Human wrists turn faster and further.
   - At k16 the robot has the larger translation (165 vs 142 mm), meaning long robot moves are steadier and longer.
3. **Stationary / small motion: a structural difference.**
   - Rows where both arms are stationary at k8: ego 19.5% vs robot 2.4%. At k16: ego 14.9% vs robot 1.0%.
   - Per arm, the robot has more near-zero rows (k16 < 1 mm: robot 37–42% vs ego 17–18%), because it often moves one arm while the other waits.
   - Humans hold both hands still at the same time far more often.
   - This bears on the "too little stop / small-motion supervision" hypothesis: ego has more "both arms still" supervision than the robot.
4. **Gripper.**
   - Same scale, but different shape:
     - Ego is clearly binary-ish (g < 0.1 is 52–68%), with almost no 0.25–0.5 band on the right arm (1.8%).
     - Robot state is 0.25–0.5 for 16%, because the follower width stays mid-range while holding.
   - Transitions per episode p50: ego L 1 / R 2 vs robot 4 / 4. **Ego has fewer clear open/close events.**
   - The robot's g_action − g_state is mostly the leader-cmd vs follower-width signal difference (k1 mean 0.10–0.12). Compared within the same signal (inside the action chunk), the change rate at k4 is ego 3.8–5.4% vs robot 7.1–7.6%: similar order of magnitude.
   - Ego action max is 0.85 because humans never open fully. The robot action uses all of 0..1.
5. **Mean action direction (bias).**
   - Robot k16 mean dx is −4.7 (L) / −5.7 (R) mm and dz is +2.5 / +5.3 mm.
   - Ego mean dx is +4.3 / +2.1 mm, dy is +2.8 (L) / −2.8 (R) mm.
   - The sign of x is opposite. This comes from the start-pose difference (the robot descends and pulls back, the human pushes forward).
6. **Wrist cameras:**
   - Sharpness: ego 1,500–1,900 vs robot 58–62.
   - Contrast: 78–81 vs 47–53.
   - Composition: ego is a wide fisheye (table + plate + cubes + room); the robot sees a close-up of the fingers and cube (contact sheet).
   - The global view is similar (brightness 119 vs 128, contrast 52 vs 51).

## 4. Right arm near the low-x boundary (base frame, R312c FK)
- The R312c right arm sits below 172 mm absolute x for about 3.5% of rows (6,075).
- Only **5.3%** of those rows have an ego sample within 2 cm in task-relative coordinates. In the 172–190 mm bin it is 8.3%; above 250 mm, 85.5%.
- **Ego barely covers the robot's boundary region.**
- In that region the robot k8 mean dxyz is about [+0.4, +4.9, +1.9] mm. The few ego matches go [+13, +14, +19] mm in a different direction (small sample).
- Caveat: ego has no absolute base frame, so this is a task-relative match only. It says nothing about where a human's absolute workspace edge is.

## 5. Data-quality findings (separate documents)
- **Ego LeRobot frames are not time-contiguous.**
  - 2,344 of 51,314 adjacent pairs have a time gap (p50 1.2 s, max 15.7 s).
  - Analyses must decide adjacency with `aux.row` / `aux.time_s`. `validation/continuity.py` does this.
- **Raw MASt3R-SLAM silent jumps.**
  - 55 steps over 3 m/s in 24 episodes; the anchor-relative state stays offset afterwards.
  - Fix: v2b (Rule C) → training rows with > 100 mm per adjacent row: 18 → 0; train −6.2%, val 0%.
- **Ego episode start ≠ anchor in 126 of 298 episodes** (first frame offset p90 0.16 cm, max 56 cm). The robot always has the anchor at frame 0.

## 6. Implications for ego pretrain → R312c FT (answering guide §30 per component)

| Component | Coverage | Basis |
|---|---|---|
| State | **partial** | Robot tail regions (raised start, large lateral, near the low-x boundary) are rare in ego. 75% of robot state rows are within the robot baseline NN distance |
| Translation action | **high** | 85–98% of samples covered; k1–k8 magnitudes match |
| Rotation action | **ego covers it more widely** | Robot rotations are mostly inside the ego range (rotation NN coverage 0.83–0.90) |
| Gripper | **scale same, dynamics differ** | Fewer ego transitions, different hold values; the robot action follows the leader signal |
| Visual | **wrist views differ greatly**, global view similar | Low-level statistics + contact sheets only |

Prediction (a hypothesis, not a measurement):
- Ego pretraining most likely transfers the "how far and in which direction to move" prior (action) and the language/global-view links.
- State, wrist views and gripper timing will probably need relearning in R312c FT.
- The **state start-pose difference** in particular is a likely source of pretrain→FT interference. The FT data quickly overrides it, so the effect should be checked from the early FT loss curve (early 10k steps).

## 7. Kinematic compatibility (IK/FK, diagnostic only; nothing feeds training)
Details:
- `kinematic_summary.md`: deployment setting, joint5 locked.
- `kinematic_nolock/kinematic_summary.md`: joint5 unlocked; the downstream metrics come from this one.

Method:
- Ego task-relative poses are placed at a real R312c episode's start TCP and solved with continuous IK. 60 episodes per dataset.
- Success means position < 2 mm and rotation < 2°.

**Validation first.** With joint5 unlocked, IK on R312c's own trajectories succeeds 100%. The IK Δq reproduces the actual robot Δq: k4 p95 0.206 vs 0.206 rad, k16 0.766 vs 0.753. So in this mode the IK stack is a faithful stand-in for the robot.

| Question (guide §19) | Result |
|---|---|
| 1. How much ego EEF is reachable | Position only: **99.7% / 99.5%**. Full pose (joint5 free): **72.3% / 74.6%** (robot 100%). Deployment setting (joint5 locked): 22.1% / 15.9% |
| 2. Is the gap position or orientation | **Orientation.** 99% of full-pose failures are recovered by position-only IK. ori=yaw recovers only 3–7% |
| 3. FK reconstruction accuracy | Successful solves are < 2 mm / 2° by definition. Across all solves (deployment setting), ego p95 123–151 mm / 20–23° vs robot 34–41 mm / 10–14° |
| 4. Joint configurations outside the robot range | **Partly.** ego-IK→robot q NN p95 is 0.26–0.29 vs robot baseline 0.06–0.07 (×4). Only 53–54% of ego configurations are within the robot baseline. Joint-limit proximity per joint is similar to the robot (j2/j3 near rest) |
| 5. Joint trajectory smoothness | **About 3× rougher.** Per-row (66.7 ms) max-joint abs(dq) p95: ego 0.21 vs robot actual 0.07 rad. p99: 0.47–0.55 vs 0.11 |
| 6. Does ego cover the robot joint region | Robot q within the ego baseline distance: **L 73% / R 92%**. Ego covers most of the robot region and also goes elsewhere |
| 7. Left/right difference | Small. R is slightly worse in joint5-locked full pose (15.9 vs 22.1%); with joint5 free it is similar (74.6 vs 72.3%) |
| 8. What FT must learn | Wrist orientation, and the joint-space "efficiency" of the same EEF motion (see below) |

**Δq per horizon (joint5 free):**

| Horizon | Ego / robot actual Δq |
|---|---|
| k4 | 0.46–0.50 vs 0.21 rad |
| k8 | 0.76–0.89 vs 0.40 rad |
| k16 | 1.18–1.28 vs 0.72–0.75 rad |

The EEF translation is about the same size (§1), yet reBot needs **about twice the joint motion**. The likely drivers:
- the larger human wrist rotation (k16 rotation p95 is 21–35% higher);
- poses close to awkward wrist configurations.

**Future targets 0.8 s ahead, solved from q_t (joint5 free):** ego 88–99% vs robot 100%. Short horizons are almost all feasible. Infeasibility accumulates as the orientation drifts from the task start.

**Workspace (2 cm, task-relative, sampled episodes):** 72–75% of ego rows are IK-feasible, spread over 64% of the ego bins, not just near the start. Robot bins covered: raw ego 0.10–0.11 → feasible ego 0.07–0.08. Feasibility removes about 20–30% of the coverage.

**Separate finding (deployment stack):**
- Under the deployment `V4_PINK_LOCK=joint5`, IK reproduces R312c's own teleop poses only 45.2% / 52.2% of the time, with p95 34–41 mm / 10–14°.
- This is not an ego-data issue. See `HANDOFF_joint5_lock_deploy.md`.

**Caveats:**
- Ego has no absolute base frame. Each ego task-relative trajectory is attached to a robot start pose, so the orientation at the start always equals the robot's. Infeasibility comes from the human wrist rotation that builds up afterwards.
- Changing the start-pose assignment moves the result by only 3–4 points (0.22/0.16 → 0.26/0.20).

### Conclusion of the kinematic section
- The human trajectory is almost entirely compatible with reBot in **position** and in **short-horizon relative motion**.
- About 1/4 of wrist **orientations** cannot be matched exactly.
- The same EEF motion needs **about 2× the joint motion, at about 3× the per-step roughness**.
- So what ego pretraining can transfer is mostly the "Cartesian intent" (CART20). Wrist orientation and the joint-space execution will have to be learned from R312c FT (or handled by the deployment IK).
- The current deployment lock fails even on the robot's own data, so it should be checked first, before any ego conclusion.
