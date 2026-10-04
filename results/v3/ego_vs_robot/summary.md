# Ego CART20 vs R312c RELCART20: dataset comparison (read-only)

- Ego: `${HOME}/c8/ego_cart20_v2_lerobot/ego_cart20_v2_train`
- Robot: `${HOME}/c8/compare_ref/r312c_relcart20_rel16_v4`
- seed 1000, joint subsample 20000 per dataset, runtime 137.6 s

## 1. Contract parity

| Item | Ego | R312c |
|---|---|---|
| state dim | 20 | 20 |
| action shape | [16, 32] | [16, 32] |
| fps (rows) | 15 | 15 |
| camera keys | ['observation.images.global', 'observation.images.left_wrist', 'observation.images.right_wrist'] | ['observation.images.global', 'observation.images.left_wrist', 'observation.images.right_wrist'] |
| action[..., 20:32] max |x| in the DATASET | 0.0 | 1.8690905570983887 |
| state at episode row 0 == identity (max err) | 1.0223748739808798 | 2.7755575615628914e-16 |
| stored rot6d rows orthonormal (max err) | 7.488670528132957e-08 | 7.499892668016983e-08 |
| gripper value range (state + action) | [0.0, 0.8470862507820129] | [0.0, 1.0] |

| Documented contract | Ego | R312c | Match |
|---|---|---|---|
| UMI_DT_ms | 50.05 (ego_cart20.config.UMI_DT, raw-track interpolation) | 50.05 (umi76_to_lerobot UMI_DT, UMI76_CONTRACT 2b) | yes |
| rot6d | first two ROWS | first two ROWS (umi pose_util.mat_to_rot6d) | yes |
| state | RELCART20 inv(T_task_start) T(t), [L9|R9|gL gR] | RELCART20 inv(T_ep_row0) T(t), [L9|R9|gL gR] | yes |
| arm_order | LEFT first | LEFT first (robot0 = LEFT) | yes |
| gripper_polarity | 0 closed / 1 open, caliper mm / 80 (state and action SAME signal) | 0 closed / 1 open; state = FOLLOWER width / 0.11441 m, action = LEADER cmd / 45 (different signals) | yes |
| aux12 | 0 in the dataset | dq12 in the dataset; the REL-only trainer hard-zeroes 20:32 at model input AND output, so the model sees 0 | yes |

Semantic parity: **PASS**.

## 2. Dataset size

| | Episodes | Training rows | Action targets | Row-time (min) | Rows/episode p5/p50/p95 |
|---|---:|---:|---:|---:|---|
| ego | 298 | 51,612 | 825,792 | 57.3 | 64 / 168 / 291 |
| robot | 312 | 174,012 | 2,784,192 | 193.3 | 434 / 542 / 710 |

## 3. State / workspace (task-relative)

See `state_distribution.csv`. Workspace occupancy, 2 cm voxels, common grid:

| Arm | Min count | Ego bins | Robot bins | Shared | Robot bins covered by ego | Robot SAMPLES in ego-visited bins |
|---|---:|---:|---:|---:|---:|---:|
| L | 1 | 2984 | 6549 | 1305 | 0.199 | 0.600 |
| L | 5 | 1220 | 3809 | 540 | 0.142 | 0.614 |
| R | 1 | 2973 | 6549 | 1668 | 0.255 | 0.653 |
| R | 5 | 1072 | 3416 | 547 | 0.160 | 0.670 |

## 4–6. CART20 action by horizon (translation mm, rotation deg)

| k (ms) | Arm | Ego trans p50/p95/p99 | R312c trans p50/p95/p99 | Ego rot p50/p95 | R312c rot p50/p95 |
|---|---|---|---|---|---|
| 1 (50) | L | 1.1 / 14.5 / 23.5 | 0.8 / 12.2 / 19.8 | 0.4 / 4.3 | 0.2 / 2.7 |
| 1 (50) | R | 0.7 / 15.2 / 24.1 | 0.5 / 13.5 / 21.2 | 0.2 / 4.5 | 0.1 / 3.5 |
| 4 (200) | L | 3.1 / 52.4 / 82.7 | 2.8 / 47.5 / 77.2 | 1.2 / 14.6 | 0.6 / 10.6 |
| 4 (200) | R | 1.9 / 55.9 / 86.0 | 1.7 / 52.7 / 81.3 | 0.7 / 16.6 | 0.3 / 13.8 |
| 8 (400) | L | 5.4 / 93.1 / 142.4 | 5.7 / 91.0 / 146.0 | 2.1 / 26.7 | 1.2 / 20.1 |
| 8 (400) | R | 3.2 / 99.2 / 149.2 | 3.6 / 99.9 / 150.0 | 1.2 / 31.3 | 0.8 / 26.4 |
| 16 (801) | L | 10.1 / 142.3 / 213.3 | 11.9 / 165.5 / 245.0 | 3.9 / 47.8 | 2.4 / 35.4 |
| 16 (801) | R | 5.8 / 150.9 / 224.9 | 8.6 / 171.5 / 243.3 | 2.2 / 56.2 | 1.8 / 46.5 |

Action occupancy (1 cm voxels of dxyz, min 1 / 5 samples):

| k | Arm | Robot bins covered by ego (min1 / min5) | Ego bins covered by robot (min1 / min5) | Robot samples in ego bins |
|---|---|---|---|---|
| 4 | L | 0.547 / 0.364 | 0.471 / 0.648 | 0.977 |
| 4 | R | 0.525 / 0.393 | 0.462 / 0.678 | 0.974 |
| 8 | L | 0.363 / 0.254 | 0.511 / 0.757 | 0.928 |
| 8 | R | 0.393 / 0.238 | 0.517 / 0.799 | 0.931 |
| 16 | L | 0.263 / 0.213 | 0.558 / 0.793 | 0.852 |
| 16 | R | 0.299 / 0.167 | 0.566 / 0.785 | 0.853 |

Motion direction (moving samples > 5 mm, 12 x 6 azimuth/elevation bins):

| Arm, k | JS divergence (bits) | Ego mean unit | R312c mean unit | Robot mass in bins ego lacks |
|---|---:|---|---|---:|
| L_k4 | 0.108 | [0.127, 0.051, -0.041] | [0.08, -0.021, 0.104] | 0.000 |
| L_k8 | 0.103 | [0.129, 0.076, -0.025] | [0.093, -0.042, 0.11] | 0.000 |
| L_k16 | 0.085 | [0.106, 0.112, 0.005] | [0.07, -0.047, 0.12] | 0.000 |
| R_k4 | 0.082 | [0.061, -0.043, 0.0] | [0.103, -0.01, 0.1] | 0.000 |
| R_k8 | 0.070 | [0.083, -0.053, 0.037] | [0.117, -0.011, 0.12] | 0.000 |
| R_k16 | 0.065 | [0.086, -0.065, 0.083] | [0.079, -0.024, 0.148] | 0.000 |

## 7. Continuous gripper (0 closed / 1 open, NOT remapped)

See `gripper_distribution.csv` for the full table.

| Dataset | Arm | Signal | Mean | p10 / p50 / p90 | g<0.1 | 0.25–0.5 | 0.5–0.75 | ≥0.75 |
|---|---|---|---:|---|---:|---:|---:|---:|
| ego | L | state | 0.259 | 0.00 / 0.08 / 0.77 | 0.515 | 0.132 | 0.111 | 0.137 |
| ego | L | action_k8 | 0.262 | 0.00 / 0.09 / 0.77 | 0.510 | 0.132 | 0.112 | 0.140 |
| robot | L | state | 0.236 | 0.00 / 0.00 / 0.82 | 0.592 | 0.166 | 0.093 | 0.119 |
| robot | L | action_k8 | 0.207 | 0.00 / 0.00 / 1.00 | 0.710 | 0.049 | 0.057 | 0.153 |
| ego | R | state | 0.238 | 0.00 / 0.07 / 0.75 | 0.679 | 0.018 | 0.173 | 0.106 |
| ego | R | action_k8 | 0.239 | 0.00 / 0.07 / 0.75 | 0.678 | 0.018 | 0.173 | 0.108 |
| robot | R | state | 0.246 | 0.00 / 0.00 / 0.86 | 0.608 | 0.157 | 0.081 | 0.133 |
| robot | R | action_k8 | 0.217 | 0.00 / 0.00 / 1.00 | 0.666 | 0.034 | 0.043 | 0.169 |

dg = g_action(t+k) - g_state(t). For R312c that mixes two signals (follower state vs leader action), so the same-signal column action-internal = g_action(k) - g_action(k1) is the like-for-like one.

| Dataset | Arm | k | fraction abs(dg) > 0.1 | abs(dg) p95 | Action-internal abs(dg) > 0.1 | Action-internal p95 | corr(abs(dg), translation) |
|---|---|---:|---:|---:|---:|---:|---:|
| ego | L | 4 | 0.048 | 0.094 | 0.038 | 0.072 | 0.250 |
| ego | L | 16 | 0.137 | 0.385 | 0.131 | 0.358 | 0.338 |
| robot | L | 4 | 0.308 | 0.595 | 0.076 | 0.154 | 0.589 |
| robot | L | 16 | 0.374 | 0.832 | 0.214 | 0.738 | 0.631 |
| ego | R | 4 | 0.065 | 0.141 | 0.054 | 0.110 | 0.339 |
| ego | R | 16 | 0.179 | 0.622 | 0.170 | 0.591 | 0.484 |
| robot | R | 4 | 0.342 | 0.616 | 0.071 | 0.157 | 0.470 |
| robot | R | 16 | 0.398 | 0.836 | 0.186 | 0.777 | 0.533 |

State gripper transitions per episode (hysteresis 0.4/0.6), p25/p50/p75: ego_L 0/1/2; robot_L 4/4/6; ego_R 2/2/4; robot_R 2/4/4

Within-dataset state vs action(k1) gripper: robot {"L": {"mean_abs_diff": 0.10337540051304911, "corr": 0.821673208007695}, "R": {"mean_abs_diff": 0.11800758745181238, "corr": 0.7968175242348637}}; ego {"L": {"mean_abs_diff": 0.0036520192294656876, "corr": 0.9985344709474594}, "R": {"mean_abs_diff": 0.004946307099161206, "corr": 0.9978665418934749}}

## 8. Left/right asymmetry

| Dataset, k | Trans p50 L/R (mm) | Ratio R/L | Rot p95 L/R | Mean dxyz L (mm) | Mean dxyz R (mm) |
|---|---|---:|---|---|---|
| ego_k1 | 1.1 / 0.7 | 0.67 | 4.3 / 4.5 | [0.16, 0.07, -0.11] | [-0.04, -0.09, 0.05] |
| ego_k4 | 3.1 / 1.9 | 0.61 | 14.6 / 16.6 | [0.74, 0.43, -0.17] | [0.01, -0.49, 0.55] |
| ego_k8 | 5.4 / 3.2 | 0.59 | 26.7 / 31.3 | [1.85, 1.14, 0.07] | [0.48, -1.24, 1.67] |
| ego_k16 | 10.1 / 5.8 | 0.57 | 47.8 / 56.2 | [4.25, 2.82, 0.85] | [2.11, -2.84, 4.0] |
| robot_k1 | 0.8 / 0.5 | 0.62 | 2.7 / 3.5 | [-0.09, -0.06, 0.03] | [-0.06, 0.11, 0.09] |
| robot_k4 | 2.8 / 1.7 | 0.61 | 10.6 / 13.8 | [-0.62, -0.21, 0.23] | [-0.65, 0.28, 0.6] |
| robot_k8 | 5.7 / 3.6 | 0.64 | 20.1 / 26.4 | [-1.8, -0.29, 0.78] | [-2.14, 0.08, 1.76] |
| robot_k16 | 11.9 / 8.6 | 0.72 | 35.4 / 46.5 | [-4.69, -0.12, 2.54] | [-5.74, -1.37, 5.27] |

## 9. Near-zero actions (fraction of rows)

| Dataset, arm, k | <1 mm | <2 mm | <5 mm | <10 mm | rot<1° | rot<2° | Both arms <2 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| ego_L_k1 | 0.482 | 0.603 | 0.780 | 0.897 | 0.651 | 0.813 | 0.373 |
| ego_R_k1 | 0.554 | 0.660 | 0.792 | 0.890 | 0.719 | 0.837 | 0.373 |
| ego_L_k4 | 0.333 | 0.439 | 0.569 | 0.687 | 0.475 | 0.572 | 0.230 |
| ego_R_k4 | 0.376 | 0.508 | 0.635 | 0.717 | 0.556 | 0.646 | 0.230 |
| ego_L_k8 | 0.257 | 0.362 | 0.491 | 0.587 | 0.404 | 0.493 | 0.195 |
| ego_R_k8 | 0.277 | 0.414 | 0.568 | 0.648 | 0.474 | 0.571 | 0.195 |
| ego_L_k16 | 0.174 | 0.272 | 0.408 | 0.499 | 0.314 | 0.415 | 0.149 |
| ego_R_k16 | 0.183 | 0.304 | 0.476 | 0.574 | 0.359 | 0.485 | 0.149 |
| robot_L_k1 | 0.527 | 0.626 | 0.787 | 0.922 | 0.808 | 0.913 | 0.284 |
| robot_R_k1 | 0.559 | 0.649 | 0.788 | 0.907 | 0.795 | 0.887 | 0.284 |
| robot_L_k4 | 0.433 | 0.474 | 0.561 | 0.670 | 0.560 | 0.680 | 0.052 |
| robot_R_k4 | 0.477 | 0.508 | 0.589 | 0.684 | 0.576 | 0.684 | 0.052 |
| robot_L_k8 | 0.394 | 0.438 | 0.490 | 0.563 | 0.488 | 0.562 | 0.024 |
| robot_R_k8 | 0.448 | 0.476 | 0.519 | 0.587 | 0.513 | 0.574 | 0.024 |
| robot_L_k16 | 0.367 | 0.410 | 0.446 | 0.484 | 0.445 | 0.483 | 0.010 |
| robot_R_k16 | 0.417 | 0.449 | 0.476 | 0.510 | 0.476 | 0.504 | 0.010 |

## 10. Coverage / nearest neighbour (common scaler on the union, both directions, with in-domain baselines)

| Block | robot→ego p50/p95 | robot→robot p50/p95 | ego→robot p50/p95 | ego→ego p50/p95 | Robot covered within robot-p95 | Ego covered within ego-p95 |
|---|---|---|---|---|---:|---:|
| k4_position | 0.20 / 1.16 | 0.13 / 0.59 | 0.18 / 2.64 | 0.17 / 1.69 | 0.857 | 0.900 |
| k4_rotation | 0.19 / 1.42 | 0.12 / 0.89 | 0.26 / 3.41 | 0.25 / 2.17 | 0.903 | 0.904 |
| k4_gripper | 0.00 / 0.52 | 0.00 / 0.00 | 0.12 / 0.34 | 0.02 / 0.21 | 0.634 | 0.905 |
| k4_full_CART20 | 1.07 / 4.06 | 0.62 / 2.20 | 1.17 / 6.69 | 0.99 / 4.52 | 0.801 | 0.871 |
| k8_position | 0.21 / 1.38 | 0.14 / 0.60 | 0.17 / 2.46 | 0.17 / 1.64 | 0.851 | 0.908 |
| k8_rotation | 0.21 / 1.80 | 0.13 / 0.94 | 0.26 / 3.43 | 0.25 / 2.15 | 0.882 | 0.900 |
| k8_gripper | 0.00 / 0.52 | 0.00 / 0.00 | 0.11 / 0.43 | 0.02 / 0.21 | 0.653 | 0.915 |
| k8_full_CART20 | 1.15 / 4.41 | 0.67 / 2.29 | 1.20 / 6.77 | 1.00 / 4.31 | 0.783 | 0.860 |
| k16_position | 0.24 / 1.96 | 0.16 / 0.63 | 0.19 / 2.23 | 0.19 / 1.54 | 0.817 | 0.916 |
| k16_rotation | 0.27 / 2.45 | 0.16 / 0.96 | 0.29 / 3.63 | 0.29 / 2.19 | 0.832 | 0.891 |
| k16_gripper | 0.00 / 0.52 | 0.00 / 0.00 | 0.12 / 0.30 | 0.02 / 0.31 | 0.659 | 0.957 |
| k16_full_CART20 | 1.39 / 4.94 | 0.77 / 2.36 | 1.34 / 6.63 | 1.10 / 4.13 | 0.730 | 0.846 |
| state_pose18 | 2.25 / 5.29 | 1.19 / 3.21 | 1.91 / 4.14 | 1.12 / 2.68 | 0.749 | – |

MMD (RBF, median heuristic, secondary): k4 0.0083, k8 0.0087, k16 0.0111

## Right-arm low-x boundary (R312c base frame)

R312c right absolute x (mm) p1/p5/p50: 126 / 182 / 297

| Abs x bin (mm) | Robot rows | Robot rows with ego within 2 cm (task-rel) | Robot k8 mean dxyz (mm) | Matched-ego k8 mean dxyz (mm) |
|---|---:|---:|---|---|
| [250, 1000000000.0] | 137381 | 0.855 | [-2.35, -0.69, 2.11] | [6.11, -0.74, 6.75] |
| [220, 250] | 12508 | 0.350 | [-1.85, 1.72, -0.77] | [7.77, 5.45, 10.5] |
| [190, 220] | 13019 | 0.168 | [-1.53, 3.08, 0.85] | [-0.66, 10.43, 11.39] |
| [172, 190] | 5029 | 0.083 | [-1.85, 3.63, 0.69] | [-0.02, 9.07, 11.71] |
| [-1000000000.0, 172] | 6075 | 0.053 | [0.39, 4.93, 1.87] | [13.44, 14.1, 18.75] |

> robot x is the ABSOLUTE base-frame TCP (FK aux.q_t). Ego has no robot base frame; ego rows are matched by TASK-RELATIVE right-arm state xyz (inv(T_start) T(t)) within 2 cm of the robot rows' task-relative xyz. That is a like-for-like comparison only if the robot's task-start pose is consistent across episodes; it says nothing about the absolute workspace edge for humans.

## Orders

- ego: episodes {"PRB": 48, "PBR": 42, "RBP": 55, "RPB": 53, "BRP": 53, "BPR": 47}
- robot: episodes {"RBP": 52, "RPB": 52, "BRP": 52, "BPR": 52, "PRB": 52, "PBR": 52}

## 11. Visual domain (low-level)

| View | Brightness p50 | Contrast p50 | Sharpness p50 | RGB mean |
|---|---:|---:|---:|---|
| ego_global | 118.8 | 51.6 | 347 | [118.68, 122.95, 121.8] |
| ego_left_wrist | 148.8 | 80.7 | 1864 | [147.45, 151.97, 147.73] |
| ego_right_wrist | 154.9 | 77.7 | 1533 | [151.69, 156.36, 156.97] |
| robot_global | 127.9 | 50.6 | 286 | [136.34, 138.26, 138.09] |
| robot_left_wrist | 163.9 | 52.6 | 58 | [154.93, 154.68, 155.65] |
| robot_right_wrist | 136.7 | 46.5 | 62 | [148.79, 148.54, 150.32] |

low-level RGB statistics only; no semantic similarity is claimed. X-VLA encoder embeddings not computed (optional section 25). Contact sheets: `plots/camera_contact_sheets.png`.

## 12–14. Interpretation

(Filled in by hand from the numbers above. See `INTERPRETATION.md`.)

