# Teleoperation improvement plan — scope fence and priority ladder (user directive, 2026-09-11)

Techniques from recent teleop systems, placed **where and when** they belong in the existing architecture.
The sensing architecture is NOT redesigned. Nothing in §1–§10 below is implemented today.

## Authoritative architecture (baseline, unchanged)

```
HEAD RGB-D → metric 3D human hand → operator calibration → official Aero DexPilot → Aero Hand
WRIST FISHEYE + ICM42688P → OpenVINS → WristPoseProvider → TrackingSupervisor → relative SE(3) → reBot
```

Raw wrist video + full-rate IMU stay available for offline refinement. `WristPoseProvider` is the canonical wrist
6-DoF interface — **no new HandVIO abstraction**.

## Priority ladder — build in this order, not in parallel

```
1. A1 head-mounted RGB-D human-hand take        <- operator, blocked on recording
2. A1 report + architecture decision (Case A/B on D_detect)
3. A3 standalone real Aero vision teleoperation
4. Real right-wrist OpenVINS validation
5. Virtual reBot dry run
6. Low-speed real reBot teleoperation
7. Aero + reBot combined teleoperation
--- ONLY after 7 is stable ---
8. grasp-phase arm stabilization      (§1)
9. IK continuity refinement           (§5)
10. singularity / manipulability gate (§8)
11. offline refined trajectories      (§4)
12. dataset collection
13. affordance-policy experiments
```

## Deferred items, with where each one lands

| § | technique | lands in | note |
|---|---|---|---|
| 1 | grasp-phase wrist stabilization (OAT) | command path only, after `TrackingSupervisor`, before arm command | suppress only SMALL wrist ΔSE(3) while fingers move fast and wrist is near-still. Never modify raw pose. Config-gated, **off by default**. Log activation + suppressed translation/rotation. Not before A3 + M1. |
| 2 | physical wrist-visual quality (ViHaTeleop) | hardware/runbook, not code | fixed FPS, manual exposure, fixed gain where practical, stable illumination. LED is **not** a mandatory dependency — measure real VIO under actual workspace lighting first, add illumination only if blur / low-feature intervals prove to be the bottleneck. |
| 3 | decoupled rates (OPEN TEACH) | coordinator + robot clients | 30 Hz is the DATASET timeline, not the servo clock. Robot loops target ~50–60 Hz off the latest valid target with interpolation. Measure capture→pose→retarget→command→execution timestamps BEFORE tuning; do not add latency-heavy filtering to fake smoothness. |
| 4 | live vs refined poses (BiDex) | episode layout + offline stage | **mandatory**, but the STORAGE CONTRACT is the part that cannot be deferred cheaply — see below. |
| 5 | IK continuity regularization | below the teleop abstraction, in the IK solver | minimize EE error + λ·‖q − q_prev‖. Canonical action stays relative SE(3). Log IK failures, joint discontinuities, joint delta norm, manipulability. |
| 6 | DexPilot / official Aero | already primary | do NOT redesign. semantic-7D stays as baseline/fallback/ablation/debug only. Never switch the primary path to joint-angle copying. |
| 7 | operator-specific hand calibration (ADAPT) | `human_hand.operator_palm_scale_m` | already decided: fixed constant from clean OPEN frames pooled across distances; live palm scale is a HEALTH signal only and must never modulate command gain. Human anatomical normalization stays separate from human→Aero morphology calibration. |
| 8 | robot-side safety (Bunny-VisionPro) | `robot/safety.py` + IK | workspace clamp and velocity/acceleration limits already exist. Add manipulability/singularity gate later: good → normal gain, near-singular → reduced Cartesian gain, unsafe → HOLD. No heavy collision planning in the first real test. |
| 9 | global egocentric observation (ActiveUMI) | already done | head RGB + depth + intrinsics + timestamps always recorded (`head_depth` is `required: true` in both profiles since 2026-09-11). **No active head control in V1** — there is no robot neck/camera arm. |
| 10 | learned contact execution (TeleDexter) | OUT OF SCOPE | no tactile, force control, learned contact controller, or RL grasp stabilization in the first paper. |

## §4 storage contract — FROZEN 2026-09-11 (the refinement itself stays at ladder 11)

The offline refinement is still deferred. What was frozen, before bulk recording, is the naming + provenance, because
every episode recorded from now on locks in the schema.

```
raw/     wrist_pose_live_{left,right}.parquet        <- canonical, written today
derived/ wrist_pose_refined_{left,right}.parquet     <- ladder 11, ADDED not substituted
         delta_tcp_live_{left,right}.parquet         <- same rule when delta_tcp appears (no stream exists yet)
         delta_tcp_refined_{left,right}.parquet
```

```
live    = value a causal/online estimator produced at recording time
refined = offline, non-causal trajectory from the episode's full raw camera + IMU

INVARIANT: refined NEVER overwrites or deletes live. They are different data.
```

* Provenance per stream in `metadata.json.pose_streams`: `pose_type`, `estimator`, `causal`, `source_camera`,
  `source_imu`; refined later adds `source_episode` / `refinement_backend` / `refinement_version`.
  `declare_pose_stream()` already accepts those — the values are deliberately not invented now.
* **Backward compatibility:** `stream_path()` prefers the canonical name and falls back to the legacy
  `wrist_pose_{side}.parquet`, which *is* a live stream (the online estimator wrote it). Old episodes stay readable;
  new recordings never write the legacy name. No bulk rewrite of existing episodes.
* Pose is strictly a derived signal, but the live one sits in the raw episode bundle because it is recording-time
  state. The invariant above matters more than the folder.
* Payoff: training configs select `action_pose_source: live | refined`, so the comparison is an ablation for free.

Scope kept: naming + loader fallback + provenance only. No refinement, smoothing, loop closure, new VIO backend,
no `HandVIOBackend`, and no change to the live estimator or `WristPoseProvider`.

## Research framing (do not let implementation obscure it)

VIO is data-acquisition infrastructure, **not** the scientific novelty. The claim under test:

> Can visual geometry and semantics alone infer actionable affordances that jointly determine robot-arm transport
> and dexterous Aero Hand configuration?

```
visual scene → visual affordance ─┬─ transport affordance → ΔTCP/SE(3) → reBot
                                  └─ grasp affordance     → Aero hand shape → Aero Hand
```

Focus on visually inferable quantities: grasp region, approach direction, wrist orientation, hand pre-shape, finger
configuration. Not slip control or contact-force inference.
