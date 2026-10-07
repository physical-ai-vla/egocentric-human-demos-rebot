# RELCART20 state contract (real side frozen 2026-09-29): handoff to the ego-dataset owner

The ego RelCart20 dataset (`ego_relcart20_rel16_v3d`) is built by the ego agent. The real dataset `r180_relcart20_rel16_v3d` is
frozen and training (run `HEAD180-RELCART20-REL16V3-D600K`, 4090 GPU0). The two datasets must match this contract exactly;
the VLA must not be able to tell ego from real by the state.

## State (20, float32), order frozen

| dims | field | definition |
|---|---|---|
| 0:3 | L_rel xyz (m) | translation of `T_rel = inv(T_L_anchor) @ T_L(t)` |
| 3:9 | L_rel rot6d | **first two ROWS** of R_rel (`umi.common.pose_util.mat_to_rot6d` = `mat[..., :2, :]`); identity = `[1,0,0, 0,1,0]` |
| 9:12 | R_rel xyz (m) | same, right arm |
| 12:18 | R_rel rot6d | same convention |
| 18 | g_L | jaw openness at t, 0 = CLOSED, 1 = OPEN, clip [0, 1] |
| 19 | g_R | same |

- **TCP:** `rebot_fk_torch.tcp(q12)` in the robot base frame. This equals the dataset TCP to 0.000 mm / 0.00°. Ego uses the same FK on its pseudo-q.
  - Deploy path: `infer_core_v4._tcp_mat` (eef_kin + V4_FRAME_FIX). It agrees with `rebot_fk_torch` to 0.0003 mm.
- **Anchor:** the pose at the **first row (frame_index 0) of each LeRobot episode**.
  - Segment = episode. On the real side, an episode is one teleop recording of the whole 3-stack task, with no finer segmentation.
  - Ego must use the same semantics: the first row of each frozen ego episode. Do not re-cut segments.
  - Do not use a sliding "current t" anchor or a global recording origin.
- **Gripper:**
  - Real: follower jaw width / `W_OPEN`, with `W_OPEN = 270 * 0.05 / 118 m` (raw −270 = fully open).
  - Ego: `open_fraction`.
  - Both: 0 = closed, 1 = open.
- **Action, aux, loss:** unchanged from REL16-v3d.
  - Action: REL16 `A_k = inv(T_t) T_{t+(k+1)dt}`, continuous grip, Δq12.
  - Plus `aux.q_t`, and the ego `aux.fk_mask` where applicable.
  - The state anchor (episode start) and the action anchor (current pose) differ by design.

## ⚠ 2026-09-29 addendum: anchor = TASK-episode start (read this before building ego)

"First row of the LeRobot episode" is only correct when the LeRobot episode IS the whole task demonstration. That holds for
real: one episode = one 3-stack teleop recording. The anchor semantics are **task-episode start-relative**:
`T_rel(t) = inv(T_task_start) @ T(t)`, with one anchor for the whole task.

1. **Audit first.** In the frozen ego-v3d, find out whether one LeRobot episode is a whole original human demonstration or a
   retarget segment (for example 65 frames). Report which one it is before building anything.
2. If episodes are **segments**, do NOT use the segment's first row as the anchor.
   - Recover each segment's `source_episode_id` and its source-frame mapping from provenance.
   - Use the pseudo-TCP of the ORIGINAL demonstration's first frame as the anchor. All segments of one source episode share that
     one anchor.
   - Example: a segment covering source frames 100..164 has `T_rel(100) = inv(T_0) T_100`, not identity.
3. New gate **QA4**: `same source_episode_id ⇒ identical T_anchor` (hash of the 2×4×4 anchor, per source episode). It must PASS.
   QA1 then applies to the first frame of each SOURCE episode, not to every segment.
4. If the original start pseudo-q/TCP cannot be recovered, **stop and report**. Do not fall back to segment-start anchors
   silently. The fallback decision (restore the source anchor vs. a matched segment-relative real dataset) is the user's.
5. The robot UI matches the task-episode semantics: the anchor is taken once at `/run`, and the whole manipulation task runs on it.

## ⚠ gripper dims 18:20 are NOT final for ego

Build and QA the 18 pose dims. Do not final-freeze g_L/g_R until the UMI open_fraction provenance has been audited:
- which raw signal it comes from
- the real full-close and full-open positions
- how the command/sensor maps to the actual aperture

It must be mapped to the same normalized aperture as real (0 = closed, 1 = open; real = follower width / 0.11441 m).

## Reference implementation

- Dataset derive (node): `umi_bridge/umi76/derive_relcart20.py`, sha256 prefix 36ecf859.
- Runtime / QA function: `holobrain-mac-model/infer_core_v4.relcart20_state(anchor, cur, w)`.
- QA script: `umi_bridge/rel16_audit/relcart20_qa.py`.

## Gates (real results; ego must pass the same)

| gate | real result |
|---|---|
| QA1 anchor rows == identity | max 2.8e-16 (PASS) |
| QA2 independent recomputation (deploy FK + relcart20_state vs stored) | pos 0.0003 mm, rot6d 6.9e-7, grip 0 (PASS) |
| QA3 all non-state columns byte-identical to the parent | PASS |

## Real distribution (for the ego↔real comparison, relcart20_qa.py)

| | L | R |
|---|---|---|
| rel xyz p1 (cm) | [-15.9, -15.8, -12.0] | [-15.8, -41.7, -21.1] |
| rel xyz p50 (cm) | [-0.0, 2.3, 0.3] | [-0.2, -3.9, 0.2] |
| rel xyz p99 (cm) | [12.0, 41.9, 22.7] | [11.0, 16.6, 23.5] |
| rel xyz mean ± std (cm) | [-0.5, 9.2, 2.6] ± [4.6, 14.5, 6.7] | [-1.7, -8.9, 2.1] ± [4.9, 13.6, 8.5] |
| rel rotation p50 / p90 / p99 (deg) | 20.3 / 69.4 / 100.0 | 24.3 / 97.5 / 141.0 |
| gripper mean g | 0.205 | 0.260 |
| p(g ≥ 0.6) | 0.136 | 0.196 |
| transitions / episode | 6.0 | 6.1 |

Frozen: data/meta are read-only on the node and on the Mac. `umi_bridge/umi76/r180_relcart20_rel16_v3d.SHA256SUMS` is on the node.
