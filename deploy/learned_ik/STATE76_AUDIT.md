# STATE76_AUDIT (2026-09-29)

Source of truth: `umi_bridge/umi76/umi76_to_lerobot.py:episode_frames` (packing) + measured `r180_umi76_rel16_v3d/meta/stats.json`.
Nothing here changes a dataset or the running HEAD180-REL16V3-D600K run.

## Headline

**state76 contains NO raw joint angles.** Every pose channel is a TCP pose (computed by FK from follower q, in the
dataset TCP convention) expressed *relative to a current TCP frame*. There is no absolute base-frame pose either.
The only robot-unit channels are the gripper widths (m).

Notation: T_i(t) = TCP pose of arm i at t (4x4), dt = 3/59.94 s (50 ms), o = other arm.
pose9 = [pos3 (m) | rot6d] with rot6d = **first two ROWS** of R (`umi.common.pose_util.mat_to_rot6d`: `mat[..., :2, :]`).
Packing is timestep-major, older frame first ([t−dt, t]).

## Per-dimension table

Class key:
- **A**: informative Cartesian channel that is robot-agnostic given a shared TCP/frame convention
- **B**: constant by construction (zero information)
- **C**: deterministic duplicate of other dims
- **D**: embodiment layout (inter-arm geometry of this robot's mounting)
- **E**: robot-unit gripper
- **F**: raw joint (none)

| idx | arm | block | time | semantic | unit | measured std | robot-specific? | Cartesian-replaceable? | class |
|---|---|---|---|---|---|---|---|---|---|
| 0–2 | L | pos | t−dt | pos of inv(T_L(t))·T_L(t−dt), self history translation | m | 2–3.5 mm | no (TCP convention only) | already Cartesian | A |
| 3–5 | L | pos | t | pos of inv(T_L(t))·T_L(t) = 0 | m | 3e-17 | no | — | **B** |
| 6–8 | L | pos_wrt | t−dt | pos of inv(T_R(t))·T_L(t−dt) | m | 0.10–0.22 | inter-arm geometry | already Cartesian | A + D |
| 9–11 | L | pos_wrt | t | pos of inv(T_R(t))·T_L(t) | m | 0.10–0.22 | inter-arm geometry | yes | A + D (≈ 6–8; they differ only by the self motion) |
| 12–17 | L | rot | t−dt | rot6d of inv(R_L(t))·R_L(t−dt), self history rotation | — | ≤1.1e-2 | no | yes | A |
| 18–23 | L | rot | t | rot6d(I) = [1,0,0,0,1,0] | — | 0 | no | — | **B** |
| 24–29 | L | rot_wrt | t−dt | rot6d of inv(R_R(t))·R_L(t−dt) | — | 0.2–0.6 | inter-arm | yes | A + D |
| 30–35 | L | rot_wrt | t | rot6d of inv(R_R(t))·R_L(t) | — | 0.2–0.6 | inter-arm | yes | A + D |
| 36 / 37 | L | grip | t−dt / t | follower jaw width | m (0–0.114) | 0.034 | **yes** (reBot jaw geometry; not the 0..1 action g) | yes → openness 0..1 | E |
| 38–40 | R | pos | t−dt | self history translation | m | 2–3.6 mm | no | yes | A |
| 41–43 | R | pos | t | 0 | m | 3e-17 | no | — | **B** |
| 44–49 | R | pos_wrt | t−dt, t | pos of inv(T_L(t))·T_R(·) | m | 0.10–0.18 | inter-arm | yes | A + D, **C** given 24–35 + 6–11 |
| 50–55 | R | rot | t−dt | self history rotation | — | ≤1.4e-2 | no | yes | A |
| 56–61 | R | rot | t | rot6d(I) | — | 0 | no | — | **B** |
| 62–73 | R | rot_wrt | t−dt, t | rot6d of inv(R_L(t))·R_R(·) | — | 0.2–0.5 | inter-arm | yes | **C**: at t it is the transpose of 30–35 (measured means match: 62=24, 63=27, 65=25, ...) |
| 74 / 75 | R | grip | t−dt / t | follower jaw width | m | 0.039 | **yes** | yes | E |

## Counts

| class | dims | notes |
|---|---|---|
| B constant | 18 (3–5, 18–23, 41–43, 56–61) | carry no information; they come from the UMI layout |
| A self-history motion | 18 (0–2, 12–17, 38–40, 50–55) | this is the only velocity-like signal; std is small, 2–3 mm and ~1° per 50 ms |
| A+D cross-arm | 36 (6–11, 24–35, 44–49, 62–73) | roughly half is a deterministic duplicate (the R-wrt-L current pose = inverse of the L-wrt-R current pose) |
| E gripper width | 4 | robot units |
| F raw joint | **0** | |

## Consequence for the "remove raw joints / cart20" proposal

1. There is no joint channel to remove. The premise that "action is cross-embodiment but observation is
   reBot-specific because of raw q" does not hold for state76. It already follows "VLA sees Cartesian only; q lives only in FK".
2. The proposed `cart20` = [p_L, R6_L, p_R, R6_R, g_L, g_R] in the **base frame** is not a subset of state76. It
   **adds** absolute base-frame pose, which state76 deliberately does not have, and it drops self-history and
   cross-arm. Absolute base-frame xyz is *more* embodiment/mounting-specific than the current relative channels,
   unless a canonical workspace frame is defined first (which the advisor also notes).
3. The existing `slim20` state (infer_core_v4 `build_state`) is yet another design. It is per arm:
   [inv(T_t)·T_{t−dt} pose9, width], i.e. self-history + gripper with no cross-arm and no absolute pose. That makes it
   the subset {A self-history + E} of state76.
4. Real robot-specific parts of state76, in order:
   - E (widths in m): replaceable by openness 0..1, the same as the action g.
   - D (inter-arm offset): depends on how the two arms are mounted; it is not a joint.
   - B (18 dead dims): harmless but useless.
5. Per the plan ("state76 dimension을 충분한 audit 없이 임의 제거 금지"), no dims are removed now. A state ablation would
   be a *new, separately named* dataset and run, with the variant fixed before training. Candidates:
   - S76: as-is (HEAD180-REL16V3)
   - S20-rel: slim20, with openness in place of width
   - S20-abs: cart20 as proposed, in the dataset TCP base frame

   This comes after LearnedIK Phase A, not now.
