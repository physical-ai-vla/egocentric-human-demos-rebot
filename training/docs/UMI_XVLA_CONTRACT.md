# UMI canonical trajectory contract (read from upstream, not from memory)

Source: `universal_manipulation_interface/` at `diffusion_policy/dataset/umi_dataset.py` and
`diffusion_policy/common/pose_repr_util.py`, with the shipped `config/task/umi.yaml`.

## 1. The anchor is the LATEST observation pose

`umi_dataset.py:344-352` converts both the observation poses and the action poses with the same base:

```python
obs_pose_mat    = convert_pose_mat_rep(pose_mat,   base_pose_mat=pose_mat[-1], pose_rep=self.obs_pose_repr)
action_pose_mat = convert_pose_mat_rep(action_mat, base_pose_mat=pose_mat[-1], pose_rep=self.obs_pose_repr)
```

`pose_mat[-1]` is the newest pose in the observation horizon, i.e. the pose at the moment the policy is asked.
Note the action line reads `obs_pose_repr`, not `action_pose_repr` — an upstream quirk that is invisible with
the shipped config because both are `relative`.

## 2. `relative` means current-anchor, not incremental

`pose_repr_util.py:62`:

```python
elif pose_rep == 'relative':
    out = np.linalg.inv(base_pose_mat) @ pose_mat        # forward  (training label)
    out = base_pose_mat @ pose_mat                        # backward (decode at deploy)
```

So with `base = T_t`:

```
A_k = inv(T_t) @ T_{t+k}        k = 1..16      every step anchored on the SAME pose
P_j = inv(T_t) @ T_{t-j}                       observation history in the same frame
```

`config/task/umi.yaml:87-89` ships `obs_pose_repr: relative`, `action_pose_repr: relative`.

The other options exist and are NOT what upstream ships:

| pose_rep | meaning | note |
|---|---|---|
| `abs` | untouched world pose | — |
| `rel` | position difference, rotation right-multiplied by inv(base_rot) | marked "legacy buggy implementation" in the source |
| `relative` | `inv(base) @ pose` | **shipped default, the canonical UMI label** |
| `delta` | `np.diff` of positions, `R_k @ inv(R_{k-1})` of rotations | sequential increments — decoded back with `cumsum` |

**Our v4 is the `delta` family**: `action[t] = inv(T_t) @ T_{t+1}` stored per row and re-composed by the loader.
UMI offers that representation but does not use it, and the decode path for it is a cumulative sum, which is
exactly where composition error accumulates.

## 3. Everything else in the label

- rotation: `mat_to_pose10d` → `[pos3, rot6d]`, rot6d = the first two ROWS of R (`mat[..., :2, :]`).
- gripper: `data['action'][..., 7r+6]` — the absolute jaw width at that action step, carried through unchanged.
- bimanual: `robot{i}_eef_pos_wrt{j}` = `inv(T_other[-1]) @ T_this`, i.e. the other arm's CURRENT pose is the
  anchor (`umi_dataset.py:289-291`).
- episode-start term: `inv(T_start) @ T_obs` with deliberate noise added to `T_start`
  (`scale 0.05` on all six components), and only the ROTATION part is kept — the position line is commented out.
- action layout per robot: `[pos3, rot6d, gripper]` = 10, so a bimanual row is 20 wide, same as ours.

## 4. What this changes for X-VLA

The architecture does not change. What changes is what the 16 rows of the target mean:

```
v4  (now)          target[k] = inv(T_{t+k})   @ T_{t+k+1}     sequential, decoded by composing
v5  (UMI)          target[k] = inv(T_t)       @ T_{t+k+1}     current-anchor, decoded in one step
state (both)       inv(T_t) @ T_{t-1}                          already current-anchor
deploy decode v5   T_target_k = T_measured_now @ A_k           no accumulation, and a latency skip is free
```

A LeRobot row cannot carry a v5 label as a single 20-vector, because the label depends on the query frame:
row `t+5`'s action differs depending on whether the query was `t` or `t+1`. So the chunk must be built per
query — either stored as a `(16,20)` action column, or generated in the collator from absolute poses.

## 5. Unit tests the converter must pass before any training

1. `A_1` equals v4's stored step-0 delta for the same frame.
2. `A_16` equals the composition `ΔT_t ΔT_{t+1} … ΔT_{t+15}` of the v4 labels (same endpoint, tolerance ~1e-9).
3. `T_t @ A_k` reproduces the source absolute pose `T_{t+k}` (round trip through rot6d included).
4. left/right channel order matches v4 (`robot0` = left).
5. gripper at step k is the width at `t+k`, and the episode's final steps do not run past its end.
6. chunk length 16 with no padding invented at the boundary — a frame without 16 real futures is dropped,
   the way v4 drops the last frame for having no `t+1`.


## 6. Names and the state definition, frozen (2026-09-23)

Call this branch **UMI current-anchor**, never just "relative": the v4 delta form was relative too, and the
two were confused once already.

The observation state is the bridge both tracks must build identically:

```
prev_rel_t = inv(T_t) @ T_{t-1}          the PREVIOUS pose expressed in the CURRENT frame
                                         (this is the inverse of the forward motion inv(T_{t-1}) @ T_t)
per arm    [prev_rel pos3+rot6d (9), current width (1)] = 10      bimanual 20
t = 0      prev_rel = identity and state_prev_valid = 0
```

Verified on real R150 frames at 2.98e-08 (float32 precision). Both tracks call `umi_action.py`; neither
re-implements it.

Deployment regression guard: every horizon pose attaches to the same measured pose,

```
T_target[k] = T_now @ A[k]           never  T_target[k-1] @ A[k]
```

`umi_action.from_current_anchor` is a single left-multiply and has no sequential path, so the forbidden form
cannot be written through the shared module.
