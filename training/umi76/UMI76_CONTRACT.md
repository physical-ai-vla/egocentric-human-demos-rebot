# HEAD180 UMI76 — frozen contract (2026-09-25)

**Experiment A: UMI76 temporal/lowdim-compatible baseline.** state76, 50.05 ms observation and action
spacing, action horizon 16, REL16 vs DELTA16.

Deliberately not called "UMI-exact": what matches upstream is the low-dim packing and the temporal
contract. The vision stack does not -- upstream bimanual ships two wrist cameras and we feed three
(global + both wrists) -- and the policy is X-VLA's flow matching, not UMI's diffusion policy. Claiming
"UMI reproduction" past those two differences would be wrong.

Goal: reproduce upstream **bimanual** UMI observation semantics on the HEAD pool, and compare ONLY the
action representation, REL32 (current-anchor) against DELTA32 (sequential body delta). No slim20 control.

Source of truth read for this: `universal_manipulation_interface/diffusion_policy/config/task/umi_bimanual.yaml`,
`diffusion_policy/dataset/umi_dataset.py`, `diffusion_policy/model/vision/timm_obs_encoder.py`,
`diffusion_policy/common/pose_repr_util.py`.

## 1. Data

HEAD only: `r150_umi.zarr` + R150 headview, `r30_umi.zarr` + R30 day4 headview. 180 episodes.
global_transform **none**, mirror **none**, wrist transform **none**. Images resize 224x224 only.
Cameras: global + left_wrist + right_wrist (upstream bimanual ships 2 wrist cameras and no global; we keep
our 3, and this is the one place the setup is richer than upstream).

## 2. Observation state — 76D, upstream packing, NOT reordered

`low_dim_obs_horizon = 2`. Per timestep 38D, flattened 76D.

`timm_obs_encoder.py:169` does `low_dim_keys = sorted(low_dim_keys)` and then, per key,
`data.reshape(B, -1)` on a `(B, T, D)` tensor. So the order is **alphabetical by key**, and within a key it
is **timestep-major**: `[t0_d0..t0_dD-1, t1_d0..t1_dD-1]`. The yaml's own key order is NOT the packing order.

| slice | key | per-step | x horizon |
|---|---|---|---|
| `[ 0: 6]` | `robot0_eef_pos` | 3 | 6 |
| `[ 6:12]` | `robot0_eef_pos_wrt1` | 3 | 6 |
| `[12:24]` | `robot0_eef_rot_axis_angle` | 6 (rot6d) | 12 |
| `[24:36]` | `robot0_eef_rot_axis_angle_wrt1` | 6 | 12 |
| `[36:38]` | `robot0_gripper_width` | 1 | 2 |
| `[38:44]` | `robot1_eef_pos` | 3 | 6 |
| `[44:50]` | `robot1_eef_pos_wrt0` | 3 | 6 |
| `[50:62]` | `robot1_eef_rot_axis_angle` | 6 | 12 |
| `[62:74]` | `robot1_eef_rot_axis_angle_wrt0` | 6 | 12 |
| `[74:76]` | `robot1_gripper_width` | 1 | 2 |

Definitions (`umi_dataset.py`):

```
self      convert_pose_mat_rep(pose_mat, base_pose_mat=pose_mat[-1],        pose_rep='relative')
cross-arm convert_pose_mat_rep(pose_mat, base_pose_mat=other_pose_mat[-1],  pose_rep='relative')
'relative' == inv(base) @ pose            (pose_repr_util.py:62)
```
Both anchor on the LATEST observation pose. At horizon 2 the self term at t1 is identity by construction;
it is kept anyway because upstream keeps it and this build does not re-pack.

Gripper is the **absolute jaw width**, both history steps, never a 0/1 label.

**There is no `wrt_start` term.** It exists in the single-arm `umi.yaml` and in dataset code, but
`umi_bimanual.yaml` never requests the key, so it never reaches the observation dict. Do not add it.

`robot0 = LEFT`, `robot1 = RIGHT` — our hardware convention, verified on HEAD by wrist-video correlation.
Never reassign arm identity from where something appears in an image.

## 2b. Temporal contract — physical time, not stored index

Upstream samples low-dim observations at `current_idx - idx * 3` on a ~59.94 Hz buffer
(`sampler.py:148`, reversed so the older frame comes first), and all latency terms are zero because
`dataset_frequeny: 0` makes `(camera_obs_latency - robot_obs_latency) * dataset_frequeny` vanish. So the
observation pair is separated by **3 / 59.94 = 50.05 ms**, and the action targets by the same 50.05 ms.

Our zarr is 30 Hz and the stored cadence is 15 Hz, so 50.05 ms is 1.5 raw frames -- no integer index
reproduces it. Copying `down_sample_steps = 3` onto stored frames would give 200 ms, four times upstream;
taking one stored frame would give 66.7 ms. **Neither is used.** History and action targets are produced by
interpolating the raw 30 Hz trajectory at the exact physical offsets:

```
query          t          = q / 30      (q = raw index of the row, stride 2 -> rows at 15 Hz)
history        t - 0.05005
action target  t + (k+1) * 0.05005      k = 0..15
position       linear interpolation
rotation       SO(3) slerp on the bracketing raw samples
gripper width  linear interpolation at the same timestamps
```

Both arms are interpolated at the SAME physical timestamps, for the self term and the cross-arm term alike.
Rows whose history falls before the episode start, or whose 16th action target falls past its end, are
dropped -- never padded.

## 2c. Gate result, 2026-09-25 (before any dataset was built)

`gate_umi76.py` on 3 real R150 episodes, 30 random rows, recomputing everything from the zarr with an
independent implementation rather than calling the converter's own interpolation:

```
state76 vs independent recomputation   2.97e-08
current self term == identity          1.42e-16
REL A[k] decodes to the source pose    2.97e-08
gripper == width at that instant       3.69e-09
observation dt / action dt             50.050 ms both
compose(D[0..k]) == A[k]               7.45e-05 mm / 1.19e-04 mm
DELTA[0] == REL[0]                     exactly 0
```

## 3. Action — the only thing that differs between the twins

Per arm `[pos3, rot6d, width] = 10`, bimanual `[LEFT10, RIGHT10] = 20`, **horizon 16** -> `(16, 20)`,
targets spaced 50.05 ms apart, so the chunk spans 0.80 s -- upstream's `action_horizon: 16` at upstream's
own spacing. This build is **experiment A, the UMI-exact temporal baseline**. A 32-step / 15 Hz version is
experiment B and must not be called an upstream reproduction; keeping both variables at once would confuse
"does the UMI formulation work" with "does the longer horizon work".

```
REL     A[k] = inv(T_t) @ T(t + (k+1)*0.05005)     k = 0..15, every step on the SAME anchor
DELTA   D[0] = A[0];  D[j] = inv(A[j-1]) @ A[j]    j = 1..15, stored per row, not stacked by the loader
both    gripper at step k = the absolute width at that future instant; D[j].width == A[j].width
deploy  T_target[k] = T_now @ A[k]                 never T_target[k-1] @ A[k]
```

Rows whose episode does not carry 16 real future targets are dropped. No padding: X-VLA does not read
`action_is_pad`, so a padded target is trained on as if it were real.

## 4. Twin invariants (gate before training)

Identical: episodes, rows, frame_index, timestamp, task, videos, state76, split, seed.
Geometry: `DELTA[0] == REL[0]`, and `compose(D[0..k]) == A[k]` for every k, tight at k=15.
Timing: sampled rows must show an observation dt of 50.05 ms and an action dt of 50.05 ms.
Identity: `self_rel[current] == I`, `cross[0_wrt1][current] == inv(T1_t) @ T0_t`, and the
mirror for `1_wrt0` -- these catch an anchor taken from the wrong arm or the wrong timestep.
Packing: `[0:10]` LEFT and `[10:20]` RIGHT in both.
Normalization: STATE identical across twins; ACTION stats computed per twin (the distributions differ by
construction); VISUAL identity.

## 5. Widening X-VLA proprio 20 -> 76

`soft_transformer.py:381` concatenates `[action_with_noise(20) | proprio(20) | time(32)] = 72` and feeds
`action_encoder = DomainAwareLinear(dim_action + dim_time + dim_propio, hidden_size, num_domains)`.

**proprio sits in the MIDDLE.** Appending the new 56 columns at the end would shift the time embedding into
the proprio slot and silently corrupt every timestep. The new input is 128 wide and the copy is:

```
new[:, :,   0: 20] = old[:, :,  0:20]     action   copied
new[:, :,  20: 96] = N(0, ref_std)        proprio  ALL 76 newly initialized, nothing copied
new[:, :,  96:128] = old[:, :, 40:72]     time     copied
bias unchanged
```

The old 20 proprio columns are **not** carried into the first 20 of UMI76. slim20 was
`[L prev_rel9, L width, R prev_rel9, R width]`; UMI76's first 20 are `robot0_eef_pos` history,
`robot0_eef_pos_wrt1` history and part of `robot0_eef_rot_axis_angle`. The count matches and the meaning
does not, so copying would install a wrong prior. `ref_std` is the std of the surviving action and time
columns, used instead of a generic xavier because the weight is stored as an Embedding of shape
`(num_domains, out*in)` and xavier would read fan-in off that flattened shape.

`DomainAwareLinear` stores the weight as `nn.Embedding(num_domains, output_size * input_size)`, so it has to
be reshaped to `(num_domains, hidden_size, input_size)`, padded on the input axis, and flattened back.

Everything else -- vision, language, policy transformer, action decoder, soft prompts -- is copied
unchanged. Gates: the action and time columns must be bit-identical to the source, and the proprio block
must NOT still equal the old slim20 weights. REL and DELTA start from the SAME widened directory.

## 6. State-dim gate (the silent failure this prevents)

`modeling_xvla.py:312` is `return pad_vector(state, self.model.dim_proprio)`. A 20-D state fed to a 76-D
model is zero-padded and trains without any error, producing a run that looks like UMI76 and is slim20 with
56 zeros. So the dataset's state width is asserted before training, and `XVLA_STRICT_STATE_DIM=1` makes
that padding raise instead of pad. The flag is off by default so other runs keep the padding behaviour.


## 7. Deployment path (built 2026-09-25, before any checkpoint existed)

`infer_core_v4.py` keeps the slim20 path untouched and branches on `V4_STATE_MODE`:

```
V4_STATE_MODE = slim20 | umi76      default slim20
V4_OBS_BUF_S  = 1.0                 ring-buffer length, seconds
UMI_HISTORY_DT = 3/59.94            50.05 ms, fixed
```

- `push_obs(joints14, t)` stores `(t, TCP(2,4,4), width(2))`; FK only, no model
- `_interp_at(tq)` linear position and width, Slerp rotation
- `build_state_umi76()` the alphabetical, timestep-major packing of section 2, cross-arm anchored on the
  other arm's CURRENT pose
- inference is refused until the buffer spans 50.05 ms

Live observations arrive at 15 Hz, so 50.05 ms falls between two samples and MUST be interpolated. Serving
the previous frame instead is a 66.7 ms history the policy never saw.

The action decode needed no change: `ACTION_MODE=umi` is already `T_now @ A[k]` and `delta` already
composes, so REL16/DELTA16 only need `V4_CHUNK=16`.

`gate_parity76.py`, model-free and robot-free, replays a synthetic 30 Hz trajectory into the deploy ring
buffer at the live 15 Hz cadence and compares against the converter's own construction:

```
deploy state76 vs converter packing   5.65e-05   (15 Hz buffer interpolated to 50.05 ms, ~0.06 mm)
REL16   T_now @ A[k] -> future pose    1.11e-16 m
DELTA16 compose then apply -> same     1.11e-16 m
refuses to build before 50.05 ms       PASS
```

First real-robot comparison: both twins at the SAME `exec_k`, `n_action=1`, rot180 false, mirror false.
