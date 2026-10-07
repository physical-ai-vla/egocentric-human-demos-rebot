# deploy/: serving X-VLA checkpoints on the reBot (Mac MPS + Pink IK + MIT robot service)

## Components

| file | role |
|---|---|
| `infer_core_v4.py` | Inference core `V4Inferencer`. It handles observation -> state (RELCART20 / slim20 / umi), X-VLA flow sampling (MPS bf16), the REL-only zero mask, chunk decode `T_k = T_now @ A_k`, the IK backends (`pink`, numerical, learned, pinkdq/policydq, cuRobo client), the gripper mapping, RTC, the one-arm (`V4_ARM_ONLY=right`) contract and the fixed anchor. |
| `mac_v4_smoke_ui.py` | Flask web UI and controller. It provides step / stream / **PLAN** execution, `/mode` live controls, the checkpoint dropdown, the cube-size stop, the trial logger and HRA safety gates. |
| `pink_ik.py` | Pinocchio/Pink QP IK. Frame tasks: pos 1.0 / ori 0.5. Posture toward the seed is 1e-3. TCP = gripper_link + 70 mm x. URDF from `REBOT_URDF`, default `~/holobrain-mac-model/reBot_B601_DM_dualarm.urdf`; the same file is in `training/robot/`. |
| `eef_kin.py` | FK and continuity IK (same URDF/TCP model as the labels). The same file is in `pipeline/pseudo_joint/`. |
| `hra_direction_probe.py` | Read-only offline probe (`/observe` only). It computes the cosine between the predicted right-TCP motion and the direction to a touched cube target. |
| `TODO_gripper_hold.md` | Open bug: generic `V4_GRIPPER=hold` sends the raw jaw reading as a command. |
| `loaders/` | `load_*_ckpt_to_ui.sh <step> [port]`: fetch a checkpoint (node or SSD), verify sha256 + identity (run, dataset, domain), restart the UI with the run's V4_* contract, then restore the per-port control settings. Active loaders: `load_hra_rightonly_ckpt_to_ui.sh` (HRA, port 8056), `load_relonly_run_ckpt_to_ui.sh` (4090 REL-only bimanual), `load_relonly_5090_run_ckpt_to_ui.sh`. `legacy/` holds the earlier per-run loaders. |
| `swappers/` | `auto_ui_d20_e5_v5.sh` (bimanual ports 8052/8053; loads each new checkpoint and skips pinned ports) and `auto_ui_hra_8056.sh` (HRA). These are the latest versions only. |
| `robot_service/robot_service_mit.py` | Our HTTP robot service on **:8021**: two arms in MIT impedance mode with an integral feed-forward, a 100 Hz control thread, `/observe`, `/execute_step` and `/plan/{open,rows,progress}`. **It depends on the colleague's reBot plugin, which is NOT included:** `bh_indy7_LeRobot` (`bh_indy7_lerobot`, `bh_indy7_lerobot_gui`, `lerobot_plugin_rebot`). Run it in that env: `~/bh_indy7_LeRobot/.venv/bin/python robot_service_mit.py`. Stop the old `robot_service.py` (:8020) first. `MIT_MOCK=1` runs without hardware. `usb_arm_ports.py` maps left/right by USB location ID. |
| `learned_ik/` | Optional learned-IK backend (`lik0_common.py` + training/eval scripts and contract docs). Weights not included. |

The core also needs these, which are not in this folder:
- `umi.common.pose_util` from the UMI upstream (`~/holobrain-mac-model/umi_pkg`, upstream part of the repo).
- `rel16_aux_relonly.py`, which the core imports from `~/umi_bridge/rel16_audit/relonly`. The same file is `training/relonly/rel16_aux_relonly.py`; point the path there or symlink it.
- LeRobot with X-VLA (`lerobot.policies.xvla`) in `~/xvla-mac` (py env).
- `learned_ik` is optional and loads from `~/umi_bridge/learned_ik`.

## Serving a checkpoint

Normally you use a loader, which sets every variable below, guards identity and restarts the UI:

```bash
# HRA right-only (port 8056); A100/CT models need the fixed anchor and zero UMI convention
RUN=HRA-RIGHTONLY-ROBOTCAM-CT7-OLD85-LOSSMASK-D20-B8-100K DS_EXPECT=ego_hra_ct7_old85_v1_train \
V4_FIXED_ANCHOR_JSON=~/c8/hra_a100/G_anchor.json UMI_TCP_PITCH_DEG=0 UMI_TCP_OFFSET_MM=0,0,0 \
  bash loaders/load_hra_rightonly_ckpt_to_ui.sh 5000 8056
# bimanual REL-only run on the 4090 (port 8053)
bash loaders/load_relonly_run_ckpt_to_ui.sh <step> 8053
```

The by-hand form is `V4_* ... ~/xvla-mac/bin/python mac_v4_smoke_ui.py`, served at `http://localhost:<V4_UI_PORT>`. The environment contract has to match how the checkpoint was trained:

| var | meaning (REL-only CART20 value) |
|---|---|
| `V4_CKPT` | local `pretrained_model` dir |
| `V4_DEVICE` / `V4_DTYPE` | `mps` / `bf16` |
| `V4_RELONLY=1` | installs the 20:32 zero mask (**required** for REL-only and loss-mask checkpoints; refuses pinkdq/policydq) |
| `V4_ACTION_MODE=umi` | `T_k = T_now @ A_k` (current-anchor chunk); `delta` is the old incremental mode |
| `V4_STATE_MODE=relcart20` | state = `inv(T_anchor) T_t` per arm + openness. The anchor is the pose at `/run`, or `V4_FIXED_ANCHOR_JSON` (`T_base_F`; B uses `B_anchor_F.json`, CT5-CT7 use `G_anchor.json`) |
| `V4_CHUNK=16`, `V4_EXEC_K` | chunk length; rows executed per inference in step mode (4) |
| `V4_DENOISE_STEPS` | 5 (bimanual) / 10 (single arm) |
| `V4_TASK` | instruction string (HRA: "Approach to the red cube") |
| `V4_GRIPPER` | `continuous` (bimanual REL-only, with `V4_GRIP_THRESH`), `binary`, `relative`, `predict`, `hold` (see TODO) |
| `V4_ARM_ONLY=right`, `V4_ARM_ONLY_GRIP=hold` | one-arm contract: dummy left state, black left wrist, left arm and both jaws held |
| `IK_BACKEND`, `V4_PINK_LOCK`, `V4_PINK_ORI`, `V4_PINK_POSTURE` | `pink`; joint5 lock (IK_PROFILE=lock) or free |
| `V4_UMI_TCP_PITCH_DEG`, `V4_UMI_TCP_OFFSET_MM` | HandUMI -> robot TCP convention. Only for HRA v1 (33.1 deg + offset). **0 / 0,0,0 for robot-TCP-label models** (robotcam v2, A100) |
| `V4_GLOBAL_ROT180`, `V4_GLOBAL_MIRROR` | global camera orientation, decided by the training relation (0 for current data) |
| `V4_WRIST_ZOOM` | right-wrist zoom (HRA used 1.5) |
| `V4_ROBOT` | robot service URL (`http://localhost:8021` = MIT service), `V4_JAW_TORQUE=0` with it |
| `V4_PLAN=1` + `V4_PLAN_*` | PLAN execution (below) |
| `V4_RTC`, `V4_RTC_*` | real-time chunking guidance (optional, roughly doubles MPS cost) |
| `V4_REMOTE_INFER`, `V4_CUROBO_URL` | optional 5090 inference / cuRobo IK. OFF since 10-03, because LAN latency made it slower than MPS |
| logs | `V4_PLAN_LOG` (~/v4_plan.jsonl), `V4_CYCLE_LOG`, `V4_IK_LOG`, `V4_STREAM_LOG`, `V4_TRIAL_LOG` |

Ports: 8053 = stacking / co-train models, 8056 = HRA right-only (the HRA loader refuses 8053). Per-port sticky settings (pin, zoom, cube stop,
IK profile, UMI pitch/offset, PLAN values) live in `~/umi_bridge/.auto_ui_d20_e5/<port>.*`. Before any UI restart, check that `/status` shows
`ui.running == false`. After the restart, compare the controls with their earlier values.

## PLAN settings that matter

PLAN mode keeps one forward pass in flight and streams IK-solved rows (50.05 ms apart) to the robot service. The service plays them as 100 Hz knots.
The seam between chunks is controlled by four values. Their **code defaults are bad** for current models:

| setting | `/mode` key | env | default | **use** |
|---|---|---|---|---|
| joint-space EMA blend | `planema` | `V4_PLAN_EMA` | 0.35 | **0.25** |
| cross-fade rows | `planxf` | `V4_PLAN_XFADE` | 4 | **0** |
| polynomial smoothing (deg) | `plansm` | `V4_PLAN_SMOOTH_DEG` | 3 | **0** |
| min executed rows before next forward | `planmin` | `V4_PLAN_MIN_EXEC` | 6 | **1** |

`curl -X POST 'http://localhost:8056/mode?planema=0.25&planxf=0&plansm=0&planmin=1'`. On 10-07 a UI restart silently reset these values to the defaults,
and the operator reported that inference quality dropped a lot. The HRA loader now captures and restores them (MODEQ). Execution settings are part of the policy, so record them with every run.

## Cube-size stop

`V4_CUBE_STOP_PX` (loader `CUBE_STOP_PX`) or `POST /cube_stop?px=<n>&follow=<0|1>` controls the stop. Every 0.1 s the UI measures the red cube's apparent size,
sqrt(area) of the largest HSV-red blob in the **raw** 640x480 right-wrist frame (before zoom). It stops after 2 consecutive readings >= px; 170-250 px were used.
With follow on (`V4_CUBE_FOLLOW=1`, the default), it resumes `/run` with the same fixed anchor once the cube is visible and below 85 % of px on 3 checks.
0 = off. This is a proximity proxy, not verified contact.

## Trial logger

`POST /trial` (JSON) logs one trial. Fields:
- `stack_level` 0-3 (required)
- `wrong_color`
- `failure_reason`
- `layout_id`
- `order_executed`
- `note`

There is also a TRIAL panel in the UI. Automatic fields come from `V4_CKPT`: model type, step and robot dataset/episodes.
The file is `V4_TRIAL_LOG` (`~/rebot_trials.jsonl`). `GET /trials` gives a per-checkpoint summary. Success = level 3 and not wrong colour.
It was not used during the HRA runs, which is why no HRA success rate exists. Use it.

## KNOWN ISSUE: bimanual PLAN mode never opens the grippers

In PLAN mode the forward job compiles each chunk row with **no gripper widths**. `mac_v4_smoke_ui.py` line 1053, inside `_plan_forward_job`:

```python
P = P_all[j].copy(); Q = Q_all[j].copy(); W = np.asarray(L_["widths"][j]).copy()   # line 1052: W is computed ...
a, ikerr, ok, clipped = INF.solve_waypoint(seed, P, Q, None)                          # line 1053: ... but widths_k=None is passed
```

`W` is only stored in the row dict (line 1068) and never reaches the command. In `infer_core_v4.py` `solve_waypoint` (def at line 1010),
the gripper command is built at lines 1190-1193:

```python
cmd[gi] = (self._grip_relative(...) if (GRIPPER_MODE == "relative" and widths_k is not None) else
           grip_to_cmd(widths_k[r]) if (GRIPPER_MODE in ("predict", "binary", "continuous", "relative") and widths_k is not None)
           else float(q_now[gi]))
```

With `widths_k=None`, every gripper mode falls through to `float(q_now[gi])`. Here `q_now` is the seed. Its gripper entries are the **measured raw jaw**
(lines 1036, 1047 and 1072), on the 0 (closed) .. -270 (open) scale. That value becomes the row's `action`, which `/plan/rows` sends to the robot unchanged (line 1164).
The robot expects a 0..45 *command*, and the bridge clips negatives to 0 = closed. This is the same failure that `TODO_gripper_hold.md` and the `jaw_hold_cmd` docstring (`infer_core_v4.py` lines 333-336) describe.

Result: in bimanual PLAN mode (`V4_PLAN=1` without `V4_ARM_ONLY`), the predicted gripper channel is ignored. A closed jaw stays closed, and an open jaw is
commanded closed, so the grippers most likely never open during a PLAN run. The streaming path does pass widths: line 1249, `solve_waypoint(q14, P, Q, W if ... gripper_mode in (...) else None)`.
The one-arm HRA path is unaffected, because it holds the jaws through `jaw_hold_cmd` anyway (lines 1194-1195).
**Status: not fixed. It was found by code reading and has not been reproduced on the robot.** The fix is to pass `W` (with the same gripper-mode condition as line 1249) at line 1053.
Also apply `jaw_hold_cmd` to the `widths_k is None` fallback, together with the regression test listed in `TODO_gripper_hold.md`.

Line numbers refer to the files in this folder, copied 2026-10-07 (`infer_core_v4.py` 10-07 13:50, `mac_v4_smoke_ui.py` 10-07 14:45).
