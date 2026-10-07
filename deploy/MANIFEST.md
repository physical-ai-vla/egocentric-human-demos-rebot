# deploy/ MANIFEST

Copied 2026-10-07 with `cp -p` (mtimes preserved). Sources were not modified. There is no overlapping deploy code in `~/egocentric-human-demos-rebot`, so every file comes from the working dirs.

| dest | source | notes |
|---|---|---|
| `infer_core_v4.py` | `~/holobrain-mac-model/infer_core_v4.py` (10-07 13:50) | latest |
| `mac_v4_smoke_ui.py` | `~/holobrain-mac-model/mac_v4_smoke_ui.py` (10-07 14:45, 3,814 lines) | latest |
| `pink_ik.py` | `~/holobrain-mac-model/pink_ik.py` (10-01) | |
| `eef_kin.py` | `~/holobrain-mac-model/eef_kin.py` (09-29) | same file as `pipeline/pseudo_joint/eef_kin.py` |
| `hra_direction_probe.py` | `~/holobrain-mac-model/hra_direction_probe.py` | |
| `TODO_gripper_hold.md` | `~/holobrain-mac-model/TODO_gripper_hold.md` | |
| `loaders/load_hra_rightonly_ckpt_to_ui.sh`, `load_relonly_run_ckpt_to_ui.sh`, `load_relonly_5090_run_ckpt_to_ui.sh` | `~/umi_bridge/` | active loaders (called by the swappers / HRA scripts) |
| `loaders/legacy/*.sh` (14) | `~/umi_bridge/load_*_ckpt_to_ui.sh` | per-run loaders of earlier runs (cart20, relcart20 v3/v3L/v4, relonly d6/ego/egov2b/ftego/r384/v4, v3, v3L, ssd, generic). Only the current file of each; every `.bak_*` dropped |
| `swappers/auto_ui_d20_e5_v5.sh`, `swappers/auto_ui_hra_8056.sh` | `~/umi_bridge/` | latest swapper versions only (v1-v4 of `auto_ui_d20_e5` dropped) |
| `robot_service/robot_service_mit.py` | `~/robot-cockpit/robot_service_mit.py` | ours; needs the colleague's `bh_indy7_LeRobot` reBot plugin (not included) |
| `robot_service/usb_arm_ports.py` | `~/robot-cockpit/usb_arm_ports.py` | imported by the service (left/right by USB location ID) |
| `learned_ik/` | `~/umi_bridge/learned_ik/{lik0_common,train_lik0,build_lik0_data,eval_ik_backends}.py`, `*.md`, `LEARNEDIK_V0_SPLIT.json` | optional IK backend code + contracts; `data/`, `runs/` (534 MB, weights), logs excluded |

## Excluded

- All `*.bak*` (many next to the loaders, UI and core), `*.log`, `ui_swap_log.tsv`, per-port state dirs (`~/umi_bridge/.auto_ui_d20_e5/`).
- Every local checkpoint copy `~/holobrain-mac-model/ckpt_UI_*`, `chunks_ep0.npz` and other replay data.
- `~/holobrain-mac-model/umi_pkg` (UMI upstream, imported as `umi.common.pose_util`). It belongs to the upstream part of the repo.
- `~/robot-cockpit/robot_service.py` (the older POS_VEL service, :8020) and the rest of robot-cockpit (recording app, owned elsewhere).
- 5090 inference/cuRobo server code (`/srv/data/johann/{ui_infer,curobo}`): turned off 10-03 and not requested.

## Notes

- The secret scan is clean. `infer_core_v4.py` contains the 5090 tailnet IP as the `V4_CUROBO_URL` default (`100.64.0.5:8790`), and the loaders use `bh-aiteam@100.64.0.2` via `-J head-lp`.
  Both are kept: a private repo, and the scripts need them.
- Loaders and swappers hard-code `~/holobrain-mac-model`, `~/umi_bridge` and `~/xvla-mac/bin/python`. In a checkout, either keep that layout or symlink.
- The PLAN gripper issue in `README.md` was found by reading the code (line citations given). It has not been reproduced on the robot.
