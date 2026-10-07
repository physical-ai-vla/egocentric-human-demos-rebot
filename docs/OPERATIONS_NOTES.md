# Operations notes (things that are easy to get wrong)

Collected from day-to-day work between September and October 2026. Each item cost at least one wrong run or a wrong
conclusion. Dates are when the fact was established.

## Robot (reBot B601) conventions

- **Gripper semantics.** Raw jaw 0 = CLOSED, −270 = OPEN. Commands are 0..45 with raw = −6·cmd; cmd ≥ 27 opens. A comment in
  `training/humanik_delta.py` saying "1 = closed" is wrong: the binarized label 1 means OPEN.
- **Gripper offset drift.** The follower gripper's open value shifts per power cycle (seen 0 / −100 / +90 / 358 wrap); closed
  clips at −270. Inference clips to [−270, 0]; check the jaw reading before `/run`.
- **TCP frame.** The dataset TCP and the `eef_kin` FK TCP differ by R_y(90°). Forgetting the frame fix caused an upward
  runaway (2026-09). The IK stack's TCP is `gripper_link` + 0.07 m along x and lies ~82 mm AHEAD of the physical jaw tip; use
  the URDF fingertip box for contact (HRA, 2026-10-06).
- **Left/right arms are bound by USB location ID**, not by port name. Use LEFT_LOC / RIGHT_LOC; never POST `execute_step` by
  hand.
- **Global camera rotation.** `V4_GLOBAL_ROT180=0`. Decide it against the measured training relation, not appearance.
- **After a robot-service restart**, `goal_interp` resets to ON; restore the UI's value.

## Deployment UI (`deploy/`)

- **PLAN settings are part of the policy.** Good values: `plan_ema 0.25, xfade 0, smooth 0, min_exec 1`
  (`/mode?planema=0.25&planxf=0&plansm=0&planmin=1`). UI restarts used to reset them to the code defaults (0.35 / 4 / 3 / 6)
  and inference quality dropped visibly (2026-10-07). The loader now restores them; still compare `/status` after a restart.
- **Never restart a UI mid-run.** Check `/status` → `ui.running == False` first.
- **Known issue:** in bimanual PLAN mode `solve_waypoint` gets `widths_k=None`, so jaws likely never open (see
  `deploy/README.md`). Single-arm HRA is unaffected (jaws held).
- **Generic `GRIPPER_MODE=hold` bug:** it sends the raw jaw (0..−270) as a 0..45 command → clipped to 0 = closed. HRA uses
  `jaw_hold_cmd` (raw/−6). See `deploy/TODO_gripper_hold.md`.
- Robotcam-trained models (HRA v2, A100) need TCP pitch/offset 0/0; anchored models need the same anchor file they were
  trained with (`B_anchor_F.json` for B, `G_anchor.json` for CT5–CT7).
- Per-port sticky settings live in `~/umi_bridge/.auto_ui_d20_e5/` (`.wristzoom`, `.cubestop`, `.ikprofile`, `.umipitch`,
  `.umioffset`, `<port>.pin`).
- Log every trial with the UI's TRIAL panel (`/trial`); without it there is no success rate (none was logged up to 10-07).

## Training (`training/`)

- **GPU nodes: RTX 4090 or RTX 5090 only, never the 5080s** (user rule, 2026-10-06). All GPU work goes through
  `ray job submit`, pinned with a node resource.
- **4090 (torch 2.6) and 5090 (torch 2.7.1) differ**: runs on different nodes are not bit-identical twins; say so when comparing.
- **steps must equal decay.** LeRobot silently rescales the cosine schedule when steps < decay; launchers refuse mismatches.
- **X-VLA domain id:** only base slots 10–17 were pretrained; 0 and 6 are untrained. Our embodiment uses slot 20 (set in the
  base's `policy_preprocessor.json`, there is no CLI flag).
- **Log cadence:** `log_freq` 100–200; logs use `\r`, parse with `tr '\r' '\n'`; the printed step is rounded ("22K"), compute
  step = line index × log_freq.
- **Never edit a running chain script.** Bash reads scripts incrementally.
- **A Mac reboot kills the archivers** (nohup on the Mac). Restart them, or the node's disk guard stops training.
- **Retention race:** a guard that deletes `training_state` must skip the newest checkpoint (it may still be writing).
- **LeRobot v3 writes:** a killed write leaves a footer-less parquet; use per-shard writers + aggregate.
- **`dataset_tools.delete_episodes` fails** on the 2-D (16, 32) action; select episodes with `--dataset.episodes`.

## Pose and scale (`pipeline/`, `hra/`)

- MASt3R-SLAM runs on the 5090 via Ray (the 4090 sm_120 build produced 194 tracked / 196 RELOC on one take; rebuilt for sm_89
  it matches the 5090). Never use a session-global scale: per-episode scale varies 0.19–0.55.
- IMU-VI scale is unobservable on slow motions (HRA); use cube PnP (edge **3.8 cm**, not 5 cm) or the origin plane.
- The exported right-wrist camera–IMU rotation was off by ~18° (`collector/configs/calibration/camera_imu_right_rotfix_20261006.npy`).
- MASt3R-SLAM can jump > 10 cm in one frame without a loss flag; always run the v2b jump filter.
- `open3d` 0.18 segfaults with numpy 2 on the Mac; use trimesh + own ICP.

## Data lineage

- Stacking ego data: 349 HandUMI episodes (09-15..17) + HRL80 (60). Gen-1 = 561 segments; Gen-2 CART20 v2b = 297/32 episodes.
- Robot data: R150, R312c (312 eps), R384, R30 = R312c episodes 150–179.
- Approach data: HRA_red (166 kept of 200), HRA_A100 (107 kept on 10-06; CT7-old = 85).
- Datasets and checkpoints are NOT in this repo (node disks, NFS, and the portable SSD archive
  `rebot_ckpts_archive/trackb_<RUN>/`; the SSD keeps 50k-multiple steps after the 10-04 prune).
