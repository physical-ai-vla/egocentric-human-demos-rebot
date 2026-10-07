# collector/ MANIFEST

Assembled 2026-10-07 from `~/ego_collector`. Git state: branch `orbbec-rgbd`, HEAD
`12db478b33c7a5c78868dcf13955542eaa0fc168`. The copy is the **working tree**, which includes about 180 uncommitted
changes, among them the current `configs/handumi/*.yaml`. The source was only read, never modified.

| source | dest | notes |
|---|---|---|
| `handumi_collector/` | `handumi_collector/` | The full package |
| `configs/handumi/` | `configs/handumi/` | Hardware profiles, `tasks.yaml`, `collector.yaml`, `robot_like_v1.yaml`, `pose.yaml`, `depth_pose.yaml`, `collection_contract_v1.yaml`, … |
| `configs/calibration/` | `configs/calibration/` | Small yaml/json/npy/txt calibration files, `kalibr_raw/`, `camera_tcp/`, README, `robot_A_right_view_v001.jpg` (14 KB) |
| `configs/charuco.yaml`, `configs/charuco_wrist.yaml` | `configs/` | ChArUco board specs used by the calibration scripts |
| `assets/handumi/` | `assets/handumi/` | Tracking-body meshes (2 × 3.9 MB STL) and yaml/json for the RGB-D tracking tools |
| `firmware/teensy_imu/`, `firmware/xiao_imu/`, `firmware/bringup/` | `firmware/` | Arduino sources, `config.h`, `PROTOCOL.md`, README |
| `scripts/<18 files>` | `scripts/` | See the script list below |
| `tests/handumi/` | `tests/handumi/` | |
| `docs/handumi_collector/*.md` | `docs/` | PLAN, POSE_V2, CAMERA_FREEZE, IMU_BRINGUP, IMU_RING_POLICY, TEMPORAL_SYNC, RGBD_POSE, RGBD_TEACHER_QA, ROBOT_LIKE_PROTOCOL |
| `pyproject.toml`, `.python-version` | same | `pyproject.toml` was **rewritten** to package only `handumi_collector`; the `handumi` extra became core dependencies, and `../experimental` is a path dependency |
| (new) | `README.md`, `MANIFEST.md` | |

Scripts copied:

- Calibration and cameras: `verify_physical_side`, `identify_board`, `charuco_calibrate`, `generate_charuco`,
  `calibrate_table_frame_charuco`, `table_frame_ui`, `fit_table_frame`, `calib_take_qa`, `exposure_sweep`,
  `wrist_brightness_sweep`, `wrist_image_quality`, `wrist_cam_diagnose`.
- robot_like_v1: `robotlike_replay_live`, `robotlike_offline_check`, `robotlike_compare_groups`,
  `robotlike_session_check.sh`, `umi_export_orbslam`.
- Helper: `provenance`.

These were chosen because they import `handumi_collector` and serve collection, calibration or collection QA.

## Excluded

| item | reason |
|---|---|
| `start_ref_HRA_A100.png` (2.4 MB), `start_ref_HRA_A100_horizontal_old.png.bak` (2.3 MB) | Images over 1 MB, and a `.bak`. **The first is functional**: it is the stored HRA_A100 start reference that the UI's start-match tile loads. Without it the UI shows no reference until the operator presses "SET START REF". Its companion `start_ref_HRA_A100.up.npy` was copied. |
| 18 `*.bak*` files in `handumi_collector/` (`ui/app.py.bak_pre_*` ×6, `session.py.bak_pre_*` ×2, `camera.py.bak_pre_*` ×2, `start_match.py.bak_v1/v1b`, …) and 5 in `configs/handumi/` | Backup copies |
| `__pycache__/`, `.DS_Store`, `.pytest_cache/` | Cruft |
| `datasets/`, `data/`, `outputs/`, `runs/`, `ckpts/`, `models/`, `E240/`, `calibration/` (V0 C922 calib takes), `Log/`, `*.log` | Recordings, data, checkpoints, logs |
| `uv.lock` | Belongs to the old combined project. Regenerate it with `uv lock`, which resolves cleanly (checked with `uv lock --dry-run`). |
| Other `scripts/*` (ego batch/ingest/export, IK retarget, LeRobot builders, audits, g6/gNN, `robotlike_resident_5080.sh`, …) | Ego pipeline, training and eval work, owned by `../pipeline` and the other folders. RGB-D body-tracking scripts went to `../experimental/scripts`. |
| Root configs `deploy_start_stats.json`, `deploy_visual_ref.npz`, `e120_*.json`, `robot_subsets.json` | Pipeline, training and deploy artifacts, not collector inputs |
| V0 configs (`world_tag_map`, `wrist_extrinsics`, `qa*`, `aperture_calibration`, `collection_plan`, `camera/`) | Copied to `../experimental/configs` |
| `README.md` (repo root), `PNP_FEASIBILITY_REPORT.md`, `WRIST_MOTION_AUDIT.md`, `record_*.sh/py`, `calibrate_c922.sh` | The root README was summarised into `README.md` here. The rest are one-off V0-era reports and scripts. |

## Notes

- **Cross-folder dependency**: `handumi_collector.devices.camera` (the UVC/Orbbec backends),
  `tools/check_rgbd_ready.py`, 5 ChArUco scripts and 2 tests import `ego_collector.camera.*` or
  `ego_collector.tracking`. That package is in `../experimental/ego_collector`.
- **Not self-contained**: `scripts/robotlike_offline_check.py` imports `c8_phase3` and `c8_collision_v1` from `~/c8`
  (`sys.path` insert). `scripts/robotlike_session_check.sh` uses the Ray GPU cluster, tailnet IPs `100.64.0.x`, and
  `/srv/data/johann` NFS paths. Some docs and scripts still contain absolute `/Users/jeonghwanlee/...` paths.
- Test check: on this copy, `tests/handumi` gives the same pass/fail list as the source tree. There are 12
  failures from before the copy and one macOS segfault in `test_virtual_view.py::test_calibrate_pinhole_on_rendered_checkerboard`.
  The single difference is `test_pilot_episodes_report_the_broken_left_jaw`, which is SKIPPED here because it needs
  recorded pilot data.
- Secret scan: clean. Configs contain device USB serials, product names and tailnet IPs, but no credentials.
