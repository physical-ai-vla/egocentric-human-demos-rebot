# experimental/ MANIFEST

Assembled 2026-10-07 from `~/ego_collector`. Git state: branch `orbbec-rgbd`, HEAD `12db478`, plus the uncommitted
working tree. The source was only read, never modified.

| source | dest | notes |
|---|---|---|
| `ego_teleop/` | `ego_teleop/` | Wearable teleop / VIO package |
| `ego_collector/` | `ego_collector/` | V0 AprilTag pipeline. Its `camera/` subpackage is also used by `../collector`. |
| `configs/ego_teleop/` | `configs/ego_teleop/` | |
| `configs/camera/` | `configs/camera/` | `c922.example.yaml` and the local `c922.yaml` (V0 C922 intrinsics, gitignored in the source) |
| `configs/{aperture_calibration,charuco,charuco_wrist,collection_plan,qa,qa_hawor,world_tag_map,wrist_extrinsics}.yaml` | `configs/` | V0 configs |
| (symlink) | `configs/calibration → ../../collector/configs/calibration`, `configs/handumi → ../../collector/configs/handumi` | `ego_teleop` resolves `configs/...` from its own root, so these are symlinks instead of duplicates |
| `assets/aero_hand_open/` | `assets/aero_hand_open/` | 2 URDFs + NOTICE (TetherIA, Apache-2.0), 60 KB |
| `scripts/{handumi_body_track,body_reference_bakeoff,body_canonical_tracker,body_center_bakeoff,body_jump_audit,body_to_ego16,body_track_export,body_tcp_fit,cube_center_planes,cube_tcp_detect,imu_orientation,rgbd_qa10_report,rgbd_tcp_qa,hand_detector_audit,cube_pnp_anchor,imu_bridge_test,camera_to_grasp_offset,provenance,generate_apriltags,mast3r_live_server}` | `scripts/` | RGB-D body tracking (parked), the MASt3R live server (tested by `tests/teleop/test_mast3r_live.py`), and the V0 tag sheets |
| `tests/teleop/` | `tests/teleop/` | |
| `tests/*.py` (V0 tests + `synth.py`) | `tests/v0/` | Moved one level down. `from synth import …` still resolves through pytest rootdir insertion. |
| `docs/ego_teleop/*.md` | `docs/ego_teleop/` | |
| (new) | `pyproject.toml` | Derived from the source pyproject. Packages `ego_collector` + `ego_teleop`. |
| (copy) | `.python-version` | |
| (new) | `README.md`, `MANIFEST.md` | |

## Excluded

| item | reason |
|---|---|
| `.DS_Store`, `__pycache__/`, `*.bak*` | Cruft |
| `assets/handumi/` | Copied to `../collector/assets/` instead, because the collector's RGB-D tools use it |
| `models/` (`hand_landmarker.task`), WiLoR/HaWoR weights | Model weights, downloaded at runtime |
| `runs/`, `outputs/`, `datasets/`, `data/`, `ckpts/`, `E240/`, `Log/` | Outputs and data |
| Ego pipeline scripts (`ego_ik_retarget.py`, `ego_batch_process.sh`, …) | Owned by `../pipeline`. **`scripts/rgbd_tcp_qa.py` references `scripts/ego_ik_retarget.py`**, which is not here. |
| `~/orbbec_recorder.py` (a home-directory script referenced by `tests/teleop/test_rgbd_arm_poc.py`) | Outside the source tree. The test skips when it is absent. |

## Notes

- Test check: on this copy, `tests/teleop` + `tests/v0` give 356 passed, the same as the source tree. That run used
  `PYTHONPATH=collector:experimental` and the source venv.
- `ego_teleop` imports `handumi_collector`, so install `../collector` in the same environment.
- Some docs and tools hardcode `/Users/jeonghwanlee/...`, `~/robot-cockpit`, or `sudo` camera access. They were left
  as-is.
