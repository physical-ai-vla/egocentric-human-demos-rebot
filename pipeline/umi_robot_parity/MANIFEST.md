# pipeline/umi_robot_parity MANIFEST

Assembled 2026-10-20 from `~/ego_collector` (working tree, uncommitted) and `~/vio/kalibr`. Sources were only read.

| source | dest |
|---|---|
| `scripts/tag_rig_*.py`, `robot_cam_solve.py`, `umi_tcp_traj.py`, `replay_parity.py`, `kalibr_bag_export.py`, `kalibr_to_camimu.py` | `scripts/` |
| `visual_teleop/tools/{tag_rig_record,robot_cam_poses,umi_replay,ready_pose}.py` | `tools/` (also present in full under `experimental/visual_teleop/tools/`) |
| `configs/calibration/umi_tag_rig_v001.json`, `robot_cam_v001.json` | `configs/` |
| `~/vio/kalibr/{run_kalibr.sh,patch_kalibr.py,remote_kalibr.sh}` | `kalibr/` |
| `outputs/umi_anchor_tags_*`, `robot_gripper_tags_*`, `aprilgrid_3x4_50mm_id300.pdf` | `print/` |

Excluded: tag-rig recordings (`datasets/tag_rig/`), depth frames, Kalibr bags.
