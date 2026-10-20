# umi_robot_parity: UMI TCP calibration and UMI → reBot replay parity (2026-10-16 → 10-20)

Goal (lead decision 10-16): motion collected with the HandUMI must replay the same on the reBot. A fixed global Orbbec sees
3-tag AprilTag rigs on each UMI (tag36h11 200–202 right, 210–212 left, printed 48 mm) and, later, tag 220 on the robot
gripper. Overview and numbers: [`docs/UPDATE_2026-10-20.md`](../../docs/UPDATE_2026-10-20.md), Sec. 5.

| step | code | status |
|---|---|---|
| record takes (`--proto rig|pivot|corners|robotcal|axes|view`) | `tools/tag_rig_record.py` | used |
| rig geometry, multi-corner rig PnP | `scripts/tag_rig_solve.py`, `tag_rig_pose.py` | done |
| UMI TCP by touching board corners | `scripts/tag_rig_touch_tcp.py` (pivot: `tag_rig_pivot*.py`, superseded) | **done 10-20**, `configs/umi_tag_rig_v001.json` |
| IMU ↔ rig rotation + time offset | `scripts/tag_rig_imu.py`, fusion `tag_rig_fuse.py` | done |
| robot base ↔ Orbbec | `tools/ready_pose.py`, `tools/robot_cam_poses.py` (dry-run default, `--go` moves), `scripts/robot_cam_solve.py` | built, synthetic PASS, **not run (robot not installed)** |
| UMI trajectories → robot replay | `scripts/umi_tcp_traj.py`, `tools/umi_replay.py` | built, not run on the robot |
| parity A/B/C | `scripts/replay_parity.py` (`--umi-only`, `--synthetic`) | built |
| Kalibr camera–IMU | `kalibr/run_kalibr.sh`, `patch_kalibr.py`, `scripts/kalibr_bag_export.py`, `kalibr_to_camimu.py` | works; used for the dropped stereo rig |

Print files in `print/` (print at 100 %, measure the black square; our printer gave 48 mm for 50 mm). `tools/` import
`visual_teleop` (see `experimental/visual_teleop`), and the robot tools need the reBot MIT bridge.
