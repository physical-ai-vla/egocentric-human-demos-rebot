"""ego_teleop — wearable (fisheye + IMU wristband) teleoperation of a reBot Damiao arm + Aero Hand Open.

Built on top of the existing packages in this repo instead of duplicating them:
  handumi_collector.devices        Arducam fisheye + Teensy ICM42688P drivers, MCAP raw recorder   (sensors/, recorder/)
  handumi_collector.pose           backend-agnostic VIO interface (PoseEstimator / PoseEstimate), SE(3), timing, sync
  ego_collector.hands              MediaPipe HandLandmarker
  ego_collector.hands3d.aero       21-joint skeleton -> human semantic 7D -> Aero compact/16/actuations
  robot-cockpit robot_service      /observe, /execute_step (joint space; rad arm + raw gripper)
  handumi-sw robots.kinematics     pyroki IK for reBot B601 (separate venv; see robot/rebot_client.py)

Two independent branches (arm transport / hand grasp) meet only in robot/coordinator.py (one 30 Hz clock, one
TeleopCommand) and in recorder/ (one synchronized episode). See docs/ego_teleop/PLAN.md.
"""
