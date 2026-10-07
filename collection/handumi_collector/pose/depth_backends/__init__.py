"""Depth-tracker backends. Each implements handumi_collector.pose.depth_hand_tracker.DepthHandPoseTracker.
Names are config values (depth_pose.yaml `backend:`), resolved through the registry in depth_hand_tracker.py.
A backend whose software is missing raises TrackerUnavailable on construction — the pipeline records that and stops;
it never substitutes another backend or fabricates poses."""
