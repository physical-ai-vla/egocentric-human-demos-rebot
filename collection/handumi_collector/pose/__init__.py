"""HandUMI pose-tracking subsystem (M2, fiducial-free).

    Arducam fisheye + ICM42688P IMU  ->  PoseEstimator (VIO / VO backend)  ->  T_world_camera
                                     ->  camera->TCP calibration            ->  T_world_TCP
                                     ->  episode-local canonicalisation      ->  head-timeline TCP + grip
                                     ->  HOME-return / static / IMU / jump QA

Strictly separate from acquisition: nothing here is imported by the recorder, and raw episodes never depend on it.
No AprilTag / ArUco / ChArUco / fiducial code lives in, or is imported by, this package (tests/handumi/test_no_fiducial.py).
"""
from .estimator import PoseEstimate, PoseEstimator, TrackingState, BackendInfo, BackendUnavailable   # noqa: F401
