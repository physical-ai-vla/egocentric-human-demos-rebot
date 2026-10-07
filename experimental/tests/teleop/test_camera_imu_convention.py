"""The camera-IMU extrinsic is a direction, not a magnitude — pin the direction, not just the length.

A lever arm of the right length pointing the wrong way is exactly what a flipped convention looks like, and no norm
check catches it. `camera_imu_right_v001` came from Kalibr and was accepted on three external checks (lever arm vs a
tape measure, rotation vs an independent board+gyro hand-eye, timeshift vs two prior measurements) plus a physical
confirmation of the translation direction against the real mount. These tests re-assert that meaning wherever the
number is consumed, so a future re-import that inverts the transform, writes millimetres, or swaps sides fails here
rather than silently producing VIO that does not work."""
import numpy as np
import pytest
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation
from ego_teleop.calibration.bundle import load_bundle
from ego_teleop.transforms.calibration import load_teleop_calibration
from ego_teleop.transforms.se3 import inv_T
from ego_teleop.tracking.backends.openvins import write_openvins_config

CAL_DIR = Path(__file__).resolve().parents[2] / "configs" / "calibration"
# Kalibr, right wrist, 2026-09-16 (calib_right-camchain-imucam.yaml). IMU expressed in the CAMERA frame, metres.
KALIBR_T_CAM_IMU = np.array([[-0.9941856423076082, -0.09761961776862138, 0.0454457792992612, -0.04765307471729195],
                             [-0.09065814574395467, 0.5310952136417725, -0.8424482029520083, 0.07698258355630894],
                             [0.058103435695979054, -0.8416699378458924, -0.5368572496360787, -0.006278258765558664],
                             [0.0, 0.0, 0.0, 1.0]])
MOUNT_LEVER_ARM_M = 0.085          # lens to IMU board, measured by hand on the unit
FROZEN_OFFSET_MS = 32.77           # camera - imu, measured twice before Kalibr and kept frozen


def _right():
    p = sorted(CAL_DIR.glob("camera_imu_right_v*.yaml"))
    if not p: pytest.skip("camera_imu_right not calibrated yet")
    return np.asarray(yaml.safe_load(p[-1].read_text())["T_camera_imu"], np.float64).reshape(4, 4)


def test_stored_transform_is_imu_in_camera_frame_not_its_inverse():
    """The file stores Kalibr's T_cam_imu as-is. Storing the inverse would also be a valid SE(3) — and wrong."""
    T = _right()
    assert np.allclose(T, KALIBR_T_CAM_IMU, atol=1e-8), "stored T_camera_imu is not the Kalibr result"
    assert not np.allclose(T, inv_T(KALIBR_T_CAM_IMU), atol=1e-6), "stored transform is the INVERSE — convention flipped"


def test_translation_direction_matches_the_physical_mount():
    """Camera optical frame: +x image-right, +y image-down, +z forward out of the lens. On the right unit the IMU board
    sits below the lens toward the forearm, inboard, and in roughly the same fore-aft plane — confirmed by eye on the
    hardware. Signs, not just magnitudes."""
    t = _right()[:3, 3]
    assert t[1] > 0, "IMU must be BELOW the lens in the camera frame (+y = image-down)"
    assert t[0] < 0, "IMU must be inboard of the lens (-x)"
    assert abs(t[1]) > abs(t[0]) > abs(t[2]), "dominant offset is vertical, then lateral, then fore-aft"
    assert abs(t[2]) < 0.02, "IMU is roughly in the lens plane fore-aft, not 6 cm behind it"


def test_lever_arm_is_metres_and_matches_the_tape_measure():
    """Catches a unit slip: the same number in millimetres would be 90.8 m, in centimetres 9.08 m."""
    n = float(np.linalg.norm(_right()[:3, 3]))
    assert 0.5 * MOUNT_LEVER_ARM_M <= n <= 2.0 * MOUNT_LEVER_ARM_M, f"lever arm {n:.4f} m implausible vs {MOUNT_LEVER_ARM_M} m measured"
    assert abs(n - MOUNT_LEVER_ARM_M) < 0.02, f"lever arm {n*100:.2f} cm drifted from the measured {MOUNT_LEVER_ARM_M*100} cm"


def test_rotation_block_is_a_proper_rotation_and_round_trips_as_a_quaternion():
    R = _right()[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "rotation block is not orthonormal"
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6), "determinant != +1 — a reflection, not a rotation"
    q = Rotation.from_matrix(R).as_quat()                      # xyzw, the convention ik() and the bundle use
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(Rotation.from_quat(q).as_matrix(), R, atol=1e-9)


def test_inverse_reproduces_t_imu_cam_and_preserves_the_lever_arm():
    # The bundle stores matrices rounded to 9 decimals, so a re-exported file is orthonormal only to ~1e-9; the bar here is
    # 1e-8 m = 10 nm on a 9 cm lever arm -- far below anything physical, far above float noise.
    T = _right(); Ti = inv_T(T)
    assert np.allclose(T @ Ti, np.eye(4), atol=1e-8) and np.allclose(Ti @ T, np.eye(4), atol=1e-8)
    assert np.linalg.norm(Ti[:3, 3]) == pytest.approx(np.linalg.norm(T[:3, 3]), abs=1e-8), "inversion changed the lever arm length"
    assert np.allclose(Ti[:3, :3], T[:3, :3].T, atol=1e-8)


def test_openvins_receives_the_inverse_because_it_wants_the_camera_in_the_imu_frame():
    """The one place the direction is deliberately flipped. OpenVINS `T_imu_cam` = camera pose in IMU frame = inv(ours)."""
    T = _right()
    intr = dict(model="kannala_brandt", K=np.array([[791.28, 0, 956.9], [0, 793.15, 0 + 523.48], [0, 0, 1]]),
                D=np.array([-0.05217, 0.0287, -0.01597, 0.00338]), image_size=(1920, 1080))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        write_openvins_config(d, intrinsics=intr, T_camera_imu=T, downscale=1)
        chain = yaml.safe_load((Path(d) / "kalibr_imucam_chain.yaml").read_text().split("\n", 1)[1])["cam0"]
    got = np.asarray(chain["T_imu_cam"], np.float64)
    assert np.allclose(got, inv_T(T), atol=1e-12), "OpenVINS got T_camera_imu un-inverted"
    assert not np.allclose(got, T, atol=1e-6)
    assert chain["distortion_model"] == "equidistant", "Kannala-Brandt must map to equidistant, not radtan"


def test_bundle_and_pose_file_agree_and_the_frozen_time_offset_survived_the_import():
    b = load_bundle("right")
    if b.T_camera_imu is None: pytest.skip("right bundle has no T_camera_imu yet")
    assert np.allclose(b.T_camera_imu, _right(), atol=1e-9), "bundle and camera_imu_right file disagree"
    assert b.camera_imu_source == "kalibr" and b.offline_camera_imu, "production requires an offline camera-IMU source"
    assert b.time_offset_ms == pytest.approx(FROZEN_OFFSET_MS, abs=1e-6), "the frozen time offset was overwritten by the import"
    cal = load_teleop_calibration("right")
    assert np.allclose(cal.T_C_I, _right(), atol=1e-9), "the teleop loader reads a different transform than the bundle"
    assert cal.camera_imu_time_offset_ms == pytest.approx(FROZEN_OFFSET_MS, abs=1e-6)


def test_left_is_not_silently_the_right_units_transform():
    """Sides are physically mirrored; an accidental copy would show up as identical matrices."""
    p = sorted(CAL_DIR.glob("camera_imu_left_v*.yaml"))
    if not p: pytest.skip("left not calibrated yet")
    L = np.asarray(yaml.safe_load(p[-1].read_text())["T_camera_imu"], np.float64).reshape(4, 4)
    assert not np.allclose(L, _right(), atol=1e-6), "left and right extrinsics are identical — one side was copied"
