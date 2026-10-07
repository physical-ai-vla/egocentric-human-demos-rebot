import numpy as np
import pytest
import yaml
from pathlib import Path
from handumi_collector.pose.calibration import SideCalibration
from ego_teleop.calibration.bundle import WristCalibrationBundle, ImuNoise, load_bundle, save_bundle
from ego_teleop.transforms.se3 import inv_T
from .conftest import T_of

FISHEYE = dict(model="kannala_brandt", K=np.array([[600.0, 0, 960.0], [0, 600.0, 540.0], [0, 0, 1]]), D=np.array([0.05, -0.01, 0.002, -0.0005]), image_size=(1920, 1080))
NOISE = ImuNoise(2e-4, 2e-5, 2e-3, 3e-3, 400.0)


def full_bundle(side="right", source="kalibr"):
    b = WristCalibrationBundle(side, hardware=dict(unit_id="wristR-01", side=side, mount_revision="v1", camera_serial="ARDU-7", imu_serial="TEENSY-3", resolution="1920x1080"))
    b.camera = dict(**FISHEYE, reprojection_rms_px=0.31, n_views=30, target="checkerboard 9x6 25mm")
    b.imu_noise = NOISE; b.T_camera_imu = T_of((0.012, -0.004, 0.021), (90, 0, 3)); b.time_offset_ms = -4.2
    for f in ("camera", "T_camera_imu", "time_offset_ms"): b.set_provenance(f, source=source, tool="kalibr_calibrate_imu_camera", runs=["camchain.yaml"])
    b.set_provenance("imu_noise", source="allan_variance", tool="ego_teleop.tools.imu_allan", runs=["imu_right.csv"])
    return b


def test_empty_bundle_lists_everything_missing_and_refuses_production():
    b = WristCalibrationBundle("right")
    assert set(b.missing) == {"camera", "imu_noise", "T_camera_imu", "time_offset_ms", "hardware.unit_id"}
    with pytest.raises(RuntimeError, match="incomplete"):
        b.require_production()


def test_partial_noise_is_incomplete():
    b = full_bundle(); b.imu_noise = ImuNoise(gyroscope_noise_density=2e-4)
    assert "imu_noise" in b.missing and not b.imu_noise.complete


def test_online_only_camera_imu_is_refused_for_production():
    b = full_bundle(source="openvins_online")
    assert b.missing == [] and not b.offline_camera_imu
    with pytest.raises(RuntimeError, match="online refinement alone is not accepted"):
        b.require_production()
    full_bundle(source="kalibr").require_production()          # offline source passes


def test_hardware_hash_tracks_the_physical_configuration():
    a, b = full_bundle(), full_bundle()
    assert a.hardware_hash() == b.hardware_hash()
    b.hardware["mount_revision"] = "v2"
    assert a.hardware_hash() != b.hardware_hash()
    ok, diff = a.matches_hardware(dict(mount_revision="v2")); assert not ok and diff == ["mount_revision"]
    assert a.matches_hardware(dict(unit_id="wristR-01"))[0]


def test_roundtrip_and_versioning(tmp_path):
    b = full_bundle()
    p1 = save_bundle(b, cal_dir=tmp_path, notes="first")
    assert p1.name == "wrist_bundle_right_v001.yaml" and b.version == "wrist_bundle_right_v001"
    b.time_offset_ms = -3.9; p2 = save_bundle(b, cal_dir=tmp_path)
    assert p2.name == "wrist_bundle_right_v002.yaml" and p1.exists()          # never overwritten
    back = load_bundle("right", cal_dir=tmp_path)
    assert back.version == "wrist_bundle_right_v002" and back.time_offset_ms == -3.9
    assert np.allclose(back.T_camera_imu, b.T_camera_imu) and back.imu_noise.complete and back.offline_camera_imu
    assert np.allclose(back.intrinsics()["K"], FISHEYE["K"]) and back.intrinsics()["model"] == "kannala_brandt"
    d = yaml.safe_load(p2.read_text()); assert d["production_ready"] and d["hardware_hash"] == b.hardware_hash() and d["provenance"]["T_camera_imu"]["source"] == "kalibr"


def test_export_pose_files_feeds_the_existing_pose_pipeline(tmp_path):
    b = full_bundle(); save_bundle(b, cal_dir=tmp_path)
    files = b.export_pose_files(cal_dir=tmp_path)
    assert set(files) == {"fisheye", "camera_imu"}
    cal = SideCalibration("right", cal_dir=tmp_path)             # what handumi_collector.pose.run actually reads
    assert cal.intrinsics is not None and cal.intrinsics["model"] == "kannala_brandt"
    assert np.allclose(cal.intrinsics["K"], FISHEYE["K"]) and np.allclose(cal.intrinsics["D"], FISHEYE["D"])
    assert np.allclose(cal.T_camera_imu, b.T_camera_imu) and cal.imu_extrinsics_calibrated
    assert cal.camera_imu_time_offset_ms == b.time_offset_ms and cal.imu_noise["gyroscope_noise_density"] == 2e-4


def test_pinhole_model_survives_export(tmp_path):
    b = full_bundle(); b.camera = dict(model="pinhole", K=np.eye(3) * 400, D=np.zeros(4), image_size=(640, 480))
    b.export_pose_files(cal_dir=tmp_path)
    assert SideCalibration("right", cal_dir=tmp_path).intrinsics["model"] == "pinhole"
