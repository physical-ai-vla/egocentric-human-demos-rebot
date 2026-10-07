import numpy as np
import pytest
import yaml
from handumi_collector.pose.estimator import BackendUnavailable
from handumi_collector.pose.backends import make_backend, available_backends
from handumi_collector.pose.se3 import inv_T
import ego_teleop.tracking.backends  # noqa: F401  registers
from ego_teleop.tracking.backends.openvins import write_openvins_config, OpenVinsBackend, ICM42688P_NOISE
from .conftest import T_of

FISHEYE = dict(model="kannala_brandt", K=np.array([[600.0, 0, 960.0], [0, 600.0, 540.0], [0, 0, 1]]), D=np.array([0.05, -0.01, 0.002, -0.0005]), image_size=(1920, 1080))


def _load(p):
    txt = p.read_text(); assert txt.startswith("%YAML:1.0")
    return yaml.safe_load(txt.split("\n", 1)[1])


def test_config_writer_equidistant_and_extrinsic_convention(tmp_path):
    T_C_I = T_of((0.01, -0.02, 0.03), (90, 0, 10))
    cfg = write_openvins_config(tmp_path, intrinsics=FISHEYE, T_camera_imu=T_C_I, downscale=2, imu_noise=dict(gyroscope_noise_density=1e-4))
    est = _load(cfg); cam = _load(tmp_path / "kalibr_imucam_chain.yaml")["cam0"]; imu = _load(tmp_path / "kalibr_imu_chain.yaml")["imu0"]
    assert cam["distortion_model"] == "equidistant" and cam["resolution"] == [960, 540] and cam["intrinsics"] == [300.0, 300.0, 480.0, 270.0]
    assert np.allclose(np.array(cam["T_imu_cam"]), inv_T(T_C_I))        # OpenVINS T_imu_cam = camera in IMU = inv(T_camera_imu)
    assert cam["timeshift_cam_imu"] == 0.0 and cam["distortion_coeffs"] == [0.05, -0.01, 0.002, -0.0005]
    assert imu["gyroscope_noise_density"] == 1e-4 and imu["accelerometer_noise_density"] == ICM42688P_NOISE["accelerometer_noise_density"] and imu["update_rate"] == 400.0
    assert est["max_cameras"] == 1 and est["use_stereo"] is False and est["calib_cam_timeoffset"] is True and est["relative_config_imucam"] == "kalibr_imucam_chain.yaml"


def test_config_writer_pinhole_falls_back_to_radtan(tmp_path):
    pin = dict(model="pinhole", K=np.array([[400.0, 0, 319.5], [0, 400.0, 239.5], [0, 0, 1]]), D=np.zeros(4), image_size=(640, 480))
    write_openvins_config(tmp_path, intrinsics=pin, T_camera_imu=np.eye(4), overrides=dict(num_pts=99))
    cam = _load(tmp_path / "kalibr_imucam_chain.yaml")["cam0"]; est = _load(tmp_path / "estimator_config.yaml")
    assert cam["distortion_model"] == "radtan" and cam["resolution"] == [640, 480] and est["num_pts"] == 99


def test_registered_in_handumi_registry_and_requires_calibration():
    assert "openvins" in available_backends()
    be = make_backend("openvins", downscale=2)
    assert isinstance(be, OpenVinsBackend) and be.info.uses_imu and be.info.metric_scale and be.info.online_capable
    with pytest.raises(BackendUnavailable, match="intrinsics"):
        be.initialize(intrinsics=None, T_camera_imu=np.eye(4))
    with pytest.raises(BackendUnavailable, match="T_camera_imu"):
        be.initialize(intrinsics=FISHEYE, T_camera_imu=None)


def test_missing_bridge_is_reported_not_faked(tmp_path, monkeypatch):
    import ego_teleop.tracking.backends.openvins as ov
    monkeypatch.setattr(ov, "DEFAULT_BRIDGE_DIRS", (str(tmp_path / "nowhere"),))
    monkeypatch.setitem(__import__("sys").modules, "ov_bridge", None)     # force ImportError
    be = OpenVinsBackend(bridge_dir=str(tmp_path / "nowhere"))
    with pytest.raises(BackendUnavailable, match="ov_bridge"):
        be.initialize(intrinsics=FISHEYE, T_camera_imu=np.eye(4))
