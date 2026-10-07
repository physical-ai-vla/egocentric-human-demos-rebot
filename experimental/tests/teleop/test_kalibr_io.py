import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path
import yaml
from ego_teleop.tools.kalibr_import import parse_camchain
from ego_teleop.tools.kalibr_export import write_target_yaml, write_imu_yaml, TARGETS, export
from ego_teleop.transforms.se3 import inv_T
from .conftest import T_of

T_CI = T_of((0.011, -0.003, 0.02), (90.0, 0.0, 2.0))


def camchain(tmp_path, *, dist="equidistant", shift=0.00412, header=True) -> Path:
    d = {"cam0": dict(T_cam_imu=[[float(x) for x in r] for r in T_CI], camera_model="pinhole", distortion_model=dist,
                      distortion_coeffs=[0.05, -0.01, 0.002, -0.0005], intrinsics=[601.2, 600.4, 959.1, 541.3], resolution=[1920, 1080],
                      timeshift_cam_imu=shift, rostopic="/cam0/image_raw")}
    p = tmp_path / "camchain-imucam.yaml"; p.write_text(("%YAML:1.0\n" if header else "") + yaml.safe_dump(d, sort_keys=False)); return p


def test_kalibr_conventions_are_not_inverted_and_timeshift_sign_flips(tmp_path):
    r = parse_camchain(camchain(tmp_path))
    assert np.allclose(r["T_camera_imu"], T_CI)                       # Kalibr T_cam_imu == our T_camera_imu, NO inversion
    assert not np.allclose(r["T_camera_imu"], inv_T(T_CI))
    assert np.isclose(r["time_offset_ms"], -4.12)                     # t_imu = t_cam + 0.00412 s  ->  camera - imu = -4.12 ms
    c = r["camera"]
    assert c["model"] == "kannala_brandt" and c["image_size"] == (1920, 1080)
    assert np.allclose(c["K"], [[601.2, 0, 959.1], [0, 600.4, 541.3], [0, 0, 1]]) and np.allclose(c["D"], [0.05, -0.01, 0.002, -0.0005])
    assert parse_camchain(camchain(tmp_path / "x" if False else tmp_path, dist="radtan"))["camera"]["model"] == "pinhole"


def test_opencv_yaml_header_and_bad_input(tmp_path):
    assert parse_camchain(camchain(tmp_path, header=False))["camera"]["model"] == "kannala_brandt"
    with pytest.raises(SystemExit):
        parse_camchain(camchain(tmp_path), cam="cam1")
    bad = tmp_path / "bad.yaml"
    d = yaml.safe_load(camchain(tmp_path).read_text().split("\n", 1)[1]); d["cam0"]["T_cam_imu"] = np.eye(4).tolist(); d["cam0"]["T_cam_imu"][0][0] = 2.0
    bad.write_text(yaml.safe_dump(d))
    with pytest.raises(SystemExit, match="proper rotation"):
        parse_camchain(bad)


def test_import_cli_writes_bundle_marked_kalibr(tmp_path):
    cc = camchain(tmp_path)
    from ego_teleop.tools import kalibr_import
    from ego_teleop.calibration.bundle import load_bundle, save_bundle, WristCalibrationBundle, ImuNoise
    b = WristCalibrationBundle("right"); b.imu_noise = ImuNoise(2e-4, 2e-5, 2e-3, 3e-3); b.set_provenance("imu_noise", source="allan_variance")
    save_bundle(b, cal_dir=tmp_path)
    rc = kalibr_import.main([str(cc), "--side", "right", "--write-bundle", "--export-pose-files", "--cal-dir", str(tmp_path),
                             "--unit-id", "wristR-01", "--mount-revision", "v1"])
    assert rc == 0
    got = load_bundle("right", cal_dir=tmp_path)
    assert got.missing == [] and got.offline_camera_imu and got.camera_imu_source == "kalibr"
    got.require_production()
    assert np.allclose(got.T_camera_imu, T_CI) and np.isclose(got.time_offset_ms, -4.12) and got.hardware["unit_id"] == "wristR-01"
    assert (tmp_path / "fisheye_right_v001.yaml").exists() and (tmp_path / "camera_imu_right_v001.yaml").exists()


def test_target_yaml_is_tag_free(tmp_path):
    assert "aprilgrid" not in TARGETS
    p = write_target_yaml(tmp_path / "target.yaml", "checkerboard", targetCols=11)
    d = yaml.safe_load(p.read_text()); assert d["target_type"] == "checkerboard" and d["targetCols"] == 11 and d["rowSpacingMeters"] == 0.025
    d2 = yaml.safe_load(write_target_yaml(tmp_path / "t2.yaml", "circlegrid").read_text())
    assert d2["target_type"] == "circlegrid" and d2["asymmetricGrid"] is True
    with pytest.raises(SystemExit, match="tag-free"):
        write_target_yaml(tmp_path / "t3.yaml", "aprilgrid")
    n = yaml.safe_load(write_imu_yaml(tmp_path / "imu.yaml", dict(gyroscope_noise_density=1.5e-4), 400).read_text())
    assert n["gyroscope_noise_density"] == 1.5e-4 and n["update_rate"] == 400.0 and n["rostopic"] == "/imu0"


@pytest.mark.slow
def test_export_bag_from_synthetic_episode(tmp_path):
    """Full-rate IMU + gray images land in a readable ROS1 bag on one clock."""
    from rosbags.rosbag1 import Reader
    ep = tmp_path / "ep"
    subprocess.run([sys.executable, "-m", "handumi_collector.tools.synth_episode", str(ep), "--seconds", "3", "--camera-imu-offset-ms", "0"], check=True, capture_output=True)
    res = export(ep, "right", tmp_path / "calib.bag")
    assert res["images"] == 90 and res["imu_samples"] == 1200 and 380 < res["imu_rate_hz"] < 420      # 400 Hz IMU, NOT resampled to 30
    with Reader(tmp_path / "calib.bag") as r:
        topics = {c.topic: c.msgcount for c in r.connections}
        assert topics == {"/cam0/image_raw": 90, "/imu0": 1200}
        ts = sorted(m[1] for m in r.messages())
        assert ts == sorted(ts) and (ts[-1] - ts[0]) / 1e9 == pytest.approx(3.0, abs=0.2)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        export(ep, "right", tmp_path / "calib.bag")
