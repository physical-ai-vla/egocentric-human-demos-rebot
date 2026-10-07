import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from handumi_collector.pose.episode_io import derived_dir
from handumi_collector.pose.se3 import T_to_pose7
from ego_teleop.tools.calib_converge import take_summary, compare, COLS
from ego_teleop.tracking.backends.openvins import MODE_OVERRIDES, OpenVinsBackend
from .conftest import T_of


def write_trace(ep: Path, T, dt_ms, *, n=200, side="right", jitter_deg=0.0, seed=0):
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed); out = derived_dir(ep, "openvins"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        Ti = np.array(T, float).copy()
        if jitter_deg: Ti[:3, :3] = Ti[:3, :3] @ Rotation.from_rotvec(rng.normal(0, np.radians(jitter_deg), 3)).as_matrix()
        p7 = T_to_pose7(Ti)
        rows.append(dict(t_ns=i * 33_000_000, frame_index=i, dt_cam_imu_ms=-dt_ms + rng.normal(0, 0.05), n_feat=150,
                         **{c: float(v) for c, v in zip(COLS, p7)}))
    pd.DataFrame(rows).to_parquet(out / f"calib_trace_{side}.parquet", index=False)
    return ep


def test_missing_trace_is_reported(tmp_path):
    with pytest.raises(SystemExit, match="run m1_vio --mode calibration"):
        take_summary(tmp_path / "nope", "right")


def test_two_agreeing_takes_pass_and_a_disagreeing_one_fails(tmp_path):
    T = T_of((0.01, 0.0, 0.02), (90, 0, 2))
    e1 = write_trace(tmp_path / "ep1", T, 4.2, jitter_deg=0.05, seed=1)
    e2 = write_trace(tmp_path / "ep2", T, 4.3, jitter_deg=0.05, seed=2)
    takes = [take_summary(e1, "right"), take_summary(e2, "right")]
    assert all(abs(t["time_offset_ms"] - 4.25) < 0.2 for t in takes)          # sign: our offset = -dt
    r = compare(takes)
    assert r["verdict"] == "PASS" and r["between_takes"]["max_rot_deg"] < 1.0 and abs(r["time_offset_ms"] - 4.25) < 0.2
    e3 = write_trace(tmp_path / "ep3", T_of((0.03, 0.0, 0.02), (93, 0, 2)), 9.0, seed=3)
    bad = compare(takes + [take_summary(e3, "right")])
    assert bad["verdict"] == "FAIL" and any("rotation spread" in x for x in bad["reasons"]) and any("time-offset spread" in x for x in bad["reasons"])


def test_single_take_never_passes(tmp_path):
    r = compare([take_summary(write_trace(tmp_path / "ep1", np.eye(4), 4.0), "right")])
    assert r["verdict"] == "FAIL" and any("only one take" in x for x in r["reasons"])


def test_converge_cli_writes_online_provenance_bundle(tmp_path):
    from ego_teleop.tools import calib_converge
    from ego_teleop.calibration.bundle import load_bundle, save_bundle, WristCalibrationBundle, ImuNoise
    T = T_of((0.01, 0, 0.02), (90, 0, 2))
    e1 = write_trace(tmp_path / "ep1", T, 4.2, seed=1); e2 = write_trace(tmp_path / "ep2", T, 4.25, seed=2)
    b = WristCalibrationBundle("right", hardware=dict(unit_id="wristR-01")); b.camera = dict(model="pinhole", K=np.eye(3) * 400, D=np.zeros(4), image_size=(640, 480))
    b.imu_noise = ImuNoise(2e-4, 2e-5, 2e-3, 3e-3); save_bundle(b, cal_dir=tmp_path)
    rc = calib_converge.main([str(e1), str(e2), "--side", "right", "--write-bundle", "--cal-dir", str(tmp_path)])
    assert rc == 0
    got = load_bundle("right", cal_dir=tmp_path)
    assert got.missing == [] and got.camera_imu_source == "openvins_online"
    with pytest.raises(RuntimeError):                       # complete but not production-ready: offline source still required
        got.require_production()


def test_openvins_modes_set_the_right_flags():
    assert MODE_OVERRIDES["production"] == dict(calib_cam_extrinsics=False, calib_cam_intrinsics=False, calib_cam_timeoffset=False)
    assert MODE_OVERRIDES["calibration"]["calib_cam_extrinsics"] and MODE_OVERRIDES["calibration"]["calib_cam_timeoffset"]
    be = OpenVinsBackend(mode="calibration", overrides=dict(num_pts=99))
    assert be.overrides["calib_cam_extrinsics"] is True and be.overrides["num_pts"] == 99 and be.calibration_estimate() is None
    prod = OpenVinsBackend()
    assert prod.mode == "production" and prod.overrides["calib_cam_timeoffset"] is False
    with pytest.raises(ValueError):
        OpenVinsBackend(mode="whatever")
