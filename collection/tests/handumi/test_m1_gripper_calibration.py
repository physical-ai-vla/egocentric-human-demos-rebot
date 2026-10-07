import time
from pathlib import Path
import yaml
from handumi_collector import calibration as cal
from handumi_collector.collector.session import CollectorSession


def test_versioned_calibration_store(tmp_path):
    assert cal.load_gripper_calibration(cal_dir=tmp_path) == (None, {})
    p1 = cal.save_gripper_calibration({"left": dict(ticks_closed=1000, ticks_open=2000), "right": dict(ticks_closed=3900, ticks_open=4600)}, cal_dir=tmp_path)
    p2 = cal.save_gripper_calibration({"left": dict(ticks_closed=1010, ticks_open=1990), "right": dict(ticks_closed=3900, ticks_open=4600)}, cal_dir=tmp_path)
    assert (p1.name, p2.name) == ("gripper_v001.yaml", "gripper_v002.yaml") and p1.exists()          # never overwritten
    ver, sides = cal.load_gripper_calibration(cal_dir=tmp_path)
    assert ver == "gripper_v002" and sides["left"]["ticks_closed"] == 1010
    ver1, s1 = cal.load_gripper_calibration("gripper_v001", cal_dir=tmp_path)
    assert ver1 == "gripper_v001" and s1["left"]["ticks_closed"] == 1000
    assert yaml.safe_load(p2.read_text())["convention"].startswith("normalized 0 = closed")


def test_set_closed_open_via_session(mock_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "CAL_DIR", tmp_path / "cal")
    for g in mock_cfg.hardware.grippers: g.ticks_closed = g.ticks_open = None; g.norm_one_is = "open"
    s = CollectorSession.create(mock_cfg)
    time.sleep(0.15)
    assert s.devices.gripper_calibration_version is None
    assert s.devices.grippers["left"].last_sample.normalized != s.devices.grippers["left"].last_sample.normalized   # NaN until calibrated
    for side in ("left", "right"):
        s.devices.grippers[side].sim_closed = 1000; s.devices.grippers[side].sim_open = 2000
        s.calibrate_gripper(side, "closed"); s.calibrate_gripper(side, "open")   # captures whatever the mock reads now
    # force meaningful values then save
    s._pending_cal = {"left": dict(ticks_closed=1000, ticks_open=2000), "right": dict(ticks_closed=1000, ticks_open=2000)}
    ver = s.save_gripper_calibration(notes="test")
    assert ver == "gripper_v001" and s.devices.grippers["left"].cfg.ticks_open == 2000
    time.sleep(0.1)
    ns = s.devices.grippers["left"].last_sample.normalized
    assert 0.0 <= ns <= 1.0
    s.start(); time.sleep(0.3); s.stop(); meta = s.keep()
    assert meta["gripper_calibration_version"] == "gripper_v001" and meta["hardware_profile"] == mock_cfg.hardware.profile
    import json
    sm = json.loads((s.manager.session_dir / "session_meta.json").read_text())
    assert "LEFT IMU" in sm["identity"] and sm["identity"]["LEFT IMU"].startswith("MOCK-LEFT")
    s.close()
