"""M1.5 pre-IMU production: profiles without IMUs, state machine incl. ERROR_REVIEW, integrity validator, hardware check,
disk gate, .complete marker, dataset modes, review summary, and an offscreen Qt UI smoke (construct, refresh, record, keep)."""
import json
import os
import time
from pathlib import Path
import pytest
from handumi_collector.collector.integrity import preliminary_qa, validate_episode
from handumi_collector.collector.session import CollectorSession
from handumi_collector.config import DEFAULT_CONFIG_DIR, AppConfig, CameraCfg, CollectorCfg, GripperCfg, HardwareCfg, TasksCfg, load_config


def _preimu_mock_cfg(tmp_path, **coll) -> AppConfig:
    hw = HardwareCfg(profile="preimu_test", cameras=[CameraCfg("head", "policy_obs", backend="mock", width=64, height=48, codec="libx264"),
                                                     CameraCfg("left_wrist", "pose_estimation", backend="mock", width=64, height=48, codec="libx264"),
                                                     CameraCfg("right_wrist", "pose_estimation", backend="mock", width=64, height=48, codec="libx264"),
                                                     CameraCfg("head_depth", "aux_depth", backend="mock", width=32, height=24, codec="libx264", required=False)],
                     imus=[], grippers=[GripperCfg("left", backend="mock", ticks_closed=1000, ticks_open=2000), GripperCfg("right", backend="mock", ticks_closed=3000, ticks_open=2000)])
    return AppConfig(hardware=hw, tasks=TasksCfg(target_per_order=2), config_dir=DEFAULT_CONFIG_DIR,
                     collector=CollectorCfg(dataset_root=str(tmp_path / "raw"), session_prefix="Htest", min_episode_s=0.1, preflight=["a", "b"], **coll))


def test_preimu_profiles_load_without_imus():
    for prof in ("handumi_preimu", "dev_current"):
        c = load_config(hardware=prof)
        assert c.hardware.imus == [] and {g.side for g in c.hardware.grippers} == {"left", "right"}
        assert {"left_wrist", "right_wrist"} <= {cam.name for cam in c.hardware.cameras}
    # handumi_preimu: the C922 head is retired, the Orbbec RGB-D IS the head observation and is now REQUIRED
    c = load_config(hardware="handumi_preimu")
    assert "head" not in {cam.name for cam in c.hardware.cameras}
    aux = [cam for cam in c.hardware.cameras if cam.role == "aux_depth"][0]
    assert aux.name == "head_depth" and aux.required is True
    assert c.collector.disk["block_gb"] > 0 and "Hpilot" in c.tasks.datasets


def test_record_without_imu_and_optional_aux_missing(tmp_path):
    cfg = _preimu_mock_cfg(tmp_path)
    s = CollectorSession.create(cfg, dataset="Hpilot")
    assert cfg.tasks.target_per_order == 50 and s.manager.session_dir.parent.name == "Hpilot"
    s.devices.cameras["head_depth"].close()                      # optional aux gone -> still READY
    assert s.can_record() == (True, "ready") and s.state_label() == "READY" and s.optional_missing() == ["head_depth"]
    s.start(); time.sleep(0.4); s.stop(); r = s.review_summary()
    assert r["imus"] == {} and set(r["grippers"]) == {"left", "right"} and r["verdict"] == "PASS" and r["streams"]["head"]["frames"] > 5
    meta = s.keep(); ep = s.manager.session_dir / meta["episode_dir"]
    assert (ep / ".complete").exists() and not (ep / ".incomplete").exists() and meta["imu_present"] is False and meta["aux_available"] == {"head_depth": False}
    assert meta["device_status_at_start"]["head"]["connected"] is True
    sm = json.loads((s.manager.session_dir / "session_meta.json").read_text()); assert sm["dataset_name"] == "Hpilot" and sm["imu_present"] is False
    v = validate_episode(ep, expect_grippers=["left", "right"], expect_imus=[]); assert v["ok"], v["problems"]
    assert "/left/imu" not in json.dumps(meta["sensor_messages"])      # channel reserved in the schema, no messages
    s.close()


def test_error_review_on_required_device_failure(tmp_path):
    cfg = _preimu_mock_cfg(tmp_path); s = CollectorSession.create(cfg)
    s.start(); time.sleep(0.2)
    s.devices.cameras["left_wrist"].fail_after_s = 0.0            # mock dropout on a REQUIRED device
    time.sleep(0.3); assert s.poll() == "ERROR_REVIEW" and s.state == "ERROR_REVIEW"
    r = s.review_summary(); assert r["verdict"] == "REVIEW" and "required device failed" in r["notes"][0]
    meta = s.keep(notes="op"); assert meta["quality"] == "REVIEW" and "KEEP_WITH_WARNING" in meta["notes"] and "critical_device_error" in meta["event_kinds"]
    assert s.state == "IDLE"; s.close()


def test_disk_gate_and_preflight_gate(tmp_path):
    cfg = _preimu_mock_cfg(tmp_path, disk={"warn_gb": 1e9, "block_gb": 1e9, "gb_per_minute_estimate": 0.6}); s = CollectorSession.create(cfg)
    ok, why = s.can_record(); assert not ok and "disk" in why and s.disk()["state"] == "RED"; s.close()
    cfg = _preimu_mock_cfg(tmp_path, preflight_required=True); s = CollectorSession.create(cfg)
    ok, why = s.can_record(); assert not ok and "preflight" in why
    s.preflight_ok = True; assert s.can_record()[0]; s.close()


def test_hardware_check_records_and_validates(tmp_path):
    cfg = _preimu_mock_cfg(tmp_path); s = CollectorSession.create(cfg)
    r = s.hardware_check(0.6)
    assert r["ready"] is True, r["problems"]
    assert set(r["streams"]) >= {"head", "left_wrist", "right_wrist"} and all(d["video_frames"] == d["frame_meta"] for d in r["streams"].values())
    assert r["grippers_live"]["left"]["hz"] > 0 and Path(r["path"]).parent.name == "_hwcheck" and s.state == "IDLE"
    assert s.manager.raw_total == 0 and s.manager.episodes() == []            # never counted as data
    s.close()


def test_integrity_validator_detects_problems(tmp_path):
    cfg = _preimu_mock_cfg(tmp_path); s = CollectorSession.create(cfg); s.start(); time.sleep(0.3); s.stop(); meta = s.keep(); ep = s.manager.session_dir / meta["episode_dir"]; s.close()
    assert validate_episode(ep)["ok"]
    (ep / "left_wrist.mp4").unlink(); v = validate_episode(ep, expect_imus=["left"])
    assert not v["ok"] and any("left_wrist.mp4 missing" in p for p in v["problems"]) and any("imu left" in p for p in v["problems"])
    verdict, notes = preliminary_qa(dict(streams={"head": dict(fps_measured=12.0, frames=10, duration_s=1)}, duration_s=1), [dict(kind="camera_drop")])
    assert verdict == "REVIEW" and len(notes) == 2


def test_ui_offscreen_smoke(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets as W
    from handumi_collector.ui.app import MainWindow
    app = W.QApplication.instance() or W.QApplication([])
    cfg = _preimu_mock_cfg(tmp_path); s = CollectorSession.create(cfg); w = MainWindow(s)
    for _ in range(3): w.refresh(); app.processEvents()
    assert w.dots["imu_left"].text().endswith("not installed") and "READY" in w.rec.text()
    w.on_start(); assert s.state == "RECORDING"; time.sleep(0.4); w.refresh(); assert "REC" in w.rec.text()
    w.on_mark(); w.on_stop(); assert s.state == "REVIEW" and not w.review.isHidden() and "Preliminary QA" in w.review.text()   # isVisible needs a shown window
    time.sleep(0.3); w.refresh(); w.on_keep(); assert s.state == "IDLE" and s.manager.raw_total == 1
    w.on_hwcheck(); assert "READY FOR COLLECTION" in w.review.text() or "HARDWARE CHECK" in w.review.text()
    w.close(); app.processEvents()


def test_ui_previews_every_configured_camera_and_renders_depth(tmp_path):
    """The preview row used to be a hardcoded head/left/right triple, so the Orbbec was a status dot only — a pilot
    could be recorded pointing at the wrong thing with no way to see it. Tiles now come from the profile, and an
    aux_depth camera renders its DEPTH map, not its colour image."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets as W
    from handumi_collector.ui.app import MainWindow
    app = W.QApplication.instance() or W.QApplication([])
    cfg = _preimu_mock_cfg(tmp_path); s = CollectorSession.create(cfg); w = MainWindow(s)
    assert set(w.previews) == {c.name for c in cfg.hardware.cameras}
    assert w.previews["head_depth"].is_depth and not w.previews["head"].is_depth
    for _ in range(3): w.refresh(); app.processEvents()
    assert w.previews["head_depth"].lab.pixmap() is not None and not w.previews["head_depth"].lab.pixmap().isNull()
    s.close(); w.close(); app.processEvents()


def test_ui_without_a_c922_still_previews_the_depth_camera(tmp_path):
    """The Stage-A RGB-D profile has no C922 at all: the depth tile must be the one that appears."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets as W
    from handumi_collector.config import AppConfig, CameraCfg, CollectorCfg, HardwareCfg, TasksCfg, DEFAULT_CONFIG_DIR
    from handumi_collector.ui.app import MainWindow
    app = W.QApplication.instance() or W.QApplication([])
    hw = HardwareCfg(profile="stagea_test", cameras=[CameraCfg("head_depth", "aux_depth", backend="mock", width=32, height=24, codec="libx264")],
                     imus=[], grippers=[])
    cfg = AppConfig(hardware=hw, tasks=TasksCfg(target_per_order=1), config_dir=DEFAULT_CONFIG_DIR,
                    collector=CollectorCfg(dataset_root=str(tmp_path / "raw"), session_prefix="Hstagea", min_episode_s=0.1))
    s = CollectorSession.create(cfg); w = MainWindow(s)
    assert list(w.previews) == ["head_depth"] and w.previews["head_depth"].is_depth
    for _ in range(3): w.refresh(); app.processEvents()
    assert not w.previews["head_depth"].lab.pixmap().isNull()
    s.close(); w.close(); app.processEvents()


# ------------------------------------------------------------------------------------- gripper raw-integrity gate (2026-09-16)
def test_gripper_integrity_gate_levels():
    """Left gripper, 2026-09-16: after a servo power cycle its unwrapped counter sat ~4000 ticks away from the calibrated span,
    normalized read a flat 0.25 for 22 episodes and preliminary QA said PASS. A reading outside the span, or one that never
    changes, is a REJECT; a jaw that simply was not used is only a note."""
    from handumi_collector.collector.integrity import gripper_integrity
    healthy = dict(n=2300, raw_min=2100, raw_max=3900, raw_changes=1800, norm_min=0.02, norm_max=0.97)
    assert gripper_integrity("right", healthy, 2110, 3946) == ("PASS", None)
    frozen = dict(n=2303, raw_min=2458, raw_max=2458, raw_changes=0, norm_min=0.25, norm_max=0.25)
    lvl, note = gripper_integrity("left", frozen, 6943, 5389); assert lvl == "REJECT" and "frozen" in note
    moved_counter = dict(n=2300, raw_min=2355, raw_max=2861, raw_changes=900, norm_min=0.23, norm_max=0.29)
    lvl, note = gripper_integrity("left", moved_counter, 6943, 5389); assert lvl == "REJECT" and "RECALIBRATE" in note
    idle_hand = dict(n=2300, raw_min=6460, raw_max=6482, raw_changes=400, norm_min=0.30, norm_max=0.32)
    lvl, note = gripper_integrity("left", idle_hand, 6943, 5389); assert lvl == "REJECT" and "signal anomaly" in note   # no one-handed episodes exist
    lvl, note = gripper_integrity("left", None, 6943, 5389, duration_s=20.0); assert lvl == "REJECT" and "samples" in note
    lvl, note = gripper_integrity("left", dict(healthy, n=30), 2110, 3946, duration_s=20.0); assert lvl == "REJECT", "30 samples in 20 s is a dead stream"
    assert gripper_integrity("left", healthy, None, None) == ("PASS", None)          # uncalibrated: no span to judge against


def test_review_verdict_uses_the_gripper_gate(tmp_path):
    cfg = _preimu_mock_cfg(tmp_path); s = CollectorSession.create(cfg, dataset="Hpilot")
    try:
        time.sleep(0.3); s.start(); time.sleep(0.6); s.stop()
        r = s.review_summary(); assert "grip_stats" in dir(s.recorder) and set(s.recorder.grip_stats) == {"left", "right"}
        # mock jaws sweep inside their calibrated span -> the gate must not object
        assert not any("gripper" in n for n in r["notes"]), r["notes"]
        # a servo whose counter moved: fake the stats and the verdict must turn REJECT with a recalibrate note
        s.recorder.grip_stats["left"] = dict(n=500, raw_min=20000, raw_max=20400, raw_changes=300, norm_min=1.0, norm_max=1.0, raw_last=20400, raw_first=20000)
        r2 = s.review_summary(); assert r2["verdict"] == "REJECT" and any("RECALIBRATE" in n for n in r2["notes"])
        s.discard(notes="test")
    finally: s.close()
