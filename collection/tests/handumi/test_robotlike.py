"""robot_like_v1 protocol: live monitor math, gripper stages, and the session wiring (protocol tag, live summary)."""
import json
import time
from types import SimpleNamespace
import numpy as np
from handumi_collector.collector.session import CollectorSession
from handumi_collector.devices.base import ImuSample, SampleBuffer
from handumi_collector.robotlike.monitor import RobotLikeMonitor, load_protocol

HZ = 15.0


class _Imu:
    def __init__(self): self.buffer = SampleBuffer(4000); self.cfg = SimpleNamespace(rate_hz=400); self.seq = 0

    def push(self, t_ns, w, a=(0.0, 0.0, 9.81)):
        self.seq += 1; self.buffer.append(ImuSample("x", self.seq, t_ns // 1000, t_ns, *a, *w, 30.0))


class _Grip:
    def __init__(self): self.norm = 0.9

    def quality(self): return SimpleNamespace(normalized=self.norm)


def _devices():
    return SimpleNamespace(imus={"left": _Imu(), "right": _Imu()}, grippers={"left": _Grip(), "right": _Grip()})


def _run(mon, dev, seconds, w_fn, t0=0, grip_fn=None):
    """Feed 400 Hz samples and tick at 15 Hz on a synthetic clock."""
    t = t0; step = int(1e9 / 400); next_tick = t0 + int(1e9 / HZ)
    while t < t0 + int(seconds * 1e9):
        for side, imu in dev.imus.items(): imu.push(t, w_fn(side, t))
        t += step
        if t >= next_tick:
            if grip_fn: grip_fn(t)
            mon.tick(t); next_tick += int(1e9 / HZ)
    return t


def _start_when_ready(s, timeout=5.0):
    """The mock jaw sweeps open/closed; REC is gated on 'closed at rest', so wait for a ready moment like the UI does."""
    t_end = time.monotonic() + timeout
    while not s.can_record()[0]:
        assert time.monotonic() < t_end, s.can_record()[1]
        time.sleep(0.02)
    s.start()


def test_protocol_file_sets_30s_and_thresholds():
    p = load_protocol("robot_like_v1")
    assert p["protocol"] == "robot_like_v1" and p["auto_loop_overrides"]["episode_s"] == 30.0
    assert abs(p["ang_w_p95"] - 0.996) < 1e-9 and abs(p["ang_a_p95"] - 4.413) < 1e-9


def test_slow_rotation_passes_fast_rotation_fails():
    p = load_protocol("robot_like_v1"); dev = _devices(); mon = RobotLikeMonitor(p, dev)
    mon.begin_episode(__import__("pathlib").Path("/tmp/_rl_unused"))
    _run(mon, dev, 3.0, lambda side, t: (0.3, 0.0, 0.0) if side == "left" else (2.5, 0.0, 0.0))
    snap = mon.snapshot()
    assert snap["sides"]["left"]["w_lamp"] == "GREEN" and abs(snap["sides"]["left"]["w"] - 0.3) < 1e-6
    assert snap["sides"]["right"]["w_lamp"] == "RED"
    with mon._lock: s = mon._summary()
    assert s["sides"]["left"]["over_frac_ang_w"] == 0.0 and s["sides"]["right"]["over_frac_ang_w"] == 1.0
    assert any("right wrist rotation fast" in r for r in s["reasons"]) and not any(r.startswith("left wrist") for r in s["reasons"])
    assert s["verdict"] == "FAIL"


def test_angular_accel_matches_probe_definition():
    """|dw|*15 on the 15 Hz grid: a 1 Hz sine of amplitude A peaks at ~2*pi*A (continuous derivative)."""
    p = load_protocol("robot_like_v1"); dev = _devices(); mon = RobotLikeMonitor(p, dev); mon.begin_episode(__import__("pathlib").Path("/tmp/_rl_unused"))
    A = 0.5
    _run(mon, dev, 4.0, lambda side, t: (A * np.sin(2 * np.pi * t / 1e9), 0.0, 0.0))
    peak = mon.sides["left"].as_ and max(mon.sides["left"].as_)
    assert 0.85 * 2 * np.pi * A < peak < 1.1 * 2 * np.pi * A


def test_sample_gap_does_not_read_as_acceleration():
    p = load_protocol("robot_like_v1"); dev = _devices(); mon = RobotLikeMonitor(p, dev)
    t = _run(mon, dev, 1.0, lambda side, t: (0.2, 0.0, 0.0))
    for k in range(10): mon.tick(t + int((k + 1) * 1e9 / HZ))          # no samples for ~0.7 s
    assert mon.sides["left"].w is None
    t2 = t + int(11 * 1e9 / HZ)
    for side, imu in dev.imus.items(): imu.push(t2 - 1000, (0.9, 0.0, 0.0))
    mon.tick(t2)
    assert mon.sides["left"].a is None                                   # first tick after a gap has no derivative


def test_grasp_release_counted_with_hysteresis():
    p = load_protocol("robot_like_v1"); dev = _devices(); mon = RobotLikeMonitor(p, dev)
    mon.tick(0)                                                          # initial state: open
    mon.begin_episode(__import__("pathlib").Path("/tmp/_rl_unused"))
    seq = [0.9, 0.5, 0.4, 0.3, 0.2, 0.4, 0.6, 0.62, 0.7, 0.9, 0.3, 0.8]   # dips to 0.4 / climbs to 0.62 are inside the band
    for i, n in enumerate(seq):
        dev.grippers["left"].norm = n; mon.tick(int((i + 1) * 1e9 / HZ))
    st = mon.sides["left"]
    assert (st.grasps, st.releases) == (2, 2)


def test_session_tags_protocol_and_writes_live_summary(mock_cfg):
    mock_cfg.collector.protocol = load_protocol("robot_like_v1")
    s = CollectorSession.create(mock_cfg)
    try:
        assert s.robotlike is not None
        time.sleep(0.3); _start_when_ready(s); time.sleep(0.8); s.stop(); r = s.review_summary()
        assert any(n.startswith("robot-like ") for n in r["notes"])
        meta = s.keep(); ep = s.manager.session_dir / meta["episode_dir"]
        assert meta["protocol"] == "robot_like_v1" and meta["robot_like_live"]["verdict"] in ("PASS", "FAIL")
        summ = json.loads((ep / "derived" / "robot_like" / "live_summary.json").read_text())
        assert summ["schema"] == "robot_like_live/v1" and summ["sides"]["left"]["n_ticks"] > 0
    finally:
        s.close()


def test_legacy_session_is_unchanged(mock_cfg):
    s = CollectorSession.create(mock_cfg)
    try:
        assert s.robotlike is None
        _start_when_ready(s); time.sleep(0.3); s.stop(); meta = s.keep()
        assert "protocol" not in meta and "robot_like_live" not in meta
        assert not (s.manager.session_dir / meta["episode_dir"] / "derived" / "robot_like").exists()
    finally:
        s.close()


def test_ui_shows_robotlike_panel_only_under_protocol(mock_cfg):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets as W
    from handumi_collector.ui.app import MainWindow
    app = W.QApplication.instance() or W.QApplication([])
    mock_cfg.collector.protocol = load_protocol("robot_like_v1")
    s = CollectorSession.create(mock_cfg); w = MainWindow(s)
    try:
        time.sleep(0.3)
        for _ in range(3): w.refresh(); app.processEvents()
        txt = w.rl_label.text()
        assert "LEFT" in txt and "RIGHT" in txt and "rot" in txt and "stage" in txt
        _start_when_ready(s); time.sleep(0.5); w.refresh(); assert "grasp" in w.rl_label.text()
        w.on_stop(); w.refresh(); assert "robot-like" in w.review.text()
        w.on_keep(); w.refresh(); assert "last episode" in w.rl_label.text()
        w.on_load_robotlike(); qa = w.qa_text.toPlainText()
        assert "live " in qa and "offline: not run yet" in qa
    finally:
        w.close(); app.processEvents()
    s2 = CollectorSession.create(mock_cfg.__class__(**{**mock_cfg.__dict__, "collector": type(mock_cfg.collector)(**{**mock_cfg.collector.__dict__, "protocol": None})}))
    w2 = MainWindow(s2)
    try: assert w2.rl_label is None
    finally: w2.close(); app.processEvents()


def test_dataset_qa_relaxes_only_the_stacking_checks():
    """HRA_red (approach-only): a still jaw passes, a frozen jaw still REJECTs; robot-like FAIL is advisory."""
    from handumi_collector.collector.integrity import gripper_integrity
    still = dict(n=300, raw_changes=4, raw_min=4316, raw_max=4320, norm_min=0.98, norm_max=0.99)
    assert gripper_integrity("right", still, 4300, 3800, duration_s=10.0)[0] == "REJECT"
    assert gripper_integrity("right", still, 4300, 3800, duration_s=10.0, require_travel=False) == ("PASS", None)
    frozen = dict(still, raw_changes=0)          # jaw at rest, servo answering at 100 Hz (2026-10-03 HRA_red ep 4/5)
    assert gripper_integrity("right", frozen, 4300, 3800, duration_s=10.0)[0] == "REJECT"
    assert gripper_integrity("right", frozen, 4300, 3800, duration_s=10.0, require_travel=False) == ("PASS", None)
    dead = dict(still, n=20)                     # a servo that stops answering stops producing samples
    assert gripper_integrity("right", dead, 4300, 3800, duration_s=10.0, require_travel=False)[0] == "REJECT"
    away = dict(still, raw_min=9000, raw_max=9001)
    assert gripper_integrity("right", away, 4300, 3800, duration_s=10.0, require_travel=False)[0] == "REJECT"


def test_hra_red_mode_loads_the_relaxations(tmp_path):
    from handumi_collector.config import DEFAULT_CONFIG_DIR, load_config
    from handumi_collector.collector.episode_manager import EpisodeManager
    cfg = load_config(DEFAULT_CONFIG_DIR, mock=True)
    EpisodeManager(cfg, session_dir=tmp_path / "s", dataset="HRA_red")
    assert cfg.collector.qa == {"gripper_checks": False, "robot_like": "advisory"}
    cfg2 = load_config(DEFAULT_CONFIG_DIR, mock=True)
    EpisodeManager(cfg2, session_dir=tmp_path / "s2", dataset="HRL80")
    assert cfg2.collector.qa == {}
