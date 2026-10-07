"""VIO init readiness gate: START waits for real stillness on every IMU, not for a timer.

The pilots that started already moving cost OpenVINS 4-12 s of initialization per episode; the gate exists so that
cannot happen again. These tests pin (1) the verdict logic on synthetic samples, (2) the session state machine around it
with mock IMUs -- still mocks open it, a shaken mock keeps it shut, cancel works -- (3) that what the gate saw is written
into the episode metadata, and (4) that a profile without IMUs starts immediately and says the gate was not applied."""
import json
import math
import time
from pathlib import Path
import pytest
from handumi_collector.collector.session import CollectorSession
from handumi_collector.collector.stillness import DEFAULTS, StillnessGate
from handumi_collector.config import DEFAULT_CONFIG_DIR, AppConfig, CameraCfg, CollectorCfg, GripperCfg, HardwareCfg, ImuCfg, TasksCfg, load_config
from handumi_collector.devices.base import ImuSample
from handumi_collector.devices.mock import MockImu

CFG = dict(DEFAULTS, min_still_s=1.0, window_s=0.25, countdown_s=0.0)
NS = 1_000_000_000


def _samples(t0_ns, dur_s, *, rate=200, gyro_deg_s=1.5, accel_std=0.05, gap_at_s=None, gap_ms=0.0, side="right"):
    out = []; n = int(dur_s * rate); dt = int(NS / rate); t = t0_ns
    for i in range(n):
        if gap_at_s is not None and abs(i / rate - gap_at_s) < 0.5 / rate: t += int(gap_ms * 1e6)
        w = math.radians(gyro_deg_s); a = 9.81 + accel_std * (1 if i % 2 else -1)       # std exactly accel_std
        out.append(ImuSample(side, i, i * 5000, t, 0.0, 0.0, a, w, 0.0, 0.0, 30.0)); t += dt
    return out


def test_window_verdict_uses_gyro_accel_and_gaps():
    g = StillnessGate(CFG, sides=("right",), rate_hz=200)
    ok, w = g.evaluate_window(_samples(0, 0.25)); assert ok and w["gyro_mean_deg_s"] == pytest.approx(1.5, abs=0.01)
    ok, w = g.evaluate_window(_samples(0, 0.25, gyro_deg_s=8.0)); assert not ok and "gyro" in w["reason"]
    ok, w = g.evaluate_window(_samples(0, 0.25, accel_std=0.5)); assert not ok and "accel" in w["reason"]
    ok, w = g.evaluate_window(_samples(0, 0.25, gap_at_s=0.1, gap_ms=60)); assert not ok and "gap" in w["reason"]
    ok, w = g.evaluate_window(_samples(0, 0.25)[:5]); assert not ok and "samples" in w["reason"]


def test_ready_only_after_min_still_and_motion_resets_the_run():
    g = StillnessGate(CFG, sides=("right",), rate_hz=200); g.arm(0)
    still = _samples(0, 3.0)
    for k in range(1, 13):                                            # poll every 0.25 s
        now = int(k * 0.25 * NS); g.update("right", [s for s in still if s.host_receive_ns <= now], now)
        assert g.ready == (g.progress_s() >= 1.0)
    assert g.ready and g.label() == "VIO INIT READY" and g.state["right"].ready_at_ns is not None
    # a moving window: run collapses to zero and the label goes back to WAIT STILL
    now = int(3.25 * NS); g.update("right", _samples(int(3.0 * NS), 0.25, gyro_deg_s=40.0), now)
    assert not g.ready and g.progress_s() == 0.0 and g.label() == "WAIT STILL"
    now = int(3.5 * NS); g.update("right", _samples(int(3.25 * NS), 0.25), now)
    assert 0 < g.progress_s() < 1.0 and g.label().startswith("STILL ")


def test_every_side_must_be_still():
    g = StillnessGate(CFG, sides=("left", "right"), rate_hz=200); g.arm(0)
    for k in range(1, 9):
        now = int(k * 0.25 * NS)
        g.update("right", _samples(now - NS // 4, 0.25, side="right"), now)
        g.update("left", _samples(now - NS // 4, 0.25, gyro_deg_s=30.0, side="left"), now)   # left keeps moving
    assert g.state["right"].still_s >= 1.0 and not g.ready and g.progress_s() == 0.0
    m = g.to_meta(started_ns=None)
    assert m["sides"]["right"]["ready"] and not m["sides"]["left"]["ready"] and m["gyro_max_deg_s"] == 6.0 and not m["satisfied"]


def test_not_armed_means_no_progress():
    g = StillnessGate(CFG, sides=("right",), rate_hz=200)
    g.update("right", _samples(0, 0.25), NS // 4); assert not g.ready and g.progress_s() == 0.0


def test_production_config_carries_the_measured_thresholds():
    st = load_config().collector.stillness
    assert st["enabled"] is True and st["min_still_s"] == 3.0 and st["gyro_max_deg_s"] == 6.0 and st["accel_std_max"] == 0.35 and st["max_gap_ms"] == 25.0
    assert st["countdown_s"] == 0.0 and st["hold_after_rec_s"] == 3.0, "stillness must be INSIDE the recording: REC at once, then hold"


# ---------------------------------------------------------------------------------------------------------- session
def _mock_cfg(tmp_path, **still) -> AppConfig:
    hw = HardwareCfg(profile="gate_test", cameras=[CameraCfg("head", "policy_obs", backend="mock", width=64, height=48, codec="libx264"),
                                                   CameraCfg("left_wrist", "pose_estimation", backend="mock", width=64, height=48, codec="libx264"),
                                                   CameraCfg("right_wrist", "pose_estimation", backend="mock", width=64, height=48, codec="libx264")],
                     imus=[ImuCfg("left", backend="mock", rate_hz=400), ImuCfg("right", backend="mock", rate_hz=400)],
                     grippers=[GripperCfg("left", backend="mock", ticks_closed=1000, ticks_open=2000), GripperCfg("right", backend="mock", ticks_closed=3000, ticks_open=2000)])
    s = dict(DEFAULTS, min_still_s=0.4, countdown_s=0.0, hold_after_rec_s=0.5); s.update(still)
    return AppConfig(hardware=hw, tasks=TasksCfg(target_per_order=2), config_dir=DEFAULT_CONFIG_DIR,
                     collector=CollectorCfg(dataset_root=str(tmp_path / "raw"), session_prefix="Htest", min_episode_s=0.1, stillness=s))


def _poll_until(s, state, timeout_s):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        s.poll()
        if s.state == state: return True
        time.sleep(0.05)
    return s.state == state


def _find(d, key):
    if isinstance(d, dict):
        if key in d: return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None: return r
    return None


def test_still_mock_units_open_the_gate_and_the_episode_records_how(tmp_path):
    cfg = _mock_cfg(tmp_path); s = CollectorSession.create(cfg, dataset="Hpilot")
    try:
        time.sleep(0.3)
        assert s.request_start() == "WAITING_FOR_STILLNESS" and s.state_label() in ("WAIT STILL",) or s.state_label().startswith("STILL")
        assert _poll_until(s, "RECORDING", 4.0), f"gate never opened: {s.state_label()} {s.gate and s.gate.state}"
        _poll_for(s, 0.8); s.stop(); m = s.keep(notes="gate test")          # poll through the 0.5 s hold so GO is reached
        ep = sorted(s.manager.session_dir.glob("episode_*"))[-1]; meta = json.loads((ep / "episode_meta.json").read_text())
        g = _find(meta, "stillness_gate"); assert g is not None, "stillness_gate missing from episode metadata"
        assert g["enabled"] and g["satisfied"] and g["min_still_s"] == 0.4 and g["gyro_max_deg_s"] == 6.0
        assert set(g["sides"]) == {"left", "right"} and all(v["ready"] and v["still_s"] >= 0.4 for v in g["sides"].values())
        assert all(v["ready_to_start_s"] is not None and v["ready_to_start_s"] >= 0 for v in g["sides"].values())
        kinds = [e["kind"] for e in json.loads((ep / "events.json").read_text())]
        assert "hold_still_start" in kinds and "go_cue" in kinds and "moved_before_go" not in kinds, kinds
        go = [e for e in json.loads((ep / "events.json").read_text()) if e["kind"] == "go_cue"][0]
        assert go["detail"]["clean"] and go["t_rel_s"] >= 0.4 and all(v >= 0.4 for v in go["detail"]["held_still"].values())   # hold 0.5 s; t_start lands a few ms after the gate clock
    finally: s.close()


def _poll_for(s, dur_s):
    t0 = time.monotonic()
    while time.monotonic() - t0 < dur_s: s.poll(); time.sleep(0.03)


def test_moving_during_the_hold_is_logged_not_blocked(tmp_path, monkeypatch):
    cfg = _mock_cfg(tmp_path); s = CollectorSession.create(cfg, dataset="Hpilot")
    try:
        time.sleep(0.3); s.request_start(); assert _poll_until(s, "RECORDING", 4.0) and s.holding_still and s.hold_left_s() > 0
        monkeypatch.setattr(MockImu, "GYRO_AMP_RAD_S", 0.6)      # the operator moves at REC, like stacking episodes 3-4
        _poll_for(s, 0.9)
        assert s.state == "RECORDING" and not s.holding_still     # recording continued; the hold phase ended on time
        s.stop(); s.keep(); ep = sorted(s.manager.session_dir.glob("episode_*"))[-1]
        ev = json.loads((ep / "events.json").read_text()); kinds = [e["kind"] for e in ev]
        assert "moved_before_go" in kinds and "go_cue" in kinds
        assert not [e for e in ev if e["kind"] == "go_cue"][0]["detail"]["clean"]
    finally: s.close()


def test_a_moving_unit_keeps_the_gate_shut_and_cancel_returns_to_idle(tmp_path, monkeypatch):
    monkeypatch.setattr(MockImu, "GYRO_AMP_RAD_S", 0.6)          # ~34 deg/s: a hand that is clearly not still
    cfg = _mock_cfg(tmp_path); s = CollectorSession.create(cfg, dataset="Hpilot")
    try:
        time.sleep(0.3); s.request_start()
        assert not _poll_until(s, "RECORDING", 1.2) and s.state == "WAITING_FOR_STILLNESS"
        assert s.gate.progress_s() == 0.0 and s.state_label() == "WAIT STILL"
        assert "gyro" in (s.gate.state["right"].last_window.get("reason") or "")
        assert not s.can_record()[0]                                # REC stays blocked while waiting
        s.cancel_start(); assert s.state == "IDLE" and s.gate is None and s.manager.raw_total == 0
    finally: s.close()


def test_device_failure_while_waiting_cancels_instead_of_recording(tmp_path):
    cfg = _mock_cfg(tmp_path); s = CollectorSession.create(cfg, dataset="Hpilot")
    try:
        time.sleep(0.3); s.request_start(); s.devices.cameras["head"].close()
        assert s.poll() == "IDLE" and s.state == "IDLE" and s.manager.raw_total == 0
    finally: s.close()


def test_profile_without_imus_starts_immediately_and_says_so(tmp_path):
    cfg = _mock_cfg(tmp_path); cfg.hardware.imus = []
    s = CollectorSession.create(cfg, dataset="Hpilot")
    try:
        time.sleep(0.3); assert s.request_start() == "RECORDING"
        time.sleep(0.3); s.stop(); m = s.keep()
        ep = sorted(s.manager.session_dir.glob("episode_*"))[-1]; g = _find(json.loads((ep / "episode_meta.json").read_text()), "stillness_gate")
        assert g is not None and g["enabled"] is False and "IMU" in g["reason"]
    finally: s.close()
