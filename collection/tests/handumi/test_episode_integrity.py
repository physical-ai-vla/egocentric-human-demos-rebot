"""End-to-end with mock devices: record ~0.7 s, KEEP, then check every artifact is consistent (this is the M0 gate)."""
import json
import time
from collections import Counter
import av
from handumi_collector.collector.episode_manager import EpisodeManager
from handumi_collector.collector.mcap_writer import read_messages
from handumi_collector.collector.recorder import EpisodeRecorder
from handumi_collector.devices.manager import DeviceManager


def _record(cfg, mgr: EpisodeManager, dm: DeviceManager, order="RBP", seconds=0.7):
    rec = EpisodeRecorder(cfg, dm, mgr.next_episode_dir(), order, cfg.tasks.instruction(order), mgr.anchor)
    rec.start(); time.sleep(seconds); rec.stop()
    return rec


def test_episode_roundtrip(mock_cfg):
    dm = DeviceManager(mock_cfg.hardware); dm.build(); dm.connect_all()
    assert dm.required_ok() == (True, []), dm.errors
    mgr = EpisodeManager(mock_cfg)
    mgr.write_session_meta(devices={n: s.detail for n, s in dm.statuses().items()}, video_listing=[])
    time.sleep(0.2)   # let buffers fill so start() must discard pre-start data
    rec = _record(mock_cfg, mgr, dm)
    assert (rec.episode_dir / ".incomplete").exists()
    meta = rec.finalize(status="KEEP"); mgr.keep(rec.episode_dir, "RBP")
    dm.close_all()
    ep = rec.episode_dir
    assert not (ep / ".incomplete").exists() and (ep / "episode_meta.json").exists() and (ep / "events.json").exists()
    # videos: one per recorded stream, frame count == frame_meta count, decodable
    for name in ("head", "left_wrist", "right_wrist", "head_depth"):
        assert (ep / f"{name}.mp4").exists()
        with av.open(str(ep / f"{name}.mp4")) as c:
            n = sum(1 for _ in c.decode(video=0))
        assert n == meta["streams"][name]["frames"] > 10, (name, n, meta["streams"][name])
    depth_pngs = list((ep / "head_depth_depth").glob("*.png"))
    assert len(depth_pngs) == meta["streams"]["head_depth"]["frames"]
    # mcap: topics, monotonic log times per topic, frame_meta ↔ video frame numbering, no pre-start samples
    msgs = list(read_messages(ep / "sensors.mcap"))
    topics = Counter(t for t, _, _ in msgs)
    for t in ("/head/frame_meta", "/left/imu", "/right/imu", "/left/gripper", "/right/gripper", "/system/sync"):
        assert topics[t] > 0, topics
    for topic in topics:
        ts = [lt for t, lt, _ in msgs if t == topic]
        assert ts == sorted(ts), topic
    fm = [m for t, _, m in msgs if t == "/head/frame_meta"]
    assert [m["video_frame"] for m in fm] == list(range(len(fm)))
    assert all(m["capture_ns"] >= meta["t_start_monotonic_ns"] for m in fm)
    imu = [m for t, _, m in msgs if t == "/left/imu"]
    assert 0.5 * 400 * 0.7 < len(imu) < 1.5 * 400 * 0.7 + 50, len(imu)
    assert all(imu[i + 1]["seq"] == imu[i]["seq"] + 1 for i in range(len(imu) - 1))     # no lost IMU samples
    assert all(imu[i + 1]["device_timestamp_us"] > imu[i]["device_timestamp_us"] for i in range(len(imu) - 1))
    grip = [m for t, _, m in msgs if t == "/right/gripper"]
    assert all(0.0 <= m["normalized"] <= 1.0 for m in grip)
    # meta consistency + counters
    assert meta["status"] == "KEEP" and meta["order"] == "RBP" and meta["hw_event_in_episode"] is False, meta["event_kinds"]
    assert meta["sensor_messages"]["/left/imu"] == len(imu)
    counters = json.loads((mgr.session_dir / "counters.json").read_text())
    assert counters["valid"]["RBP"] == 1 and counters["raw_total"] == 1


def test_discard_and_rebuild_and_incomplete(mock_cfg):
    dm = DeviceManager(mock_cfg.hardware); dm.build(); dm.connect_all()
    mgr = EpisodeManager(mock_cfg)
    r1 = _record(mock_cfg, mgr, dm, "RPB", 0.3); r1.finalize(status="KEEP"); mgr.keep(r1.episode_dir, "RPB")
    r2 = _record(mock_cfg, mgr, dm, "RPB", 0.3); r2.finalize(status="KEEP"); mgr.discard(r2.episode_dir)
    r3 = _record(mock_cfg, mgr, dm, "BRP", 0.3)          # simulate crash: stopped but never finalized
    dm.close_all()
    assert not r2.episode_dir.exists() and (mgr.session_dir / "_discarded" / r2.episode_dir.name / "sensors.mcap").exists()
    assert json.loads((mgr.session_dir / "_discarded" / r2.episode_dir.name / "episode_meta.json").read_text())["status"] == "DISCARD"
    fresh = EpisodeManager(mock_cfg, session_dir=mgr.session_dir)       # restart → counters rebuilt from disk
    assert fresh.counts["RPB"] == 1 and fresh.counts["BRP"] == 0 and fresh.rejected == 1
    assert fresh.incomplete_episodes() == [r3.episode_dir]
    assert fresh.next_episode_dir().name == "episode_000004"           # numbering never reuses ids
    assert fresh.next_order() == "RBP"                                  # least collected, config order tie-break


def test_dropout_produces_event(mock_cfg):
    dm = DeviceManager(mock_cfg.hardware); dm.build(); dm.connect_all()
    dm.imus["left"].fail_after_s = 0.2
    mgr = EpisodeManager(mock_cfg)
    rec = _record(mock_cfg, mgr, dm, "RBP", 0.6); meta = rec.finalize(status="KEEP")
    dm.close_all()
    assert meta["hw_event_in_episode"] is True
    assert {"device_error", "imu_timeout"} & set(meta["event_kinds"]), meta["event_kinds"]


def test_a_jaw_that_never_moved_is_not_a_grasp_signal(tmp_path):
    """A broken jaw does not look like a fault. The encoder answers at its normal rate and reports a perfectly steady
    number, so sample counts, NaN checks and gap detectors all pass. Five pilot episodes were recorded on
    2026-09-15 with the left jaw pinned at exactly 1.000 because its bearing had snapped, and nothing objected --
    which would have put a constant into the grasp channel of an R150-compatible export as if it were data.

    Travel is what separates 'closed the whole time' from 'not measuring': a real episode opens and closes."""
    import numpy as np
    from handumi_collector.collector.integrity import MIN_GRIPPER_TRAVEL

    def travel_of(swing):
        vals = [1.0 - swing * (i % 2) for i in range(400)]
        return float(np.percentile(vals, 95) - np.percentile(vals, 5))

    stuck, working = travel_of(0.0), travel_of(0.5)
    assert stuck < MIN_GRIPPER_TRAVEL <= working, (stuck, working)


def test_pilot_episodes_report_the_broken_left_jaw(tmp_path):
    """The real five. Kept as a test because the failure is invisible without this check."""
    import glob, json
    from pathlib import Path
    from handumi_collector.collector.integrity import validate_episode
    eps = sorted(glob.glob("datasets/human_handumi_raw/Hpilot/Hpilot_20260915_171744/episode_*"))
    if not eps:
        import pytest; pytest.skip("pilot episodes not present")
    ep = Path(eps[0])
    meta = json.loads((ep / "episode_meta.json").read_text())
    v = validate_episode(ep, expect_streams=list(meta["streams"]), expect_imus=["left", "right"],
                         expect_grippers=["left", "right"])
    assert v["grippers"]["left"]["moved"] is False and v["grippers"]["left"]["travel"] == 0.0
    assert v["grippers"]["right"]["moved"] is True
    assert any("jaw never moved" in p for p in v["problems"]), v["problems"]
