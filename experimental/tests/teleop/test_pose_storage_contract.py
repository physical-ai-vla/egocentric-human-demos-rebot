"""live / refined pose storage contract (frozen 2026-09-11, before bulk recording).

The offline refinement itself is ladder step 11 and is NOT implemented. What is frozen here is the naming and
provenance, because every episode recorded from now on locks in the schema: deferring the *names* would later force
either a rename (old episodes stop matching) or an unmarked `live` stream sitting next to a `_refined` sibling."""
import json
import pandas as pd
import pytest
from ego_teleop.recorder.episode_logger import TeleopEpisodeLogger, EPISODE_LAYOUT, stream_path
from ego_teleop.tracking.interfaces import WristPose
import numpy as np


def _wp(t_ns=0, source="openvins"):
    return WristPose.from_T(t_ns, np.eye(4), source=source)


def test_live_suffix_is_the_canonical_name(tmp_path):
    lg = TeleopEpisodeLogger(tmp_path / "ep")
    lg.add_wrist_pose("left", _wp()); lg.add_wrist_pose("right", _wp())
    lg.close()
    assert (tmp_path / "ep/raw/wrist_pose_live_left.parquet").exists()
    assert (tmp_path / "ep/raw/wrist_pose_live_right.parquet").exists()
    # a new recording must NEVER write the pre-2026-09-11 name
    assert not (tmp_path / "ep/raw/wrist_pose_left.parquet").exists()
    assert "wrist_pose_left" not in EPISODE_LAYOUT["raw"]


def test_provenance_is_recorded_per_stream(tmp_path):
    lg = TeleopEpisodeLogger(tmp_path / "ep")
    lg.add_wrist_pose("left", _wp(source="openvins"))
    meta = lg.close()
    prov = meta["pose_streams"]["wrist_pose_live_left"]
    assert prov["pose_type"] == "live" and prov["causal"] is True
    assert prov["estimator"] == "openvins"
    assert prov["source_camera"] == "left_wrist" and prov["source_imu"] == "left_wrist"
    assert json.loads((tmp_path / "ep/metadata.json").read_text())["pose_streams"] == meta["pose_streams"]


def test_legacy_episode_still_resolves_and_counts_as_live(tmp_path):
    """A pre-contract episode is readable; its unmarked file IS a live stream (written by the online estimator)."""
    ep = tmp_path / "old"; (ep / "raw").mkdir(parents=True)
    pd.DataFrame([{"t_ns": 1, "side": "left"}]).to_parquet(ep / "raw/wrist_pose_left.parquet", index=False)
    p = stream_path(ep, "wrist_pose_live_left")
    assert p is not None and p.name == "wrist_pose_left.parquet"
    assert stream_path(ep, "wrist_pose_live_right") is None          # absent stays absent, never invented


def test_canonical_file_wins_over_legacy(tmp_path):
    ep = tmp_path / "both"; (ep / "raw").mkdir(parents=True)
    for n in ("wrist_pose_left", "wrist_pose_live_left"):
        pd.DataFrame([{"t_ns": 1}]).to_parquet(ep / f"raw/{n}.parquet", index=False)
    assert stream_path(ep, "wrist_pose_live_left").name == "wrist_pose_live_left.parquet"


def test_refined_stream_can_be_declared_without_touching_live(tmp_path):
    """Ladder step 11 will add this; the contract must already accept it. Live must remain untouched."""
    lg = TeleopEpisodeLogger(tmp_path / "ep")
    lg.add_wrist_pose("left", _wp())
    lg.declare_pose_stream("wrist_pose_refined_left", pose_type="refined", causal=False,
                           estimator="offline", source_camera="left_wrist", source_imu="left_wrist",
                           refinement_backend="TBD", refinement_version="TBD")
    meta = lg.close()
    assert meta["pose_streams"]["wrist_pose_live_left"]["pose_type"] == "live"
    assert meta["pose_streams"]["wrist_pose_refined_left"]["causal"] is False
    assert (tmp_path / "ep/raw/wrist_pose_live_left.parquet").exists()   # live not overwritten or removed
