from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd

from ego_collector.camera.capture import Frame
from ego_collector.io.parquet import read_parquet, stack_column, write_parquet
from ego_collector.recording.episode import EpisodePaths, episode_dir_name, list_episodes, next_episode_id
from ego_collector.recording.metadata import EpisodeMetadata
from ego_collector.recording.recorder import EpisodeWriter


def test_episode_ids_and_paths(tmp_path):
    assert next_episode_id(tmp_path) == 1
    (tmp_path / episode_dir_name(7)).mkdir()
    (tmp_path / "episode_000012").mkdir()
    (tmp_path / "junk").mkdir()
    assert next_episode_id(tmp_path) == 13
    eps = list_episodes(tmp_path)
    assert [e.episode_id for e in eps] == [7, 12]
    p = EpisodePaths(tmp_path / "episode_000007")
    assert p.video.name == "head.mp4" and p.qa_report.parent.name == "qa"
    p.write_metadata({"a": 1})
    assert p.update_metadata(b=2) == {"a": 1, "b": 2}


def test_metadata_roundtrip():
    m = EpisodeMetadata(episode_id=3, task="single_cube_pick_place", instruction="Pick up the red cube")
    d = m.to_dict()
    assert d["outcome"] == "unknown" and d["dataset_tier"] == "unprocessed" and d["resolution"] == [1920, 1080]
    back = EpisodeMetadata.from_dict({**d, "unknown_field": 1})
    assert back == m


def test_parquet_list_columns_roundtrip(tmp_path):
    lm = np.random.default_rng(0).normal(size=(5, 21, 2))
    path = write_parquet(tmp_path / "x.parquet", {"t": np.arange(5), "lm": lm, "flag": [True, False, True, True, False]})
    df = read_parquet(path)
    assert list(df.columns) == ["t", "lm", "flag"]
    back = stack_column(df, "lm")
    np.testing.assert_allclose(back, lm.reshape(5, -1))


def test_episode_writer_writes_video_and_timestamps(tmp_path):
    paths = EpisodePaths(tmp_path / "episode_000001")
    w = EpisodeWriter(paths=paths, width=320, height=240, fps=30)
    t0 = time.monotonic_ns()
    for i in range(12):
        img = np.full((240, 320, 3), i * 20, dtype=np.uint8)
        cap_idx = i if i < 6 else i + 1  # simulate one skipped capture index
        w.put(Frame(index=cap_idx, timestamp_ns=t0 + i * 33_333_333, image=img))
    stats = w.close()
    assert stats["frame_count"] == 12 and stats["dropped_frames"] == 1 and stats["video_codec"] in ("avc1", "mp4v")
    assert abs(stats["fps_measured"] - 30.0) < 0.5
    ts = pd.read_parquet(paths.timestamps)
    assert list(ts.columns) == ["frame_index", "capture_index", "timestamp_ns"] and len(ts) == 12
    cap = cv2.VideoCapture(str(paths.video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    ok, frame = cap.read()
    assert ok and frame.shape == (240, 320, 3)
