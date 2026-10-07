"""Index-style curation: quality/embedding/dedup/caption/tier on tiny synthetic episodes."""

from __future__ import annotations

import json

import cv2
import numpy as np

from ego_collector.engine.curate import QualityThresholds, curate_episode, hierarchical_caption
from ego_collector.io.parquet import write_parquet
from ego_collector.recording.episode import EpisodePaths


class FakeEncoder:
    """Deterministic embedding = mean color direction (same video -> same vector)."""

    def __call__(self, frames):
        v = np.concatenate([f.mean(axis=(0, 1)) for f in frames[:3]])
        v = np.resize(v, 16).astype(np.float64)
        return v / (np.linalg.norm(v) + 1e-9)


def make_ep(root, i, *, seed, instruction, tier_meta=None, visible=True):
    paths = EpisodePaths(root / f"episode_{i:06d}")
    paths.root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    w = cv2.VideoWriter(str(paths.video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for _ in range(30):
        w.write(base)
    w.release()
    n = 30
    write_parquet(paths.hands3d_pose, {
        "frame_index": np.arange(n), "timestamp_ns": np.arange(n) * 33_000_000,
        "right_visible": np.full(n, visible), "right_confidence": np.full(n, 0.9),
        "left_visible": np.zeros(n, bool), "left_confidence": np.full(n, np.nan),
    })
    meta = {"episode_id": i, "task": "cube_stack", "instruction": instruction, "order": "RBP",
            "frame_count": n, "dropped_frames": 0, "fps_measured": 30.0, "duration_s": 1.0,
            "dataset_tier": "action_labelled"}
    meta.update(tier_meta or {})
    paths.write_metadata(meta)
    return paths


def test_curation_tiers_and_dedup(tmp_path):
    root = tmp_path
    th = QualityThresholds(min_blur=0.0, sides="right", dup_similarity=0.99)
    enc = FakeEncoder()
    a = make_ep(root, 1, seed=1, instruction="stack RBP")
    r1 = curate_episode(a, encoder=enc, root=root, th=th)
    assert r1["tier"] == "A" and not r1["dedup"]["duplicate"]
    # same video, same instruction -> duplicate -> D
    b = make_ep(root, 2, seed=1, instruction="stack RBP")
    r2 = curate_episode(b, encoder=enc, root=root, th=th)
    assert r2["dedup"]["duplicate"] and r2["tier"] == "D"
    # same video, DIFFERENT instruction -> kept
    c = make_ep(root, 3, seed=1, instruction="stack PBR")
    r3 = curate_episode(c, encoder=enc, root=root, th=th)
    assert not r3["dedup"]["duplicate"] and r3["tier"] == "A"
    # pose failed (hand rarely visible) but video fine -> B (video-only, never rejected)
    d = make_ep(root, 4, seed=7, instruction="stack BRP", visible=False, tier_meta={"dataset_tier": "video_only"})
    r4 = curate_episode(d, encoder=enc, root=root, th=th)
    assert r4["tier"] == "B"
    # bad video (fps) -> D
    e = make_ep(root, 5, seed=9, instruction="stack BPR", tier_meta={"fps_measured": 12.0})
    r5 = curate_episode(e, encoder=enc, root=root, th=th)
    assert r5["tier"] == "D"
    # artifacts written + metadata updated
    m = a.read_metadata()
    assert m["training_tier"] == "A" and (a.curation_dir / "quality_score.json").exists()
    assert np.load(a.curation_dir / "embedding.npy").shape == (16,)
    cap = json.loads((a.curation_dir / "caption.json").read_text())
    assert len(cap["subtasks"]) == 3 and "red cube" in cap["objects"][0]


def test_hierarchical_caption_orders(tmp_path):
    p = make_ep(tmp_path, 9, seed=3, instruction="stack PRB", tier_meta={"order": "PRB", "outcome": "failure"})
    cap = hierarchical_caption(p)
    assert cap["objects"] == ["purple cube", "red cube", "blue cube"]
    assert "on the purple cube" in cap["subtasks"][1]["actions"][2]
    assert cap["outcome_note"].endswith("failure")
