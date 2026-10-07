"""Index-style data engine: quality score -> embedding -> dedup -> tier -> hierarchical caption.

Per episode (written under <episode>/curation/):
  quality_score.json     blur, brightness, fps, drops, hands-visible %, pose confidence
  embedding.npy          pooled visual embedding of start/middle/end frames (DINOv2 by default)
  caption.json           hierarchical caption (template-based; swap in a VLM later)
  report.json            nearest-neighbour episode + similarity + duplicate verdict

metadata gains: quality (dict), training_tier: A (action-supervised) | B (video-only) | D (reject),
duplicate_of (episode name or null). Dedup rule (Figure-style): high visual similarity alone never
drops an episode - it must ALSO share the instruction; same scene with a different instruction is
exactly the prompt-conditioned data we want.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ego_collector.io.parquet import read_parquet
from ego_collector.recording.episode import EpisodePaths

log = logging.getLogger("ego_collector.curate")
TIERS = ("A", "B", "D")


@dataclass
class QualityThresholds:
    min_fps: float = 27.0
    max_drop_fraction: float = 0.02
    min_blur: float = 40.0  # Laplacian variance, motion-blurred video sits well below
    min_hands_visible: float = 0.6  # fraction of frames with the required hand(s) visible
    min_pose_conf: float = 0.5
    sides: str = "right"
    dup_similarity: float = 0.965


def _sample_frames(video: Path, n: int = 9) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            out.append(img)
    cap.release()
    return out


def quality_score(paths: EpisodePaths, th: QualityThresholds) -> dict:
    meta = paths.read_metadata()
    frames = _sample_frames(paths.video)
    blur = [float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) for f in frames]
    bright = [float(f.mean()) for f in frames]
    q: dict = {
        "fps_measured": float(meta.get("fps_measured", 0.0) or 0.0),
        "drop_fraction": int(meta.get("dropped_frames", 0)) / max(int(meta.get("frame_count", 0)) + int(meta.get("dropped_frames", 0)), 1),
        "blur_median": float(np.median(blur)) if blur else float("nan"),
        "blur_p75": float(np.percentile(blur, 75)) if blur else float("nan"),
        "blur_per_frame": [round(b, 1) for b in blur],
        "brightness_median": float(np.median(bright)) if bright else float("nan"),
        "hands_visible_fraction": float("nan"),
        "pose_confidence_median": float("nan"),
        "pose_source": None,
    }
    sides = ("left", "right") if th.sides == "both" else (th.sides,)
    if paths.finger_pose.exists() and (paths.root / "hands3d" / "hawor_export.npz").exists():
        # HaWoR-tracked episode: wrist/finger validity IS the hand-visibility signal
        wp = read_parquet(paths.wrist_pose)
        vis = np.ones(len(wp), bool)
        for s in sides:
            vis &= wp[f"{s}_wrist_valid"].to_numpy(bool)
        q["hands_visible_fraction"] = float(vis.mean()) if len(wp) else 0.0
        q["pose_confidence_median"] = float(np.nanmedian(wp[[f"{s}_tag_decision_margin" for s in sides]].to_numpy()))
        q["pose_source"] = "hawor"
    elif paths.hands3d_pose.exists():
        df = read_parquet(paths.hands3d_pose)
        vis = np.ones(len(df), bool)
        confs = []
        for s in sides:
            vis &= df[f"{s}_visible"].to_numpy(bool)
            c = df[f"{s}_confidence"].to_numpy(dtype=np.float64)
            confs.append(np.nanmedian(c[df[f"{s}_visible"].to_numpy(bool)]) if vis.any() else float("nan"))
        q["hands_visible_fraction"] = float(vis.mean()) if len(df) else 0.0
        q["pose_confidence_median"] = float(np.nanmean(confs))
        q["pose_source"] = "wilor"
    elif paths.hand_pose.exists():
        hp = read_parquet(paths.hand_pose)
        vis = np.ones(len(hp), bool)
        for s in sides:
            vis &= hp[f"{s}_hand_visible"].to_numpy(bool)
        q["hands_visible_fraction"] = float(vis.mean()) if len(hp) else 0.0
        q["pose_source"] = "mediapipe"
    q["checks"] = {
        "fps": q["fps_measured"] >= th.min_fps,
        "drops": q["drop_fraction"] <= th.max_drop_fraction,
        # ego video is motion-blurred WHILE the head moves (normal); gate on the sharp frames
        # (p75 of sampled frames) so we reject focus/lighting problems, not motion.
        "blur": not np.isfinite(q["blur_p75"]) or q["blur_p75"] >= th.min_blur,
        "hands_visible": not np.isfinite(q["hands_visible_fraction"]) or q["hands_visible_fraction"] >= th.min_hands_visible,
    }
    q["video_ok"] = bool(q["checks"]["fps"] and q["checks"]["drops"] and q["checks"]["blur"])
    return q


class TimmEncoder:
    """DINOv2 ViT-S/14 pooled features (any timm model name works)."""

    def __init__(self, model_name: str = "vit_small_patch14_dinov2.lvd142m", device: str = "auto") -> None:
        import timm
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
        cfg = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**cfg, is_training=False)

    def __call__(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image

        with torch.no_grad():
            x = torch.stack([self.transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames_bgr]).to(self.device)
            feats = self.model(x).float().cpu().numpy()
        v = feats.mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-9)


def episode_embedding(paths: EpisodePaths, encoder) -> np.ndarray:
    frames = _sample_frames(paths.video, n=3)  # start / middle / end
    emb = encoder(frames)
    out = paths.curation_dir / "embedding.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb.astype(np.float32))
    return emb


def dedup_report(paths: EpisodePaths, emb: np.ndarray, root: Path, th: QualityThresholds) -> dict:
    from ego_collector.recording.episode import list_episodes

    meta = paths.read_metadata()
    best = {"episode": None, "similarity": float("nan"), "same_instruction": None}
    for other in list_episodes(root):
        if other.root == paths.root:
            continue
        f = other.curation_dir / "embedding.npy"
        if not f.exists():
            continue
        sim = float(np.dot(emb, np.load(f)))
        if not np.isfinite(best["similarity"]) or sim > best["similarity"]:
            om = other.read_metadata()
            best = {"episode": other.root.name, "similarity": sim, "same_instruction": om.get("instruction") == meta.get("instruction")}
    dup = bool(np.isfinite(best["similarity"]) and best["similarity"] >= th.dup_similarity and best["same_instruction"])
    return {"nearest": best, "duplicate": dup, "threshold": th.dup_similarity, "rule": "visual-similar AND same instruction -> duplicate; different instruction always kept"}


ORDER_VERBS = ("at the bottom", "in the middle", "on top")


def hierarchical_caption(paths: EpisodePaths, colors: list[str] | None = None) -> dict:
    """Template-based hierarchy from metadata (swap in a VLM for real captions later)."""
    meta = paths.read_metadata()
    cap: dict = {"task": meta.get("instruction") or meta.get("task", ""), "source": "template", "subtasks": []}
    order = str(meta.get("order") or "")
    colors = colors or ["red", "blue", "purple"]
    by_initial = {c[0].upper(): c for c in colors}
    if order and all(ch in by_initial for ch in order):
        seq = [by_initial[ch] for ch in order]
        cap["objects"] = [f"{c} cube" for c in seq]
        for i, c in enumerate(seq):
            sub = {"subtask": f"Place the {c} cube {ORDER_VERBS[min(i, 2)]}.",
                   "actions": [f"Reach toward the {c} cube.", f"Pinch the {c} cube between thumb and index finger.",
                               ("Place it on the table." if i == 0 else f"Place it on the {seq[i-1]} cube."), "Release the cube."]}
            cap["subtasks"].append(sub)
        cap["result"] = f"The cubes are stacked {', '.join(seq)} from bottom to top."
    if meta.get("outcome") in ("failure", "partial"):
        cap["outcome_note"] = f"episode outcome: {meta.get('outcome')}"
    return cap


def assign_tier(paths: EpisodePaths, q: dict, dup: dict, th: QualityThresholds) -> str:
    """A = action-supervised, B = video-only (world-model), D = reject.

    Dyna-style: a failed pose never rejects the video - bad video quality or duplicates do.
    """
    if dup.get("duplicate") or not q["video_ok"]:
        return "D"
    pose_ok = (
        q.get("pose_source") in ("wilor", "hawor")
        and np.isfinite(q["hands_visible_fraction"]) and q["hands_visible_fraction"] >= th.min_hands_visible
        and (not np.isfinite(q["pose_confidence_median"]) or q["pose_confidence_median"] >= th.min_pose_conf)
        and paths.read_metadata().get("dataset_tier") == "action_labelled"
    )
    return "A" if pose_ok else "B"


def curate_episode(paths: EpisodePaths, *, encoder, root: Path, th: QualityThresholds = QualityThresholds(), colors: list[str] | None = None) -> dict:
    q = quality_score(paths, th)
    emb = episode_embedding(paths, encoder)
    dup = dedup_report(paths, emb, root, th)
    cap = hierarchical_caption(paths, colors)
    tier = assign_tier(paths, q, dup, th)
    cur = paths.curation_dir
    cur.mkdir(parents=True, exist_ok=True)
    (cur / "quality_score.json").write_text(json.dumps(q, indent=2, default=float))
    (cur / "report.json").write_text(json.dumps(dup, indent=2, default=float))
    (cur / "caption.json").write_text(json.dumps(cap, indent=2))
    paths.update_metadata(quality={k: q[k] for k in ("fps_measured", "blur_median", "hands_visible_fraction", "pose_confidence_median", "video_ok")},
                          training_tier=tier, duplicate_of=dup["nearest"]["episode"] if dup["duplicate"] else None)
    return {"tier": tier, "quality": q, "dedup": dup}


__all__ = ["QualityThresholds", "TimmEncoder", "assign_tier", "curate_episode", "dedup_report", "episode_embedding", "hierarchical_caption", "quality_score"]
