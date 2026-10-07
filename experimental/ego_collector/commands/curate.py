"""Index-style curation over episodes: quality -> embedding -> dedup -> caption -> tier A/B/D.

    ego-collector curate datasets/raw/episode_* [--sides right] [--colors red,blue,purple]

Run AFTER process-hands3d (pose-aware tiering) - or before, for video-only triage.
Tier A = RGB + pseudo-action (WAM action supervision), B = RGB only (video prediction),
D = reject (bad video or duplicate). Duplicates require visual similarity AND the same
instruction; the same scene with a different prompt is kept on purpose.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ego_collector.engine.curate import QualityThresholds, TimmEncoder, curate_episode
from ego_collector.recording.episode import EpisodePaths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector curate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", type=Path, nargs="+")
    p.add_argument("--root", type=Path, default=Path("datasets/raw"), help="Corpus to dedup against")
    p.add_argument("--sides", choices=("left", "right", "both"), default="right")
    p.add_argument("--colors", default="red,blue,purple")
    p.add_argument("--dup-similarity", type=float, default=0.965)
    p.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    th = QualityThresholds(sides=args.sides, dup_similarity=args.dup_similarity)
    encoder = TimmEncoder(args.model, device=args.device)
    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    for ep in args.episodes:
        r = curate_episode(EpisodePaths(ep), encoder=encoder, root=args.root, th=th, colors=colors)
        q, d = r["quality"], r["dedup"]
        near = d["nearest"]
        sim = f"{near['similarity']:.3f} vs {near['episode']}" if near["episode"] else "-"
        print(f"{ep.name}: tier {r['tier']}  video_ok={q['video_ok']}  blur {q['blur_median']:.0f}  hands {q['hands_visible_fraction']*100 if q['hands_visible_fraction']==q['hands_visible_fraction'] else float('nan'):.0f}%  nn {sim}{'  DUPLICATE' if d['duplicate'] else ''}")


if __name__ == "__main__":
    main()
