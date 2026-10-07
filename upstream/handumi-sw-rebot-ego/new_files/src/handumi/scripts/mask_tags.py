#!/usr/bin/env python3
"""Create ``observation.images.head_clean`` (AprilTags hidden) next to the raw head video.

::

    handumi dataset mask-tags outputs/cubes_A --camera head --method fill

Uses the per-row tag corners stored by the AprilTag backend; if corners were
detected on the rectified fisheye view (``corner_space == 1``) they are mapped
back to raw pixels with the head intrinsics from ``--spatial``. The raw video
and rows are left untouched; ``meta/info.json`` gains the new video feature
and ``meta/episodes`` gets copied video references.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from handumi.processing.tag_mask import corners_from_row, mask_tags


def _iter_rows(root: Path):
    import pandas as pd

    for path in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(path)
        yield path, df.sort_values("index")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path)
    p.add_argument("--camera", default="head", help="Video key suffix (observation.images.<camera>).")
    p.add_argument("--suffix", default="_clean")
    p.add_argument("--method", choices=("fill", "blur", "noise"), default="fill")
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--spatial", type=Path, default=Path("outputs/calibration/spatial.yaml"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    import pandas as pd

    src_key = f"observation.images.{args.camera}"
    dst_key = f"{src_key}{args.suffix}"
    info_path = args.root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if src_key not in info["features"]:
        raise SystemExit(f"{src_key} not in dataset features")
    if dst_key in info["features"] and not args.overwrite:
        raise SystemExit(f"{dst_key} already exists (use --overwrite)")

    geometry = None
    try:
        from handumi.scripts.setup.calibrate_head_camera import load_head_intrinsics
        from handumi.tracking.apriltag import CameraGeometry

        geometry = CameraGeometry.for_intrinsics(load_head_intrinsics(args.spatial))
    except SystemExit:
        print("note: no head intrinsics; assuming corners are already in raw pixels", file=sys.stderr)

    # Map episode -> video file for the source key from meta/episodes.
    ep_files = sorted((args.root / "meta" / "episodes").rglob("*.parquet"))
    ep_meta = pd.concat([pd.read_parquet(f) for f in ep_files]) if ep_files else None
    col_chunk, col_file = f"videos/{src_key}/chunk_index", f"videos/{src_key}/file_index"
    video_of_episode: dict[int, tuple[int, int]] = {}
    if ep_meta is not None and col_chunk in ep_meta.columns:
        for _, r in ep_meta.iterrows():
            video_of_episode[int(r["episode_index"])] = (int(r[col_chunk]), int(r[col_file]))

    # Group rows by video file (rows are ordered by global index; each video file holds whole episodes).
    rows_by_video: dict[tuple[int, int], list[dict]] = {}
    for _, df in _iter_rows(args.root):
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            key = video_of_episode.get(ep, (0, 0))
            rows_by_video.setdefault(key, []).append(row.to_dict())

    total_frames = 0
    for (chunk, file_idx), rows in sorted(rows_by_video.items()):
        src = args.root / "videos" / src_key / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4"
        dst = args.root / "videos" / dst_key / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4"
        if not src.exists():
            print(f"missing {src}", file=sys.stderr)
            continue
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or info.get("fps", 30)
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_video != len(rows):
            print(f"warning: {src.name}: {n_video} video frames vs {len(rows)} rows; masking by order up to the shorter", file=sys.stderr)
        dst.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        for i, row in enumerate(rows):
            ok, frame = cap.read()
            if not ok:
                break
            quads = corners_from_row(row)
            if geometry is not None and int(np.asarray(row.get("observation.apriltag.corner_space", [0])).reshape(-1)[0]) == 1:
                quads = [geometry.to_raw_pixels(q) for q in quads]
            out = mask_tags(frame, quads, method=args.method, margin=args.margin)
            if writer is None:
                writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
            writer.write(out)
            total_frames += 1
        cap.release()
        if writer is not None:
            writer.release()
        print(f"{src.name}: {len(rows)} rows -> {dst}")

    info["features"][dst_key] = dict(info["features"][src_key])
    info["features"][dst_key]["info"] = {**info["features"][src_key].get("info", {}), "derived_from": src_key, "tag_mask": args.method}
    info_path.write_text(json.dumps(info, indent=2))
    stats_path = args.root / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if src_key in stats and dst_key not in stats:
            stats[dst_key] = stats[src_key]
            stats_path.write_text(json.dumps(stats, indent=2))
    for f in ep_files:
        df = pd.read_parquet(f)
        changed = False
        for col in df.columns:
            if col.startswith(f"videos/{src_key}/"):
                new_col = col.replace(f"videos/{src_key}/", f"videos/{dst_key}/")
                if new_col not in df.columns:
                    df[new_col] = df[col]
                    changed = True
        if changed:
            shutil.copy2(f, f.with_suffix(".parquet.bak")) if not f.with_suffix(".parquet.bak").exists() else None
            df.to_parquet(f)
    print(f"done: {total_frames} frames -> feature {dst_key}")


if __name__ == "__main__":
    main()
