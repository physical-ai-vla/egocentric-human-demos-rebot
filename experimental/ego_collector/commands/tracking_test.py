"""Integration tests A-E on a processed episode (run process-tags first).

    A static jitter       : camera + tag stationary        -> --static 0:30 (seconds)
    B trajectory sanity   : left/right/forward/backward   -> look at `visualize`
    C return-to-point     : revisit one point ~20 times     -> dwell repeatability (auto)
    D both wrists visible : availability per side
    E world tags visible while the head moves : world availability / reprojection
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ego_collector.io.parquet import read_parquet
from ego_collector.qa.tracking import dwell_repeatability, pose_array, static_jitter, tracking_summary
from ego_collector.recording.episode import EpisodePaths


def _window(df, spec: str | None):
    if not spec:
        return np.ones(len(df), bool)
    a, b = spec.split(":")
    t = (df["timestamp_ns"].to_numpy() - df["timestamp_ns"].iloc[0]) / 1e9
    return (t >= float(a or 0)) & (t <= float(b or 1e9))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ego_collector tracking-test", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode", type=Path)
    p.add_argument("--static", default=None, help="seconds window 'a:b' where camera and tags were stationary (test A)")
    p.add_argument("--side", choices=("left", "right", "both"), default="both")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)
    paths = EpisodePaths(args.episode)
    cam = read_parquet(paths.camera_pose)
    wrist = read_parquet(paths.wrist_pose)
    finger = read_parquet(paths.finger_pose) if paths.finger_pose.exists() else None
    ts = wrist["timestamp_ns"].to_numpy()
    report: dict = {"episode": paths.root.name, "summary": tracking_summary(cam, wrist, finger)}
    sides = ("left", "right") if args.side == "both" else (args.side,)
    for side in sides:
        poses = pose_array(wrist, f"{side}_wrist")
        valid = wrist[f"{side}_wrist_valid"].to_numpy(bool)
        entry: dict = {}
        if args.static:
            w = _window(wrist, args.static)
            entry["static_jitter"] = asdict(static_jitter(poses, valid & w))
            hv = cam["world_pose_valid"].to_numpy(bool) & w
            entry["head_static_jitter"] = asdict(static_jitter(pose_array(cam, "head"), hv))
        entry["repeatability"] = asdict(dwell_repeatability(poses, valid, ts))
        if finger is not None and args.static:
            w = _window(finger, args.static)
            a = finger[f"{side}_aperture_m"].to_numpy(dtype=np.float64)[w]
            a = a[np.isfinite(a)]
            entry["aperture_static_std_mm"] = float(a.std() * 1000) if len(a) >= 5 else float("nan")
        report[side] = entry
    s = report["summary"]
    print(f"frames {s['frames']}  world valid {s['world_pose_valid_fraction']*100:.1f}%  reproj {s['world_reprojection_error_median_px']:.2f}px  head jumps {s['head']['catastrophic_jumps']}")
    no_world = not paths.read_metadata().get("tracking", {}).get("world_tags", True)
    if no_world:
        s["all_required_visible_ratio"] = float(np.mean(wrist["left_tag_visible"].to_numpy(bool) & wrist["right_tag_visible"].to_numpy(bool))) if len(wrist) else 0.0
        print("(no world tags: ALL-REQUIRED = left AND right wrist visible; poses are head-frame)")
    arv = s["all_required_visible_ratio"]
    go = "GO" if arv > 0.95 else ("ADJUST tags/FOV" if arv >= 0.90 else "NO-GO (C922 FOV likely insufficient)")
    print(
        f"visible: world {s['world_visible_ratio']*100:.1f}%  left {s['left_wrist_visible_ratio']*100:.1f}%  "
        f"right {s['right_wrist_visible_ratio']*100:.1f}%  ALL-REQUIRED {arv*100:.1f}%  -> {go}"
    )
    for side in sides:
        e = report[side]
        line = f"{side:5s} valid {s[f'{side}_wrist_valid_fraction']*100:.1f}%  jumps {s[f'{side}_wrist']['catastrophic_jumps']}"
        if "static_jitter" in e:
            j = e["static_jitter"]
            line += f"  jitter RMS {j['translation_rms_mm']:.2f} mm / {j['rotation_rms_deg']:.2f} deg (n={j['samples']})"
        r = e["repeatability"]
        line += f"  dwells {r['dwells']} max spread {r['max_spread_mm']:.1f} mm"
        if finger is not None:
            fs = s["fingers"]
            line += (
                f"\n      fingers: thumb {fs[f'{side}_thumb_visible_fraction']*100:.1f}%  index {fs[f'{side}_index_visible_fraction']*100:.1f}%"
                f"  aperture {fs[f'{side}_aperture_valid_fraction']*100:.1f}%  noise {fs[f'{side}_aperture_noise_mm']:.2f} mm"
                f"  range [{fs[f'{side}_aperture_min_mm']:.0f}, {fs[f'{side}_aperture_max_mm']:.0f}] mm"
            )
            if "aperture_static_std_mm" in e:
                line += f"  static std {e['aperture_static_std_mm']:.2f} mm"
        print(line)
    verdict = {
        "all_required_visible_gt_95": s["all_required_visible_ratio"] > 0.95,
        "world_availability_gt_95": no_world or s["world_pose_valid_fraction"] > 0.95,
        "wrist_availability_gt_95": all((s[f"{sd}_tag_visible_fraction"] if no_world else s[f"{sd}_wrist_valid_fraction"]) > 0.95 for sd in sides),
        "no_catastrophic_jumps": (no_world or s["head"]["catastrophic_jumps"] == 0) and all(s[f"{sd}_wrist"]["catastrophic_jumps"] == 0 for sd in sides),
        "repeatability_lt_10mm": all(not np.isfinite(report[sd]["repeatability"]["max_spread_mm"]) or report[sd]["repeatability"]["max_spread_mm"] < 10 for sd in sides),
    }
    if finger is not None:
        fs = s["fingers"]
        verdict["aperture_availability_gt_80"] = all(fs[f"{sd}_aperture_valid_fraction"] > 0.80 for sd in sides)
        verdict["aperture_noise_lt_5mm"] = all(not np.isfinite(fs[f"{sd}_aperture_noise_mm"]) or fs[f"{sd}_aperture_noise_mm"] < 5.0 for sd in sides)
    report["verdict"] = verdict
    print("verdict:", ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in verdict.items()))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=float))
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
