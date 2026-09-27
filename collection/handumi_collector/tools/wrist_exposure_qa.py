"""Pick a wrist exposure by TRACKING QUALITY, not by how the image looks.

    python -m handumi_collector.tools.wrist_exposure_qa --episode EP --stream left_wrist

The question an exposure sweep has to answer is not "is it bright enough" but "do features survive a fast wrist
rotation". A 15.6 ms exposure (the Arducam factory default) is fine on a static scene and smears at speed, and a mean
brightness number cannot see that. So every frame is bucketed by its OWN measured motion, and feature retention is
reported PER BUCKET: the right exposure is the shortest one that still holds features in the fast bucket.

Metrics per frame: mean brightness, blur proxy (variance of Laplacian), detected corners, and the fraction of corners
that survive forward-backward KLT into the next frame — the same tracker head_odometry and the VO backends rely on, so
this measures what the pose pipeline will actually get.

Timing (fps, median gap, p95 jitter, max gap) is NOT duplicated here: `tools.inspect_episode --validate` already
reports it. Run both on each take."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

# Motion buckets by median optical-flow magnitude between consecutive frames (px). Boundaries are descriptive, not
# thresholds to pass: they exist so "features survive fast motion" is answered separately from the easy frames.
BUCKETS = ((0.0, 1.0, "static"), (1.0, 5.0, "slow"), (5.0, 15.0, "normal"), (15.0, 1e9, "fast"))


def bucket_of(px: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= px < hi:
            return name
    return "fast"


def analyse(episode: Path, stream: str, *, max_corners: int = 800, downscale: int = 1, limit: int | None = None) -> dict:
    import cv2
    from ..pose.episode_io import RawEpisode
    ep = RawEpisode.load(episode)
    if stream not in ep.frames:
        raise SystemExit(f"{episode}: no stream {stream!r} (have {sorted(ep.frames)})")
    rows = []
    prev_gray = None
    for _vf, _fi, t_ns, img in ep.iter_frames(stream, downscale=downscale, gray=True):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r = dict(t_ns=int(t_ns), brightness=float(gray.mean()),
                 blur_var=float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        p0 = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 12, blockSize=5)
        r["corners"] = 0 if p0 is None else int(len(p0))
        if prev_gray is not None and r["corners"] > 0:
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts.reshape(-1, 1, 2), None,
                                                  winSize=(21, 21), maxLevel=3)
            p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, winSize=(21, 21), maxLevel=3)
            ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1)
            ok &= np.linalg.norm(p0b.reshape(-1, 2) - prev_pts, axis=1) < 1.0
            flow = np.linalg.norm(p1.reshape(-1, 2)[ok] - prev_pts[ok], axis=1) if ok.any() else np.zeros(1)
            r["survived"] = float(ok.mean())
            r["flow_px"] = float(np.median(flow))
            r["bucket"] = bucket_of(r["flow_px"])
        else:
            r["survived"] = np.nan; r["flow_px"] = np.nan; r["bucket"] = "n/a"
        rows.append(r)
        prev_gray = gray
        prev_pts = (np.zeros((0, 2), np.float32) if p0 is None else p0.reshape(-1, 2).astype(np.float32))
        if limit and len(rows) >= limit:
            break
    return dict(episode=str(episode), stream=stream, n_frames=len(rows), rows=rows)


def report(res: dict) -> str:
    rows = res["rows"]
    if not rows:
        return "no frames"
    b = np.array([r["brightness"] for r in rows])
    bl = np.array([r["blur_var"] for r in rows])
    co = np.array([r["corners"] for r in rows])
    out = [f"episode {res['episode']}",
           f"stream  {res['stream']}   frames {res['n_frames']}",
           "",
           f"brightness   mean {b.mean():6.1f}   p05 {np.percentile(b,5):6.1f}   p95 {np.percentile(b,95):6.1f}",
           f"blur (varLap) med {np.median(bl):8.1f}   p05 {np.percentile(bl,5):8.1f}  (LOWER = blurrier)",
           f"corners       med {np.median(co):6.0f}   p05 {np.percentile(co,5):6.0f}   min {co.min():6.0f}",
           "",
           "feature retention by measured motion  (this is what decides the exposure)",
           f"  {'bucket':8s} {'frames':>7s} {'flow px':>9s} {'survived':>9s} {'corners':>8s} {'blur':>9s}"]
    for _lo, _hi, name in BUCKETS:
        sel = [r for r in rows if r.get("bucket") == name]
        if not sel:
            out.append(f"  {name:8s} {0:7d}         -         -        -         -")
            continue
        f = np.array([r["flow_px"] for r in sel]); s = np.array([r["survived"] for r in sel])
        c = np.array([r["corners"] for r in sel]); v = np.array([r["blur_var"] for r in sel])
        out.append(f"  {name:8s} {len(sel):7d} {np.median(f):9.2f} {np.median(s):9.3f} {np.median(c):8.0f} {np.median(v):9.1f}")
    fast = [r for r in rows if r.get("bucket") == "fast"]
    out += ["", ("no fast-motion frames in this take — the sweep cannot tell you anything about blur; redo the take "
                 "with a genuinely fast wrist rotation" if not fast else
                 f"fast bucket: {len(fast)} frames, median retention {np.median([r['survived'] for r in fast]):.3f}")]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--stream", default="left_wrist")
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", type=Path, help="also write the per-frame rows here")
    a = ap.parse_args(argv)
    res = analyse(Path(a.episode), a.stream, downscale=a.downscale, limit=a.limit)
    print(report(res))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
