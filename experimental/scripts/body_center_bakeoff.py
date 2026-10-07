#!/usr/bin/env python3
"""[2026-09-18] Which body-centre estimator? full-blob centroid vs uniform trim vs mean-shift core, same 8 episodes.

The forearm is connected to the HandUMI, so no darkness threshold separates them and tightening `DARK_MAX` only eats
the body. What separates them is density and extent: the body is a compact 0.14 x 0.11 x 0.08 m lump, the arm is a
long tail leaving the frame. So the question is which centre estimator ignores the tail.

Two rules this bake-off obeys, because breaking either makes the comparison meaningless:

  * **Identity association is identical in every arm.** It runs once, on the full-blob centroid, and every estimator
    is then applied to the same assigned components. Otherwise a better estimator could look worse purely because it
    associated differently.
  * **The estimate must not be dragged by the previous frame.** The previous centroid seeds mean-shift only when
    nothing else is available; the iteration starts from the component's own median and walks to its own densest
    mode, so the output is a property of this frame's point cloud. An estimator that leans on frame t-1 smooths its
    own error and then reports a small jump rate for the wrong reason.

Frames are decoded once and the component point clouds cached, so all arms score the same data.

    .venv/bin/python scripts/body_center_bakeoff.py --session <session>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from handumi_collector.pose.rgbd_io import RgbdEpisode                      # noqa: E402
import handumi_body_track as B                                              # noqa: E402

MAX_PTS = 3000          # per component; mean-shift on a subsample is the same mode, at a fraction of the cost
BANDWIDTHS = (0.05, 0.07, 0.09, 0.12)


def cache_episode(ep_path: Path, stream: str | None, rng) -> dict:
    """One decode pass -> per side, per frame: the component's subsampled 3D points. Identity is decided here, once."""
    ep = RgbdEpisode.load(ep_path, stream=stream)
    nrm, pd, plane = B.fit_table_plane(ep)
    K = ep.intrinsics.K
    n = ep.n_frames
    pts = {s: [None] * n for s in ("left", "right")}
    tracks: dict = {}
    switches = {s: 0 for s in ("left", "right")}
    for f in ep.iter_frames(0, n, 1):
        dets = B.detect_bodies(f, K, nrm, pd)
        for d in dets:                                   # association must see the UNREFINED centroid, in every arm
            d["xyz"] = np.median(d["points"], 0)
        a = B.associate(tracks, dets, f.rgb.shape[1])
        for side in ("left", "right"):
            if side in a:
                d = dets[a[side]]
                prev = tracks.get(side)
                if prev is not None and prev["miss"] == 0 and np.linalg.norm(d["xyz"] - prev["xyz"]) > B.GATE_M:
                    switches[side] += 1
                tracks[side] = dict(xyz=d["xyz"], area=float(d["area"]), miss=0)
                P = d["points"]
                if len(P) > MAX_PTS:
                    P = P[rng.choice(len(P), MAX_PTS, replace=False)]
                pts[side][f.index] = P
            elif side in tracks:
                tracks[side]["miss"] += 1
    return dict(episode=ep_path.name, n=n, pts=pts, switches=switches, plane=plane)


def est_full(P, prev):
    return np.median(P, 0)


def est_trim(P, prev, r=0.09):
    """One trim around the previous centroid -- the version that inherits whatever error frame t-1 had."""
    if prev is None:
        return np.median(P, 0)
    k = np.linalg.norm(P - prev, axis=1) < r
    return np.median(P[k], 0) if k.sum() >= 200 else np.median(P, 0)


def est_meanshift(P, prev, bw=0.09, iters=12):
    """Walk to this component's own densest mode. Seeded from the component median, NOT from the previous frame."""
    c = np.median(P, 0)
    for _ in range(iters):
        k = np.linalg.norm(P - c, axis=1) < bw
        if k.sum() < 150:
            break
        c2 = np.median(P[k], 0)
        if np.linalg.norm(c2 - c) < 2e-4:
            c = c2
            break
        c = c2
    return c


def run(cache: list[dict], est, **kw) -> dict:
    jumps50 = jumps100 = 0
    steps, stills, covs = [], [], []
    for c in cache:
        for side in ("left", "right"):
            seq = c["pts"][side]
            X = np.full((c["n"], 3), np.nan)
            prev = None
            for i, P in enumerate(seq):
                if P is None:
                    prev = None
                    continue
                X[i] = est(P, prev, **kw)
                prev = X[i]
            v = np.all(np.isfinite(X), 1)
            covs.append(float(v.mean()))
            idx = np.nonzero(v)[0]
            st = []
            for a, b in zip(idx[:-1], idx[1:]):
                if b == a + 1:
                    st.append(float(np.linalg.norm(X[b] - X[a])) * 1e3)
            st = np.asarray(st)
            if st.size:
                steps.append(st)
                jumps50 += int((st > 50).sum()); jumps100 += int((st > 100).sum())
            # quiet window, same rule as the tracker: quietest gap-free 1.0 s, ranked on p95 then median speed
            w, best = 30, None
            for a0 in range(0, c["n"] - w + 1):
                if not v[a0:a0 + w].all():
                    continue
                Q = X[a0:a0 + w]
                sp = np.linalg.norm(np.diff(Q, axis=0), axis=1) * 1e3
                key = (float(np.percentile(sp, 95)), float(np.median(sp)))
                if best is None or key < best[0]:
                    best = (key, Q)
            if best is not None:
                stills.append(float(np.linalg.norm(best[1].std(0))) * 1e3)
    allst = np.concatenate(steps) if steps else np.zeros(1)
    return dict(jumps50=jumps50, jumps100=jumps100,
                step_p50=float(np.percentile(allst, 50)), step_p95=float(np.percentile(allst, 95)),
                step_p99=float(np.percentile(allst, 99)),
                still_med=float(np.median(stills)) if stills else float("nan"),
                still_max=float(np.max(stills)) if stills else float("nan"),
                coverage_min=float(np.min(covs)), coverage_mean=float(np.mean(covs)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--stream", default=None); ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    eps = sorted(p for p in a.session.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    cache = []
    for p in eps:
        cache.append(cache_episode(p, a.stream, rng)); print(f"  cached {p.name}", flush=True)
    sw = sum(v for c in cache for v in c["switches"].values())

    # Mean-shift was measured and rejected on 2026-09-18: the HandUMI is a hollow multi-part mechanism, not a single
    # dense lump, so small bandwidths hop between spurious internal modes (bw 0.05 -> 285 jumps vs 78 for the plain
    # centroid) and large ones converge back to the full blob. Kept as one reference arm, not swept again.
    arms = [("full-blob centroid", est_full, {})]
    arms += [(f"uniform trim r={r:.2f}", est_trim, dict(r=r)) for r in (0.06, 0.07, 0.09, 0.12, 0.15)]
    arms += [("mean-shift bw=0.09 (rejected)", est_meanshift, dict(bw=0.09))]

    print("\n" + "=" * 104)
    print("BODY-CENTRE BAKE-OFF   identical frames, identical identity association, 8 episodes x 2 sides")
    print("=" * 104)
    print(f"\n{'estimator':>34s} {'>50mm':>7s} {'>100mm':>7s} {'step p50':>9s} {'p95':>7s} {'p99':>7s} "
          f"{'still med':>10s} {'still max':>10s} {'cover min':>10s}")
    rows = []
    for name, fn, kw in arms:
        r = run(cache, fn, **kw); r["estimator"] = name; rows.append(r)
        print(f"{name:>34s} {r['jumps50']:7d} {r['jumps100']:7d} {r['step_p50']:8.2f}m {r['step_p95']:6.1f} "
              f"{r['step_p99']:6.1f} {r['still_med']:9.2f}m {r['still_max']:9.2f}m {r['coverage_min']*100:9.1f}%")
    print(f"\n  identity switches (shared by every arm): {sw}")
    print("\n  choose on >100 mm first (those are the events that reach the IK as a Δq spike), then >50 mm, and only\n"
          "  among estimators whose stationary std and coverage did not move.")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1, default=float)); print(f"\nrecord -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
