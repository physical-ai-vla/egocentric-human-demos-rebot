#!/usr/bin/env python3
"""[2026-09-18] What are the >50 mm and >100 mm single-frame jumps in the HandUMI body track?

At 30 Hz a 100 mm step is 3 m/s. A hand carrying a gripper does not do that, so those frames are almost certainly the
detector's own failures -- but "almost certainly" is not a classification, and the fix is different for each cause.
So every jump is re-examined against the evidence in the +/-5 frames around it and put in one of four buckets:

    true_fast_motion       speed rises and falls smoothly, the blob keeps its size and depth coverage
    segmentation_change    the blob's AREA steps at the same frame -- the body merged with the arm, or split
    depth_failure          valid-depth coverage inside the blob collapses at that frame
    identity_or_gap        the track had just re-acquired, or the two units came within the association gate

The bucket decides the remedy: fast motion needs nothing, a segmentation step needs the mask, a depth collapse needs
a temporal filter, an identity event needs the association rule. Lumping them together as "noise" would hide all four.

    .venv/bin/python scripts/body_jump_audit.py --session <session>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from handumi_collector.pose.rgbd_io import RgbdEpisode                      # noqa: E402
import handumi_body_track as B                                              # noqa: E402

W = 5            # frames either side of the event that are allowed to explain it
JUMP_MM = 50.0


def track(ep: RgbdEpisode, nrm, pd, K) -> dict:
    n = ep.n_frames
    out = {s: dict(xyz=np.full((n, 3), np.nan), area=np.zeros(n), cov=np.zeros(n), valid=np.zeros(n, bool),
                   reacq=np.zeros(n, bool)) for s in ("left", "right")}
    tracks: dict = {}
    for f in ep.iter_frames(0, n, 1):
        dets = B.detect_bodies(f, K, nrm, pd)
        a = B.associate(tracks, dets, f.rgb.shape[1])
        reacq = {s: bool(tracks.get(s, {}).get("miss", 99)) for s in ("left", "right")}
        B.update_tracks(tracks, dets, a)
        for side in ("left", "right"):
            if side in a:
                d = dets[a[side]]
                out[side]["reacq"][f.index] = reacq[side]
                out[side]["xyz"][f.index] = d["xyz"]; out[side]["area"][f.index] = d["area"]
                out[side]["cov"][f.index] = d["depth_cov"]        # depth validity of the blob, unaffected by the repair
                out[side]["valid"][f.index] = True
        if "left" in a and "right" in a:
            sep = np.linalg.norm(dets[a["left"]]["xyz"] - dets[a["right"]]["xyz"])
            for side in ("left", "right"):
                out[side].setdefault("sep", np.full(n, np.nan))
                out[side]["sep"][f.index] = sep
    return out


def classify(t: dict, i: int) -> tuple[str, dict]:
    """`i` is the frame the body ARRIVED at; the step is i-1 -> i."""
    n = len(t["valid"])
    lo, hi = max(0, i - W), min(n, i + W + 1)
    v = t["valid"][lo:hi]
    area, cov = t["area"][lo:hi], t["cov"][lo:hi]
    P = t["xyz"][lo:hi]
    ev = {}
    # area step: median area before vs after, relative
    k = i - lo
    a_before = np.median(area[:k][v[:k]]) if v[:k].any() else np.nan
    a_after = np.median(area[k:][v[k:]]) if v[k:].any() else np.nan
    ev["area_ratio"] = float(a_after / a_before) if np.isfinite(a_before) and a_before > 0 else float("nan")
    ev["cov_at"] = float(cov[k]); ev["cov_before"] = float(np.median(cov[:k][v[:k]])) if v[:k].any() else float("nan")
    ev["reacq"] = bool(t["reacq"][i])
    sep = t.get("sep")
    ev["sep_m"] = float(sep[i]) if sep is not None and np.isfinite(sep[i]) else float("nan")
    # speed profile around the event: a real reach accelerates and decelerates over several frames
    sp = np.linalg.norm(np.diff(P, axis=0), axis=1) * 1e3
    fin = np.isfinite(sp)
    ev["neighbour_speed_p50_mm"] = float(np.median(sp[fin])) if fin.any() else float("nan")
    ev["step_mm"] = float(np.linalg.norm(t["xyz"][i] - t["xyz"][i - 1]) * 1e3)

    if ev["reacq"] or (np.isfinite(ev["sep_m"]) and ev["sep_m"] < B.GATE_M):
        return "identity_or_gap", ev
    if np.isfinite(ev["cov_before"]) and ev["cov_at"] < 0.6 * ev["cov_before"]:
        return "depth_failure", ev
    if np.isfinite(ev["area_ratio"]) and (ev["area_ratio"] > 1.35 or ev["area_ratio"] < 0.74):
        return "segmentation_change", ev
    if np.isfinite(ev["neighbour_speed_p50_mm"]) and ev["neighbour_speed_p50_mm"] > 0.35 * ev["step_mm"]:
        return "true_fast_motion", ev
    return "segmentation_change", ev            # an isolated spike with no support is a mask artefact, not a reach


def audit(ep_path: Path, stream: str | None) -> dict:
    ep = RgbdEpisode.load(ep_path, stream=stream)
    nrm, pd, _pl = B.fit_table_plane(ep)
    T = track(ep, nrm, pd, ep.intrinsics.K)
    out = {"episode": ep_path.name, "sides": {}}
    for side, t in T.items():
        v = t["valid"]
        idx = np.nonzero(v)[0]
        events = []
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                continue                         # a gap is not a jump; it is already counted as a gap
            step = float(np.linalg.norm(t["xyz"][b] - t["xyz"][a]) * 1e3)
            if step > JUMP_MM:
                cls, ev = classify(t, int(b))
                events.append(dict(frame=int(b), cls=cls, **ev))
        buckets: dict[str, int] = {}
        for e in events:
            buckets[e["cls"]] = buckets.get(e["cls"], 0) + 1
        out["sides"][side] = dict(n_events=len(events), buckets=buckets,
                                  over_100=sum(1 for e in events if e["step_mm"] > 100),
                                  events=events[:40])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", type=Path); g.add_argument("--episode", type=Path)
    ap.add_argument("--stream", default=None); ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    eps = [a.episode] if a.episode else sorted(p for p in a.session.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    rows = []
    for p in eps:
        rows.append(audit(p, a.stream)); print(f"  audited {p.name}", flush=True)

    print("\n" + "=" * 104)
    print("LARGE TRANSLATION JUMP AUDIT  --  every >50 mm single-frame step, classified from its own neighbourhood")
    print("=" * 104)
    print(f"\n{'episode':>14s} {'side':>6s} {'>50mm':>6s} {'>100mm':>7s}  buckets")
    tot: dict[str, int] = {}
    for r in rows:
        for side, s in r["sides"].items():
            for k, n in s["buckets"].items():
                tot[k] = tot.get(k, 0) + n
            print(f"{r['episode'][-11:]:>14s} {side:>6s} {s['n_events']:6d} {s['over_100']:7d}  "
                  + ", ".join(f"{k}={n}" for k, n in sorted(s["buckets"].items())))
    n_all = sum(tot.values())
    print(f"\n  {n_all} events total: " + ", ".join(f"{k} {n} ({n/max(n_all,1)*100:.0f}%)" for k, n in sorted(tot.items(), key=lambda kv: -kv[1])))

    print("\n  largest events, with the evidence that classified them")
    ev = sorted(((r["episode"], side, e) for r in rows for side, s in r["sides"].items() for e in s["events"]),
                key=lambda x: -x[2]["step_mm"])[:12]
    print(f"    {'episode':>18s} {'side':>6s} {'frame':>6s} {'step':>8s} {'area x':>7s} {'cov':>6s} "
          f"{'nbr spd':>8s} {'sep m':>7s}  class")
    for epn, side, e in ev:
        print(f"    {epn:>18s} {side:>6s} {e['frame']:6d} {e['step_mm']:7.1f}m {e['area_ratio']:7.2f} "
              f"{e['cov_at']:6.2f} {e['neighbour_speed_p50_mm']:7.1f}m {e['sep_m']:7.3f}  {e['cls']}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1, default=float)); print(f"\nrecord -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
