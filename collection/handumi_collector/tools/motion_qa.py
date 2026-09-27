"""Motion QA for a recorded take: record the DISTRIBUTION, fail only on catastrophe.

    python -m handumi_collector.tools.motion_qa --episode EP [--hardware handumi_rgbd] [--json out.json]

The point of a final motion take is to learn what the rig actually produces, so there are deliberately NO pass/fail
quality thresholds here: every number is reported and nothing is judged against a guess. What IS checked is the short
list of ways a take can be worthless, where continuing would waste the operator's time:

    wrong camera              the stream was not the physical camera the profile names
    format mismatch           the camera delivered a size other than the frozen one
    imaging not frozen        a UVC control in the preset did not actually apply
    frame-loss burst          consecutive dropped frames, not isolated hiccups
    timestamp discontinuity   non-monotonic, duplicated, or a gap of several frame periods
    feature collapse          almost no trackable corners
    massive blur              even the still frames are smeared
    depth collapse            the head RGB-D returned almost nothing

Reuses integrity.validate_episode for timing, wrist_exposure_qa for the per-motion-bucket vision distribution, and
rgbd_io for the head depth. Run it on every take before moving the rig."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

# CATASTROPHIC FLOORS ONLY. These are not quality gates and must not be read as ones: they are the levels below which a
# take carries no information at all. Quality thresholds come later, from the measured distributions.
CATASTROPHIC = dict(min_corners_median=60, min_static_blur_var=20.0, max_consecutive_drops=5,
                    max_gap_frame_periods=5.0, min_depth_coverage=0.30)


def _fail(out: list, what: str, detail: str) -> None:
    out.append(f"{what}: {detail}")


def check_provenance(session_meta: dict, expected: dict | None, fails: list) -> dict:
    """Identity, delivered format and the UVC preset, straight out of what the recorder wrote."""
    rep = {}
    for name, det in (session_meta.get("devices") or {}).items():
        if not isinstance(det, dict) or "actual" not in det:
            continue
        d = dict(index=det.get("index"), actual=det.get("actual"))
        ident = det.get("identity") or {}
        d["identity"] = ident
        if ident:
            if ident.get("matches") not in (None, 1):
                _fail(fails, "wrong camera", f"{name}: {ident['matches']} devices matched {ident.get('match_name')!r} — "
                                             "an ordinal cannot name a physical camera")
            exp_name = ident.get("match_name")
            got = ident.get("listing_name")
            if exp_name and got and exp_name.lower() not in got.lower():
                _fail(fails, "wrong camera", f"{name}: opened {got!r}, profile names {exp_name!r}")
        exp = (expected or {}).get(name)
        a = det.get("actual") or {}
        if exp and a:
            if (int(a.get("width", 0)), int(a.get("height", 0))) != (exp["width"], exp["height"]):
                _fail(fails, "format mismatch", f"{name}: delivered {int(a.get('width',0))}x{int(a.get('height',0))}, "
                                                f"profile freezes {exp['width']}x{exp['height']}")
        bad = [c for c in (det.get("uvc_controls") or []) if not c.get("applied")]
        if bad:
            _fail(fails, "imaging not frozen", f"{name}: " + "; ".join(f"{c['control']} ({c.get('reason')})" for c in bad))
        d["uvc_controls"] = {c["control"]: c.get("readback") for c in (det.get("uvc_controls") or [])}
        rep[name] = d
    return rep


def check_timing(ep: Path, fails: list) -> dict:
    from ..collector.integrity import validate_episode
    v = validate_episode(ep)
    rep = {}
    for name, st in (v.get("streams") or {}).items():
        rep[name] = st
        gap, nominal = st.get("max_gap_ms"), 1000.0 / max(st.get("fps") or 30.0, 1e-6)
        if gap and gap > CATASTROPHIC["max_gap_frame_periods"] * nominal:
            _fail(fails, "timestamp discontinuity", f"{name}: max gap {gap:.1f} ms = "
                                                    f"{gap/nominal:.1f} frame periods")
        if st.get("frame_meta") is not None and st.get("video_frames") is not None and st["frame_meta"] != st["video_frames"]:
            _fail(fails, "episode integrity", f"{name}: {st['frame_meta']} frame_meta rows vs {st['video_frames']} "
                                              "encoded video frames")
    ev_p = ep / "events.json"
    bursts = []
    if ev_p.exists():
        try:
            for e in json.loads(ev_p.read_text()):
                if e.get("kind") == "camera_drop":
                    n = int((e.get("detail") or {}).get("skipped", 0))
                    bursts.append((e.get("device"), n))
        except Exception:
            pass
    worst = max((n for _d, n in bursts), default=0)
    rep["_camera_drop_events"] = [dict(device=d, skipped=n) for d, n in bursts]
    if worst > CATASTROPHIC["max_consecutive_drops"]:
        _fail(fails, "frame-loss burst", f"{worst} consecutive frames dropped")
    if not v.get("ok"):
        for p in v.get("problems", []):
            _fail(fails, "episode integrity", p)
    return rep


def check_wrist(ep: Path, stream: str, fails: list, *, limit: int | None) -> dict:
    from .wrist_exposure_qa import BUCKETS, analyse, report
    res = analyse(ep, stream, limit=limit)
    rows = res["rows"]
    if not rows:
        _fail(fails, "feature collapse", f"{stream}: no frames analysed")
        return {}
    co = np.array([r["corners"] for r in rows])
    if np.median(co) < CATASTROPHIC["min_corners_median"]:
        _fail(fails, "feature collapse", f"{stream}: median {np.median(co):.0f} corners "
                                         f"(< {CATASTROPHIC['min_corners_median']})")
    static = [r for r in rows if r.get("bucket") == "static"]
    if static and np.median([r["blur_var"] for r in static]) < CATASTROPHIC["min_static_blur_var"]:
        _fail(fails, "massive blur", f"{stream}: still frames have median varLap "
                                     f"{np.median([r['blur_var'] for r in static]):.1f}")
    out = dict(n_frames=len(rows), text=report(res), buckets={})
    for _lo, _hi, name in BUCKETS:
        sel = [r for r in rows if r.get("bucket") == name]
        out["buckets"][name] = dict(
            n=len(sel),
            flow_px=(float(np.median([r["flow_px"] for r in sel])) if sel else None),
            retention=(float(np.median([r["survived"] for r in sel])) if sel else None),
            corners=(float(np.median([r["corners"] for r in sel])) if sel else None),
            blur_var=(float(np.median([r["blur_var"] for r in sel])) if sel else None))
    b = np.array([r["brightness"] for r in rows])
    out["brightness"] = dict(p05=float(np.percentile(b, 5)), mean=float(b.mean()), p95=float(np.percentile(b, 95)))
    return out


def check_head_depth(ep: Path, stream: str, fails: list, *, stride: int) -> dict:
    from ..pose.rgbd_io import RgbdEpisode
    rep = RgbdEpisode.load(ep, stream=stream)
    cov, med = [], []
    for f in rep.iter_frames(0, None, stride):
        d = f.depth_m
        cov.append(float((d > 0).mean()))
        v = d[d > 0]
        if len(v):
            med.append(float(np.median(v)))
    if not cov:
        _fail(fails, "depth collapse", f"{stream}: no depth frames")
        return {}
    out = dict(n_sampled=len(cov), coverage_mean=float(np.mean(cov)), coverage_p05=float(np.percentile(cov, 5)),
               distance_median_m=(float(np.median(med)) if med else None),
               intrinsics=rep.intrinsics.to_dict())
    if out["coverage_mean"] < CATASTROPHIC["min_depth_coverage"]:
        _fail(fails, "depth collapse", f"{stream}: mean valid-depth coverage {out['coverage_mean']:.1%}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--hardware", default=None, help="profile to check the delivered format against (e.g. handumi_rgbd)")
    ap.add_argument("--wrists", default="left_wrist,right_wrist")
    ap.add_argument("--depth-stream", default="head_depth")
    ap.add_argument("--limit", type=int, default=None, help="analyse only the first N wrist frames (quick look)")
    ap.add_argument("--depth-stride", type=int, default=10)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    ep = Path(a.episode)
    fails: list[str] = []
    expected = None
    if a.hardware:
        from ..config import load_config
        expected = {c.name: dict(width=c.width, height=c.height, fps=c.fps)
                    for c in load_config(hardware=a.hardware).hardware.cameras}
    sm_p = ep.parent / "session_meta.json"
    session_meta = json.loads(sm_p.read_text()) if sm_p.exists() else {}
    if not session_meta:
        fails.append(f"provenance missing: no {sm_p}")

    out = dict(episode=str(ep), hardware=a.hardware,
               provenance=check_provenance(session_meta, expected, fails),
               timing=check_timing(ep, fails), wrists={}, head_depth=None,
               catastrophic_floors=CATASTROPHIC)
    from ..pose.episode_io import RawEpisode
    present = set(RawEpisode.load(ep).frames)
    for w in [s.strip() for s in a.wrists.split(",") if s.strip()]:
        if w in present:
            out["wrists"][w] = check_wrist(ep, w, fails, limit=a.limit)
    if a.depth_stream in present:
        out["head_depth"] = check_head_depth(ep, a.depth_stream, fails, stride=a.depth_stride)

    print(f"episode {ep}")
    print(f"streams {sorted(present)}")
    print()
    for name, d in out["provenance"].items():
        a_ = d.get("actual") or {}
        ident = d.get("identity") or {}
        print(f"  {name:12s} {int(a_.get('width',0))}x{int(a_.get('height',0))}@{a_.get('fps',0):.2f}  "
              f"name={ident.get('listing_name')!r} matches={ident.get('matches')} serials={ident.get('observed_serials')}")
        if d.get("uvc_controls"):
            print(f"               preset {d['uvc_controls']}")
    print()
    for name, st in out["timing"].items():
        if name.startswith("_"):
            continue
        print(f"  {name:12s} frames={st.get('frame_meta')}/{st.get('video_frames')} fps={st.get('fps')} "
              f"gap_med={st.get('median_gap_ms')} jitter_p95={st.get('p95_jitter_ms')} "
              f"gap_max={st.get('max_gap_ms')} skipped={st.get('skipped')}")
    for w, d in out["wrists"].items():
        print()
        print("\n".join(d.get("text", "").splitlines()[1:]))      # its own episode header would repeat ours
    if out["head_depth"]:
        h = out["head_depth"]
        print(); print(f"  head depth   coverage mean {h['coverage_mean']:.1%} p05 {h['coverage_p05']:.1%}   "
                       f"median distance {h['distance_median_m']:.3f} m   ({h['n_sampled']} frames sampled)")
    print()
    if fails:
        print("CATASTROPHIC — this take carries no information:")
        for f in fails:
            print(f"  - {f}")
    else:
        print("no catastrophic failure. Distribution recorded above; quality thresholds are NOT judged here by design.")
    out["catastrophic"] = fails
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=str))
        print(f"\nwrote {a.json}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
