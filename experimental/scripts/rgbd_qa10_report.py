#!/usr/bin/env python3
"""[2026-09-18] Sensor QA for the RGB-D teacher-sensor pilot (dataset HRGBD_qa10).

The depth camera in this dataset is NOT a policy input. Its whole job is to produce metric human xyz / EEF labels
after every RGB-only route to translation failed (MASt3R rejected, monocular metric depth rejected, cube PnP
insufficient, IMU bridge insufficient). A label sensor is held to a different standard than an observation sensor: a
policy can learn around a noisy observation, but a biased label is baked into every episode trained on it and no
amount of data removes it. So this reports the label-quality questions, per episode, before a larger take is
authorised:

    1  depth valid ratio            can it see the working volume at all
    2  depth dropout rate           frames and runs where it stops seeing
    3  RGB-depth alignment          does a pixel's depth belong to that pixel
    4  depth / RGB timestamp sync   do the streams describe the same instant
    5  depth bias / std, 10-50 cm   the accuracy of the label at grasp distance
    6  stationary / held jitter     how much a label wanders when nothing moves
   10  grip / camera stream sync    does the grip label line up in time with the frames

Items 7-9 (grasp-TCP trajectory after camera_to_tcp_v1, chunk displacement percentiles, non-physical chunk share)
need a metric TCP track and live in `rgbd_tcp_qa.py`; this script prints what it would need for them.

Nothing here has a pass/fail threshold baked in where the rig has never been measured. Where a limit IS known --
structural dropout, gross edge offset, a stream that stops -- it is called a FAIL. Everything else is reported as a
number so the limits can be set from what this rig actually produces.

    .venv/bin/python scripts/rgbd_qa10_report.py --session datasets/human_handumi_raw/HRGBD_qa10/HRGBD_qa10_<stamp>
    .venv/bin/python scripts/rgbd_qa10_report.py --episode <episode_dir>            # one episode
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handumi_collector.pose.rgbd_io import RgbdEpisode, backproject          # noqa: E402
from handumi_collector.pose.episode_io import RawEpisode                     # noqa: E402
from handumi_collector.pose.rgbd_align import edge_alignment, summarise      # noqa: E402

# The band the label actually has to be right in: a grasped 50 mm cube seen from the head mount sits here. Wider bands
# are reported too, because a bias that only appears far away is still evidence about the sensor.
BANDS = ((0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.80), (0.80, 1.50))
WORK_LO, WORK_HI = 0.10, 0.80


def pct(a, q):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of every True run -- a dropout's LENGTH is what matters, not its count: thirty scattered bad
    frames are noise, thirty consecutive ones are a hole the label cannot be interpolated across."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


# --------------------------------------------------------------------------------------------- 1, 2, 5, 6
def depth_stats(ep: RgbdEpisode, *, sample: int, static_n: int) -> dict:
    """Valid ratio, dropout, per-band noise and plane bias, from one decode pass.

    `static_n` frames at the START of the episode are also kept in full. The collector's stillness gate means the
    operator has been still for at least 3 s before REC begins, so those frames are a real static scene: any
    variation across them is the sensor's own, which is the only way to separate sensor jitter from hand motion
    without a motion-capture reference."""
    step = max(1, ep.n_frames // sample)
    valid_ratio, work_ratio, med_dist, idx = [], [], [], []
    static: list[np.ndarray] = []
    for f in ep.iter_frames(0, ep.n_frames, 1):
        d = f.depth_m
        if f.index < static_n:
            static.append(d.copy())
        if f.index % step and f.index >= static_n:
            continue
        v = d > 0
        valid_ratio.append(float(v.mean()))
        work_ratio.append(float(((d >= WORK_LO) & (d <= WORK_HI)).mean()))
        dv = d[v]
        med_dist.append(float(np.median(dv)) if dv.size else float("nan"))
        idx.append(f.index)
    valid_ratio = np.array(valid_ratio); work_ratio = np.array(work_ratio); idx = np.array(idx)

    # (2) dropout. Two different failures, reported apart: a frame the recorder never wrote a depth map for, and a
    # frame whose depth map is present but has collapsed. Only the first is structural, and only it is a hard FAIL.
    n_meta = len(ep.t_ns)
    collapsed = valid_ratio < 0.5 * float(np.median(valid_ratio)) if valid_ratio.size else np.zeros(0, bool)
    rr = runs_of(collapsed)

    out = dict(
        n_frames=ep.n_frames, n_sampled=int(len(valid_ratio)),
        valid_ratio_mean=float(valid_ratio.mean()) if valid_ratio.size else float("nan"),
        valid_ratio_p05=pct(valid_ratio, 5), valid_ratio_min=float(valid_ratio.min()) if valid_ratio.size else float("nan"),
        work_band_ratio_mean=float(work_ratio.mean()) if work_ratio.size else float("nan"),
        median_distance_m=float(np.nanmedian(med_dist)) if med_dist else float("nan"),
        collapsed_frames=int(collapsed.sum()), collapsed_share=float(collapsed.mean()) if collapsed.size else 0.0,
        longest_collapsed_run=int(max((b - a for a, b in rr), default=0)),
        depth_maps_present=n_meta,
    )

    # (5)/(6) per-band noise and plane bias on the static window
    out["static_frames"] = len(static)
    if len(static) >= 8:
        S = np.stack(static)                                   # (T, H, W) metres, 0 = invalid
        ok = (S > 0).all(0)                                    # only pixels valid in EVERY static frame
        with np.errstate(invalid="ignore"):
            mu = np.where(ok, S.mean(0), np.nan)
            sd = np.where(ok, S.std(0), np.nan)
        bands = {}
        for lo, hi in BANDS:
            m = ok & (mu >= lo) & (mu < hi)
            n = int(m.sum())
            if n < 200:
                bands[f"{lo:.2f}-{hi:.2f}"] = dict(pixels=n)
                continue
            b = dict(pixels=n,
                     temporal_std_mm_p50=float(np.nanmedian(sd[m])) * 1e3,
                     temporal_std_mm_p95=pct(sd[m], 95) * 1e3)
            b.update(plane_residual_mm(mu, m, ep))
            bands[f"{lo:.2f}-{hi:.2f}"] = b
        out["bands"] = bands
    else:
        out["bands"] = {}
        out["static_note"] = ("fewer than 8 frames at the head of the episode; per-band noise and bias not measured. "
                              "The stillness gate should guarantee these -- check the take started from HOME.")
    return out


def plane_residual_mm(mu: np.ndarray, mask: np.ndarray, ep: RgbdEpisode) -> dict:
    """Systematic bending, which is what a *bias* looks like on a label sensor.

    The desk is the one large flat thing every episode contains. Fit a plane to the backprojected points of this
    distance band and report the residual: temporal std says how much a point wanders, this says whether the surface
    is being reconstructed with a curve. A sensor can be beautifully repeatable and still bend a table by a
    centimetre, and a bent table is a biased label."""
    ys, xs = np.nonzero(mask)
    if len(xs) > 40000:
        sel = np.random.default_rng(0).choice(len(xs), 40000, replace=False)
        ys, xs = ys[sel], xs[sel]
    K = ep.intrinsics.K
    z = mu[ys, xs]
    X = (xs - K[0, 2]) * z / K[0, 0]
    Y = (ys - K[1, 2]) * z / K[1, 1]
    P = np.stack([X, Y, z], 1)
    c = P.mean(0)
    # total-least-squares plane through the centroid; the smallest singular direction is the normal
    _, sv, Vt = np.linalg.svd(P - c, full_matrices=False)
    nrm = Vt[-1]
    r = (P - c) @ nrm
    # A band is only "a plane" if the fit is thin compared with the patch; otherwise the residual is describing
    # scene geometry, not the sensor, and reporting it as bias would be a lie.
    planar = float(sv[-1] / max(sv[0], 1e-9))
    return dict(plane_residual_mm_rms=float(np.sqrt((r ** 2).mean())) * 1e3,
                plane_residual_mm_p95=float(np.percentile(np.abs(r), 95)) * 1e3,
                plane_flatness=planar,
                plane_trustworthy=bool(planar < 0.02))


# --------------------------------------------------------------------------------------------- 3
def alignment(ep: RgbdEpisode, n: int) -> dict:
    step = max(1, ep.n_frames // n)
    per = [edge_alignment(f.rgb, f.depth_m) for f in ep.iter_frames(0, ep.n_frames, step)]
    a = summarise(per)
    a["n_frames"] = len(per)
    return a


# --------------------------------------------------------------------------------------------- 4, 10
def timing(ep_path: Path, depth_stream: str) -> dict:
    """Do the streams describe the same instant, and does the grip label line up with them?

    The Orbbec's colour and depth leave the SDK inside one frame object and are stamped once, so depth-vs-its-own-RGB
    sync is structural and cannot drift -- asserting it would be theatre. What can drift, and what the label pipeline
    actually joins on, is head_depth against the wrist cameras and against the Feetech gripper, each on its own clock.
    That is what is measured here."""
    raw = RawEpisode.load(ep_path)
    out: dict = {"streams": {}}
    ref = raw.frames.get(depth_stream)
    if ref is None:
        return {"error": f"no frame_meta for {depth_stream!r}"}
    rt = np.asarray(ref.capture_ns, np.int64)
    d = np.diff(rt) / 1e6
    out["streams"][depth_stream] = dict(n=len(rt), fps=float(1e9 * (len(rt) - 1) / max(rt[-1] - rt[0], 1)),
                                        gap_ms_p50=pct(d, 50), gap_ms_p95=pct(d, 95), gap_ms_max=float(d.max()) if d.size else float("nan"))
    for name, fm in raw.frames.items():
        if name == depth_stream:
            continue
        t = np.asarray(fm.capture_ns, np.int64)
        dd = np.diff(t) / 1e6
        k = np.searchsorted(t, rt).clip(1, len(t) - 1)
        off = np.minimum(np.abs(rt - t[k - 1]), np.abs(t[k] - rt)) / 1e6      # nearest-neighbour offset, ms
        out["streams"][name] = dict(n=len(t), fps=float(1e9 * (len(t) - 1) / max(t[-1] - t[0], 1)),
                                    gap_ms_p50=pct(dd, 50), gap_ms_p95=pct(dd, 95),
                                    offset_to_depth_ms_p50=pct(off, 50), offset_to_depth_ms_p95=pct(off, 95),
                                    offset_to_depth_ms_max=float(off.max()) if off.size else float("nan"))
    # (10) the grip label's own clock, and whether it spans the frames it will be joined to
    grips = {}
    for side, g in (raw.grip or {}).items():
        t = np.asarray(g.t_ns, np.int64)
        if not len(t):
            grips[side] = dict(n=0, note="gripper stream empty -- the grip label cannot be produced")
            continue
        dd = np.diff(t) / 1e6
        k = np.searchsorted(t, rt).clip(1, len(t) - 1)
        off = np.minimum(np.abs(rt - t[k - 1]), np.abs(t[k] - rt)) / 1e6
        nrm = np.asarray(g.normalized, np.float64)
        grips[side] = dict(n=len(t), hz=float(1e9 * (len(t) - 1) / max(t[-1] - t[0], 1)),
                           gap_ms_p50=pct(dd, 50), gap_ms_p95=pct(dd, 95), gap_ms_max=float(dd.max()) if dd.size else float("nan"),
                           offset_to_depth_ms_p50=pct(off, 50), offset_to_depth_ms_p95=pct(off, 95),
                           covers_first_frame=bool(t[0] <= rt[0]), covers_last_frame=bool(t[-1] >= rt[-1]),
                           norm_min=float(np.nanmin(nrm)), norm_max=float(np.nanmax(nrm)),
                           # a jaw that never travels is not a grip label, it is a constant
                           norm_travel=float(np.nanmax(nrm) - np.nanmin(nrm)))
    out["grippers"] = grips
    return out


# --------------------------------------------------------------------------------------------- report
def qa_episode(ep_path: Path, *, stream: str | None, sample: int, static_n: int, align_n: int) -> dict:
    r: dict = {"episode": ep_path.name, "path": str(ep_path)}
    ep = RgbdEpisode.load(ep_path, stream=stream)
    r["stream"] = ep.stream
    k = ep.intrinsics
    r["intrinsics"] = dict(fx=k.fx, fy=k.fy, cx=k.cx, cy=k.cy, width=k.width, height=k.height,
                           depth_unit_mm=k.depth_unit_m * 1e3, aligned_to_rgb=k.aligned_to_rgb, source=k.source)
    # A guessed focal length is a silent metric-scale error in every label, so it is a hard FAIL, not a warning.
    r["intrinsics_reported_by_device"] = bool(k.fx and k.fy and k.source not in ("", "fallback", "guess"))
    r["depth"] = depth_stats(ep, sample=sample, static_n=static_n)
    r["alignment"] = alignment(ep, align_n)
    r["timing"] = timing(ep_path, ep.stream)
    meta = ep.meta or {}
    r["duration_s"] = meta.get("duration_s")
    r["order"] = meta.get("order")
    r["status"] = meta.get("status")
    r["calibration_version"] = meta.get("calibration_version")
    r["verdict"] = verdict(r)
    return r


def verdict(r: dict) -> dict:
    """Only what is knowable without a rig baseline is judged. Everything else is a number for the record."""
    fails, warns = [], []
    if not r.get("intrinsics_reported_by_device"):
        fails.append("intrinsics were not reported by the device -- every metric label from this episode is on a guessed scale")
    d = r["depth"]
    if d["depth_maps_present"] < 0.98 * d["n_frames"]:
        fails.append(f"structural depth dropout: {d['depth_maps_present']} depth maps for {d['n_frames']} frames")
    if d["longest_collapsed_run"] >= 15:                     # half a second at 30 Hz: longer than a label can bridge
        fails.append(f"a run of {d['longest_collapsed_run']} consecutive frames with collapsed depth")
    elif d["collapsed_share"] > 0.02:
        warns.append(f"{d['collapsed_share']:.1%} of sampled frames have collapsed depth")
    if d["valid_ratio_mean"] < 0.30:
        fails.append(f"mean valid-depth ratio {d['valid_ratio_mean']:.1%} -- the sensor is not seeing the workspace")
    a = r["alignment"]
    if a.get("problems"):
        fails += [f"alignment: {p}" for p in a["problems"]]
    warns += [f"alignment: {w}" for w in a.get("warnings", [])]
    t = r["timing"]
    for name, s in (t.get("streams") or {}).items():
        if s.get("offset_to_depth_ms_p95", 0) > 20:          # more than half a frame at 30 Hz
            warns.append(f"{name}: p95 offset to depth {s['offset_to_depth_ms_p95']:.1f} ms")
        if not (25 <= s.get("fps", 0) <= 35):
            fails.append(f"{name}: {s.get('fps', 0):.1f} fps")
    for side, g in (t.get("grippers") or {}).items():
        if not g.get("n"):
            fails.append(f"gripper {side}: no samples")
            continue
        if g.get("norm_travel", 0) < 0.05:
            fails.append(f"gripper {side}: jaw never travelled (norm range {g['norm_travel']:.3f}) -- not a grip label")
        if not (g.get("covers_first_frame") and g.get("covers_last_frame")):
            warns.append(f"gripper {side}: does not span the whole episode")
    return dict(fail=fails, warn=warns, ok=not fails)


def fmt(v, nd=1, unit=""):
    return "   n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}{unit}"


def print_report(rows: list[dict]) -> None:
    print("\n" + "=" * 108)
    print("RGB-D TEACHER-SENSOR QA  --  depth is a LABEL source here, not a policy input")
    print("=" * 108)

    print("\n[1][2] depth valid ratio and dropout")
    print(f"{'episode':>16s} {'frames':>7s} {'valid%':>7s} {'p05%':>6s} {'work%':>6s} {'dist m':>7s} "
          f"{'collapsed':>10s} {'run':>5s} {'maps':>6s}")
    for r in rows:
        d = r["depth"]
        print(f"{r['episode'][-12:]:>16s} {d['n_frames']:7d} {d['valid_ratio_mean']*100:6.1f}% {d['valid_ratio_p05']*100:5.1f}% "
              f"{d['work_band_ratio_mean']*100:5.1f}% {fmt(d['median_distance_m'],2):>7s} "
              f"{d['collapsed_frames']:4d} ({d['collapsed_share']*100:.1f}%) {d['longest_collapsed_run']:5d} {d['depth_maps_present']:6d}")

    print("\n[3] RGB-depth alignment (edge offset, px)")
    print(f"{'episode':>16s} {'median':>8s} {'p95':>8s} {'overlap':>8s} {'invalid@edge':>13s}  problems")
    for r in rows:
        a = r["alignment"]
        print(f"{r['episode'][-12:]:>16s} {fmt(a.get('median_offset_px'),1):>8s} {fmt(a.get('p95_offset_px'),1):>8s} "
              f"{fmt((a.get('overlap') or 0)*100,0,'%'):>8s} {fmt((a.get('invalid_depth_near_edges') or 0)*100,1,'%'):>13s}"
              f"  {'; '.join(a.get('problems') or []) or '-'}")

    print("\n[4][10] stream timing and grip sync (ms)")
    for r in rows:
        t = r["timing"]
        print(f"  {r['episode']}")
        for name, s in (t.get("streams") or {}).items():
            extra = ("" if "offset_to_depth_ms_p50" not in s else
                     f"   offset->depth p50 {s['offset_to_depth_ms_p50']:.1f} p95 {s['offset_to_depth_ms_p95']:.1f} max {s['offset_to_depth_ms_max']:.1f}")
            print(f"      {name:>12s}  n {s['n']:5d}  {s['fps']:5.2f} fps  gap p50 {s['gap_ms_p50']:.1f} p95 {s['gap_ms_p95']:.1f}{extra}")
        for side, g in (t.get("grippers") or {}).items():
            if not g.get("n"):
                print(f"      grip {side:>7s}  {g.get('note','')}")
                continue
            print(f"      grip {side:>7s}  n {g['n']:5d}  {g['hz']:5.1f} Hz  gap p95 {g['gap_ms_p95']:.1f} max {g['gap_ms_max']:.1f}"
                  f"   offset->depth p50 {g['offset_to_depth_ms_p50']:.1f} p95 {g['offset_to_depth_ms_p95']:.1f}"
                  f"   norm {g['norm_min']:.2f}..{g['norm_max']:.2f} (travel {g['norm_travel']:.2f})"
                  f"   spans {'yes' if g['covers_first_frame'] and g['covers_last_frame'] else 'NO'}")

    print("\n[5][6] depth noise and plane bias by distance band  (static window at the head of each episode)")
    print(f"{'episode':>16s} {'band m':>12s} {'pixels':>9s} {'std p50':>9s} {'std p95':>9s} {'plane rms':>10s} {'plane p95':>10s} {'flat?':>6s}")
    for r in rows:
        b = r["depth"].get("bands") or {}
        if not b:
            print(f"{r['episode'][-12:]:>16s}   {r['depth'].get('static_note','no static window')}")
            continue
        first = True
        for band, v in b.items():
            if v.get("pixels", 0) < 200:
                continue
            print(f"{(r['episode'][-12:] if first else ''):>16s} {band:>12s} {v['pixels']:9d} "
                  f"{v['temporal_std_mm_p50']:8.2f}mm {v['temporal_std_mm_p95']:8.2f}mm "
                  f"{v['plane_residual_mm_rms']:9.2f}mm {v['plane_residual_mm_p95']:9.2f}mm "
                  f"{'yes' if v['plane_trustworthy'] else 'no':>6s}")
            first = False

    print("\nVERDICT")
    nf = 0
    for r in rows:
        v = r["verdict"]
        nf += len(v["fail"])
        mark = "PASS" if v["ok"] and not v["warn"] else ("PASS (warn)" if v["ok"] else "FAIL")
        print(f"  {r['episode']:>18s}  {mark}")
        for f in v["fail"]:
            print(f"      FAIL  {f}")
        for w in v["warn"]:
            print(f"      warn  {w}")
    print(f"\n  {sum(1 for r in rows if r['verdict']['ok'])}/{len(rows)} episodes usable as a label source; {nf} failures total")
    print("\n[7][8][9] grasp-TCP trajectory, chunk displacement percentiles and non-physical chunk share are NOT in this\n"
          "          report: they need a metric TCP track. Run scripts/rgbd_tcp_qa.py once this report passes -- a TCP\n"
          "          trajectory built on a sensor that failed items 1-6 would only measure the sensor's failure.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", type=Path, help="session directory holding episode_* subdirectories")
    g.add_argument("--episode", type=Path, help="a single episode directory")
    ap.add_argument("--stream", default=None, help="depth stream name (default: the episode's only *_depth)")
    ap.add_argument("--sample", type=int, default=120, help="frames sampled per episode for valid-ratio/dropout")
    ap.add_argument("--static-frames", type=int, default=45, help="frames at the head of the episode treated as the static window")
    ap.add_argument("--align-frames", type=int, default=12, help="frames used for the RGB-depth edge alignment")
    ap.add_argument("--json", type=Path, help="write the full per-episode record here")
    a = ap.parse_args()

    if a.episode:
        eps = [a.episode]
    else:
        eps = sorted(p for p in a.session.iterdir() if p.is_dir() and p.name.startswith("episode_"))
        if not eps:
            print(f"no episode_* directories under {a.session}"); return 1
    rows = []
    for p in eps:
        try:
            rows.append(qa_episode(p, stream=a.stream, sample=a.sample, static_n=a.static_frames, align_n=a.align_frames))
            print(f"  read {p.name}", flush=True)
        except Exception as exc:
            print(f"  {p.name}: UNREADABLE -- {exc}", flush=True)
            rows.append(dict(episode=p.name, path=str(p), error=str(exc),
                             verdict=dict(fail=[f"unreadable: {exc}"], warn=[], ok=False),
                             depth={}, alignment={}, timing={}))
    good = [r for r in rows if "error" not in r]
    if good:
        print_report(good)
    bad = [r for r in rows if "error" in r]
    if bad:
        print("\nUNREADABLE EPISODES")
        for r in bad:
            print(f"  {r['episode']}: {r['error']}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1, default=float))
        print(f"\nfull record -> {a.json}")
    return 0 if all(r["verdict"]["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
