"""Vision-only V0 end to end on ONE real episode, and the three numbers that decide whether it works.

    head RGB-D --> head_odometry ----------> T_world_head(t)
                                                  |
                        cross_view(head depth, wrist RGB) --> T_head_wrist at sparse instants
                                                  |
                                        T_world_wrist anchors (METRIC)
                                                  |
    wrist RGB --> opencv_vo -----> dense relative trajectory (scale ARBITRARY)
                                                  |
                                           metric_align (sim3)
                                                  |
                                           T_world_wrist

This module contains NO tracking algorithm. It loads an episode, calls head_odometry, cross_view, run_side (opencv_vo)
and metric_align in that order, and reports. Every estimator lives in pose/; if a number here looks wrong, the fix
belongs in the module that produced it, never here.

The headline is deliberately three numbers, printed before anything else:

    A_crossview        do head and wrist ever see the same surface? (this is what supplies SCALE)
    broken_links       how often does either temporal chain break? (this is where an IMU would go)
    E_HOME             does the trajectory come back to where it started?

Their COMBINATION is the finding, not any one alone: sparse anchors with an unbroken VO chain is a working system;
dense anchors with a chain that snaps at every fast motion is a clear place for the IMU.

Gaps are never interpolated. A frame whose motion could not be measured is reported as UNMEASURED, and a broken chain
is reported as broken -- smoothing it away would turn an observability failure into a quiet accuracy failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..pose.calibration import SideCalibration
from ..config import load_pose_cfg
from ..pose.cross_view import DEFAULT_GATES, anchor_from_pair
from ..pose.depth_config import load_depth_pose_cfg
from ..pose.depth_run import static_segments_from_events
from ..pose.episode_io import RawEpisode, derived_dir, write_table
from ..pose.head_odometry import HeadRgbdOdometry
from ..pose.metric_align import align_to_anchors
from ..pose.rgbd_io import RgbdEpisode
from ..pose.run import STREAM, run_side
from ..pose.se3 import inv_T


class _WorldAnchor:
    """A cross-view anchor moved into the world frame.

    cross_view returns T_head_wrist -- the wrist expressed in the head camera AT THAT INSTANT. The head is on a moving
    human head, so those poses do not share a frame and cannot be fitted against a trajectory until head odometry has
    placed each one: T_world_wrist = T_world_head(t) @ T_head_wrist(t). align_to_anchors is duck-typed on
    (.t_ns, .valid, .T_head_wrist), so the composed pose is handed over through the same attribute rather than by
    changing metric_align -- the alignment maths is identical, only the frame is now common.
    """

    __slots__ = ("t_ns", "valid", "T_head_wrist", "src")

    def __init__(self, anchor, T_world_head) -> None:
        self.t_ns = anchor.t_ns
        self.valid = True
        self.T_head_wrist = T_world_head @ anchor.T_head_wrist
        self.src = anchor


def _chain_stats(valid: np.ndarray) -> dict:
    """Broken links, longest unbroken run and longest gap, from a per-frame validity mask."""
    v = np.asarray(valid, bool)
    if not len(v):
        return dict(n=0, n_valid=0, broken_links=0, longest_valid_run=0, longest_gap=0)
    broken = int(np.sum(v[1:] & ~v[:-1])) + (0 if v[0] else 1)   # every entry INTO a valid run after an invalid one
    runs_v, runs_g, cur_v, cur_g = 0, 0, 0, 0
    for x in v:
        if x:
            cur_v += 1; runs_g = max(runs_g, cur_g); cur_g = 0
        else:
            cur_g += 1; runs_v = max(runs_v, cur_v); cur_v = 0
    return dict(n=int(len(v)), n_valid=int(v.sum()), broken_links=broken,
                longest_valid_run=int(max(runs_v, cur_v)), longest_gap=int(max(runs_g, cur_g)))


def _median_pose(Ts: np.ndarray) -> np.ndarray:
    """Median translation + the rotation closest to the mean, for a static window. Median, not mean, so one bad frame
    in a 'still' window does not drag the HOME reference."""
    from scipy.spatial.transform import Rotation
    Ts = np.asarray(Ts, np.float64).reshape(-1, 4, 4)
    T = np.eye(4)
    T[:3, 3] = np.median(Ts[:, :3, 3], axis=0)
    T[:3, :3] = Rotation.from_matrix(Ts[:, :3, :3]).mean().as_matrix()
    return T


def _jitter(Ts: np.ndarray) -> dict:
    """Spread of a supposedly-static window: how much the tracker moves when the hand does not."""
    from scipy.spatial.transform import Rotation
    Ts = np.asarray(Ts, np.float64).reshape(-1, 4, 4)
    if len(Ts) < 2:
        return dict(n=len(Ts), translation_mm=None, rotation_deg=None)
    med = _median_pose(Ts)
    d = np.linalg.norm(Ts[:, :3, 3] - med[:3, 3], axis=1) * 1e3
    ang = Rotation.from_matrix(np.einsum("ij,njk->nik", med[:3, :3].T, Ts[:, :3, :3])).magnitude()
    return dict(n=int(len(Ts)), translation_mm=float(np.percentile(d, 95)), rotation_deg=float(np.degrees(np.percentile(ang, 95))))


def _pose_error(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    from scipy.spatial.transform import Rotation
    E = inv_T(Ta) @ Tb
    return float(np.linalg.norm(E[:3, 3]) * 1e3), float(np.degrees(Rotation.from_matrix(E[:3, :3]).magnitude()))


def _summ(v, scale=1.0) -> dict:
    a = np.asarray([x for x in v if x is not None], float)
    if not len(a):
        return dict(n=0)
    return dict(n=int(len(a)), median=float(np.median(a) * scale), p05=float(np.percentile(a, 5) * scale),
                p95=float(np.percentile(a, 95) * scale), max=float(a.max() * scale))


def analyze(episode: Path, side: str = "left", *, stride: int = 3, max_dt_ms: float = 25.0,
            head_stream: str = "head_depth", gates: dict | None = None, cal_dir: Path | None = None,
            progress=print) -> dict:
    t_start = time.time()
    ep = RawEpisode.load(episode)
    head = RgbdEpisode.load(episode, stream=head_stream)
    cal = SideCalibration(side, cal_dir=cal_dir)
    if cal.intrinsics is None:
        raise SystemExit(f"no fisheye intrinsics for {side}; calibrate before analysing")
    if not cal.physical_side_verified:
        progress(f"  WARNING: {side} has physical_side_verified=false — the pose may be attributed to the wrong hand")

    out: dict = dict(episode=str(episode), side=side, head_stream=head_stream, stride=stride,
                     calibration=cal.summary(), valid_radius_px=cal.intrinsics.get("valid_radius_px"))

    # ---------------------------------------------------------------- 1. head RGB-D odometry
    progress(f"  head odometry over {head.n_frames} RGB-D frames...")
    odo = HeadRgbdOdometry(head.intrinsics)
    head_est, head_T, head_valid, head_t = [], [], [], []
    for f in head.iter_frames():
        e = odo.push(f.rgb, f.depth_m, f.t_ns, f.index)
        head_est.append(e)
        head_valid.append(bool(e.valid))
        head_t.append(int(f.t_ns))
        head_T.append(e.T_world_head if e.T_world_head is not None else np.eye(4))
    head_T = np.asarray(head_T, np.float64); head_valid = np.asarray(head_valid, bool); head_t = np.asarray(head_t, np.int64)
    out["head"] = _chain_stats(head_valid)
    out["head"].update(fps=head.fps,
                       inlier_ratio=_summ([e.inlier_ratio for e in head_est if e.valid]),
                       reprojection_px=_summ([e.reprojection_error_px for e in head_est if e.valid]),
                       step_translation_mm=_summ([e.step_translation_mm for e in head_est if e.valid]))

    # ---------------------------------------------------------------- 2. wrist VO (existing runner, unchanged)
    progress(f"  wrist VO ({side})...")
    cfg = load_pose_cfg()
    sr = run_side(ep, side, cfg, cal=cal)
    Ts_vo, valid_vo, t_vo = sr.Ts_episode_camera, sr.valid, sr.t_ns
    out["wrist_vo"] = _chain_stats(valid_vo)
    out["wrist_vo"].update(backend=cfg.backend, canonical_anchor=sr.canonical_anchor, runtime_s=round(sr.runtime_s, 1))

    # ---------------------------------------------------------------- 3. cross-view metric anchors
    rect = _rectifier(cal)
    stream = STREAM[side]
    if stream not in ep.frames:
        raise SystemExit(f"{episode}: no {stream} stream")
    progress(f"  cross-view anchors every {stride} head frames (wrist masked to r<={cal.intrinsics.get('valid_radius_px')})...")
    anchors, world_anchors, attempted = [], [], 0
    wrist_iter = _wrist_frames(ep, stream, cfg)
    for i, f in enumerate(head.iter_frames(step=stride)):
        j = int(np.argmin(np.abs(t_vo - f.t_ns)))
        if abs(int(t_vo[j]) - int(f.t_ns)) > max_dt_ms * 1e6:
            continue
        img = wrist_iter(j)
        if img is None:
            continue
        attempted += 1
        a = anchor_from_pair(f.rgb, f.depth_m, head.intrinsics, rect.rectify(img), rect.K_out,
                             t_ns=f.t_ns, side=side, gates=gates, wrist_valid_mask=rect.valid_mask,
                             head_frame_index=f.index, wrist_frame_index=j)
        anchors.append(a)
        k = int(np.argmin(np.abs(head_t - f.t_ns)))
        if a.valid and head_valid[k]:
            world_anchors.append(_WorldAnchor(a, head_T[k]))
        elif a.valid:
            a.reason = "cross-view solved but head odometry had no pose at this instant — cannot place it in world"
        if progress and attempted % 25 == 0:
            progress(f"    {attempted} attempted, {sum(x.valid for x in anchors)} valid")

    n_valid_anchor = int(sum(a.valid for a in anchors))
    out["cross_view"] = dict(
        attempted_frames=attempted, valid_anchors=n_valid_anchor,
        A_crossview=(n_valid_anchor / attempted) if attempted else 0.0,
        placed_in_world=len(world_anchors),
        matches=_summ([a.n_matches for a in anchors]),
        depth_backed_matches=_summ([a.n_depth_valid for a in anchors]),
        inliers=_summ([a.n_inliers for a in anchors if a.valid]),
        inlier_ratio=_summ([a.inlier_ratio for a in anchors if a.valid]),
        reprojection_px=_summ([a.reprojection_error_px for a in anchors if a.valid]),
        depth_m=_summ([a.depth_median_m for a in anchors if a.valid]),
        rejection_reasons=_reasons([a.reason for a in anchors if not a.valid]),
        gates=dict(DEFAULT_GATES, **(gates or {})))

    # ---------------------------------------------------------------- 4. sim(3) into metric
    progress(f"  aligning VO to {len(world_anchors)} world anchors...")
    al = align_to_anchors(t_vo, Ts_vo, valid_vo, world_anchors, max_dt_ms=max_dt_ms)
    out["metric_align"] = dict(
        ok=al.ok, reason=al.reason, n_anchors=al.n_anchors, scale=al.scale,
        position_rmse_mm=al.position_rmse_mm, rotation_rmse_deg=al.rotation_rmse_deg,
        anchor_geometry=al.anchor_geometry,
        position_residual_mm=_summ(al.position_residuals_mm if al.position_residuals_mm is not None else []),
        rotation_residual_deg=_summ(al.rotation_residuals_deg if al.rotation_residuals_deg is not None else []),
        scale_consistency=al.scale_consistency)

    # ---------------------------------------------------------------- 5. HOME return, on the METRIC trajectory
    out["home"] = _home(head, t_vo, al.apply(Ts_vo) if al.ok else None, valid_vo, world_anchors)
    out["runtime_s"] = round(time.time() - t_start, 1)
    return out, anchors, (al, t_vo, Ts_vo, valid_vo)


def _reasons(rs) -> dict:
    c: dict = {}
    for r in rs:
        k = (r or "unknown").split("(")[0].strip()[:70]
        c[k] = c.get(k, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def _rectifier(cal: SideCalibration):
    from ..pose.backends.opencv_vo import FisheyeRectifier
    return FisheyeRectifier(cal.intrinsics, fov_deg=90.0)


def _wrist_frames(ep: RawEpisode, stream: str, cfg):
    """Random access to decoded wrist frames by index, decoded once and cached — anchors are sparse, so decoding the
    whole stream per anchor would dominate the runtime."""
    cache: dict[int, np.ndarray] = {}
    state = dict(it=ep.iter_frames(stream, downscale=cfg.image_downscale), done=False, last=-1)

    def get(j: int):
        # iter_frames yields (video_frame, frame_index, capture_ns, image); frame_index indexes the same arrays as
        # t_vo, so it is the key to cache on. Decoding is forward-only, so a request for an already-passed index that
        # was never cached returns None rather than rewinding the decoder.
        if j in cache:
            return cache[j]
        while not state["done"] and state["last"] < j:
            try:
                vf, fi, t_ns, img = next(state["it"])
            except StopIteration:
                state["done"] = True
                break
            state["last"] = int(fi)
            cache[int(fi)] = img
        return cache.get(j)

    return get


def _home(head: RgbdEpisode, t_vo: np.ndarray, Ts_metric, valid_vo: np.ndarray, world_anchors=None) -> dict:
    """HOME return on the metric wrist trajectory, from the operator's own marks.

    This is NOT pure tracker drift: without a mechanical dock the operator does not put the hand back in exactly the
    same place. It is reported as `tracker drift + physical replacement error`, and the static jitter alongside it
    separates how much of the number is the tracker moving while the hand does not.
    """
    segs = static_segments_from_events(head, head.t_ns, load_depth_pose_cfg())
    r: dict = dict(marks_found=len(segs), source="events.json home_leave/home_return")
    if len(segs) < 2:
        r["reason"] = ("fewer than two operator HOME windows — press H at the end of the opening still period and "
                       "again when you return; without both, HOME return cannot be computed and must not be guessed")
        return r
    if Ts_metric is None:
        r["reason"] = "no metric alignment, so there is no metric trajectory to evaluate HOME on"
        return r
    win = []
    for seg in (segs[0], segs[-1]):
        m = (t_vo >= seg.t0_ns) & (t_vo <= seg.t1_ns) & valid_vo
        win.append((seg, m))
    r["start_window"] = dict(n_frames=int(win[0][1].sum()), duration_s=round((win[0][0].t1_ns - win[0][0].t0_ns) / 1e9, 2))
    r["end_window"] = dict(n_frames=int(win[1][1].sum()), duration_s=round((win[1][0].t1_ns - win[1][0].t0_ns) / 1e9, 2))
    if win[0][1].sum() < 3 or win[1][1].sum() < 3:
        # A monocular VO needs parallax to initialise, so a PERFECTLY still opening window can legitimately contain no
        # VO pose at all. The cross-view anchors do not: they are metric world poses that need no motion. Fall back to
        # them, and say so -- this is a different measurement, not the same one relabelled.
        alt = _home_from_anchors(world_anchors, segs)
        if alt is not None:
            alt["reason"] = ("VO had <3 poses in a HOME window (a still camera gives a monocular VO no parallax to "
                             "initialise on); measured on the cross-view anchors instead")
            r.update(alt)
            return r
        r["reason"] = ("a HOME window has fewer than 3 tracked frames and too few anchors — the take did not hold "
                       "still long enough")
        return r
    Ta, Tb = _median_pose(Ts_metric[win[0][1]]), _median_pose(Ts_metric[win[1][1]])
    mm, deg = _pose_error(Ta, Tb)
    r.update(translation_mm=mm, rotation_deg=deg, source="metric VO trajectory",
             start_jitter=_jitter(Ts_metric[win[0][1]]), end_jitter=_jitter(Ts_metric[win[1][1]]),
             note="HOME return error = tracker drift + physical replacement error (no mechanical dock)")
    return r


def _home_from_anchors(world_anchors, segs):
    """HOME from the metric cross-view anchors alone -- independent of the VO and of the sim(3) fit."""
    if not world_anchors:
        return None
    t = np.array([a.t_ns for a in world_anchors], np.int64)
    T = np.array([a.T_head_wrist for a in world_anchors], np.float64)
    sel = [(t >= sg.t0_ns) & (t <= sg.t1_ns) for sg in (segs[0], segs[-1])]
    if sel[0].sum() < 3 or sel[1].sum() < 3:
        return None
    Ta, Tb = _median_pose(T[sel[0]]), _median_pose(T[sel[1]])
    mm, deg = _pose_error(Ta, Tb)
    return dict(translation_mm=mm, rotation_deg=deg, source="cross-view anchors (VO unavailable in the window)",
                start_window=dict(n_frames=int(sel[0].sum()), duration_s=None),
                end_window=dict(n_frames=int(sel[1].sum()), duration_s=None),
                start_jitter=_jitter(T[sel[0]]), end_jitter=_jitter(T[sel[1]]),
                note="HOME return error = tracker drift + physical replacement error (no mechanical dock)")


def _f(v, spec=".1f", na="--"):
    return na if v is None else format(v, spec)


def format_report(o: dict) -> str:
    cv, hd, wv, ma, hm = o["cross_view"], o["head"], o["wrist_vo"], o["metric_align"], o["home"]
    L = []
    L.append("=" * 78)
    L.append(f"VISION-ONLY V0  —  {Path(o['episode']).name}  [{o['side']}]")
    L.append("=" * 78)
    L.append("")
    L.append(f"  A_crossview          {cv['A_crossview']*100:5.1f} %      ({cv['valid_anchors']} valid / {cv['attempted_frames']} attempted)")
    L.append(f"  broken_links_head    {hd['broken_links']:5d}        ({hd['n_valid']}/{hd['n']} frames tracked)")
    L.append(f"  broken_links_wrist   {wv['broken_links']:5d}        ({wv['n_valid']}/{wv['n']} frames tracked)")
    L.append("")
    L.append(f"  HOME_translation     {_f(hm.get('translation_mm')):>5} mm")
    L.append(f"  HOME_rotation        {_f(hm.get('rotation_deg'), '.2f'):>5} deg")
    if "reason" in hm:
        L.append(f"      ({hm['reason']})")
    L.append("")
    L.append("-" * 78)
    L.append(f"HEAD RGB-D ODOMETRY    {hd['fps']:.1f} fps, longest valid run {hd['longest_valid_run']}, longest gap {hd['longest_gap']}")
    L.append(f"  inlier ratio         median {_f(hd['inlier_ratio'].get('median'), '.2f')}   p05 {_f(hd['inlier_ratio'].get('p05'), '.2f')}")
    L.append(f"  reprojection px      median {_f(hd['reprojection_px'].get('median'), '.2f')}   p95 {_f(hd['reprojection_px'].get('p95'), '.2f')}")
    L.append(f"  step translation mm  median {_f(hd['step_translation_mm'].get('median'), '.1f')}   p95 {_f(hd['step_translation_mm'].get('p95'), '.1f')}")
    if hd["longest_gap"]:
        L.append(f"  {hd['longest_gap']} consecutive frames have NO head pose — that motion is UNMEASURED, not interpolated")
    L.append("")
    L.append(f"WRIST VO ({wv['backend']})   longest valid run {wv['longest_valid_run']}, longest gap {wv['longest_gap']}, {wv['runtime_s']} s")
    if wv["longest_gap"]:
        L.append(f"  {wv['longest_gap']} consecutive frames UNMEASURED")
    L.append("")
    L.append(f"CROSS-VIEW ANCHORS     wrist masked to r <= {o['valid_radius_px']} px before detection")
    L.append(f"  matches              median {_f(cv['matches'].get('median'))}   p05 {_f(cv['matches'].get('p05'))}")
    L.append(f"  depth-backed         median {_f(cv['depth_backed_matches'].get('median'))}")
    L.append(f"  PnP inliers          median {_f(cv['inliers'].get('median'))}   ratio {_f(cv['inlier_ratio'].get('median'), '.2f')}")
    L.append(f"  reprojection px      median {_f(cv['reprojection_px'].get('median'), '.2f')}   p95 {_f(cv['reprojection_px'].get('p95'), '.2f')}")
    L.append(f"  anchor depth m       median {_f(cv['depth_m'].get('median'), '.2f')}")
    L.append(f"  placed in world      {cv['placed_in_world']} of {cv['valid_anchors']} valid")
    if cv["rejection_reasons"]:
        L.append("  why anchors failed:")
        for k, n in list(cv["rejection_reasons"].items())[:6]:
            L.append(f"      {n:4d}  {k}")
    L.append("")
    L.append(f"METRIC ALIGNMENT       {'OK' if ma['ok'] else 'FAILED: ' + (ma['reason'] or '?')}")
    if ma["n_anchors"]:
        sc = ma.get("scale_consistency") or {}
        L.append(f"  sim(3) scale         {_f(ma['scale'], '.4f')}   from {ma['n_anchors']} anchors")
        L.append(f"  position residual    median {_f(ma['position_residual_mm'].get('median'))}   p95 {_f(ma['position_residual_mm'].get('p95'))}   max {_f(ma['position_residual_mm'].get('max'))} mm")
        L.append(f"  rotation residual    median {_f(ma['rotation_residual_deg'].get('median'), '.2f')}   p95 {_f(ma['rotation_residual_deg'].get('p95'), '.2f')}   max {_f(ma['rotation_residual_deg'].get('max'), '.2f')} deg")
        L.append(f"  scale consistency    pairs {sc.get('n_pairs', 0)}   median {_f(sc.get('median'), '.4f')}"
                 f"   p05 {_f(sc.get('p05'), '.4f')}   p95 {_f(sc.get('p95'), '.4f')}"
                 f"   spread {_f(sc.get('spread_pct'), '.1f')} %")
        if sc.get("note"):
            L.append(f"      {sc['note']}")
        ag = ma.get("anchor_geometry") or {}
        if ag:
            e = ag.get("extent_m", [])
            L.append(f"  anchor geometry      extent {' x '.join(f'{x*1e3:.0f}' for x in e)} mm"
                     f"   collinearity {_f(ag.get('collinearity'), '.3f')}   planarity {_f(ag.get('planarity'), '.3f')}")
            if ag.get("warning"):
                L.append(f"  !! {ag['warning']}")
        L.append("  (a good single scale with an inconsistent per-pair scale is NOT a pass)")
    L.append("")
    if hm.get("translation_mm") is not None:
        L.append(f"HOME  [{hm.get('source', '?')}]")
        L.append(f"                       start window {hm['start_window']['n_frames']} frames / {hm['start_window']['duration_s']} s,"
                 f" end {hm['end_window']['n_frames']} frames / {hm['end_window']['duration_s']} s")
        for w in ("start", "end"):
            j = hm[f"{w}_jitter"]
            L.append(f"  {w:5s} static jitter  p95 {_f(j.get('translation_mm'))} mm / {_f(j.get('rotation_deg'), '.2f')} deg over {j['n']} frames")
        L.append(f"  {hm['note']}")
    L.append("=" * 78)
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vision-only V0 end to end on one episode (orchestration only)")
    ap.add_argument("episode", type=Path)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--stride", type=int, default=3, help="attempt a cross-view anchor every N head frames")
    ap.add_argument("--head-stream", default="head_depth")
    ap.add_argument("--max-dt-ms", type=float, default=25.0)
    ap.add_argument("--json", type=Path, default=None, help="also write the full result as JSON")
    ap.add_argument("--cal-dir", type=Path, default=None, help="calibration directory (default: configs/calibration)")
    ap.add_argument("--no-write", action="store_true", help="do not write into the episode's derived/ directory")
    a = ap.parse_args(argv)

    out, anchors, (al, t_vo, Ts_vo, valid_vo) = analyze(
        a.episode, a.side, stride=a.stride, max_dt_ms=a.max_dt_ms, head_stream=a.head_stream, cal_dir=a.cal_dir,
        progress=lambda s: (print(s, flush=True)))
    print()
    print(format_report(out))

    if not a.no_write:
        d = derived_dir(a.episode, f"vision_v0_{a.side}")
        d.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        write_table(pd.DataFrame([x.to_row() for x in anchors]), d / "anchors")
        if al.ok:
            Tm = al.apply(Ts_vo)
            from ..pose.se3 import T_to_pose7
            rows = []
            for i in range(len(t_vo)):
                p = T_to_pose7(Tm[i])
                rows.append(dict(t_ns=int(t_vo[i]), valid=bool(valid_vo[i]),
                                 x=p[0], y=p[1], z=p[2], qx=p[3], qy=p[4], qz=p[5], qw=p[6]))
            write_table(pd.DataFrame(rows), d / "T_world_wrist")
        (d / "report.txt").write_text(format_report(out))
        (d / "summary.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {d}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
