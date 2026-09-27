"""Put the head camera and the wrist IMUs on one clock, from a purpose-shot calibration take.

    python -m handumi_collector.tools.pose_process EP --backend opencv_vo     # first: wrist camera poses
    python -m handumi_collector.tools.temporal_sync EP

There is exactly one constant to calibrate here, and it is not the one you would reach for first:

    head frame at capture_ns
      -> nearest wrist frame, looked up per frame        no calibration; record the delta each time
      -> + the wrist camera <-> wrist IMU offset         ONE constant, from gyro correlation
      -> IMU sample interpolated to that instant

Adding a measured `head <-> wrist camera` offset to a measured `wrist camera <-> wrist IMU` offset and storing the sum
is wrong in its first term. The three cameras free-run at genuinely different rates -- 30.0260, 30.0032 and 30.0047 Hz
-- so their relative phase winds by most of a frame period over a 30 s take and the offset measured in five-second
windows circulates through zero rather than drifting one way. Sampled once it looks like a calibration; applied later
it is wrong by up to a frame. See docs/handumi_collector/TEMPORAL_SYNC.md.

So this reports the head-to-wrist relationship as what it is -- a per-frame lookup with a delta worth printing -- and
estimates only the wrist camera to wrist IMU constant, which is a real latency between two devices bolted to the same
unit. The existing estimator correlates a camera's rotation against a gyro, which needs the two to move together: true
of a wrist pair, false of a head camera on a table that never rotates, which is why the head is reached by lookup and
not by correlation. Table taps at the end of the take check the result against something physical rather than against
the arithmetic that produced it."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from ..collector.integrity import validate_episode
from ..pose.episode_io import RawEpisode


def stream_brightness(ep: RawEpisode, stream: str, *, until_s: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """(t_s, mean luminance) over the opening seconds. Covering a lens is the cheapest way to settle which physical
    wrist a camera is on, and the two units here share a USB serial, so nothing else in the recording says."""
    import cv2
    fm = ep.frames[stream]
    t0 = int(fm.capture_ns[0])
    ts, ys = [], []
    for _vf, _i, cap_ns, img in ep.iter_frames(stream, downscale=4, gray=True):
        dt = (cap_ns - t0) / 1e9
        if dt > until_s: break
        ts.append(dt); ys.append(float(img.mean()))
    return np.asarray(ts), np.asarray(ys)


def covered_at_start(ep: RawEpisode, *, until_s: float = 4.0, dark_ratio: float = 0.45) -> dict:
    """Which stream was blacked out at the start, if any. Reported as a ratio against that camera's own later level,
    because the two wrist cameras are not equally bright to begin with.

    AUTO-EXPOSURE DEFEATS THIS. Cover the lens and the camera opens up until the mean brightness is back near where
    it started: three takes on 2026-09-15 read 82%, 53% and 100% with the lens properly covered, and a live bench
    test read 99-100% until exposure was frozen, at which point the covered stream went to 0%. So a ratio that never
    reaches `dark_ratio` does NOT mean nothing was covered -- with auto exposure on, it cannot. Freeze exposure on
    the wrist cameras for the first seconds of the take, or read the result as void rather than as a verdict."""
    out: dict = {}
    for stream in ep.frames:
        t, y = stream_brightness(ep, stream, until_s=until_s)
        if len(t) < 10: continue
        late = float(np.median(y[t > t.max() * 0.6])) or 1.0
        early = float(np.median(y[t < min(2.0, t.max() * 0.4)]))
        out[stream] = dict(early=round(early, 2), late=round(late, 2), ratio=round(early / late, 3),
                           covered=bool(early < late * dark_ratio))
    return out


def impulses(t_ns: np.ndarray, signal: np.ndarray, *, min_gap_s: float = 0.6, k: float = 6.0,
             min_ratio: float = 2.0) -> np.ndarray:
    """Times of sharp transients: points that are both `k` robust deviations above the median AND at least `min_ratio`
    times it, thinned so each burst contributes once.

    The deviation test alone is not enough. A signal that is genuinely flat has a tiny MAD, so `k` deviations is also
    tiny: head-camera frame difference sitting at 0.59 with 0.70 peaks produced 39 "transients" in 30 s, and matching a
    handful of real accelerometer taps against one candidate every 0.8 s finds a partner for everything by chance. A
    real tap in that same signal reached 6.5, eleven times the floor. Requiring both tests keeps the second kind and
    discards the first."""
    s = np.asarray(signal, np.float64)
    if len(s) < 5: return np.empty(0, np.int64)
    med = np.median(s)
    mad = np.median(np.abs(s - med)) or (s.std() or 1.0)
    hot = np.flatnonzero((s > med + k * 1.4826 * mad) & (s > med * min_ratio))
    if not len(hot): return np.empty(0, np.int64)
    keep, last = [], -np.inf
    for i in hot:
        if (t_ns[i] - last) / 1e9 >= min_gap_s:
            keep.append(i); last = t_ns[i]
        elif s[i] > s[keep[-1]]:
            keep[-1] = i                                     # a later, stronger sample in the same burst
    return t_ns[np.asarray(keep, int)]


def match_impulses(a_ns: np.ndarray, b_ns: np.ndarray, *, max_dt_ms: float = 250.0, max_spread_ms: float = 60.0) -> dict:
    """Pair each impulse in `a` with its nearest in `b` and report the offset between them (b - a).

    A median is only worth quoting if the pairs agree. Chance matches scatter: one run produced deltas of 136, 198,
    -94 and -102 ms and a median of +21 ms that meant nothing. The spread is therefore reported alongside, and a
    verdict is withheld when it is wide."""
    if not len(a_ns) or not len(b_ns): return dict(pairs=0, reason="no impulses on one side")
    d = []
    for t in a_ns:
        j = int(np.argmin(np.abs(b_ns - t)))
        dt = (b_ns[j] - t) / 1e6
        if abs(dt) <= max_dt_ms: d.append(float(dt))
    if not d: return dict(pairs=0, reason=f"no impulse pair within {max_dt_ms:.0f} ms")
    spread = float(np.percentile(d, 95) - np.percentile(d, 5)) if len(d) > 1 else 0.0
    out = dict(pairs=len(d), median_ms=round(float(np.median(d)), 2), spread_ms=round(spread, 2),
               deltas_ms=[round(x, 1) for x in d], agrees=bool(len(d) >= 3 and spread <= max_spread_ms))
    if not out["agrees"]:
        out["reason"] = (f"the {len(d)} pairs disagree by {spread:.0f} ms, past {max_spread_ms:.0f} — these are chance "
                         f"matches, not the same events seen twice" if len(d) >= 3 else f"only {len(d)} pair(s)")
    return out


def head_motion(ep: RawEpisode, stream: str, *, downscale: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """(capture_ns, mean absolute frame difference). A wrist striking the table is a transient the head camera sees
    even though it never moves itself."""
    ts, e, prev = [], [], None
    for _vf, _i, cap_ns, img in ep.iter_frames(stream, downscale=downscale, gray=True):
        f = img.astype(np.float32)
        if prev is not None:
            ts.append(cap_ns); e.append(float(np.abs(f - prev).mean()))
        prev = f
    return np.asarray(ts, np.int64), np.asarray(e)


def accel_transient(imu) -> tuple[np.ndarray, np.ndarray]:
    """(host_ns, |accel| with gravity removed). Gravity is a constant offset here, not a signal; what a tap adds is the
    departure from it."""
    mag = np.linalg.norm(imu.accel, axis=1)
    return imu.host_ns, np.abs(mag - np.median(mag))


def imu_at(imu, t_ns, *, max_extrapolation_ms: float = 10.0) -> dict | None:
    """Gyro and accel linearly interpolated to `t_ns`, or None if that instant is outside the record.

    At 200 Hz the nearest sample is within 2.5 ms, which is already small -- but gyro and accel are smooth, so
    interpolating costs nothing and removes a quantisation that would otherwise be carried into every downstream
    calculation for no reason."""
    t = np.asarray(imu.host_ns, np.int64)
    if len(t) < 2: return None
    if t_ns < t[0] - max_extrapolation_ms * 1e6 or t_ns > t[-1] + max_extrapolation_ms * 1e6: return None
    j = int(np.clip(np.searchsorted(t, t_ns), 1, len(t) - 1))
    t0, t1 = t[j - 1], t[j]
    w = 0.0 if t1 == t0 else float((t_ns - t0) / (t1 - t0))
    w = min(max(w, 0.0), 1.0)
    lerp = lambda a: (a[j - 1] * (1 - w) + a[j] * w)
    return dict(gyro=lerp(imu.gyro), accel=lerp(imu.accel), nearest_ms=round(float(min(abs(t_ns - t0), abs(t1 - t_ns))) / 1e6, 3))


def lookup_deltas(ep: RawEpisode, head: str, other: str, *, window_s: float = 5.0) -> dict:
    """What the per-frame nearest-wrist-frame lookup actually costs, and whether it stays put.

    It does not: three cameras free-running at 30.0260, 30.0032 and 30.0047 Hz wind their relative phase through most
    of a frame period over half a minute, so this is reported per window rather than as one number to store. A single
    sample of it looks like a calibration and is wrong by up to a frame when applied later."""
    h = np.asarray(ep.frames[head].capture_ns, np.int64)
    w = np.asarray(ep.frames[other].capture_ns, np.int64)
    if len(h) < 2 or len(w) < 2: return dict(error="one of the streams has no frames")
    j = np.clip(np.searchsorted(w, h), 1, len(w) - 1)
    d_ms = np.where(np.abs(w[j] - h) < np.abs(w[j - 1] - h), w[j] - h, w[j - 1] - h) / 1e6
    t_s = (h - h[0]) / 1e9
    windows = []
    for lo in np.arange(0, t_s.max(), window_s):
        m = (t_s >= lo) & (t_s < lo + window_s)
        if m.sum(): windows.append(dict(t_s=float(lo), median_ms=round(float(np.median(d_ms[m])), 2)))
    per = float(np.diff(w).mean()) / 1e6
    return dict(pair=f"{head}->{other}", abs_median_ms=round(float(np.median(np.abs(d_ms))), 2),
                abs_p95_ms=round(float(np.percentile(np.abs(d_ms), 95)), 2),
                bound_ms=round(per / 2, 2), frame_period_ms=round(per, 4),
                windows=windows,
                # numpy 2 removed ndarray.ptp, so this is np.ptp: signs that span both means the phase wrapped
                circulates=bool(len(windows) > 2 and np.ptp(np.sign([x["median_ms"] for x in windows])) > 1))


MASTER_PREFERENCE = ("head", "head_depth")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--backend", default=None, help="pose backend whose wrist camera poses to use")
    ap.add_argument("--head", default=None, help="head stream name; default: whichever of head / head_depth the episode has")
    ap.add_argument("--max-offset-ms", type=float, default=100.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    ep_path = Path(a.episode).expanduser()
    ep = RawEpisode.load(ep_path)
    # The head camera changed from the Orbbec to a C922 on 2026-09-15, and with it the stream name, so a default of
    # either name is wrong for half the episodes on disk. Take it from the episode, in the same preference order
    # inspect_episode uses for the master timeline.
    if a.head is None:
        a.head = next((n for n in MASTER_PREFERENCE if n in ep.frames and len(ep.frames[n])), MASTER_PREFERENCE[0])
    report: dict = dict(episode=ep_path.name, head=a.head)

    print("== 1. camera identity (which lens was covered at the start)")
    cov = covered_at_start(ep)
    report["lens_cover"] = cov
    for s, d in cov.items():
        print(f"   {s:12s} first 2 s at {d['ratio']:.0%} of its own later brightness" + ("   <- COVERED" if d["covered"] else ""))
    covered = [s for s, d in cov.items() if d["covered"]]
    if len(covered) == 1:
        print(f"   -> the stream recorded as {covered[0]} is the wrist whose lens was covered")
    elif not covered:
        print("   -> VOID, not negative: no stream reached the dark threshold, which is exactly what a camera "
              "compensating with auto-exposure looks like (see covered_at_start).")
        print("      Freeze wrist exposure for the first seconds of the take, or settle identity on the bench by "
              "covering one unit's lens and tapping that same unit.")
    else:
        print(f"   -> more than one stream darkened ({covered}); inconclusive")

    print("\n== 2. wrist camera <-> wrist IMU (gyro correlation, existing estimator)")
    from ..config import DEFAULT_CONFIG_DIR, load_pose_cfg
    from .camera_imu_offset import offset_for_episode
    cfg = load_pose_cfg(str(DEFAULT_CONFIG_DIR / "pose.yaml"))
    backend = a.backend or cfg.backend
    cam_imu: dict = {}
    for side in sorted(ep.imu):
        try:
            r = offset_for_episode(ep_path, side, backend, max_offset_ms=a.max_offset_ms)
        except FileNotFoundError:
            print(f"   {side}: no processed poses — run tools.pose_process {ep_path} --backend {backend} first")
            continue
        cam_imu[side] = r
        print(f"   {side:6s} offset {r.get('offset_ms')} ms   confidence {r.get('confidence', 0):.2f}   {r.get('reason','')}")
    report["camera_imu"] = cam_imu

    print("\n== 3. head <-> wrist camera (frame timestamps already recorded)")
    val = validate_episode(ep_path, expect_imus=list(ep.imu), decode_video=False)
    skew = val.get("stream_skew", {})
    report["stream_skew"] = skew
    for pair, d in skew.items():
        print(f"   {pair:28s} start {d['start_offset_ms']:+7.2f} ms   median nearest {d['median_nn_ms']:.2f} ms   p95 {d['p95_nn_ms']:.2f}")

    print("\n== 4. head -> wrist lookup (no constant to store; see TEMPORAL_SYNC.md)")
    lookups = {}
    for side in sorted(ep.imu):
        wrist = f"{side}_wrist"
        if wrist not in ep.frames: continue
        lk = lookup_deltas(ep, a.head, wrist)
        lookups[side] = lk
        if "error" in lk: print(f"   {side}: {lk['error']}"); continue
        print(f"   {a.head} -> {wrist}: |delta| median {lk['abs_median_ms']:.2f} ms, p95 {lk['abs_p95_ms']:.2f} ms  "
              f"(structural bound {lk['bound_ms']:.2f} ms = half a {lk['frame_period_ms']:.2f} ms frame)")
        print("      per 5 s: " + "  ".join(f"{x['median_ms']:+.1f}" for x in lk["windows"]) +
              ("   <- circulates, so there is no fixed offset to apply" if lk["circulates"] else ""))
    report["lookup"] = lookups

    print("\n== 5. table taps (physical check, independent of the timestamps)")
    if a.head not in ep.frames:
        print(f"   no {a.head} stream")
    else:
        t_head, e_head = head_motion(ep, a.head)
        head_taps = impulses(t_head, e_head)
        print(f"   head video transients: {len(head_taps)}")
        taps: dict = {}
        for side in sorted(ep.imu):
            t_imu, e_imu = accel_transient(ep.imu[side])
            imu_taps = impulses(t_imu, e_imu)
            m = match_impulses(imu_taps, head_taps)
            taps[side] = dict(imu_impulses=len(imu_taps), **m)
            if m.get("agrees"):
                print(f"   {side:6s} {len(imu_taps)} accel impulses, {m['pairs']} matched   "
                      f"median {m['median_ms']:+.2f} ms  spread {m['spread_ms']:.1f} ms  <- usable")
            elif m.get("pairs"):
                print(f"   {side:6s} {len(imu_taps)} accel impulses, {m['pairs']} matched, NO VERDICT: {m['reason']}")
            else:
                print(f"   {side:6s} {len(imu_taps)} accel impulses — {m.get('reason')}")
        report["taps"] = taps

    print("\n== 6. look at it")
    print(f"   python -m handumi_collector.tools.inspect_episode --episode {ep_path} --replay")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=float) + "\n")
        print(f"   wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
