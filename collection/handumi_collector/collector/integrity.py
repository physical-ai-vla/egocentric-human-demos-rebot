"""Raw-episode integrity validation (used by the HARDWARE CHECK button, the REVIEW quick-QA and tools.inspect_episode).
Checks what can be checked without any pose processing: every video decodes to exactly its frame_meta count, timestamps are
monotonic per stream, sensor channels present for configured devices, no NaN in gripper normalized when calibrated,
no episode id reuse. Returns a dict with `ok`, `problems`, and per-stream numbers."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .mcap_writer import read_messages


MIN_GRIPPER_TRAVEL = 0.05      # normalized aperture; the working jaw spans ~0.64 over an episode


def validate_episode(ep: Path, *, expect_streams: list[str] | None = None, expect_grippers: list[str] | None = None,
                     expect_imus: list[str] | None = None, expect_imu_serials: dict[str, str] | None = None,
                     imu_rate_hz: float | None = None, decode_video: bool = True, min_fps: float = 20.0) -> dict:
    ep = Path(ep); problems: list[str] = []; info: dict = dict(episode=ep.name)
    if (ep / ".incomplete").exists(): problems.append("episode still marked .incomplete")
    mp = ep / "episode_meta.json"
    if not mp.exists(): return dict(ok=False, problems=["episode_meta.json missing"], **info)
    meta = json.loads(mp.read_text()); info["duration_s"] = meta.get("duration_s")
    if not (ep / "sensors.mcap").exists(): return dict(ok=False, problems=["sensors.mcap missing"], **info)
    msgs = list(read_messages(ep / "sensors.mcap")); topics = Counter(t for t, _, _ in msgs)
    per_topic_t: dict[str, list[int]] = {}
    for t, lt, _ in msgs: per_topic_t.setdefault(t, []).append(lt)
    for t, ts in per_topic_t.items():
        if any(b < a for a, b in zip(ts, ts[1:])): problems.append(f"{t}: non-monotonic log_time")
    streams = expect_streams or list(meta.get("streams", {}).keys())
    info["streams"] = {}
    for st in streams:
        fm = [m for t, _, m in msgs if t == f"/{st}/frame_meta"]
        vf = [m["video_frame"] for m in fm]
        d = dict(frame_meta=len(fm), skipped=sum(m.get("skipped_before", 0) for m in fm))
        if vf != list(range(len(vf))): problems.append(f"{st}: video_frame numbering not contiguous")
        cap = [m["capture_ns"] for m in fm]
        if any(b <= a for a, b in zip(cap, cap[1:])): problems.append(f"{st}: capture_ns not strictly increasing")
        if len(cap) > 2:
            d["fps"] = round((len(cap) - 1) / ((cap[-1] - cap[0]) / 1e9), 2)
            if d["fps"] < min_fps: problems.append(f"{st}: measured fps {d['fps']} < {min_fps}")
            gaps = [(b - a) / 1e6 for a, b in zip(cap, cap[1:])]; d["max_gap_ms"] = round(max(gaps), 1)
            # VIO cares about interval *stability*, not just the worst gap: a backend integrates IMU between frames,
            # so jitter against the nominal period is the number that matters (P0-4).
            import statistics as _st
            nominal = 1000.0 / d["fps"]
            d["median_gap_ms"] = round(_st.median(gaps), 2)
            d["p95_jitter_ms"] = round(sorted(abs(g - nominal) for g in gaps)[max(int(0.95 * len(gaps)) - 1, 0)], 2)
        vp = ep / f"{st}.mp4"
        if not vp.exists(): problems.append(f"{st}.mp4 missing")
        elif decode_video:
            try:
                import av
                with av.open(str(vp)) as c: n = sum(1 for _ in c.decode(video=0))
                d["video_frames"] = n
                if n != len(fm): problems.append(f"{st}: video has {n} frames but frame_meta has {len(fm)}")
            except Exception as exc: problems.append(f"{st}.mp4 not decodable: {exc}")
        if len(fm) == 0: problems.append(f"{st}: no frames")
        d["_cap"] = cap
        info["streams"][st] = d
    # Cross-stream timestamp skew: VIO and the 30 Hz dataset timeline both assume the cameras are talking about the
    # same instant. A constant offset is a fixable calibration; a drifting one is a clock problem. Reported per pair as
    # the start offset and the median nearest-neighbour distance (which is what a resampler actually pays).
    import itertools as _it
    caps = {n: d.pop("_cap", []) for n, d in info["streams"].items()}
    skew = {}
    for a, b in _it.combinations([n for n, c in caps.items() if len(c) > 2], 2):
        ca, cb = np.asarray(caps[a], np.int64), np.asarray(caps[b], np.int64)
        j = np.searchsorted(cb, ca)
        j = np.clip(j, 1, len(cb) - 1)
        nn = np.minimum(np.abs(ca - cb[j - 1]), np.abs(ca - cb[j]))
        skew[f"{a}~{b}"] = dict(start_offset_ms=round((ca[0] - cb[0]) / 1e6, 2),
                                median_nn_ms=round(float(np.median(nn)) / 1e6, 2),
                                p95_nn_ms=round(float(np.percentile(nn, 95)) / 1e6, 2))
    if skew: info["stream_skew"] = skew

    # Depth frames are written as separate PNGs, outside the mp4 the video check covers, so nothing looked at them at
    # all: a truncated write, a size change mid-episode or a frame that never landed would survive QA untouched.
    # Structural damage is a failure; how much of the map is empty is a number to record and watch, not yet a gate.
    for st in streams:
        ddir = ep / f"{st}_depth"
        if not ddir.exists(): continue
        pngs = sorted(ddir.glob("[0-9]*.png"))
        dd: dict = dict(files=len(pngs), has_intrinsics=(ddir / "intrinsics.json").exists())
        n_rgb = info["streams"].get(st, {}).get("frame_meta")
        if n_rgb is not None and len(pngs) != n_rgb:
            problems.append(f"{st}_depth: {len(pngs)} depth frames against {n_rgb} colour frames")
        # Only real cameras are held to this: mock backends never report intrinsics, and failing every synthetic episode
        # would just train people to ignore the check. On real hardware it is fatal -- a guessed focal length is a silent
        # metric-scale error in every pose downstream, and check_rgbd_ready --live exists to catch it before recording.
        synthetic = bool(meta.get("synthetic")) or "mock" in str(meta.get("hardware_profile", ""))
        if not dd["has_intrinsics"] and not synthetic:
            problems.append(f"{st}_depth: intrinsics.json missing — the camera did not report intrinsics at open()")
        idx = [int(q.stem) for q in pngs]
        if idx != list(range(len(idx))): problems.append(f"{st}_depth: frame numbering not contiguous")
        if pngs:
            import cv2
            shapes, zero, bad = set(), [], 0
            for q in (pngs[:: max(1, len(pngs) // 40)] + [pngs[-1]]):          # sample ~40 frames plus the last one
                img = cv2.imread(str(q), cv2.IMREAD_UNCHANGED)
                if img is None or img.dtype != np.uint16 or img.ndim != 2:
                    bad += 1; continue
                shapes.add(img.shape); zero.append(float((img == 0).mean()))
            if bad: problems.append(f"{st}_depth: {bad} of the sampled PNGs are unreadable or not 16-bit single channel")
            if len(shapes) > 1: problems.append(f"{st}_depth: frame size changes mid-episode {sorted(shapes)}")
            if shapes: dd["size"] = list(sorted(shapes)[0])
            if zero:
                dd["zero_depth_fraction"] = dict(median=round(float(np.median(zero)), 4), max=round(max(zero), 4))
        info.setdefault("depth", {})[st] = dd

    info["grippers"] = {}
    for side in (expect_grippers or []):
        g = [m for t, _, m in msgs if t == f"/{side}/gripper"]
        d = dict(samples=len(g), nan_normalized=sum(1 for m in g if m.get("normalized") is None))
        # A jaw that never moves is not a grasp signal, and it does not look like a fault: the encoder answers at a
        # healthy rate and reports a perfectly steady number. The 2026-09-15 pilot recorded five episodes with the
        # left jaw pinned at exactly 1.000 because its bearing had broken, and every gate passed. Travel is what
        # separates "closed the whole time" from "not measuring" -- a real episode opens and closes repeatedly.
        vals = [float(m["normalized"]) for m in g if m.get("normalized") is not None and m["normalized"] == m["normalized"]]
        if vals:
            travel = float(np.percentile(vals, 95) - np.percentile(vals, 5))
            d.update(travel=round(travel, 4), moved=bool(travel >= MIN_GRIPPER_TRAVEL),
                     median=round(float(np.median(vals)), 4))
            if travel < MIN_GRIPPER_TRAVEL:
                problems.append(f"gripper {side}: jaw never moved (travel {travel:.3f} over {len(vals)} samples, "
                                f"held at {np.median(vals):.3f}) — this episode carries no grasp signal for {side}")
        info["grippers"][side] = d
        if not g: problems.append(f"gripper {side}: no samples")
    # IMU health, not just presence. An episode can carry a full-rate, gap-free-looking IMU stream that is 18 % spurious
    # data-ready edges, or one that stopped halfway, and counting samples sees neither. Same statistics and the same
    # thresholds tools.validate_session applies to a bench session, so a recording and a bring-up log are judged alike.
    info["imus"] = {}
    for side in (expect_imus or []):
        im = [m for t, _, m in msgs if t == f"/{side}/imu"]
        d: dict = dict(samples=len(im))
        if not im:
            problems.append(f"imu {side}: no samples"); info["imus"][side] = d; continue
        from ..devices.imu_stats import stream_stats
        seq = np.array([m["seq"] for m in im], np.int64)
        dev = np.array([m["device_timestamp_us"] for m in im], np.int64)
        host = np.array([m["host_receive_ns"] for m in im], np.int64)
        acc = np.array([[m["ax"], m["ay"], m["az"]] for m in im], np.float64)
        gyr = np.array([[m["gx"], m["gy"], m["gz"]] for m in im], np.float64)
        rate = float(meta.get("imu_rate_hz") or imu_rate_hz or 200.0)
        st = stream_stats(seq=seq, device_us=dev, host_ns=host, accel=acc, gyro=gyr, expected_rate_hz=rate)
        d.update(rate_hz=st.get("rate_hz_mean"), drop_rate=st["drops"]["drop_rate"],
                 short_intervals=st.get("short_intervals", {}).get("count", 0),
                 p99_interval_ms=st.get("interval_ms", {}).get("p99"),
                 max_interval_ms=st.get("interval_ms", {}).get("max"),
                 clock_drift_ppm=(st.get("clock_fit") or {}).get("drift_ppm"))
        nominal = 1000.0 / rate
        if st["n_samples"] > 1:
            if not (rate * 0.975 <= st["rate_hz_mean"] <= rate * 1.025):
                problems.append(f"imu {side}: {st['rate_hz_mean']:.1f} Hz, expected {rate:g} +/-2.5 %")
            if st["drops"]["drop_rate"] > 0.001:
                problems.append(f"imu {side}: {st['drops']['lost_samples']} samples lost ({st['drops']['drop_rate']*100:.3f} %)")
            if st["short_intervals"]["count"]:
                problems.append(f"imu {side}: {st['short_intervals']['count']} spurious data-ready edges "
                                f"({st['short_intervals']['fraction']*100:.2f} %) — check the INT1 routing")
            if st["interval_ms"]["max"] > nominal * 20:
                problems.append(f"imu {side}: {st['interval_ms']['max']:.0f} ms gap in the stream")
            span = (dev[-1] - dev[0]) / 1e6
            ep_dur = float(meta.get("duration_s") or 0)
            if ep_dur > 0:
                d["coverage"] = round(span / ep_dur, 4)
                if span < ep_dur * 0.95:
                    problems.append(f"imu {side}: covers {span:.1f} s of a {ep_dur:.1f} s episode — the stream stopped early")
        info["imus"][side] = d

    # Which board was on which wrist. The serials are recorded in the mcap at episode start; nothing compared them until
    # now, so a swapped pair would silently mislabel every episode with nothing in the data to catch it afterwards.
    start = next((m for t, _, m in msgs if t == "/system/sync" and m.get("kind") == "episode_start"), None)
    recorded = {n.replace("imu_", ""): (dv or {}).get("serial_number")
                for n, dv in ((start or {}).get("devices") or {}).items() if n.startswith("imu_")}
    if recorded:
        info["imu_serials"] = recorded
        for side, want in (expect_imu_serials or {}).items():
            got = recorded.get(side)
            if want and got and want not in str(got):
                problems.append(f"imu {side}: recorded serial {got} is not the configured {want} — LEFT/RIGHT may be swapped")
            if want and not got:
                problems.append(f"imu {side}: no serial recorded, identity cannot be verified after the fact")
    elif expect_imus:
        problems.append("no IMU identity recorded at episode start — which board was on which wrist is unverifiable")
    info["events"] = meta.get("event_kinds", []); info["hw_event_in_episode"] = meta.get("hw_event_in_episode")
    return dict(ok=not problems, problems=problems, **info)


def gripper_integrity(side: str, st: dict | None, ticks_closed, ticks_open, *, duration_s: float | None = None, min_rate_hz: float = 20.0,
                      margin_ticks: int = 300) -> tuple[str, str | None]:
    """Raw-integrity verdict for one jaw from the recorder's running stats. Returns (level, note); level in PASS|REVIEW|REJECT.

    REJECT  frozen reading (the value never changed over the episode) -- the servo stopped answering, or
            the whole episode's raw range lies outside the calibrated [open, closed] tick span (+margin) -- the servo's
            unwrapped counter moved with a power cycle and the calibration no longer applies (normalized is meaningless).
    REJECT  jaw never travelled (normalized span < MIN_GRIPPER_TRAVEL). Protocol (user, 2026-09-16): both hands are used in
            every episode, so "the left jaw did not move" is a gripper signal anomaly, never an operator choice. On the
            2026-09-16 raw re-audit the left jaw's raw range was 90-220 ticks (a real grasp is 300-500) in the very
            episodes that read travel 0 -- telemetry, not behaviour."""
    min_samples = max(5, int(min_rate_hz * duration_s)) if duration_s else 5          # the stream must exist at a sane rate, however short the take
    if not st or st.get("n", 0) < min_samples: return "REJECT", f"gripper {side}: only {0 if not st else st.get('n', 0)} samples in {duration_s or 0:.1f} s"
    if st["raw_changes"] == 0: return "REJECT", f"gripper {side}: reading frozen at {st['raw_min']} for {st['n']} samples (servo not answering?)"
    if ticks_closed is not None and ticks_open is not None:
        lo, hi = min(ticks_closed, ticks_open) - margin_ticks, max(ticks_closed, ticks_open) + margin_ticks
        if st["raw_max"] < lo or st["raw_min"] > hi:
            return "REJECT", (f"gripper {side}: raw {st['raw_min']}..{st['raw_max']} is entirely outside the calibrated span "
                              f"{min(ticks_closed, ticks_open)}..{max(ticks_closed, ticks_open)} -- servo counter moved (power cycle?), RECALIBRATE this gripper")
    if st["norm_max"] >= st["norm_min"] and (st["norm_max"] - st["norm_min"]) < MIN_GRIPPER_TRAVEL:
        return "REJECT", (f"gripper {side}: jaw travel {st['norm_max'] - st['norm_min']:.3f} < {MIN_GRIPPER_TRAVEL} over the episode "
                          f"(raw {st['raw_min']}..{st['raw_max']}) -- both hands are used every episode, so this is a signal anomaly")
    return "PASS", None


def preliminary_qa(meta: dict, events: list[dict]) -> tuple[str, list[str]]:
    """Quick verdict shown in REVIEW before KEEP: PASS / REVIEW from what the recorder already knows (no video decode)."""
    notes = []
    kinds = Counter(e["kind"] for e in events)
    for k in ("device_error", "camera_drop", "timestamp_jump", "gripper_gap", "imu_timeout", "recorder_error"):
        if kinds.get(k): notes.append(f"{kinds[k]}x {k}")
    st = meta.get("streams", {})
    fps = {n: s.get("fps_measured", 0) for n, s in st.items()}
    if len(set(round(s.get("frames", 0) / max(s.get("duration_s", 1e-9), 1e-9)) for s in st.values())) > 1: pass
    for n, f in fps.items():
        # `if f and f < 25` skipped zero, because zero is falsy -- so a stream that produced no frames at all, which is
        # the worst outcome there is, was the single case that could not be flagged. A take with two empty cameras came
        # back PASS and was kept.
        if not st[n].get("frames"): notes.append(f"{n} RECORDED NO FRAMES")
        elif f < 25: notes.append(f"{n} fps {f}")
    for n, s in st.items():
        if s.get("encoder_error"): notes.append(f"{n} encoder error")
    return ("REVIEW" if notes else "PASS"), notes
