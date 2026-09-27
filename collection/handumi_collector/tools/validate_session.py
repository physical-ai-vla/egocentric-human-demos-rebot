"""PASS / FAIL gate for one IMU recording — a bring-up session directory or a recorded raw episode.

    python -m handumi_collector.tools.validate_session data/imu_right_20260914_120000
    python -m handumi_collector.tools.validate_session ~/handumi_sessions/<session>/episode_000001 --side right

Both containers carry the same schema (`pose.episode_io.ImuArrays`), so the gates are applied to one stats block from
`devices.imu_stats.stream_stats` either way. Gates are deliberately blunt and printed with the measured value next to the
limit: this is the thing that says a flashed board is fit to record with, not a diagnosis tool."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from ..devices.imu_stats import stream_stats

RATE_TOL = 0.025          # +/-2.5 % of the expected ODR (200 Hz -> 195..205)
MAX_DROP_RATE = 0.001     # 0.1 %
MAX_DOUBLE_TRIGGER = 1e-4 # 0.01 % of intervals; a bad data-ready line produces percent, not parts per million
MAX_P99_FACTOR = 1.4      # p99 interval <= 1.4 x nominal (200 Hz -> 7.0 ms)
MIN_DURATION_S = 5.0
MIN_COVERAGE = 0.95       # device-time span / session wall time: catches a stream that dies partway through
MAX_GAP_FACTOR = 20.0     # longest interval, in nominal periods: catches a stall inside the record
ACCEL_NORM_RANGE = (0.8, 1.2)     # g, valid only for a mostly-static recording
MAX_SATURATION = 0.9      # |a| should stay below 90 % of full scale


def rate_from_hardware_profile(hardware_version: str | None, side: str) -> tuple[float | None, str]:
    """Episodes do not record the ODR, so fall back to the profile they name — never to a bare constant, which would
    silently keep gating at 200 Hz after the firmware moves to 500."""
    if not hardware_version: return None, ""
    try:
        from ..config import DEFAULT_CONFIG_DIR, load_config
        cfg = load_config(str(DEFAULT_CONFIG_DIR), hardware=hardware_version)
        for imu in cfg.hardware.imus:
            if imu.side == side: return float(imu.rate_hz), f"hardware profile {hardware_version}"
    except Exception:
        pass
    return None, ""


def load_session(path: Path, side: str | None) -> tuple[dict, dict]:
    """Returns (stats, provenance). A session directory is read from imu.csv; an episode from its mcap."""
    if (path / "imu.csv").exists():
        meta = json.loads((path / "metadata.json").read_text()) if (path / "metadata.json").exists() else {}
        # A 3 h dual run is 2.2 M rows a side; csv.DictReader would build that many dicts before any of it is needed.
        try:
            import pandas as pd
            df = pd.read_csv(path / "imu.csv")
            if side: df = df[df["side"] == side]
            if df.empty: raise SystemExit(f"{path}/imu.csv: no rows" + (f" for side={side}" if side else ""))
            g = lambda k: df[k].to_numpy(np.float64)
            found_side = str(df["side"].iloc[0])
        except ImportError:
            import csv as _csv
            with open(path / "imu.csv", newline="") as fh:
                rows = [r for r in _csv.DictReader(fh) if not side or r["side"] == side]
            if not rows: raise SystemExit(f"{path}/imu.csv: no rows" + (f" for side={side}" if side else ""))
            g = lambda k: np.array([float(r[k]) for r in rows])
            found_side = rows[0]["side"]
        stats = stream_stats(seq=g("seq"), device_us=g("device_timestamp_us"), host_ns=g("host_receive_ns"),
                             accel=np.stack([g("ax"), g("ay"), g("az")], 1), gyro=np.stack([g("gx"), g("gy"), g("gz")], 1),
                             expected_rate_hz=meta.get("stats", {}).get("expected_rate_hz", 200.0),
                             accel_fs_g=meta.get("accel_fs_g"),
                             crc_errors=meta.get("stats", {}).get("crc_errors", 0), resyncs=meta.get("stats", {}).get("resyncs", 0))
        return stats, dict(kind="session", meta=meta, side=side or found_side, drops_detectable=meta.get("drops_detectable", True))
    if (path / "sensors.mcap").exists():
        from ..pose.episode_io import RawEpisode
        ep = RawEpisode.load(path)
        if not ep.imu: raise SystemExit(f"{path}: episode has no IMU streams")
        s = side or sorted(ep.imu)[0]
        if s not in ep.imu: raise SystemExit(f"{path}: no IMU for side={s} (have {sorted(ep.imu)})")
        imu = ep.imu[s]
        if "imu_rate_hz" in ep.meta:
            rate, src = float(ep.meta["imu_rate_hz"]), "episode_meta.json"
        else:
            rate, src = rate_from_hardware_profile(ep.meta.get("hardware_version"), s)
        if rate is None:
            rate, src = 200.0, "DEFAULT — neither the episode nor a hardware profile stated the ODR"
        stats = stream_stats(seq=imu.seq, device_us=imu.device_us, host_ns=imu.host_ns, accel=imu.accel, gyro=imu.gyro,
                             expected_rate_hz=rate, accel_fs_g=ep.meta.get("accel_fs_g"))
        synthetic = bool(ep.meta.get("synthetic")) or "mock" in str(ep.meta.get("hardware_profile", ""))
        return stats, dict(kind="episode", meta=ep.meta, side=s, drops_detectable=True, rate_source=src, synthetic=synthetic)
    raise SystemExit(f"{path}: neither imu.csv nor sensors.mcap — not an IMU session or raw episode")


def gates(stats: dict, prov: dict, *, static: bool, allow_synthetic: bool = False) -> list[tuple[str, bool, str]]:
    """(name, ok, detail) in report order. `static` enables the gravity check, which only means anything at rest."""
    out: list[tuple[str, bool, str]] = []
    if prov.get("synthetic") and not allow_synthetic:
        out.append(("not_synthetic", False, "mock/synthetic devices — this proves nothing about hardware (pass --allow-synthetic to gate it anyway)"))
    n = stats.get("n_samples", 0)
    if n < 2:
        return [("samples", False, f"{n} samples")]
    exp = stats["expected_rate_hz"]; nominal_ms = 1000.0 / exp
    dur = stats["duration_s"]; rate = stats["rate_hz_mean"]
    lo, hi = exp * (1 - RATE_TOL), exp * (1 + RATE_TOL)
    out.append(("duration", dur >= MIN_DURATION_S, f"{dur:.1f} s (need >= {MIN_DURATION_S:.0f} s)"))
    # A stream that dies partway through still satisfies every other gate, computed as they are over whatever arrived:
    # a 60 s dual run where one wrist went silent at 30 s passed all nine. Compare what the device timeline covers
    # against how long the session actually ran.
    meta = prov.get("meta") or {}
    t0, t1 = meta.get("t_start_monotonic_ns"), meta.get("t_stop_monotonic_ns")
    if t0 is not None and t1 is not None and t1 > t0:      # `and t0` would skip the gate for a zero start stamp
        wall = (t1 - t0) / 1e9
        cov = dur / wall
        out.append(("coverage", cov >= MIN_COVERAGE,
                    f"{cov*100:.1f}% of the {wall:.1f} s session ({dur:.1f} s of samples) — a side that stops early fails here"))
    # The device keeps sampling through a host sleep and the host's monotonic clock does not, so the two disagree by
    # exactly the time the machine was away. An overnight run showed 436 minutes of device time inside 46 minutes of
    # host time across 28 naps, and every other gate was happy with it.
    wall = meta.get("wall_duration_s")
    if wall and dur > 0:
        ratio = dur / wall
        out.append(("clock_sanity", 0.97 <= ratio <= 1.03,
                    f"{dur:.0f} s of samples over {wall:.0f} s of wall clock ({ratio:.2f}x) — a device that outran the "
                    f"host means the machine slept mid-run"))
    gap = stats["interval_ms"]["max"]
    out.append(("max_gap", gap <= nominal_ms * MAX_GAP_FACTOR,
                f"longest gap {gap:.1f} ms (limit {nominal_ms * MAX_GAP_FACTOR:.0f} ms) — a stall inside the record"))
    out.append(("rate", lo <= rate <= hi, f"{rate:.2f} Hz (expect {lo:.1f}..{hi:.1f})"))
    p99 = stats["interval_ms"]["p99"]; p99_lim = nominal_ms * MAX_P99_FACTOR
    out.append(("jitter_p99", p99 <= p99_lim, f"{p99:.3f} ms (limit {p99_lim:.2f} ms, nominal {nominal_ms:.2f})"))
    si = stats.get("short_intervals")
    if si:
        # A count of zero is the right bar for a minute and the wrong one for two hours. A bad INT1 routing produced
        # 10.8 % of a 60 s record; a good rig over 7200 s produced three samples, 0.0002 %. Judge the rate, and say
        # whether they cluster -- crosstalk fires continuously, a transport hiccup does not.
        ok = si["fraction"] <= MAX_DOUBLE_TRIGGER and not si.get("clustered")
        why = f"{si['count']} samples arrived < {si['threshold_ms']:.2f} ms after the previous one " \
              f"({si['fraction']*100:.4f}%, limit {MAX_DOUBLE_TRIGGER*100:.2f}%)"
        if si["count"] > 2: why += f", spread over {si['span_s']:.0f} s" + (" — CLUSTERED" if si.get("clustered") else "")
        out.append(("double_trigger", ok, why + " — spurious data-ready edges, not drops"))
    d = stats["drops"]
    if prov.get("drops_detectable", True):
        out.append(("drops", d["drop_rate"] <= MAX_DROP_RATE,
                    f"{d['lost_samples']} lost in {d['gap_events']} gaps = {d['drop_rate']*100:.4f}% (limit {MAX_DROP_RATE*100:.1f}%)"))
    else:
        out.append(("drops", False, "not detectable — bring-up text stream has no device sequence number"))
    out.append(("crc", stats["crc_errors"] == 0, f"{stats['crc_errors']} errors, {stats['resyncs']} resyncs"))
    a = stats["accel_g"]
    if static:
        ok = ACCEL_NORM_RANGE[0] < a["norm_median"] < ACCEL_NORM_RANGE[1]
        out.append(("gravity", ok, f"median |a| {a['norm_median']:.4f} g (expect {ACCEL_NORM_RANGE[0]}..{ACCEL_NORM_RANGE[1]} at rest)"))
    if "headroom_used" in a:
        out.append(("accel_headroom", a["headroom_used"] <= MAX_SATURATION,
                    f"peak |a| {a['abs_max']:.2f} g = {a['headroom_used']*100:.0f}% of +/-{a['full_scale']:.0f} g full scale"))
    # Fitness to record with and fitness to measure noise from are different questions. A session can fail the first --
    # 1901 samples lost to one transport event -- and still hold 87 minutes of unbroken data the second can use.
    seg = stats.get("longest_clean_segment")
    if seg and not seg["is_whole_session"]:
        out.append(("allan_eligible", seg["duration_s"] >= 1800,
                    f"longest unbroken run {seg['duration_s']/60:.1f} min starting at {seg['start_s']/60:.1f} min "
                    f"({seg['n_samples']:,} samples) — use this for noise analysis, not the whole session"))
    fit = stats.get("clock_fit") or {}
    if "drift_ppm" in fit:
        out.append(("clock_fit", fit["residual_std_ms"] < 5.0, f"drift {fit['drift_ppm']:+.1f} ppm, residual {fit['residual_std_ms']:.3f} ms"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="session directory (imu.csv) or raw episode directory (sensors.mcap)")
    ap.add_argument("--side", default=None, help="which side to check (default: the only / first one present)")
    ap.add_argument("--moving", action="store_true", help="the recording was not static — skip the 1 g gravity gate")
    ap.add_argument("--json", default=None, help="also write the stats block here")
    ap.add_argument("--allow-synthetic", action="store_true", help="gate a mock recording anyway (it proves nothing about hardware)")
    a = ap.parse_args(argv)
    path = Path(a.path).expanduser()
    stats, prov = load_session(path, a.side)
    rows = gates(stats, prov, static=not a.moving, allow_synthetic=a.allow_synthetic)
    print(f"{path}  [{prov['kind']}, side={prov['side']}]")
    fw = (prov.get("meta") or {}).get("firmware")
    if fw:
        boot = fw.get("cumulative_since_boot") or {}
        print(f"  firmware {fw.get('version')}  device rate {fw.get('device_rate_hz')} Hz  device-side dropped {fw.get('samples_dropped')} this session"
              + (f" ({boot.get('dropped')} since boot, mostly while no host was reading)" if boot else ""))
        if fw.get("ring_capacity"):
            print(f"  tx ring high-water {fw['ring_high_water']}/{fw['ring_capacity']} packets "
                  f"({fw['ring_high_water_percent']:.1f}%) — how close the device came to dropping")
    if prov.get("rate_source"): print(f"  expected rate {stats['expected_rate_hz']:g} Hz from {prov['rate_source']}")
    for name, ok, detail in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:15s} {detail}")
    failed = [n for n, ok, _ in rows if not ok]
    print(("PASS" if not failed else "FAIL: " + ", ".join(failed)))
    if a.json: Path(a.json).write_text(json.dumps(stats, indent=1) + "\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
