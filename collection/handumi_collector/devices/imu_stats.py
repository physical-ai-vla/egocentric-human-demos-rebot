"""Stream-health statistics for one IMU record — rate, jitter, sequence drops, CRC, accel headroom.

One implementation for every container the stream lands in: the live logger writes it to `metadata.json`,
`tools/validate_session.py` re-derives it from `imu.csv` or from a raw episode's mcap, and both then apply the same gates.
Timing comes from the device clock (`device_timestamp_us`) because `host_receive_ns` carries USB latency jitter; the
device->host relationship is reported separately via `pose.timing.fit_device_to_host`, which is the same fit the VIO
pipeline uses, so a drift number here and a drift number there can never disagree."""
from __future__ import annotations
import numpy as np
from .teensy_imu import G0, SEQ_MOD


def seq_losses(seq: np.ndarray) -> tuple[int, int]:
    """(lost_samples, gap_events) from a uint32 sequence column, wrap-aware and ignoring reorders — SeqTracker's rule."""
    s = np.asarray(seq, np.int64)
    if len(s) < 2: return 0, 0
    lost = (s[1:] - s[:-1] - 1) % SEQ_MOD
    lost[lost > SEQ_MOD // 2] = 0                       # duplicate / reorder, not a loss
    return int(lost.sum()), int((lost > 0).sum())


def stream_stats(*, seq, device_us, host_ns, accel, gyro, expected_rate_hz: float,
                 accel_fs_g: float | None = None, crc_errors: int = 0, resyncs: int = 0) -> dict:
    """accel in m/s^2, gyro in rad/s, both (N,3) raw sensor axes. Returns a JSON-safe dict."""
    seq = np.asarray(seq, np.int64); t_us = np.asarray(device_us, np.int64)
    accel = np.asarray(accel, np.float64).reshape(-1, 3); gyro = np.asarray(gyro, np.float64).reshape(-1, 3)
    n = len(t_us)
    out: dict = dict(n_samples=n, expected_rate_hz=float(expected_rate_hz), crc_errors=int(crc_errors), resyncs=int(resyncs))
    lost, gaps = seq_losses(seq)
    out["drops"] = dict(lost_samples=lost, gap_events=gaps, drop_rate=round(lost / (n + lost), 6) if n + lost else 0.0)
    if n < 2:
        out["duration_s"] = 0.0; return out

    dt_ms = np.diff(t_us) / 1000.0
    kept = dt_ms[dt_ms > 0]                             # a gap inflates one dt; percentiles below show it rather than hide it
    out["duration_s"] = round(float(t_us[-1] - t_us[0]) / 1e6, 3)
    out["rate_hz_mean"] = round(float(n - 1) / (float(t_us[-1] - t_us[0]) / 1e6), 3) if t_us[-1] > t_us[0] else 0.0
    out["interval_ms"] = dict(median=round(float(np.median(kept)), 4), std=round(float(kept.std()), 4),
                              min=round(float(kept.min()), 4), max=round(float(kept.max()), 4),
                              p50=round(float(np.percentile(kept, 50)), 4), p95=round(float(np.percentile(kept, 95)), 4),
                              p99=round(float(np.percentile(kept, 99)), 4))
    # Samples arriving far closer together than the ODR allows are spurious data-ready edges, not fast sampling: the
    # sequence numbers stay perfectly contiguous, so no drop check sees them, and p99 looks only at the slow tail. Seen
    # on a wrist harness whose INT1 jumper ran alongside the 8 MHz SPI clock: ~1600 extra samples 19-20 us behind their
    # partner, which pulled the mean rate to 226 Hz while the median interval stayed exactly on nominal.
    nominal_ms = 1000.0 / float(expected_rate_hz)
    short_at = np.flatnonzero(dt_ms < nominal_ms * 0.5)
    short = int(len(short_at))
    # Whether they cluster matters as much as how many. Crosstalk on the data-ready line fires continuously while the
    # geometry is bad; a handful scattered across two hours is a different phenomenon from a burst inside one second.
    span_s = 0.0
    if short > 1:
        ts = t_us[short_at] / 1e6
        span_s = float(ts.max() - ts.min())
    out["short_intervals"] = dict(count=short, fraction=round(short / len(dt_ms), 8),
                                  threshold_ms=round(nominal_ms * 0.5, 4),
                                  clustered=bool(short > 2 and span_s < 60.0), span_s=round(span_s, 1))

    # The longest run with no lost samples. A session can be unfit to record with and still hold a long, clean stretch
    # that a noise measurement can use -- those are different questions and deserve different answers.
    if len(seq) > 1:
        brk = np.flatnonzero(lost_per_step := ((np.diff(seq) - 1) % SEQ_MOD))
        brk = brk[(lost_per_step[brk] <= SEQ_MOD // 2) & (lost_per_step[brk] > 0)]
        edges = np.concatenate([[0], brk + 1, [len(seq)]])
        lens = np.diff(edges)
        k = int(np.argmax(lens))
        i0, i1 = int(edges[k]), int(edges[k + 1])
        out["longest_clean_segment"] = dict(start_index=i0, n_samples=int(i1 - i0),
                                            duration_s=round(float(t_us[i1 - 1] - t_us[i0]) / 1e6, 1),
                                            start_s=round(float(t_us[i0] - t_us[0]) / 1e6, 1),
                                            is_whole_session=bool(i1 - i0 == len(seq)))

    norm_g = np.linalg.norm(accel, axis=1) / G0
    abs_max_g = float(np.abs(accel).max() / G0)
    out["accel_g"] = dict(norm_median=round(float(np.median(norm_g)), 4), norm_std=round(float(norm_g.std()), 4),
                          abs_max=round(abs_max_g, 4), per_axis_abs_max=[round(float(v), 4) for v in np.abs(accel).max(axis=0) / G0])
    if accel_fs_g:                                      # saturation watch: widen the range only when real motion approaches it
        out["accel_g"]["full_scale"] = float(accel_fs_g)
        out["accel_g"]["headroom_used"] = round(abs_max_g / accel_fs_g, 4)
    out["gyro_dps"] = dict(mean=[round(float(v), 4) for v in np.degrees(gyro.mean(axis=0))],
                           std=[round(float(v), 4) for v in np.degrees(gyro.std(axis=0))],
                           abs_max=round(float(np.abs(np.degrees(gyro)).max()), 3))
    if host_ns is not None and len(np.asarray(host_ns)) == n:
        from ..pose.timing import fit_device_to_host
        try: out["clock_fit"] = fit_device_to_host(t_us, np.asarray(host_ns, np.int64)).to_dict()
        except Exception as exc: out["clock_fit"] = dict(error=str(exc))
    return out
