"""Four diagnostic plots for one IMU recording — accel, gyro, |accel|, and the sampling-interval histogram.

    python -m handumi_collector.tools.plot_imu data/imu_right_20260914_120000 [--save imu.png]

Reads the same two containers as `validate_session` (session imu.csv / raw episode mcap). Time is the device clock in
seconds from the first sample; the interval histogram is the picture behind the jitter gate, with the nominal period and
the p99 limit drawn on it so a marginal board is obvious at a glance."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from .validate_session import MAX_P99_FACTOR, load_session


def load_arrays(path: Path, side: str | None):
    """(t_s, accel m/s^2, gyro rad/s, expected_rate_hz) from a session directory or a raw episode."""
    if (path / "imu.csv").exists():
        import csv as _csv, json
        rows = list(_csv.DictReader(open(path / "imu.csv", newline="")))
        if side: rows = [r for r in rows if r["side"] == side]
        if not rows: raise SystemExit(f"{path}/imu.csv: no rows" + (f" for side={side}" if side else ""))
        g = lambda k: np.array([float(r[k]) for r in rows])
        t_us = g("device_timestamp_us")
        accel = np.stack([g("ax"), g("ay"), g("az")], 1); gyro = np.stack([g("gx"), g("gy"), g("gz")], 1)
        meta = json.loads((path / "metadata.json").read_text()) if (path / "metadata.json").exists() else {}
        return (t_us - t_us[0]) / 1e6, accel, gyro, meta.get("stats", {}).get("expected_rate_hz", 200.0)
    from ..pose.episode_io import RawEpisode
    ep = RawEpisode.load(path)
    s = side or sorted(ep.imu)[0]
    imu = ep.imu[s]
    return (imu.device_us - imu.device_us[0]) / 1e6, imu.accel, imu.gyro, float(ep.meta.get("imu_rate_hz", 200.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--side", default=None)
    ap.add_argument("--save", default=None, help="write a PNG here instead of opening a window")
    a = ap.parse_args(argv)
    import matplotlib
    if a.save: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..devices.teensy_imu import G0

    path = Path(a.path).expanduser()
    t, accel, gyro, exp_hz = load_arrays(path, a.side)
    dt_ms = np.diff(t) * 1000.0
    nominal_ms = 1000.0 / exp_hz

    fig, ax = plt.subplots(4, 1, figsize=(11, 12))
    for i, lab in enumerate("xyz"):
        ax[0].plot(t, accel[:, i] / G0, lw=0.6, label=f"a{lab}")
        ax[1].plot(t, np.degrees(gyro[:, i]), lw=0.6, label=f"g{lab}")
    ax[0].set_ylabel("accel [g]"); ax[1].set_ylabel("gyro [dps]")
    norm_g = np.linalg.norm(accel, axis=1) / G0
    ax[2].plot(t, norm_g, lw=0.6, color="k")
    ax[2].axhline(1.0, color="tab:green", ls="--", lw=0.8, label="1 g")
    ax[2].set_ylim(min(0.9, norm_g.min() - 0.02), max(1.1, norm_g.max() + 0.02))   # keep the 1 g line on screen when the board is still
    ax[2].set_ylabel("|accel| [g]"); ax[2].set_xlabel("device time [s]")
    # Fixed bins around the nominal period, with everything slower piled into the last one. A healthy INT1-driven board
    # produces a single interval value, and auto-ranged bins over a zero-width spread draw a bar too thin to see.
    lo, hi = nominal_ms * 0.8, nominal_ms * MAX_P99_FACTOR * 1.2
    edges = np.linspace(lo, hi, 61)
    ax[3].hist(np.clip(dt_ms, lo, hi - 1e-9), bins=edges, color="tab:blue")
    ax[3].set_xlim(lo, hi)
    ax[3].axvline(nominal_ms, color="tab:green", ls="--", lw=1.0, label=f"nominal {nominal_ms:.2f} ms")
    ax[3].axvline(nominal_ms * MAX_P99_FACTOR, color="tab:red", ls="--", lw=1.0, label=f"p99 limit {nominal_ms*MAX_P99_FACTOR:.2f} ms")
    ax[3].set_xlabel("sampling interval [ms]"); ax[3].set_ylabel("count"); ax[3].set_yscale("log")
    for x in ax: x.grid(alpha=0.25); x.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{path.name}  —  {len(t)} samples, {t[-1]:.1f} s, expected {exp_hz:g} Hz")
    fig.tight_layout()
    if a.save: fig.savefig(a.save, dpi=130); print(f"wrote {a.save}")
    else: plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
