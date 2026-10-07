"""IMU noise characterisation (Allan deviation) from a static log — step 2 of M1-0.

    python -m handumi_collector.tools.imu_monitor --hardware handumi_v1 --seconds 7200 --log imu_right.csv   # >= 2 h, dead still
    python -m ego_teleop.tools.imu_allan imu_right.csv --side right [--rate 400] [--write-bundle]

Overlapping Allan deviation of the averaged rate (imu_utils / Kalibr convention):
    sigma^2(tau) = 1/(2 tau^2 (N-2m)) * sum (theta_{k+2m} - 2 theta_{k+m} + theta_k)^2 ,  theta = cumulative sum of samples * dt
White noise dominates the -1/2 slope region  -> noise density  N = sigma(tau=1 s)
Bias random walk dominates the +1/2 region   -> random walk    K from a fit of sigma(tau) = K sqrt(tau/3) over the long-tau decade
Values are only as good as the log: needs a long, truly static, thermally settled recording."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np


def allan_deviation(x: np.ndarray, dt: float, *, n_taus: int = 40, tau_min: float | None = None, tau_max: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """x: (N,) rate samples at 1/dt Hz. Returns (taus_s, adev) using the overlapping estimator on the integrated signal."""
    x = np.asarray(x, np.float64); N = len(x)
    if N < 100: raise ValueError("need >= 100 samples")
    theta = np.concatenate([[0.0], np.cumsum(x) * dt])
    m_min = max(1, int(round((tau_min or dt) / dt))); m_max = int((tau_max or (N * dt / 5)) / dt)
    ms = np.unique(np.round(np.geomspace(m_min, max(m_max, m_min + 1), n_taus)).astype(int))
    taus, adev = [], []
    for m in ms:
        if N - 2 * m < 1: continue
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        tau = m * dt
        adev.append(float(np.sqrt(np.sum(d ** 2) / (2 * tau ** 2 * len(d))))); taus.append(float(tau))
    return np.asarray(taus), np.asarray(adev)


def noise_density(taus: np.ndarray, adev: np.ndarray) -> float:
    """sigma(1 s) — log-log interpolation of the -1/2 slope region onto tau = 1 s."""
    lt, la = np.log10(taus), np.log10(adev)
    return float(10 ** np.interp(0.0, lt, la))


def random_walk(taus: np.ndarray, adev: np.ndarray, *, frac: float = 0.3) -> float:
    """K from sigma(tau) = K sqrt(tau/3) fitted over the longest `frac` of the (log) tau range."""
    lo = np.quantile(np.log10(taus), 1 - frac)
    m = np.log10(taus) >= lo
    if m.sum() < 3: m = np.ones_like(taus, bool)
    return float(np.mean(adev[m] / np.sqrt(taus[m] / 3.0)))


def characterize(t_ns: np.ndarray, gyro: np.ndarray, accel: np.ndarray, *, rate_hz: float | None = None) -> dict:
    t = np.asarray(t_ns, np.int64)
    dt = float(np.median(np.diff(t)) / 1e9) if len(t) > 1 else (1.0 / (rate_hz or 400.0))
    if rate_hz: dt = 1.0 / float(rate_hz)
    out = dict(n_samples=int(len(t)), duration_s=round(float((t[-1] - t[0]) / 1e9), 1) if len(t) > 1 else 0.0, dt_s=dt, rate_hz=round(1 / dt, 2))
    per_axis = {}
    for name, data in (("gyroscope", gyro), ("accelerometer", accel)):
        nd, rw = [], []
        for ax in range(3):
            taus, adev = allan_deviation(np.asarray(data)[:, ax], dt)
            nd.append(noise_density(taus, adev)); rw.append(random_walk(taus, adev))
            per_axis[f"{name}_axis{ax}"] = dict(noise_density=nd[-1], random_walk=rw[-1])
        out[f"{name}_noise_density"] = float(np.mean(nd)); out[f"{name}_random_walk"] = float(np.mean(rw))
    out["per_axis"] = per_axis
    out["static_check"] = dict(gyro_mean_dps=np.degrees(np.mean(gyro, axis=0)).round(4).tolist(), accel_norm_mean=float(np.linalg.norm(np.mean(accel, axis=0))),
                               accel_norm_std=float(np.std(np.linalg.norm(accel, axis=1))))
    return out


def read_imu_csv(path: Path, side: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """imu_monitor --log CSV: side,seq,device_timestamp_us,host_receive_ns,ax,ay,az,gx,gy,gz,temp_c (device clock is used)."""
    t, g, a = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("device_timestamp_us"): continue
            if side and row.get("side") not in (None, "", side): continue
            t.append(int(float(row["device_timestamp_us"])) * 1000)
            a.append([float(row["ax"]), float(row["ay"]), float(row["az"])]); g.append([float(row["gx"]), float(row["gy"]), float(row["gz"])])
    if not t: raise SystemExit(f"{path}: no samples for side={side}")
    o = np.argsort(t)
    return np.asarray(t, np.int64)[o], np.asarray(g, np.float64)[o], np.asarray(a, np.float64)[o]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--rate", type=float, default=None, help="override the sample rate (default: median of the device timestamps)")
    ap.add_argument("--min-minutes", type=float, default=30.0, help="refuse a log shorter than this (random walk needs a long log)")
    ap.add_argument("--write-bundle", action="store_true", help="store the result in configs/calibration/wrist_bundle_<side>_vNNN.yaml")
    ap.add_argument("--cal-dir", default=None); ap.add_argument("--json", default=None)
    ap.add_argument("--notes", default="", help="recorded in the bundle provenance: how this log was selected, and why")
    a = ap.parse_args(argv)
    t, g, ac = read_imu_csv(Path(a.csv), a.side)
    res = characterize(t, g, ac, rate_hz=a.rate)
    print(json.dumps({k: v for k, v in res.items() if k != "per_axis"}, indent=1))
    if res["duration_s"] < a.min_minutes * 60:
        print(f"WARNING: log is {res['duration_s']/60:.1f} min < {a.min_minutes:.0f} min — random-walk terms are unreliable", file=sys.stderr)
    if a.json: Path(a.json).write_text(json.dumps(res, indent=1))
    if a.write_bundle:
        from ..calibration.bundle import load_bundle, save_bundle, ImuNoise
        b = load_bundle(a.side, cal_dir=Path(a.cal_dir) if a.cal_dir else None)
        b.imu_noise = ImuNoise(gyroscope_noise_density=res["gyroscope_noise_density"], gyroscope_random_walk=res["gyroscope_random_walk"],
                               accelerometer_noise_density=res["accelerometer_noise_density"], accelerometer_random_walk=res["accelerometer_random_walk"],
                               update_rate=res["rate_hz"])
        # The session directory, not the file name: every session's log is called imu.csv, so recording that told you
        # nothing about which recording these numbers came from -- and the distinction can matter a great deal, as when
        # one side's figures come from a clean segment of a session that failed its integrity gates.
        run = Path(a.csv).resolve().parent
        b.set_provenance("imu_noise", source="allan_variance", tool="ego_teleop.tools.imu_allan",
                         runs=[f"{run.parent.name}/{run.name}"],
                         notes=f"{res['duration_s']/3600:.2f} h static log, {res['n_samples']} samples, "
                               f"{res['rate_hz']:.2f} Hz" + (f". {a.notes}" if a.notes else ""))
        p = save_bundle(b, cal_dir=Path(a.cal_dir) if a.cal_dir else None, notes="imu_allan")
        print(f"wrote {p} (missing: {b.missing or 'nothing'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
