#!/usr/bin/env python3
"""[2026-09-19] Wrist-IMU orientation on the camera timeline: integrate on the device clock, THEN map to the host.

The two clocks answer different questions and must not be swapped:

    device_us   the IMU's own elapsed time. Its deltas are the true dt between samples, so this is the ONLY clock
                that may drive integration. The two wrist units tick at 197.9 Hz and 201.6 Hz -- a property of their
                crystals, not a defect, and not a claim that the data drifts apart.
    host_ns     when the sample reached this machine. It shares a timebase with the cameras, so it is the
                synchronisation anchor -- but it carries USB batching (3.9-7.9% of steps are exact ties, measured on
                CALIB_bodytcp_20260919_121002), and feeding those ties into integration injects that jitter straight
                into the orientation.

So the order is fixed:  integrate on device_us  ->  affine device_us -> host_ns  ->  SLERP onto the depth timeline.

WHAT IS AND IS NOT OBSERVABLE, stated because a rotation matrix looks equally confident either way:

  roll, pitch   OBSERVABLE. Gravity is a constant world vector, so accelerometer readings during quiet stretches
                pin the two axes perpendicular to it. Drift in these is corrected continuously.
  yaw           NOT OBSERVABLE from this sensor. Nothing in a gyro+accel IMU sees the world's heading, so yaw is
                pure integration and it drifts. The drift is MEASURED here and reported; the constant part is
                absorbed by body_tcp_fit's yaw search, but the drift WITHIN a take is a real error and is not
                hidden behind a plausible-looking matrix.

    .venv/bin/python scripts/imu_orientation.py --selftest
    .venv/bin/python scripts/imu_orientation.py --episode <ep>
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

G = 9.80665
ACCEL_GAIN = 0.02          # per-sample pull of the gravity correction; 0.02 at ~200 Hz is a ~0.25 s time constant
# Two DIFFERENT questions, so two different masks. Sharing one threshold made a deliberate slow 60 deg turn
# (12 deg/s) read as gyro drift in the selftest, because 0.35 rad/s is 20 deg/s -- far above a real turn rate.
GRAVITY_ACCEL_TOL = 0.5    # m/s^2 from |g|: the accelerometer is reading gravity, so roll/pitch may be corrected
GRAVITY_GYRO_TOL = 0.35    # rad/s: not being swung hard enough for centripetal terms to pollute the reading
STILL_GYRO_TOL = 0.05      # rad/s (~3 deg/s): genuinely not turning. ONLY for the drift audit, never for correction


def clock_map(device_us: np.ndarray, host_ns: np.ndarray) -> dict:
    """Affine device_us -> host_ns, least squares, with the fit residual reported.

    The residual is the honest statement of how well the IMU can be placed on the camera timeline. A small
    residual means a sample's host timestamp is predictable from its device timestamp, which is exactly what
    interpolating onto depth frames assumes.

    BOTH columns are centred before the fit. Uncentred, this silently failed on the right wrist unit: its Teensy
    had been up ~20 h, so device_us was 7.16e10 while the episode spanned only 3.8e7, and against a constant
    column of ones the normal equations lost all conditioning -- lstsq returned 662 ns/us where the two-point
    slope was 999.73, with a 3.2 s median residual. The left unit (device_us 9e8) happened to survive. Centring
    removes the problem outright rather than leaving it to depend on how long a microcontroller has been on.
    """
    x = device_us.astype(np.float64)
    y = host_ns.astype(np.float64)
    x0, y0 = x.mean(), y.mean()
    A = np.stack([x - x0, np.ones_like(x)], 1)
    (a, b0), *_ = np.linalg.lstsq(A, y - y0, rcond=None)
    b = y0 + b0 - a * x0
    res_ms = (y - (a * x + b)) / 1e6
    mapped = a * x + b
    return dict(scale_ns_per_us=float(a), offset_ns=float(b),
                residual_ms_p50=float(np.percentile(np.abs(res_ms), 50)),
                residual_ms_p95=float(np.percentile(np.abs(res_ms), 95)),
                residual_ms_max=float(np.abs(res_ms).max()),
                # The mapped timeline is what SLERP is keyed on, so it has to be strictly increasing. A negative
                # or zero-slope fit would still report a small residual on a short span while producing a
                # timeline that runs backwards, and the interpolation would silently reorder the rotations.
                mapped_monotonic=bool((np.diff(mapped) > 0).all()),
                # a/1000 is how many host nanoseconds pass per device microsecond; 1.0 = the clocks agree
                rate_error_ppm=float((a / 1000.0 - 1.0) * 1e6))


def integrate(gyro: np.ndarray, accel: np.ndarray, device_us: np.ndarray) -> dict:
    """Complementary filter on the DEVICE clock. Returns quaternions (N,4, xyzw) world<-sensor, and the drift audit.

    World here is gravity-aligned: +Z is up. Yaw starts at zero by construction because nothing observes it.
    """
    n = len(device_us)
    dt = np.diff(device_us.astype(np.float64)) * 1e-6
    an = np.linalg.norm(accel, axis=1)
    gn = np.linalg.norm(gyro, axis=1)
    gravity_ok = (np.abs(an - G) < GRAVITY_ACCEL_TOL) & (gn < GRAVITY_GYRO_TOL)   # safe to correct roll/pitch
    still = gravity_ok & (gn < STILL_GYRO_TOL)                                    # safe to call it "not turning"
    quiet = gravity_ok

    # Seed from gravity in the first quiet stretch: roll and pitch are pinned, yaw is left at zero.
    seed = np.flatnonzero(quiet)
    if len(seed) < 10:
        raise RuntimeError(f"no quiet window to seed orientation ({quiet.sum()} quiet samples of {n}) -- "
                           "the unit was never still enough for gravity to be read")
    up = accel[seed[:200]].mean(0)
    up /= np.linalg.norm(up)
    R0 = _align_to_up(up)

    q = np.zeros((n, 4)); q[0] = Rotation.from_matrix(R0).as_quat()
    corrected = 0
    for i in range(1, n):
        dq = Rotation.from_rotvec(gyro[i - 1] * dt[i - 1])
        r = Rotation.from_quat(q[i - 1]) * dq
        if quiet[i]:
            # Gravity says which way is up; nudge the estimate towards it without touching yaw, because the
            # accelerometer carries no heading information and pretending otherwise would fabricate one.
            meas_up = r.apply(accel[i] / max(np.linalg.norm(accel[i]), 1e-9))
            axis = np.cross(meas_up, np.array([0.0, 0.0, 1.0]))
            s = np.linalg.norm(axis)
            if s > 1e-9:
                ang = np.arctan2(s, float(np.dot(meas_up, [0, 0, 1])))
                r = Rotation.from_rotvec(axis / s * (ang * ACCEL_GAIN)) * r
                corrected += 1
        q[i] = r.as_quat()

    # Yaw drift audit: over quiet stretches the unit is not turning, so any yaw change there is drift.
    yaw = Rotation.from_quat(q).as_euler("xyz")[:, 2]
    drift = _quiet_yaw_drift(yaw, still, dt)
    return dict(quat=q, quiet=quiet, quiet_share=float(quiet.mean()), still=still,
                still_share=float(still.mean()), corrections=corrected,
                yaw_drift_deg_per_min=drift, seed_up=up)


def _align_to_up(up: np.ndarray) -> np.ndarray:
    """Smallest rotation taking the measured up-vector (sensor frame) to world +Z. Yaw is left unset, by design."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z); s = np.linalg.norm(v); c = float(np.dot(up, z))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    return Rotation.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()


def _quiet_yaw_drift(yaw: np.ndarray, still: np.ndarray, dt: np.ndarray) -> float:
    """Degrees per minute of yaw change accumulated across STILL stretches only.

    `still` must be the tight mask. With a loose one a deliberate slow turn is counted as drift and the number
    says nothing -- which is exactly what the selftest caught before the masks were separated.
    """
    d = np.diff(np.unwrap(yaw))
    m = still[1:] & still[:-1]
    if m.sum() < 50:
        return float("nan")
    return float(np.degrees(d[m].sum()) / max(dt[m].sum(), 1e-9) * 60.0)


def resample(quat: np.ndarray, t_src_ns: np.ndarray, t_dst_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SLERP the orientation onto a target timeline. Targets outside the source span are marked invalid, not
    extrapolated -- an extrapolated rotation is indistinguishable from a measured one once it is in the array."""
    ok = (t_dst_ns >= t_src_ns[0]) & (t_dst_ns <= t_src_ns[-1])
    out = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(t_dst_ns), 1))
    if ok.any():
        # SLERP needs strictly increasing keys; host ties are removed here rather than earlier so that the
        # integration above still sees every sample.
        keep = np.r_[True, np.diff(t_src_ns) > 0]
        sl = Slerp(t_src_ns[keep].astype(np.float64), Rotation.from_quat(quat[keep]))
        out[ok] = sl(t_dst_ns[ok].astype(np.float64)).as_quat()
    return out, ok


def selftest() -> int:
    rng = np.random.default_rng(0)
    n, hz = 4000, 200.0
    device_us = (np.arange(n) / hz * 1e6).astype(np.int64)
    # a known trajectory: still, then a 60 deg yaw, then still
    ang = np.zeros(n); ang[1000:2000] = np.linspace(0, np.radians(60), 1000); ang[2000:] = np.radians(60)
    truth = Rotation.from_rotvec(np.stack([np.zeros(n), np.zeros(n), ang], 1))
    gyro = np.zeros((n, 3)); gyro[1:, 2] = np.diff(ang) * hz
    accel = truth.inv().apply(np.tile([0, 0, G], (n, 1))) + rng.normal(0, 0.01, (n, 3))
    gyro = gyro + rng.normal(0, 0.002, (n, 3))

    r = integrate(gyro, accel, device_us)
    est = Rotation.from_quat(r["quat"])
    err = (est * truth.inv()).magnitude()
    print(f"  orientation error  p50 {np.degrees(np.median(err)):.3f} deg   p95 {np.degrees(np.percentile(err,95)):.3f} deg"
          f"   max {np.degrees(err.max()):.3f} deg")
    print(f"  gravity-usable {r['quiet_share']*100:.1f}%   genuinely still {r['still_share']*100:.1f}%"
          f"   corrections {r['corrections']}   yaw drift {r['yaw_drift_deg_per_min']:+.2f} deg/min")
    assert np.degrees(err.max()) < 3.0, f"integration diverged: {np.degrees(err.max()):.2f} deg"
    # The synthetic trajectory's still segments really are still, so the drift audit must report ~0. A loose
    # still-mask reported +180 deg/min here by counting the deliberate 60 deg turn as drift.
    assert abs(r["yaw_drift_deg_per_min"]) < 1.0, f"drift audit counted real motion: {r['yaw_drift_deg_per_min']:+.2f}"
    assert r["still_share"] < r["quiet_share"], "the still mask must be strictly tighter than the gravity mask"

    host_ns = (device_us.astype(np.float64) * 1000.0 * 1.0004 + 1.7e12).astype(np.int64)
    cm = clock_map(device_us, host_ns)
    print(f"  clock map  scale {cm['scale_ns_per_us']:.6f} ns/us  rate error {cm['rate_error_ppm']:+.0f} ppm"
          f"  residual p95 {cm['residual_ms_p95']:.6f} ms")
    assert abs(cm["rate_error_ppm"] - 400.0) < 5.0, cm["rate_error_ppm"]
    assert cm["residual_ms_p95"] < 1e-3, cm["residual_ms_p95"]

    assert cm["mapped_monotonic"], "mapped timeline is not strictly increasing"

    # The real failure this guards: the SAME synthetic signal, only the device clock's absolute offset changes.
    # Uncentred, the 19.9 h offset returned 662 ns/us against the 999.73 the two-point slope showed. The two
    # fits must now agree to the last significant digit, because nothing about the signal differs.
    late = device_us + 71_572_996_218
    cm2 = clock_map(late, (late.astype(np.float64) * 1000.0 * 1.0004 + 1.7e12).astype(np.int64))
    print(f"  clock map  small offset slope {cm['scale_ns_per_us']:.6f}   "
          f"19.9 h offset slope {cm2['scale_ns_per_us']:.6f}   "
          f"difference {abs(cm2['scale_ns_per_us'] - cm['scale_ns_per_us']):.3e} ns/us")
    assert abs(cm2["rate_error_ppm"] - 400.0) < 5.0, f"large device_us offset broke the fit: {cm2['rate_error_ppm']:+.0f} ppm"
    assert cm2["residual_ms_p95"] < 1e-3, cm2["residual_ms_p95"]
    assert cm2["mapped_monotonic"], "mapped timeline is not strictly increasing at a large device offset"
    assert abs(cm2["scale_ns_per_us"] - cm["scale_ns_per_us"]) < 1e-6, (
        f"same signal, different device offset, different slope: "
        f"{cm['scale_ns_per_us']:.6f} vs {cm2['scale_ns_per_us']:.6f}")

    t_dst = np.linspace(host_ns[0] - 5e8, host_ns[-1] + 5e8, 500).astype(np.int64)
    q, ok = resample(r["quat"], host_ns, t_dst)
    print(f"  resample   {ok.sum()}/{len(ok)} targets inside the IMU span, {len(ok)-ok.sum()} refused as extrapolation")
    assert ok.sum() < len(ok), "targets outside the span must be refused"
    print("\nSELFTEST PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--episode", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.episode:
        ap.error("--episode or --selftest")
    from handumi_collector.pose.episode_io import RawEpisode
    raw = RawEpisode.load(a.episode)
    print(f"\n{a.episode.name}")
    for side, im in sorted(raw.imu.items()):
        d = np.asarray(im.device_us, np.int64); h = np.asarray(im.host_ns, np.int64)
        r = integrate(np.asarray(im.gyro, float), np.asarray(im.accel, float), d)
        cm = clock_map(d, h)
        print(f"  imu_{side}: n {len(d)}  gravity-usable {r['quiet_share']*100:5.1f}%  "
              f"still {r['still_share']*100:5.1f}%  yaw drift {r['yaw_drift_deg_per_min']:+7.2f} deg/min")
        print(f"            clock map rate error {cm['rate_error_ppm']:+8.0f} ppm   "
              f"residual p50/p95/max {cm['residual_ms_p50']:.3f}/{cm['residual_ms_p95']:.3f}/{cm['residual_ms_max']:.3f} ms"
              f"   mapped monotonic {'yes' if cm['mapped_monotonic'] else 'NO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
