"""P0 of the multi-sensor ladder: do all three sensors actually run AT THE SAME TIME, at rate, on one host?

    chest RGB-D   30 FPS       Orbbec, colour + aligned depth
    wrist Arducam 30 FPS       the wrist unit's fisheye
    wrist IMU     416 Hz       XIAO nRF52840 Sense (LSM6DS3TR-C) or the Teensy/ICM42688P unit — same USB protocol
                               (firmware/teensy_imu/PROTOCOL.md v1), so the host needs only `rate_hz` changed.

Each of these is known to work alone. P0 exists because "alone" is not the question: three USB streams on one bus,
one of them 416 Hz, is exactly where frames start being dropped, and finding that out during a fusion run means
debugging two things at once.

    .venv/bin/python -m ego_teleop.tools.f0_sensors --seconds 30
    .venv/bin/python -m ego_teleop.tools.f0_sensors --hardware handumi_rgbd --imu-hz 416 --json p0.json

No robot, no tracking, no fusion: rates, drops, and the IMU device->host clock fit (spec section 10), which is the
one number that decides whether the three streams can be put on a common timeline at all."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

from handumi_collector.config import DEFAULT_CONFIG_DIR, load_hardware
from handumi_collector.devices.manager import DeviceManager
from handumi_collector.pose.timing import fit_device_to_host

DEFAULTS = dict(head_depth=30.0, left_wrist=30.0, right_wrist=30.0)


def clock_quality(samples) -> dict:
    """device_us -> host_ns fit over the window (section 10): the host NEVER uses packet arrival as the measurement
    time, so this fit is the timeline. Its residual bounds how precisely the camera<->IMU offset can be measured."""
    if len(samples) < 50: return dict(n=len(samples), note="too few samples")
    dev = np.asarray([s.device_timestamp_us for s in samples], np.float64)
    host = np.asarray([s.host_receive_ns for s in samples], np.float64)
    f = fit_device_to_host(dev, host)
    pred = dev * f.slope_ns_per_us + f.offset_ns
    res_ms = (host - pred) / 1e6
    gaps_us = np.diff(dev)
    med = float(np.median(gaps_us)) if len(gaps_us) else float("nan")
    return dict(n=len(samples), slope_ns_per_us=float(f.slope_ns_per_us),
                drift_ppm=float((f.slope_ns_per_us / 1000.0 - 1.0) * 1e6),
                residual_ms_p95=float(np.percentile(np.abs(res_ms), 95)), residual_ms_max=float(np.abs(res_ms).max()),
                device_rate_hz=1e6 / med if med and np.isfinite(med) and med > 0 else float("nan"),
                gap_us_p99=float(np.percentile(gaps_us, 99)) if len(gaps_us) else float("nan"),
                dropped_gaps=int((gaps_us > 3 * med).sum()) if med and np.isfinite(med) else 0)


def run(hardware: str, *, seconds: float, imu_hz: float | None, config_dir: Path) -> dict:
    hw = load_hardware(Path(hardware) if Path(hardware).exists() else config_dir / f"hardware_{hardware}.yaml")
    dm = DeviceManager(hw); dm.build(); dm.connect_all()
    targets = dict(DEFAULTS)
    for c in hw.cameras: targets[c.name] = float(c.fps)
    try:
        counts = {n: (c.buffer.total if hasattr(c, "buffer") else 0) for n, c in dm.cameras.items()}
        imu0 = {s: d.buffer.total for s, d in dm.imus.items()}
        t0 = time.monotonic()
        time.sleep(seconds)                                  # a plain wall-clock window: every stream is free-running
        el = time.monotonic() - t0
        out = dict(hardware=hw.profile, seconds=round(el, 2), errors=dict(dm.errors), devices={})
        for n, c in dm.cameras.items():
            got = (c.buffer.total if hasattr(c, "buffer") else 0) - counts[n]
            st = c.status()
            out["devices"][n] = dict(kind="camera", target_hz=targets.get(n, 30.0), measured_hz=got / el,
                                     frames=got, overflow=getattr(c.buffer, "overflow", None), detail=dict(st.detail))
        for side, d in dm.imus.items():
            got = d.buffer.total - imu0[side]
            q = d.quality()
            out["devices"][f"{side}_imu"] = dict(kind="imu", target_hz=float(imu_hz or d.cfg.rate_hz),
                                                 measured_hz=got / el, samples=got, loss_ratio=q.loss_ratio,
                                                 crc_errors=q.crc_errors, seq_gaps=q.seq_gaps, max_gap_ms=q.max_gap_ms,
                                                 clock=clock_quality(d.buffer.snapshot()))
    finally:
        dm.close_all()
    for n, d in out["devices"].items():
        d["passed"] = bool(d["measured_hz"] >= 0.95 * d["target_hz"]) and not out["errors"].get(n)
    out["passed"] = all(d["passed"] for d in out["devices"].values()) and not out["errors"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware", default="handumi_rgbd", help="profile name or a path to a hardware yaml")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--imu-hz", type=float, default=None, help="expected IMU rate (default: the profile's rate_hz)")
    ap.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    rep = run(a.hardware, seconds=a.seconds, imu_hz=a.imu_hz, config_dir=a.config_dir)
    print(f"\nP0 sensors — profile {rep['hardware']}, {rep['seconds']} s\n")
    print(f"  {'device':14s} {'target':>8s} {'measured':>9s}  {'n':>7s}  note")
    for n, d in rep["devices"].items():
        note = ""
        if d["kind"] == "imu":
            c = d["clock"]
            note = (f"loss {d['loss_ratio']*100:.2f}% crc {d['crc_errors']} gaps {d['seq_gaps']} | "
                    f"clock residual p95 {c.get('residual_ms_p95', float('nan')):.2f} ms, drift {c.get('drift_ppm', float('nan')):.0f} ppm")
        elif d.get("overflow"): note = f"buffer overflow {d['overflow']}"
        print(f"  [{'PASS' if d['passed'] else 'FAIL'}] {n:12s} {d['target_hz']:7.1f} {d['measured_hz']:8.1f}  "
              f"{d.get('frames', d.get('samples', 0)):7d}  {note}")
    for n, e in rep["errors"].items(): print(f"  ERROR {n}: {e}")
    print(f"\n  {'PASS' if rep['passed'] else 'FAIL'}: three sensors at rate, simultaneously")
    if a.json: a.json.write_text(json.dumps(rep, indent=1, default=str)); print(f"  report -> {a.json}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
