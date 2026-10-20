"""IMU bring-up (2026-10-14, spec 41 step 5). READ-ONLY: opens wrist IMU boards by USB serial, runs the live AHRS for N seconds,
reports rate, sequence gaps, max inter-sample gap, gyro bias, g_rest, accel-correction share, tilt, and the yaw drift while
still (yaw is unobservable -> this IS the drift the init alignment has to live with). No robot, no camera.

    .venv/bin/python -m visual_teleop.tools.imu_bringup --list
    .venv/bin/python -m visual_teleop.tools.imu_bringup --serial 4A65C2BC893657D2 --label probe --seconds 20
Keep the board STILL for the first second (bias), then still for the rest to read drift (or move it to see tilt follow)."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from ..imu.imu_reader import ImuReader
from ..fusion.wrist_pose_fusion import yaw_of


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true"); ap.add_argument("--serial", action="append", default=[])
    ap.add_argument("--label", action="append", default=[]); ap.add_argument("--seconds", type=float, default=20.0); ap.add_argument("--rate", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from handumi_collector.devices.teensy_imu import list_teensy_ports, TeensyImu
    from handumi_collector.config import ImuCfg
    if a.list:
        print(json.dumps([p for p in list_teensy_ports() if p.get("serial_number")], indent=1)); return
    labels = a.label + [f"imu{i}" for i in range(len(a.label), len(a.serial))]
    readers = []
    for sn, lab in zip(a.serial, labels):
        dev = TeensyImu(ImuCfg(side=lab, serial_number=sn, rate_hz=a.rate, required=True)); dev.open(); dev.start()
        r = ImuReader(lab, dev, camera_imu_offset_ms=0.0); r.start(); readers.append((r, dev))
    t0 = time.monotonic(); yaw_trace = {r.side: [] for r, _ in readers}; last_print = 0
    while time.monotonic() - t0 < a.seconds:
        time.sleep(0.05)
        for r, _ in readers:
            x = r.latest()
            if x is not None: yaw_trace[r.side].append((time.monotonic() - t0, yaw_of(x.R_G_I), r.ahrs.tilt_deg()))
        if time.monotonic() - last_print > 2.0:
            last_print = time.monotonic()
            print(" | ".join(f"{r.side}: {r.rate_hz():.0f}Hz ready={r.ahrs.ready} tilt={r.ahrs.tilt_deg():.1f}" for r, _ in readers), flush=True)
    rep = {}
    for r, dev in readers:
        r.stop(); dev.stop(); dev.close()
        S = list(r.samples_for_recorder); dev_us = np.array([s.device_timestamp_us for s in S]); host = np.array([s.host_receive_ns for s in S])
        tr = np.array(yaw_trace[r.side]) if yaw_trace[r.side] else np.zeros((0, 3))
        yaw = np.unwrap(tr[:, 1]) if len(tr) else np.zeros(0)
        rep[r.side] = dict(serial=dev.serial_number, port=dev.port, samples=len(S), rate_hz_device=float((len(S) - 1) / ((dev_us[-1] - dev_us[0]) * 1e-6)) if len(S) > 1 else 0.0,
                           max_gap_ms_device=float(np.diff(dev_us).max() / 1e3) if len(S) > 1 else None,
                           max_gap_ms_host=float(np.diff(host).max() / 1e6) if len(S) > 1 else None,
                           seq_lost=int(getattr(dev.seq, "lost_total", 0) or 0), ahrs_ready=r.ahrs.ready, gyro_bias_rad_s=r.ahrs.bias.tolist(),
                           g_rest=r.ahrs.g_rest, acc_correction_share=r.ahrs.acc_correction_share, ahrs_gaps=r.ahrs.n_gaps,
                           tilt_deg_final=r.ahrs.tilt_deg(),
                           yaw_drift_deg=float(np.degrees(yaw[-1] - yaw[0])) if len(yaw) > 1 else None,
                           yaw_drift_deg_per_min=float(np.degrees(yaw[-1] - yaw[0]) / ((tr[-1, 0] - tr[0, 0]) / 60)) if len(yaw) > 1 and tr[-1, 0] > tr[0, 0] else None,
                           device_status=dev.last_status)
    print(json.dumps(rep, indent=1, default=str))
    if a.out: open(a.out, "w").write(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
