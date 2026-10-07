"""Hardware smoke tool: python -m handumi_collector.tools.imu_monitor [--hardware handumi_v1|mock] [--list] [--seconds N] [--log CSV]
Prints per-side rate / loss / age / accel / gyro at 2 Hz without the UI. --list shows every USB serial port with serial numbers."""
from __future__ import annotations
import argparse
import csv
import math
import sys
import time
from ..config import DEFAULT_CONFIG_DIR, load_config
from ..devices.manager import DeviceManager
from ..devices.teensy_imu import G0, list_teensy_ports

R2D = 180.0 / math.pi


def fmt_side(name: str, imu, th: dict) -> str:
    q = imu.quality(); s = imu.last_sample
    grade = q.grade(th)
    head = f"{name:5s} {grade:6s} {q.hz:6.1f} Hz  loss {q.loss_ratio*100:5.2f}%  age {q.age_ms if q.age_ms is not None else float('nan'):6.1f} ms  crc {q.crc_errors}  gaps {q.seq_gaps}  wraps {q.ts_wraps}  reconn {q.reconnects}"
    ident = f"  serial={getattr(imu, 'serial_number', None)} port={getattr(imu, 'port', None)}"
    if s is None: return head + ident + "\n  (no samples)"
    dev = getattr(imu, "last_status", None) or {}
    return (head + ident + f"\n  accel {s.ax/G0:+6.3f} {s.ay/G0:+6.3f} {s.az/G0:+6.3f} g   |a| {math.sqrt(s.ax**2+s.ay**2+s.az**2)/G0:5.3f} g"
            f"\n  gyro  {s.gx*R2D:+7.2f} {s.gy*R2D:+7.2f} {s.gz*R2D:+7.2f} dps   temp {s.temperature_c:5.1f} C   seq {s.seq}  t_us {s.device_timestamp_us}"
            + (f"   fw {dev.get('fw')} dev_dropped {dev.get('dropped')}" if dev else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR)); ap.add_argument("--hardware", default="handumi_v1")
    ap.add_argument("--list", action="store_true", help="list serial ports (Teensy serial numbers) and exit")
    ap.add_argument("--seconds", type=float, default=0, help="run for N seconds then print a summary (0 = until Ctrl-C)")
    ap.add_argument("--log", default=None, help="append every sample to this CSV (soak tests)")
    a = ap.parse_args(argv)
    if a.list:
        for p in list_teensy_ports():
            print(f"{p['device']:32s} serial={p['serial_number']} vid={p['vid']} pid={p['pid']} product={p['product']} loc={p['location']}")
        return 0
    cfg = load_config(a.config_dir, hardware=a.hardware)
    hw = cfg.hardware; hw.cameras = []; hw.grippers = []          # IMU only
    dm = DeviceManager(hw); dm.build(); dm.connect_all()
    for n, e in dm.errors.items(): print(f"!! {n}: {e}")
    writer = None
    if a.log:
        fh = open(a.log, "a", newline=""); writer = csv.writer(fh)
        writer.writerow(["side", "seq", "device_timestamp_us", "host_receive_ns", "ax", "ay", "az", "gx", "gy", "gz", "temp_c"])
    t0 = time.time(); worst = {s: "GREEN" for s in dm.imus}
    try:
        while not a.seconds or time.time() - t0 < a.seconds:
            time.sleep(0.5)
            lines = []
            for side, imu in dm.imus.items():
                if writer:
                    for s in imu.buffer.drain(): writer.writerow([s.side, s.seq, s.device_timestamp_us, s.host_receive_ns, s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.temperature_c])
                else:
                    imu.buffer.drain()
                lines.append(fmt_side(side.upper(), imu, cfg.collector.imu_quality))
                g = imu.quality().grade(cfg.collector.imu_quality)
                if ["GREEN", "YELLOW", "RED"].index(g) > ["GREEN", "YELLOW", "RED"].index(worst[side]): worst[side] = g
            sys.stdout.write("\x1b[2J\x1b[H" if not a.seconds else ""); print(f"elapsed {time.time()-t0:7.1f}s"); print("\n".join(lines)); sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    print("\nSUMMARY worst grade per side:", worst, {s: imu.quality().__dict__ for s, imu in dm.imus.items()})
    dm.close_all()
    return 0 if all(g == "GREEN" for g in worst.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
