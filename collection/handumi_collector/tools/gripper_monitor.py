"""Hardware smoke tool: python -m handumi_collector.tools.gripper_monitor [--hardware ...] [--seconds N] [--log CSV]
Prints raw / unwrapped / normalized ticks and rate per side; use it to check wrap handling and monotonic trajectories."""
from __future__ import annotations
import argparse
import csv
import sys
import time
from ..config import DEFAULT_CONFIG_DIR, load_config
from ..devices.manager import DeviceManager


def scan(ids=range(16), bauds=(1_000_000, 500_000, 115_200)) -> int:
    """Which port answers as a Feetech bus? Prints port / baud / servo id / raw position for every responder."""
    import glob
    from ..devices.feetech_bus import FeetechBus
    found = 0
    for port in sorted(glob.glob("/dev/cu.usb*")):
        for baud in bauds:
            try:
                bus = FeetechBus(port, baud); bus.open()
            except Exception as exc:
                print(f"{port} @{baud}: cannot open ({str(exc)[:50]})"); break
            hits = []
            for sid in ids:
                try:
                    hits.append((sid, bus.read_position(sid)))
                except Exception:
                    pass
            bus.close()
            if hits:
                found += len(hits); print(f"{port} @{baud}: " + ", ".join(f"id {i} raw {r}" for i, r in hits)); break
        else:
            print(f"{port}: no Feetech servo answered")
    print("responders:", found, "-> set grippers[].port / servo_id in configs/handumi/hardware_handumi_preimu.yaml"); return 0 if found else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR)); ap.add_argument("--hardware", default="handumi_v1")
    ap.add_argument("--seconds", type=float, default=0); ap.add_argument("--log", default=None)
    ap.add_argument("--scan", action="store_true", help="probe every /dev/cu.usb* port for Feetech servos (ids 0-15, 1M/500k/115200) and exit")
    if "--scan" in (argv if argv is not None else __import__("sys").argv): return scan()
    a = ap.parse_args(argv)
    cfg = load_config(a.config_dir, hardware=a.hardware)
    hw = cfg.hardware; hw.cameras = []; hw.imus = []
    dm = DeviceManager(hw); dm.build(); dm.connect_all()
    for n, e in dm.errors.items(): print(f"!! {n}: {e}")
    print("gripper calibration:", dm.gripper_calibration_version)
    writer = None
    if a.log:
        fh = open(a.log, "a", newline=""); writer = csv.writer(fh); writer.writerow(["side", "seq", "sample_ns", "raw", "unwrapped", "normalized"])
    t0 = time.time(); nonmono = {s: 0 for s in dm.grippers}; prev = {}
    try:
        while not a.seconds or time.time() - t0 < a.seconds:
            time.sleep(0.5)
            lines = [f"elapsed {time.time()-t0:7.1f}s"]
            for side, g in dm.grippers.items():
                for s in g.buffer.drain():
                    if writer: writer.writerow([s.side, s.seq, s.sample_ns, s.raw_position_mod, s.raw_position, s.normalized])
                    if side in prev and abs(s.raw_position - prev[side]) > 2048: nonmono[side] += 1   # unwrap failure signature
                    prev[side] = s.raw_position
                q = g.quality(); st = g.status()
                lines.append(f"{side.upper():5s} {q.grade(cfg.collector.gripper_quality):6s} {q.hz:6.1f} Hz age {q.age_ms if q.age_ms is not None else float('nan'):6.1f} ms  "
                             f"raw {q.raw}  unwrapped {q.unwrapped}  norm {q.normalized if q.normalized is None else round(q.normalized, 3)}  "
                             f"cal closed/open {st.detail.get('ticks_closed')}/{st.detail.get('ticks_open')}  err {q.errors} jumps>{2048} {nonmono[side]}")
            sys.stdout.write("\x1b[2J\x1b[H" if not a.seconds else ""); print("\n".join(lines)); sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    dm.close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
