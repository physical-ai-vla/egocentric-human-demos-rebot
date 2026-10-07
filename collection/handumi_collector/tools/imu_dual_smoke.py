"""Both wrists at once, through the production device path, gated per side.

    python -m handumi_collector.tools.imu_dual_smoke --seconds 60
    python -m handumi_collector.tools.imu_dual_smoke --hardware mock --seconds 5     # exercise the tool without boards

One board passing its gates says the part works; it says nothing about whether the rig records two hands at once, which
is the state the collector actually runs in. This drives `DeviceManager` — the same builder, driver threads and serial
resolution the recorder uses — so a pass here is a statement about the production path, not about a bespoke test harness.
It writes one standard session directory per side, so `validate_session` and `plot_imu` read the result unchanged.

Identity is the first thing it prints and the first thing that can fail: which hand a board belongs to is a claim, and a
wrong one silently mislabels every episode. DeviceManager refuses two sides that name the same board, and TeensyImu
refuses to resolve a side by port order, so getting here at all means both serials were explicit and distinct."""
from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from ..config import DEFAULT_CONFIG_DIR, load_config
from ..devices.base import now_ns
from ..devices.imu_stats import stream_stats
from ..devices.manager import DeviceManager
from ..devices.teensy_imu import TX_RING_PACKETS

STATUS_PERIOD_S = 1.0        # firmware STATUS_PERIOD_MS
from .imu_logger import CSV_COLUMNS, DRAIN_S, Session
from .validate_session import gates, load_session


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--hardware", default="handumi_v1", help="hardware profile; IMUs only, cameras and grippers are ignored")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out-dir", default="data", help="session root; one imu_<side>_<stamp>/ directory per side")
    ap.add_argument("--accel-fs-g", type=float, default=8.0)
    ap.add_argument("--moving", action="store_true", help="the rig was not static — skip the 1 g gravity gate")
    ap.add_argument("--drain", type=float, default=DRAIN_S, help="discard this long before recording, to clear the device-side ring backlog")
    a = ap.parse_args(argv)

    cfg = load_config(a.config_dir, hardware=a.hardware)
    hw = cfg.hardware; hw.cameras = []; hw.grippers = []
    if len(hw.imus) < 2:
        print(f"profile {a.hardware} declares {len(hw.imus)} IMU(s); a dual-wrist smoke test needs two", file=sys.stderr); return 2
    dm = DeviceManager(hw)
    dm.build()
    dm.connect_all()                                  # raises if two sides claim one board
    if dm.errors:
        for n, e in dm.errors.items(): print(f"!! {n}: {e}", file=sys.stderr)
        for imu in dm.imus.values(): imu.stop()      # one side failing must not leave the other holding its port
        return 2

    print("identity")
    for side, imu in sorted(dm.imus.items()):
        print(f"  {side:5s}  serial={getattr(imu, 'serial_number', None)}  port={getattr(imu, 'port', None)}  "
              f"configured={next(i.serial_number for i in hw.imus if i.side == side)!r}")
    ports = {side: getattr(imu, "port", None) for side, imu in dm.imus.items()}
    real = {side: p for side, p in ports.items() if dm.imus[side].cfg.backend != "mock"}   # mock devices all report port "mock"
    if len(set(real.values())) != len(real):
        print(f"!! two sides resolved to the same port: {real}", file=sys.stderr)
        for imu in dm.imus.values(): imu.stop()
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    sessions = {side: Session(Path(a.out_dir) / f"imu_{side}_{stamp}", 200) for side in dm.imus}
    for side, s in sessions.items(): s.log(f"dual smoke, profile {a.hardware}, port {ports[side]}")
    # Draining the host buffer once is not enough: the Teensy's own ring still holds samples taken while nobody was
    # reading, it drops the newest when full, and the driver thread feeds them through afterwards. Keep discarding for a
    # moment so recording starts on live data — same reason as imu_logger's drain, one level further up the stack.
    t_drain = time.monotonic()
    while time.monotonic() - t_drain < a.drain:
        time.sleep(0.05)
        for imu in dm.imus.values(): imu.buffer.drain()
    # The device reports once a second, so the first status packet after the drain still describes the second *before* it
    # -- backlog and all. Resetting against that one leaked the pre-session peak and drops into the session's numbers:
    # a 500 Hz run, where the ring fills in 0.51 s, came out as high-water 255/256 and 16270 device drops next to a
    # host-side loss of 0.0000 %. Wait for a genuinely fresh packet from each side before taking the baseline.
    stale = {side: ((getattr(imu, "last_status", None) or {}).get("t_us")) for side, imu in dm.imus.items()}
    t_wait = time.monotonic()
    while time.monotonic() - t_wait < 1.5 * STATUS_PERIOD_S:
        time.sleep(0.05)
        for imu in dm.imus.values(): imu.buffer.drain()
        if all((getattr(imu, "last_status", None) or {}).get("t_us") not in (None, stale[side])
               for side, imu in dm.imus.items()): break
    dev_at_start = {}
    for side, imu in dm.imus.items():
        imu.ring_high_water = 0
        st = getattr(imu, "last_status", None) or {}
        dev_at_start[side] = (st.get("emitted", 0), st.get("dropped", 0))   # counters are cumulative since boot
    # monotonic time does not advance while macOS is asleep, so a duration measured on it alone counts only the hours
    # the machine was awake. An overnight run asked for three hours and stopped after seven and a quarter.
    t_start = now_ns(); t_start_wall = time.time()
    t_wall = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"\nrecording {a.seconds:g} s into {a.out_dir}/imu_<side>_{stamp}/ ...")
    t_print = time.monotonic()
    while (now_ns() - t_start) / 1e9 < a.seconds and (time.time() - t_start_wall) < a.seconds:
        time.sleep(0.1)
        for side, imu in dm.imus.items():
            for s in imu.buffer.drain():
                sessions[side].write(s.side, s.seq, s.device_timestamp_us, s.host_receive_ns,
                                     s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.temperature_c)
        if time.monotonic() - t_print >= 2.0:
            t_print = time.monotonic()
            print("\r  " + "   ".join(f"{side} {sessions[side].n:7d} @ {imu.quality().hz:6.1f} Hz"
                                      for side, imu in sorted(dm.imus.items())), end="", flush=True)
    t_stop = now_ns(); t_stop_wall = time.time()
    for imu in dm.imus.values(): imu.stop()

    print("\n")
    failed_sides = []
    for side, sess in sorted(sessions.items()):
        imu = dm.imus[side]
        q = imu.quality()
        stats = stream_stats(**sess.arrays(), expected_rate_hz=float(next(i.rate_hz for i in hw.imus if i.side == side)),
                             accel_fs_g=a.accel_fs_g, crc_errors=q.crc_errors, resyncs=getattr(imu, "parser", None) and imu.parser.resyncs or 0)
        dev = getattr(imu, "last_status", None) or {}
        hi = getattr(imu, "ring_high_water", 0)
        sess.finish(dict(tool="handumi_collector.tools.imu_dual_smoke", schema_version=1, created_utc=t_wall, side=side,
                         firmware=dict(version=dev.get("fw"), device_rate_hz=dev.get("sample_rate_hz"),
                                       samples_emitted=(dev.get("emitted") or 0) - dev_at_start[side][0],
                                       samples_dropped=(dev.get("dropped") or 0) - dev_at_start[side][1],
                                       cumulative_since_boot=dict(emitted=dev.get("emitted"), dropped=dev.get("dropped")),
                                       ring_high_water=hi, ring_capacity=TX_RING_PACKETS,
                                       ring_high_water_percent=round(100.0 * hi / TX_RING_PACKETS, 2)),
                         port=ports[side], serial_number=getattr(imu, "serial_number", None), hardware_profile=hw.profile,
                         wire_format="binary", drops_detectable=True, synthetic=imu.cfg.backend == "mock",
                         host=dict(clock="time.monotonic_ns", platform=platform.platform()),
                         t_start_monotonic_ns=t_start, t_stop_monotonic_ns=t_stop,
                         wall_duration_s=round(t_stop_wall - t_start_wall, 3),
                         monotonic_duration_s=round((t_stop - t_start) / 1e9, 3),
                         accel_fs_g=a.accel_fs_g, csv_columns=CSV_COLUMNS, stats=stats))
        print(f"{sess.path}  [{side}]")
        if sess.n == 0:
            why = getattr(imu, "last_device_log", None)
            print(f"  [FAIL] samples         0 recorded in {a.seconds:g} s"
                  + (f" — device says: {why!r}" if why else " — the board sent nothing at all"))
            failed_sides.append(f"{side}: no samples"); print(); continue
        st, prov = load_session(sess.path, side)
        rows = gates(st, prov, static=not a.moving, allow_synthetic=True)
        for name, ok, detail in rows: print(f"  [{'PASS' if ok else 'FAIL'}] {name:15s} {detail}")
        bad = [n for n, ok, _ in rows if not ok]
        if bad: failed_sides.append(f"{side}: {', '.join(bad)}")
        print()

    if failed_sides:
        print("FAIL — " + "; ".join(failed_sides)); return 1
    print(f"PASS — both wrists, {a.seconds:g} s, distinct boards"); return 0


if __name__ == "__main__":
    sys.exit(main())
