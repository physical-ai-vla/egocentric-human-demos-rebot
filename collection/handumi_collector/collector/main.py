"""python -m handumi_collector.collector.main [--hardware handumi_v1|dev_3c922|mock] [--mock] [--session DIR] [--headless-smoke S]"""
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
from ..config import DEFAULT_CONFIG_DIR, load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="HandUMI egocentric collector")
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--hardware", default="handumi_v1", help="hardware profile: handumi_v1 (production) | dev_3c922 | mock, or a yaml path")
    ap.add_argument("--mock", action="store_true", help="force every backend to mock (no hardware)")
    ap.add_argument("--session", default=None, help="resume/append to an existing session directory")
    ap.add_argument("--dataset", default=None, help="dataset mode from tasks.yaml `datasets` (Hpilot | H120): session prefix + target per order")
    ap.add_argument("--headless-smoke", type=float, default=None, metavar="SECONDS", help="record one episode without UI, then exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(a.config_dir, mock=a.mock, hardware=a.hardware)
    sd = Path(a.session) if a.session else None
    if a.headless_smoke:
        from .session import CollectorSession
        s = CollectorSession.create(cfg, session_dir=sd, dataset=a.dataset)
        # Let the devices deliver for a moment before asking whether they are ready. Readiness compares the measured
        # rate against the configured one, and a camera that has been running for a millisecond has no measured rate --
        # so checking immediately after connect_all passes everything, which is how two cameras producing nothing were
        # recorded for thirty seconds and kept.
        time.sleep(2.5)
        ok, why = s.can_record()
        print("devices:", {n: (st.connected, st.error) for n, st in s.devices.statuses().items()})
        if not ok: print("cannot record:", why); s.close(); return 2
        s.start(); time.sleep(a.headless_smoke); s.stop(); m = s.keep(notes="headless smoke")
        print("episode:", m["episode_dir"], "quality:", m["quality"], "streams:", {k: v["frames"] for k, v in m["streams"].items()},
              "sensors:", m["sensor_messages"], "events:", m["event_kinds"])
        s.close(); return 0
    from ..ui.app import run
    return run(cfg, session_dir=sd, dataset=a.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
