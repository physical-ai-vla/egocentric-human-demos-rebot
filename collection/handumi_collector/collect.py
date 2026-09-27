"""Production entry point:  .venv/bin/python -m handumi_collector.collect [--hardware handumi_preimu] [--dataset Hpilot|H120] [--mock]
Defaults to the pre-IMU profile (C922 + L/R Arducam + L/R Feetech; Orbbec/IMU optional)."""
from __future__ import annotations
import sys
from .collector.main import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--hardware" not in argv and "--mock" not in argv: argv = ["--hardware", "handumi_preimu", *argv]
    raise SystemExit(main(argv))
