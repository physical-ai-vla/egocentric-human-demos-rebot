"""Offline pose processing (M2).  python -m handumi_collector.tools.pose_process EPISODE_OR_SESSION [--backend opencv_vo] [--side left]
[--cal-dir DIR] [--allow-mock] [--offset-check]. Writes <episode>/derived/pose_<backend>/ (raw untouched) and prints the QA verdict."""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from ..collector.episode_manager import EP_RE
from ..config import DEFAULT_CONFIG_DIR, load_pose_cfg
from ..pose.backends import available_backends
from ..pose.run import process_episode


def episodes_under(p: Path) -> list[Path]:
    if (p / "episode_meta.json").exists() or (p / "sensors.mcap").exists(): return [p]
    return sorted(q for q in p.iterdir() if q.is_dir() and EP_RE.match(q.name) and not (q / ".incomplete").exists())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="HandUMI offline pose pipeline (fiducial-free VIO/VO -> TCP -> QA)")
    ap.add_argument("path", help="episode dir or session dir")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR / "pose.yaml"))
    ap.add_argument("--backend", default=None, help=f"override pose.yaml backend: {available_backends()}")
    ap.add_argument("--side", action="append", default=None, choices=["left", "right"])
    ap.add_argument("--cal-dir", default=None, help="calibration directory (default configs/calibration)")
    ap.add_argument("--allow-mock", action="store_true", help="the mock backend replays ground truth; refused for data unless explicitly allowed")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_pose_cfg(a.config); backend = a.backend or cfg.backend
    if backend == "mock" and not a.allow_mock: print("refusing backend=mock without --allow-mock (mock replays ground truth; it is not a tracker)"); return 2
    eps = episodes_under(Path(a.path))
    if not eps: print("no episodes under", a.path); return 2
    rc = 0
    for ep in eps:
        try:
            s = process_episode(ep, cfg, backend_name=backend, sides=a.side, cal_dir=Path(a.cal_dir) if a.cal_dir else None)
        except Exception as exc:
            print(f"{ep.name}: FAILED {exc!r}"); rc = 1; continue
        sides = "  ".join(f"{k}:{v['verdict']} valid={v['valid_ratio']:.3f} ret={v['return_translation_mm'] and round(v['return_translation_mm'], 1)}mm/"
                          f"{v['return_rotation_deg'] and round(v['return_rotation_deg'], 2)}deg" for k, v in s["sides"].items())
        print(f"{ep.name}: {s['verdict']}  [{backend}]  {sides}")
        for k, v in s["sides"].items():
            for r in v["reasons"]: print(f"    {k}: {r}")
    return rc


if __name__ == "__main__": raise SystemExit(main())
