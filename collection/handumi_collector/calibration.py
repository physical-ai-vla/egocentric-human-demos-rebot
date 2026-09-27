"""Versioned calibration files. Gripper: configs/calibration/gripper_vNNN.yaml (ticks_closed/ticks_open per side).
Sessions record only the version name; raw ticks are always stored, so a calibration can be re-applied offline."""
from __future__ import annotations
import re
import time
from pathlib import Path
import yaml
from .config import REPO_ROOT

CAL_DIR = REPO_ROOT / "configs" / "calibration"
_GRIP_RE = re.compile(r"^gripper_v(\d{3})\.ya?ml$")


def gripper_versions(cal_dir: Path | None = None) -> list[Path]:
    cal_dir = cal_dir or CAL_DIR
    return sorted((p for p in cal_dir.glob("gripper_v*.y*ml") if _GRIP_RE.match(p.name)), key=lambda p: int(_GRIP_RE.match(p.name).group(1)))


def load_gripper_calibration(which: str = "latest", cal_dir: Path | None = None) -> tuple[str | None, dict]:
    """Returns (version_name, {side: {ticks_closed, ticks_open}}). (None, {}) when nothing is calibrated yet."""
    cal_dir = cal_dir or CAL_DIR
    vs = gripper_versions(cal_dir)
    if which != "latest":
        vs = [p for p in vs if p.stem == which]
    if not vs: return None, {}
    p = vs[-1]
    d = yaml.safe_load(p.read_text()) or {}
    return p.stem, {k: v for k, v in d.get("sides", {}).items()}


def save_gripper_calibration(sides: dict, *, cal_dir: Path | None = None, hardware_version: str = "", notes: str = "") -> Path:
    """Write the next gripper_vNNN.yaml (never overwrites an existing version)."""
    cal_dir = cal_dir or CAL_DIR
    cal_dir.mkdir(parents=True, exist_ok=True)
    vs = gripper_versions(cal_dir)
    n = int(_GRIP_RE.match(vs[-1].name).group(1)) + 1 if vs else 1
    p = cal_dir / f"gripper_v{n:03d}.yaml"
    p.write_text(yaml.safe_dump(dict(schema="handumi_gripper_calibration/v1", created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                     hardware_version=hardware_version, notes=notes,
                                     convention="normalized 0 = closed, 1 = open (human raw convention; exporters transform to robot grip01)",
                                     sides={s: dict(ticks_closed=int(v["ticks_closed"]), ticks_open=int(v["ticks_open"])) for s, v in sides.items()}),
                                sort_keys=False))
    return p
