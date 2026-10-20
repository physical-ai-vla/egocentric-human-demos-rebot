"""Config for visual teleop (2026-10-14). One YAML (config/teleop_rebot.yaml) holds every hardware constant (spec section 40);
this module only loads it and refuses obvious mistakes. It returns plain nested dicts (attribute access via `Cfg`), so the YAML
is the schema and no constant is duplicated in Python."""
from __future__ import annotations
from pathlib import Path
import copy
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

DEFAULT_PATH = Path(__file__).resolve().parent / "config" / "teleop_rebot.yaml"
CALIB_PATH = Path(__file__).resolve().parent / "config" / "calibration.yaml"          # written ONLY by tools/calib_right.py
CALIB_LOG = Path(__file__).resolve().parent / "config" / "calibration_log.jsonl"
SIDES = ("left", "right")


class Cfg(dict):
    """dict with attribute access (nested)."""
    def __getattr__(self, k):
        try: v = self[k]
        except KeyError: raise AttributeError(k) from None
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out: raise KeyError(f"unknown config key {path}{k}")
        out[k] = _merge(out[k], v, f"{path}{k}.") if isinstance(v, dict) and isinstance(out[k], dict) else copy.deepcopy(v)
    return out


def load(path: str | Path | None = None, overrides: dict | None = None) -> Cfg:
    """Default YAML, then `path` (same schema, partial), then `overrides`. Unknown keys raise."""
    d = yaml.safe_load(DEFAULT_PATH.read_text())
    if CALIB_PATH.exists(): d = _merge(d, yaml.safe_load(CALIB_PATH.read_text()) or {})
    if path is not None and Path(path).resolve() != DEFAULT_PATH: d = _merge(d, yaml.safe_load(Path(path).read_text()) or {})
    if overrides: d = _merge(d, overrides)
    c = Cfg(d); validate(c); return c


def validate(c: Cfg) -> None:
    g = c.gripper
    if not g["close_threshold"] < g["open_threshold"]: raise ValueError("gripper.close_threshold must be < open_threshold (hysteresis)")
    if not 0 < g["short_dropout_s"] < g["long_dropout_s"]: raise ValueError("gripper dropout: 0 < short < long")
    for s in SIDES:
        ws = c.workspace[s]
        if not np.all(np.asarray(ws["min_xyz"]) < np.asarray(ws["max_xyz"])): raise ValueError(f"workspace.{s}: min must be < max")
    if c.workspace["mode"] not in ("hold", "clamp"): raise ValueError("workspace.mode must be hold|clamp")
    if c.robot["live"]: raise ValueError("robot.live must not be set in a config file; the operator arms live mode at runtime")


def T_W_C(c: Cfg) -> tuple[np.ndarray, bool]:
    e = c.cameras["T_W_C"]; return np.asarray(e["matrix"], np.float64), bool(e["measured"])


def R_I_H(c: Cfg, side: str) -> tuple[np.ndarray, bool]:
    e = c.imu["R_I_H"][side]; return Rotation.from_rotvec(np.radians(e["rotvec_deg"])).as_matrix(), bool(e["measured"])


def tracker_missing(c: Cfg) -> list[str]:
    tc = c.fusion["tracker"]; miss = []
    if not tc["T_TB_TCP"].get("pivot_measured"): miss.append("fusion.tracker.T_TB_TCP.t (calib_tracker pivot)")
    if not tc["T_W_TW"].get("measured"): miss.append("fusion.tracker.T_W_TW (calib_tracker world)")
    if not tc["T_TB_TCP"].get("mount_measured"): miss.append("fusion.tracker.T_TB_TCP.R (calib_tracker mount)")
    return miss


def unmeasured(c: Cfg, sides=SIDES) -> list[str]:
    """Calibrations still at their placeholder for the given arm side(s) AND the active pose source. Arm stages refuse while non-empty."""
    if c.fusion["position"]["source"] == "tracker": return tracker_missing(c)   # 10-15: a 6DoF tracker needs only its own extrinsics
    out = [] if T_W_C(c)[1] else ["cameras.T_W_C"]
    out += [f"imu.serial_number.{s}" for s in sides if not c.imu[s]["serial_number"]]
    out += [f"imu.R_I_H.{s}" for s in sides if not R_I_H(c, s)[1]]
    out += [f"time_sync.camera_imu_offset_ms.{s}" for s in sides if c.time_sync["camera_imu_offset_ms"][s] is None]
    if c.init["method"] == "known" and not c.init["measured"]: out.append("init.R_W_H_rotvec_deg")   # motion: measured every INIT
    return out


def R_W_H_init(c: Cfg, side: str) -> np.ndarray:
    return Rotation.from_rotvec(np.radians(c.init["R_W_H_rotvec_deg"][side])).as_matrix()


def save_calibration(updates: dict, provenance: dict) -> None:
    """Merge `updates` (same schema as the YAML, partial) into config/calibration.yaml and log provenance."""
    import json, time
    base = _merge(yaml.safe_load(DEFAULT_PATH.read_text()), {})          # schema check: unknown keys raise
    _merge(base, updates)
    cur = yaml.safe_load(CALIB_PATH.read_text()) if CALIB_PATH.exists() else {}
    def deep(a, b):
        for k, v in b.items(): a[k] = deep(dict(a.get(k) or {}), v) if isinstance(v, dict) else v
        return a
    CALIB_PATH.write_text("# written by visual_teleop/tools/calib_right.py -- see calibration_log.jsonl for provenance\n" + yaml.safe_dump(deep(cur or {}, updates), sort_keys=False))
    with open(CALIB_LOG, "a") as f: f.write(json.dumps(dict(t_wall=time.time(), updates=updates, **provenance), default=str) + "\n")

