"""Session + episode bookkeeping: directory layout, episode numbering, per-order valid counters, KEEP / DISCARD.
Counters are rebuilt from episode_meta.json on start, so a crash or restart never loses the count."""
from __future__ import annotations
import json
import platform
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from .timeline import ClockAnchor
from ..config import AppConfig
from .ownership import give_back

EP_RE = re.compile(r"^episode_(\d{6})$")


def session_name(prefix: str, t: float | None = None) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(t))}"


@dataclass
class SessionMeta:
    name: str
    operator: str
    started_wall_iso: str
    hardware_version: str
    calibration_version: str
    cube_set: str
    orders: list[str]
    target_per_order: int
    devices: dict = field(default_factory=dict)
    video_listing: list[str] = field(default_factory=list)
    host: dict = field(default_factory=dict)
    clock_anchor: dict = field(default_factory=dict)
    notes: str = ""
    collector_version: str = "0.1.0"

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=1)


class EpisodeManager:
    def __init__(self, cfg: AppConfig, *, session_dir: Path | None = None, dataset: str | None = None) -> None:
        self.cfg = cfg
        root = cfg.resolve(cfg.collector.dataset_root)
        self.dataset = dataset or cfg.collector.session_prefix
        if dataset and dataset in cfg.tasks.datasets:
            spec = cfg.tasks.datasets[dataset]
            cfg.tasks.target_per_order = int(spec["target_per_order"])
            # a pilot may want a fixed episode count that six balanced orders cannot express (10 = 5 x 2), so a
            # mode may also narrow the order set. Unknown orders are refused rather than silently dropped.
            if spec.get("orders"):
                bad = [o for o in spec["orders"] if o not in cfg.tasks.orders]
                if bad: raise ValueError(f"dataset {dataset!r} lists orders not in tasks.orders: {bad}")
                cfg.tasks.orders = list(spec["orders"])
        self.session_dir = session_dir or (root / self.dataset / session_name(self.dataset))
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "_discarded").mkdir(exist_ok=True)
        give_back(self.session_dir)          # the head camera needs sudo; the data does not belong to root
        self.anchor = ClockAnchor.now()
        self.counts: dict[str, int] = {o: 0 for o in cfg.tasks.orders}
        self.raw_total = 0
        self.rejected = 0
        self.rebuild_counts()

    # ---------------------------------------------------------------- session
    def write_session_meta(self, *, devices: dict, video_listing: list[str], extra: dict | None = None) -> Path:
        p = self.session_dir / "session_meta.json"
        if p.exists():   # resumed session: keep the original, append a resume record
            d = json.loads(p.read_text()); d.setdefault("resumes", []).append(dict(wall_iso=time.strftime("%Y-%m-%dT%H:%M:%S%z"), clock_anchor=self.anchor.to_dict()))
            p.write_text(json.dumps(d, indent=1)); return p
        t = self.cfg.tasks; hw = self.cfg.hardware
        m = SessionMeta(self.session_dir.name, self.cfg.collector.operator, time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        hw.hardware_version, hw.calibration_version, t.cube_set, list(t.orders), t.target_per_order,
                        devices=devices, video_listing=video_listing,
                        host=dict(platform=platform.platform(), python=platform.python_version(), node=platform.node()),
                        clock_anchor=self.anchor.to_dict(), notes=self.cfg.collector.notes)
        d = json.loads(m.to_json()); d.update(dataset_name=self.dataset, imu_present=any(k.startswith("imu_") for k in devices), 
                                              orbbec_present=any(k == "head_depth" and v.get("actual") for k, v in devices.items()))
        d.update(extra or {})
        p.write_text(json.dumps(d, indent=1)); return p

    # ---------------------------------------------------------------- episodes
    def episodes(self) -> list[Path]:
        return sorted(p for p in self.session_dir.iterdir() if p.is_dir() and EP_RE.match(p.name))

    def next_episode_dir(self) -> Path:
        ids = [int(EP_RE.match(p.name).group(1)) for p in self.episodes()]
        ids += [int(EP_RE.match(p.name).group(1)) for p in (self.session_dir / "_discarded").iterdir() if EP_RE.match(p.name)]
        return self.session_dir / f"episode_{(max(ids) + 1 if ids else 1):06d}"

    def rebuild_counts(self) -> None:
        self.counts = {o: 0 for o in self.cfg.tasks.orders}; self.raw_total = 0; self.rejected = 0
        for p in self.episodes():
            mp = p / "episode_meta.json"
            if (p / ".incomplete").exists() or not mp.exists():
                continue     # crashed mid-episode: left in place for inspection, never counted
            m = json.loads(mp.read_text()); self.raw_total += 1
            if m.get("status") == "KEEP" and m.get("order") in self.counts:
                self.counts[m["order"]] += 1
        self.rejected += sum(1 for p in (self.session_dir / "_discarded").iterdir() if EP_RE.match(p.name))
        self._write_counters()

    # --- reBot-feasible quota -----------------------------------------------------------------
    # The 2026-09-22 feasibility audit found only 36% of bimanual chunks executable on the reBot, so a
    # raw episode quota over-states how much trainable data a session actually produced. The audit runs
    # offline (SLAM takes minutes per episode) and drops its result here; the collector reads it every
    # time so a session in progress picks up new results, and falls back to the raw counts when the file
    # is absent. Never blocks recording on the audit.
    FEASIBLE_FILE = "feasible_counters.json"

    def feasible_counts(self) -> dict:
        """{order: reBot-feasible episode count} from the audit, or {} if it has not run."""
        p = self.session_dir / self.FEASIBLE_FILE
        if not p.exists():
            return {}
        try:
            d = json.loads(p.read_text())
        except Exception:
            return {}
        return {o: int(d.get("feasible", {}).get(o, 0)) for o in self.cfg.tasks.orders}

    def quota_counts(self) -> tuple[dict, str]:
        """the counts the quota is judged on, and which basis is in use"""
        f = self.feasible_counts()
        return (f, "feasible") if f else (self.counts, "valid")

    def _write_counters(self) -> None:
        f = self.feasible_counts()
        body = dict(valid=self.counts, raw_total=self.raw_total, rejected=self.rejected,
                    target_per_order=self.cfg.tasks.target_per_order)
        if f:
            body["feasible"] = f
            body["quota_basis"] = "feasible"
        else:
            body["quota_basis"] = "valid"
            body["note"] = "run the reBot feasibility audit to switch the quota onto feasible counts"
        (self.session_dir / "counters.json").write_text(json.dumps(body, indent=1))

    def keep(self, episode_dir: Path, order: str) -> None:
        self.raw_total += 1; self.counts[order] = self.counts.get(order, 0) + 1; self._write_counters()

    def discard(self, episode_dir: Path) -> Path:
        """Move (never delete) to _discarded/; meta status updated if present."""
        mp = episode_dir / "episode_meta.json"
        if mp.exists():
            m = json.loads(mp.read_text()); m["status"] = "DISCARD"; mp.write_text(json.dumps(m, indent=1))
        dst = self.session_dir / "_discarded" / episode_dir.name
        if dst.exists(): shutil.rmtree(dst)
        shutil.move(str(episode_dir), str(dst))
        self.rejected += 1; self.raw_total += 1; self._write_counters()
        return dst

    def incomplete_episodes(self) -> list[Path]:
        return [p for p in self.episodes() if (p / ".incomplete").exists()]

    def next_order(self) -> str:
        """Least-collected order, counted on the reBot-feasible quota when the audit has run.

        Balancing on raw KEEP counts spends effort on orders that already have enough *usable* data while
        starving one whose episodes keep failing the feasibility audit -- which is exactly the failure the
        36% audit result predicts. Ties fall back to config order.
        """
        c, _ = self.quota_counts()
        return min(self.cfg.tasks.orders, key=lambda o: (c.get(o, 0), self.cfg.tasks.orders.index(o)))
