"""Backend benchmark over Hpilot episodes (design §23): runs each backend on every episode, aggregates
valid ratio / HOME return drift / static jitter / jumps / IMU agreement / lost events / runtime, writes benchmark.json + a
markdown table. Trajectories that merely *look* smooth do not win — only these metrics are compared.

    python -m handumi_collector.tools.pose_benchmark SESSION --backends opencv_vo orbslam3 [--out DIR]"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from ..config import DEFAULT_CONFIG_DIR, load_pose_cfg
from ..pose.episode_io import write_json
from ..pose.run import process_episode
from .pose_process import episodes_under

METRICS = [("valid_ratio", "valid", "mean"), ("return_translation_mm", "ret_mm", "median"), ("return_rotation_deg", "ret_deg", "median"),
           ("home_start_pos_std_mm", "jit_mm", "median"), ("home_start_rot_std_deg", "jit_deg", "median"), ("jumps_translation", "jmp_t", "sum"),
           ("jumps_rotation", "jmp_r", "sum"), ("imu_residual_deg_median", "imu_deg", "median"), ("lost_events", "lost", "sum"), ("runtime_s", "rt_s", "mean")]


def aggregate(per_episode: list[dict]) -> dict:
    out = {}
    for key, short, how in METRICS:
        vals = [s[key] for e in per_episode for s in e["sides"].values() if s.get(key) is not None]
        if not vals: out[short] = None; continue
        out[short] = float(dict(mean=np.mean, median=np.median, sum=np.sum)[how](vals))
    verdicts = [e["verdict"] for e in per_episode]
    out["episodes"] = len(per_episode); out["pass"] = verdicts.count("PASS"); out["warn"] = verdicts.count("WARN"); out["reject"] = verdicts.count("REJECT")
    out["recovery_rate"] = _recovery(per_episode)
    return out


def _recovery(per_episode) -> float | None:
    lost = sum(s["lost_events"] for e in per_episode for s in e["sides"].values())
    if lost == 0: return None
    recovered = sum(s["lost_events"] - (1 if s["tracking_states"].get("lost", 0) and s["valid_ratio"] < 1 and s["reasons"] and any("lost for" in r for r in s["reasons"]) else 0)
                    for e in per_episode for s in e["sides"].values())
    return float(recovered / lost)


def run_benchmark(session: Path, backends: list[str], cfg, *, out: Path | None = None, cal_dir: Path | None = None, allow_mock=False, backend_factories=None) -> dict:
    eps = episodes_under(session); result = dict(schema="handumi_pose_benchmark/v1", session=str(session), n_episodes=len(eps), backends={})
    for b in backends:
        if b == "mock" and not allow_mock: raise SystemExit("mock backend refused without allow_mock")
        per = []; t0 = time.time()
        for ep in eps:
            s = process_episode(ep, cfg, backend_name=b, cal_dir=cal_dir, backend_factory=(backend_factories or {}).get(b))
            per.append(s)
        result["backends"][b] = dict(aggregate=aggregate(per), wall_s=round(time.time() - t0, 1), per_episode=[dict(episode=e["episode"], verdict=e["verdict"],
                                     sides={k: dict(verdict=v["verdict"], valid=v["valid_ratio"], ret_mm=v["return_translation_mm"], ret_deg=v["return_rotation_deg"],
                                                    imu_deg=v["imu_residual_deg_median"], lost=v["lost_events"], error=v["error"]) for k, v in e["sides"].items()}) for e in per])
    out = out or (session / "derived_benchmark"); out.mkdir(parents=True, exist_ok=True)
    write_json(result, out / "benchmark.json"); (out / "benchmark.md").write_text(markdown(result))
    return result


def markdown(result: dict) -> str:
    cols = [s for _, s, _ in METRICS] + ["pass", "warn", "reject", "recovery_rate"]
    lines = [f"# Pose backend benchmark — {Path(result['session']).name} ({result['n_episodes']} episodes)", "",
             "| backend | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for b, r in result["backends"].items():
        a = r["aggregate"]; lines.append(f"| {b} | " + " | ".join("—" if a.get(c) is None else (f"{a[c]:.3g}" if isinstance(a[c], float) else str(a[c])) for c in cols) + " |")
    lines += ["", "ret_mm is only metric for visual-inertial backends (opencv_vo / dpvo scale is arbitrary).",
              "Selection rule: relative trajectory accuracy > local continuity (valid, lost, jumps) > HOME return drift > global map."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("session"); ap.add_argument("--backends", nargs="+", default=["opencv_vo"]); ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR / "pose.yaml")); ap.add_argument("--cal-dir", default=None); ap.add_argument("--allow-mock", action="store_true")
    a = ap.parse_args(argv); cfg = load_pose_cfg(a.config)
    r = run_benchmark(Path(a.session), a.backends, cfg, out=Path(a.out) if a.out else None, cal_dir=Path(a.cal_dir) if a.cal_dir else None, allow_mock=a.allow_mock)
    print(markdown(r)); return 0


if __name__ == "__main__": raise SystemExit(main())
