"""Pose-source quality report for one recording (2026-10-15): quality histogram, transitions, dropout runs, detector stats.
    .venv/bin/python -m visual_teleop.tools.pose_report <recording_dir>
Reads pose_log.npz (100 Hz, hub fusion ticks) and det_log.npz (per camera frame, kpt mode). Korean summary for the operator."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def runs(mask: np.ndarray, t_s: np.ndarray) -> list[float]:
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None: start = i
        if start is not None and (not m or i == len(mask) - 1):
            end = i if m else i - 1; out.append(float(t_s[end] - t_s[start])); start = None
    return out


def report(rec: Path) -> dict:
    out = {}
    if (rec / "pose_log.npz").exists():
        P = np.load(rec / "pose_log.npz"); r = P["is_right"]; q = P["quality"][r]; t = (P["t_ns"][r] - P["t_ns"][r][0]) / 1e9; names = list(P["quality_names"])
        live = q != 0                                                       # after INIT
        hist = {names[k]: round(100 * float(np.mean(q[live] == k)), 1) for k in range(len(names)) if live.any() and np.any(q[live] == k)}
        tr = {}
        for a, b in zip(q[:-1], q[1:]):
            if a != b: tr[f"{names[a]}->{names[b]}"] = tr.get(f"{names[a]}->{names[b]}", 0) + 1
        dur = float(t[-1]) if len(t) else 0.0
        bad = runs(np.isin(q, [3, 4]) & live, t)                           # LOST or REACQUIRE stretches
        out["pose"] = dict(duration_s=round(dur, 1), quality_pct_after_init=hist, transitions=tr,
                           tracking_to_degraded_or_lost_per_min=round(60 * sum(v for k, v in tr.items() if k.startswith("TRACKING->")) / max(dur, 1e-9), 1),
                           lost_runs=len(bad), lost_median_s=round(float(np.median(bad)), 2) if bad else 0.0, lost_max_s=round(max(bad), 2) if bad else 0.0)
    if (rec / "det_log.npz").exists():
        D = np.load(rec / "det_log.npz"); acc = D["accepted"]; t = (D["t_ns"] - D["t_ns"][0]) / 1e9; miss = runs(~acc, t)
        out["detector"] = dict(frames=int(len(acc)), accepted_pct=round(100 * float(acc.mean()), 1), any_detection_pct=round(100 * float(np.mean(D["n_det"] > 0)), 1),
                               rejected_jump=int(D["rejected_jump"].sum()), rejected_depth=int(D["rejected_depth"].sum()),
                               dropout_median_s=round(float(np.median(miss)), 2) if miss else 0.0, dropout_max_s=round(max(miss), 2) if miss else 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("rec"); a = ap.parse_args(); r = report(Path(a.rec).expanduser())
    p, d = r.get("pose"), r.get("detector")
    if p:
        print(f"[포즈 품질] {p['duration_s']} s, INIT 이후 비율: " + ", ".join(f"{k} {v}%" for k, v in p["quality_pct_after_init"].items()))
        print(f"  TRACKING→DEGRADED/LOST 전이: 분당 {p['tracking_to_degraded_or_lost_per_min']}회 | 전이 {p['transitions']}")
        print(f"  LOST/REACQUIRE 구간 {p['lost_runs']}개, 중앙값 {p['lost_median_s']} s, 최장 {p['lost_max_s']} s")
    if d:
        print(f"[검출기] {d['frames']}프레임: 채택 {d['accepted_pct']}%, 검출 자체 {d['any_detection_pct']}%, 점프 거부 {d['rejected_jump']}, 깊이 거부 {d['rejected_depth']}, "
              f"끊김 중앙값 {d['dropout_median_s']} s / 최장 {d['dropout_max_s']} s")
    (Path(a.rec).expanduser() / "pose_report.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
