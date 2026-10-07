"""Compare OpenVINS online camera-IMU estimates ACROSS takes — step 4 of M1-0 (refine / validate, never accept one run).

    python -m ego_teleop.tools.m1_vio EP1 --side right --backend openvins --mode calibration     # per take
    python -m ego_teleop.tools.calib_converge EP1 EP2 EP3 --side right [--prior-from-bundle] [--write-bundle]

For each take it reads derived/pose_openvins/calib_trace_<side>.parquet (written in calibration mode), takes the converged tail
(last `tail_frac` of the tracked frames), and reports per-take mean/std plus the spread BETWEEN takes. Agreement gates default to
1 deg / 5 mm / 2 ms, matching the runbook. `--write-bundle` stores the cross-take mean, and its provenance is recorded as
`openvins_online`, which `require_production()` refuses on its own — an offline Kalibr result must be the primary source."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from handumi_collector.pose.episode_io import derived_dir
from handumi_collector.pose.se3 import mean_pose, pose7_to_T, rotation_angle_deg

COLS = ("T_camera_imu_x", "T_camera_imu_y", "T_camera_imu_z", "T_camera_imu_qx", "T_camera_imu_qy", "T_camera_imu_qz", "T_camera_imu_qw")


def take_summary(ep_path: Path, side: str, *, backend: str = "openvins", tail_frac: float = 0.3, min_feat: int = 20) -> dict:
    p = derived_dir(Path(ep_path), backend) / f"calib_trace_{side}.parquet"
    if not p.exists(): raise SystemExit(f"{p} missing — run m1_vio --mode calibration on this episode first")
    df = pd.read_parquet(p)
    df = df[df["n_feat"].fillna(0) >= min_feat] if "n_feat" in df else df
    if len(df) < 10: raise SystemExit(f"{p}: only {len(df)} usable rows")
    tail = df.iloc[int(len(df) * (1 - tail_frac)):]
    Ts = np.array([pose7_to_T(r) for r in tail[list(COLS)].to_numpy(np.float64)])
    T = mean_pose(Ts)
    spread_deg = float(np.max([rotation_angle_deg(T[:3, :3], X[:3, :3]) for X in Ts]))
    return dict(episode=Path(ep_path).name, n_rows=int(len(df)), n_tail=int(len(tail)), T_camera_imu=T,
                t_mm=(T[:3, 3] * 1e3).round(3).tolist(), within_take_rot_spread_deg=round(spread_deg, 3),
                within_take_trans_std_mm=float(np.linalg.norm(Ts[:, :3, 3].std(axis=0)) * 1e3),
                time_offset_ms=float(-tail["dt_cam_imu_ms"].mean()), time_offset_std_ms=float(tail["dt_cam_imu_ms"].std()),
                d_rot_deg_from_prior=float(tail["d_rot_deg"].mean()) if "d_rot_deg" in tail else None,
                d_trans_mm_from_prior=float(tail["d_trans_mm"].mean()) if "d_trans_mm" in tail else None)


def compare(takes: list[dict], *, gates: dict | None = None) -> dict:
    g = dict(rot_deg=1.0, trans_mm=5.0, time_ms=2.0); g.update(gates or {})
    Ts = np.array([t["T_camera_imu"] for t in takes]); T = mean_pose(Ts) if len(Ts) > 1 else Ts[0]
    rot = [rotation_angle_deg(T[:3, :3], X[:3, :3]) for X in Ts]
    trans = np.linalg.norm(Ts[:, :3, 3] - T[:3, 3], axis=1) * 1e3
    dt = np.array([t["time_offset_ms"] for t in takes])
    out = dict(n_takes=len(takes), T_camera_imu=T.round(9).tolist(), time_offset_ms=float(dt.mean()),
               between_takes=dict(max_rot_deg=round(float(np.max(rot)), 3), max_trans_mm=round(float(np.max(trans)), 2),
                                  time_offset_spread_ms=round(float(dt.max() - dt.min()), 3) if len(dt) > 1 else 0.0), gates=g)
    fails = []
    if len(takes) < 2: fails.append("only one take — convergence cannot be judged (record at least two excitation takes)")
    if out["between_takes"]["max_rot_deg"] > g["rot_deg"]: fails.append(f"rotation spread {out['between_takes']['max_rot_deg']:.2f} deg > {g['rot_deg']}")
    if out["between_takes"]["max_trans_mm"] > g["trans_mm"]: fails.append(f"translation spread {out['between_takes']['max_trans_mm']:.1f} mm > {g['trans_mm']}")
    if out["between_takes"]["time_offset_spread_ms"] > g["time_ms"]: fails.append(f"time-offset spread {out['between_takes']['time_offset_spread_ms']:.2f} ms > {g['time_ms']}")
    out["verdict"] = "FAIL" if fails else "PASS"; out["reasons"] = fails
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("episodes", nargs="+"); ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--backend", default="openvins"); ap.add_argument("--tail-frac", type=float, default=0.3)
    ap.add_argument("--write-bundle", action="store_true", help="store the cross-take mean (provenance: openvins_online — NOT production-ready on its own)")
    ap.add_argument("--cal-dir", default=None); ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    takes = [take_summary(Path(e), a.side, backend=a.backend, tail_frac=a.tail_frac) for e in a.episodes]
    for t in takes:
        print(f"{t['episode']:>28}  t={t['t_mm']} mm  dt={t['time_offset_ms']:+.2f}±{t['time_offset_std_ms']:.2f} ms  "
              f"within-take {t['within_take_rot_spread_deg']:.2f} deg/{t['within_take_trans_std_mm']:.1f} mm"
              + (f"  moved {t['d_rot_deg_from_prior']:.2f} deg/{t['d_trans_mm_from_prior']:.1f} mm from the prior" if t.get("d_rot_deg_from_prior") is not None else ""))
    res = compare(takes)
    b = res["between_takes"]
    print(f"between takes: rot {b['max_rot_deg']} deg, trans {b['max_trans_mm']} mm, dt spread {b['time_offset_spread_ms']} ms -> {res['verdict']}")
    for r in res["reasons"]: print(f"  - {r}")
    if a.json: Path(a.json).write_text(json.dumps(dict(takes=[{k: v for k, v in t.items() if k != 'T_camera_imu'} for t in takes], comparison=res), indent=1))
    if a.write_bundle:
        from ..calibration.bundle import load_bundle, save_bundle
        cal_dir = Path(a.cal_dir) if a.cal_dir else None
        bundle = load_bundle(a.side, cal_dir=cal_dir)
        bundle.T_camera_imu = np.asarray(res["T_camera_imu"], np.float64); bundle.time_offset_ms = res["time_offset_ms"]
        bundle.set_provenance("T_camera_imu", source="openvins_online", tool="ego_teleop.tools.calib_converge", runs=[t["episode"] for t in takes],
                              notes=f"cross-take mean, {res['verdict']}: {b}")
        bundle.set_provenance("time_offset_ms", source="openvins_online", tool="ego_teleop.tools.calib_converge", runs=[t["episode"] for t in takes])
        p = save_bundle(bundle, cal_dir=cal_dir, notes="calib_converge (online refinement)")
        print(f"wrote {p} — production_ready={not bundle.missing and bundle.offline_camera_imu} (offline Kalibr source still required)")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
