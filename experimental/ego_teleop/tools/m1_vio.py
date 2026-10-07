"""M1: recorded episode -> VIO backend -> wrist pose + M1 report. No robot involved.

    python -m ego_teleop.tools.m1_vio EPISODE_DIR --side right [--backend openvins] [--downscale 1] [--allow-mock]
                                      [--pose-config configs/handumi/pose.yaml] [--override key=value ...]

Runs handumi_collector.pose.run.process_episode (offline, episode_reset) with the chosen backend, then computes the M1
protocol metrics (ego_teleop.tracking.m1_eval) from derived/pose_<backend>/<side>_camera_pose.parquet, using
ground_truth.json when the episode is synthetic. Writes derived/pose_<backend>/m1_report_<side>.json and prints a summary."""
from __future__ import annotations
import argparse
import inspect
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import ego_teleop.tracking.backends  # noqa: F401  (registers openvins)
from handumi_collector.config import load_pose_cfg
from handumi_collector.pose.run import process_episode
from handumi_collector.pose.episode_io import derived_dir, read_table
from handumi_collector.pose.se3 import poses7_to_T
from ..tracking.m1_eval import (evaluate, parse_segments, segments_from_events, segment_metrics, segment_verdict,
                                still_windows_from_imu, stationary_jitter, horizon_consistency)
import yaml


def _parse_overrides(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try: out[k] = json.loads(v)
        except json.JSONDecodeError: out[k] = v
    return out


def gt_mock_factory(ep_dir: Path, side: str, **mock_kw):
    """Plumbing check only: a MockBackend that replays the synthetic ground-truth camera trajectory (never for real data)."""
    from handumi_collector.pose.backends.mock import MockBackend
    from handumi_collector.pose.se3 import interp_pose
    g = json.loads((Path(ep_dir) / "ground_truth.json").read_text())["sides"][side]
    gt_t = np.asarray(g["t_ns"], np.int64); gt_T = poses7_to_T(np.asarray(g["pose7"], np.float64))
    def traj(t_ns):
        k = int(np.clip(np.searchsorted(gt_t, t_ns), 1, len(gt_t) - 1)); a = (t_ns - gt_t[k - 1]) / max(gt_t[k] - gt_t[k - 1], 1)
        return interp_pose(gt_T[k - 1], gt_T[k], float(np.clip(a, 0, 1)))
    return lambda **_: MockBackend(traj, init_frames=1, **mock_kw)


def synth_cal_dir(ep_dir: Path, side: str, out: Path) -> Path:
    """Synthetic episodes: write fisheye_<side>_v001 (pinhole from ground_truth.json) + camera_imu_<side>_v001 (identity: the
    synthetic IMU is simulated in the camera frame) into a scratch calibration dir, so the real backend path can be exercised."""
    from handumi_collector.pose.calibration import write_fisheye, write_camera_imu
    g = json.loads((Path(ep_dir) / "ground_truth.json").read_text()); intr = g["intrinsics"]
    out.mkdir(parents=True, exist_ok=True)
    if not list(out.glob(f"fisheye_{side}_v*.yaml")):
        write_fisheye(side, intr["K"], intr["D"], intr["image_size"], notes="synthetic pinhole", source="ground_truth.json", cal_dir=out)
        # write_fisheye stores model=kannala_brandt; synthetic data is pinhole → patch the model field
        import yaml
        f = sorted(out.glob(f"fisheye_{side}_v*.yaml"))[-1]; d = yaml.safe_load(f.read_text()); d["model"] = intr.get("model", "pinhole"); f.write_text(yaml.safe_dump(d, sort_keys=False))
    if not list(out.glob(f"camera_imu_{side}_v*.yaml")):
        write_camera_imu(side, np.eye(4), time_offset_ms=0.0, notes="synthetic: IMU simulated in the camera frame", source="synth_episode", cal_dir=out)
    return out


def run(ep_dir: Path, side: str, *, backend: str, downscale: int | None, pose_config: str | None, overrides: dict, allow_mock: bool = False,
        backend_factory=None, rpe_window_s: float = 1.0, cal_dir: Path | None = None, segments: str | None = None, protocol: str | None = None,
        mode: str = "production", segmented: bool = False) -> dict:
    if backend == "mock" and not allow_mock and backend_factory is None: raise SystemExit("mock backend refused for M1 (use --allow-mock for plumbing tests only)")
    if backend == "mock" and backend_factory is None and (Path(ep_dir) / "ground_truth.json").exists(): backend_factory = gt_mock_factory(ep_dir, side)
    cfg = load_pose_cfg(pose_config) if pose_config else load_pose_cfg()
    cfg.backend = backend
    if segmented: cfg.segmented = dict(cfg.segmented, enabled=True)
    if downscale is not None: cfg.image_downscale = int(downscale)
    opts = dict(overrides or {})
    if backend == "openvins": opts.setdefault("mode", mode)
    captured: dict = {}
    if backend == "openvins" and backend_factory is None:                       # keep a handle on the backend to save its calibration trace
        from ..tracking.backends.openvins import OpenVinsBackend, ESTIMATOR_DEFAULTS
        # OpenVinsBackend.__init__ ends in **_ignored, so an unrecognised --override would be swallowed without a word.
        # Route anything that names an OpenVINS estimator setting into `overrides` (the estimator_config.yaml), and refuse
        # anything that is neither a constructor argument nor a known estimator key rather than dropping it silently.
        ctor = set(inspect.signature(OpenVinsBackend.__init__).parameters) - {"self", "overrides", "_ignored"}
        est_over = dict(opts.pop("overrides", None) or {})
        for k in [k for k in opts if k not in ctor]:
            if k not in ESTIMATOR_DEFAULTS:
                raise SystemExit(f"--override {k}=...: not an OpenVinsBackend argument ({', '.join(sorted(ctor))}) "
                                 f"nor an OpenVINS estimator setting — refusing to drop it silently")
            est_over[k] = opts.pop(k)
        if est_over: print(f"openvins estimator overrides: {est_over}")
        def backend_factory(**_):
            be = OpenVinsBackend(downscale=cfg.image_downscale, overrides=est_over, **opts); captured["be"] = be; return be
    qa = process_episode(ep_dir, cfg, backend_name=backend, sides=[side], backend_options=opts or None, backend_factory=backend_factory, cal_dir=cal_dir)
    from handumi_collector.pose.run import output_backend_name
    out = derived_dir(Path(ep_dir), output_backend_name(cfg, backend))
    cam = read_table(out / f"{side}_camera_pose")
    t = cam["t_ns"].to_numpy(np.int64); valid = cam["valid"].to_numpy(bool)
    p7 = cam[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy(np.float64); p7 = np.where(np.isfinite(p7), p7, [0, 0, 0, 0, 0, 0, 1])
    Ts = poses7_to_T(p7)
    gt_t = gt_T = None; gtp = Path(ep_dir) / "ground_truth.json"
    if gtp.exists():
        g = json.loads(gtp.read_text())["sides"].get(side)
        if g: gt_t = np.asarray(g["t_ns"], np.int64); gt_T = poses7_to_T(np.asarray(g["pose7"], np.float64))
    side_qa = qa["sides"][side]
    hs = side_qa.get("home_start") or {}
    rep = evaluate(t, Ts, valid, gt_t_ns=gt_t, gt_Ts=gt_T, static_t0_ns=hs.get("t0_ns"), static_t1_ns=hs.get("t1_ns"), rpe_window_s=rpe_window_s)
    result = dict(schema="ego_teleop_m1_report/v1", episode=Path(ep_dir).name, side=side, backend=backend, downscale=cfg.image_downscale, mode=opts.get("mode"),
                  pose_qa_verdict=qa["verdict"], pose_qa=side_qa, m1=rep.to_dict())
    # Ground-truth-free relative metrics: the wrist has no external tracker, so these judge self-consistency and agreement
    # with the gyro, never an absolute trajectory. See m1_eval for what each one can and cannot prove.
    try:
        from handumi_collector.pose.episode_io import RawEpisode
        from handumi_collector.pose.timing import fit_device_to_host, imu_host_times_ns
        from handumi_collector.pose.calibration import SideCalibration
        raw = RawEpisode.load(ep_dir); im = raw.imu.get(side)
        if im is not None and len(im) > 10:
            it = imu_host_times_ns(im, fit_device_to_host(im.device_us, im.host_ns), cfg.camera_imu_offset_ns(side))
            sc = SideCalibration(side, cal_dir=cal_dir) if cal_dir else SideCalibration(side)
            bias = (side_qa.get("home_start") or {}).get("gyro_bias")
            wins = still_windows_from_imu(it, im.gyro, im.accel)
            result["relative"] = dict(stationary=stationary_jitter(t, Ts, valid, wins),
                                      horizons=horizon_consistency(t, Ts, valid, imu_t_ns=it, gyro=im.gyro, gyro_bias=bias,
                                                                   R_camera_imu=sc.T_camera_imu))
    except Exception as exc:                                  # never let the extra metrics lose a run
        result["relative"] = dict(error=repr(exc))
    be = captured.get("be")
    if be is not None and getattr(be, "trace", None):
        pd.DataFrame(be.trace).to_parquet(out / f"calib_trace_{side}.parquet", index=False)
        result["calibration_estimate"] = be.calibration_estimate()
    # protocol segments (real-hardware validation): explicit --segments wins, else events.json `segment` marks
    proto = yaml.safe_load(Path(protocol).read_text()) if protocol else None
    segs = parse_segments(segments, int(t[0])) if segments else None
    if segs is None:
        evp = Path(ep_dir) / "events.json"
        if evp.exists(): segs = segments_from_events(json.loads(evp.read_text()), int(t[-1]) + 1) or None
    if segs:
        sm = segment_metrics(t, Ts, valid, segs, proto, fps=cfg.fps); sv, sr = segment_verdict(sm, proto)
        result["segments"] = dict(verdict=sv, reasons=sr, metrics=sm)
    (out / f"m1_report_{side}.json").write_text(json.dumps(result, indent=1, default=_default))
    return result


def _default(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return str(o)


def _f(v, fmt=".1f", none="n/a"):
    return none if v is None else format(v, fmt)


def summary_line(res: dict) -> str:
    m = res["m1"]; q = res["pose_qa"]
    if q.get("error") or m["n_valid"] == 0:
        return f"[{res['backend']} {res['side']}] M1 {m['verdict']} | pose_qa {res['pose_qa_verdict']} | frames {m['n_frames']} valid {m['n_valid']} | {'; '.join(m['reasons'])} | error: {q.get('error')}"
    gt = f" ATE {_f(m['ate_rmse_mm'])}mm RPE {_f(m['rpe_trans_rmse_mm'])}mm/{_f(m['rpe_rot_rmse_deg'], '.2f')}deg scale {_f(m['scale_ratio'], '.3f')}" if m["gt_available"] else ""
    st = f" static drift {_f(m['static']['drift_mm'])}mm/{_f(m['static']['rot_drift_deg'], '.2f')}deg" if m["static"] else ""
    return (f"[{res['backend']} {res['side']}] M1 {m['verdict']} | pose_qa {res['pose_qa_verdict']} | init {_f(m['time_to_first_valid_s'], '.2f')}s valid {m['valid_ratio']:.2f} "
            f"lost_runs {m['lost_runs']} longest_lost {m['longest_lost_s']:.2f}s | return {_f(m['return_to_start_mm'])}mm/{_f(m['return_to_start_deg'], '.2f')}deg{st}{gt} | {'; '.join(m['reasons']) or 'ok'}"
            + (f" | error: {q.get('error')}" if q.get("error") else ""))


def segment_line(s: dict) -> str:
    extra = ""
    if "scale_ratio" in s: extra += f" pp {s['pp_m']*100:.1f}cm / phys {s['physical_pp_m']*100:.0f}cm -> scale {s['scale_ratio']:.3f}"
    if "rotation_ratio" in s: extra += f" rot {s['max_rot_deg_from_start']:.0f}deg / phys {s['physical_deg']:.0f} -> {s['rotation_ratio']:.2f}"
    if "runaway" in s: extra += f" max_speed {s['max_speed_m_s']:.2f}m/s runaway={s['runaway']}"
    if s.get("kind") == "stationary" and "drift_mm" in s: extra += f" drift {s['drift_mm']:.1f}mm std {s['pos_std_mm']:.1f}mm"
    return f"  {s['name']:>3} {s['kind']:<11} {s['duration_s']:5.1f}s valid {s['valid_ratio']:.2f} lost_runs {s['lost_runs']} jumps {s.get('jumps', 0)}{extra}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--backend", default="openvins"); ap.add_argument("--downscale", type=int, default=None); ap.add_argument("--pose-config", default=None)
    ap.add_argument("--override", action="append", default=[], help="backend option key=value (JSON values ok)"); ap.add_argument("--allow-mock", action="store_true")
    ap.add_argument("--rpe-window-s", type=float, default=1.0)
    ap.add_argument("--cal-dir", default=None, help="calibration dir (default configs/calibration)")
    ap.add_argument("--synth-cal", action="store_true", help="synthetic episode: derive a scratch calibration from ground_truth.json")
    ap.add_argument("--mode", default="production", choices=["production", "calibration"], help="openvins: calibration = optimise extrinsic + time offset and trace them")
    ap.add_argument("--segments", default=None, help='protocol segments, relative seconds: "S0:0-10,S1:10-25,..."')
    ap.add_argument("--segmented", action="store_true", help="segmented relative VIO: IMU-static windows hold (delta p = 0), moving segments re-init the backend; outputs to derived/pose_<backend>_seg")
    ap.add_argument("--protocol", default=str(Path(__file__).resolve().parents[2] / "configs/ego_teleop/m1_protocol.yaml"))
    a = ap.parse_args(argv)
    cal_dir = Path(a.cal_dir) if a.cal_dir else None
    if a.synth_cal: cal_dir = synth_cal_dir(Path(a.episode), a.side, cal_dir or Path(a.episode) / "_synth_calibration")
    res = run(Path(a.episode), a.side, backend=a.backend, downscale=a.downscale, pose_config=a.pose_config, overrides=_parse_overrides(a.override), allow_mock=a.allow_mock, segmented=a.segmented,
              rpe_window_s=a.rpe_window_s, cal_dir=cal_dir, segments=a.segments, protocol=a.protocol, mode=a.mode)
    print(summary_line(res))
    if res.get("calibration_estimate"): ce = res["calibration_estimate"]; print(f"calibration estimate ({ce['mode']}): dt {ce['time_offset_ms']:+.2f} ms, extrinsic moved {_f(ce['d_rot_deg'], '.2f')} deg / {_f(ce['d_trans_mm'])} mm from the prior over {ce['n_frames']} frames")
    for sg in res.get("segments", {}).get("metrics", []): print(segment_line(sg))
    if "segments" in res: print(f"segments: {res['segments']['verdict']} {'; '.join(res['segments']['reasons']) or 'ok'}"); return 0 if res["m1"]["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
