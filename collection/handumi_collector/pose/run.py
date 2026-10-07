"""Offline pose pipeline for one raw episode (tracking_mode = episode_reset):

    raw wrist video + IMU  ->  backend  ->  T_session_camera  ->  episode-local canonicalisation (first valid pose = identity
    unless a HOME-start window is available, then HOME-start mean = identity)  ->  T_episode_TCP = T_episode_camera · T_camera_tcp
    ->  head-timeline resampling (+grip)  ->  QA  ->  derived/pose_<backend>/{<side>_camera_pose, <side>_tcp_pose, canonical}.parquet + pose_qa.json

Raw is never modified. Any failure is captured per side in pose_qa.json (`error`), so a broken backend cannot lose an episode."""
from __future__ import annotations
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from .backends import make_backend
from .calibration import SideCalibration
from .episode_io import RawEpisode, derived_dir, write_json, write_table
from .estimator import BackendUnavailable, PoseEstimate, TrackingState
from .imu import detect_home_windows
from .qa import SideQA, evaluate_side, worst
from .se3 import Ts_to_pose7, inv_T, mean_pose
from .sync import resample_poses, resample_scalar
from .timing import ClockFit, fit_device_to_host, imu_host_times_ns, interleave
from ..config import PoseCfg

log = logging.getLogger("handumi.pose")
STREAM = {"left": "left_wrist", "right": "right_wrist"}


def output_backend_name(cfg: PoseCfg, backend_name: str) -> str:
    """derived/pose_<name>: segmented results live next to, never over, the continuous ones."""
    return f"{backend_name}_seg" if cfg.segmented.get("enabled", False) else backend_name


@dataclass
class SideResult:
    side: str
    estimates: list[PoseEstimate]
    Ts_episode_camera: np.ndarray            # (N,4,4) canonicalised (invalid rows = identity, see valid)
    Ts_episode_tcp: np.ndarray
    valid: np.ndarray
    t_ns: np.ndarray                          # frame capture times (host clock)
    canonical_anchor: str                     # "home_start_mean" | "first_valid"
    clock: ClockFit | None
    imu_t_ns: np.ndarray | None
    gyro_bias: np.ndarray | None
    home_start: object
    home_end: object
    qa: SideQA
    runtime_s: float
    calibration: dict
    error: str | None = None


def canonicalize(estimates: list[PoseEstimate], t_ns: np.ndarray, home_start) -> tuple[np.ndarray, np.ndarray, str]:
    """T_episode,t = inv(T_anchor) · T_session,t. Anchor = mean pose over the HOME-start window when the backend was already
    valid there, else the first valid pose. Poses are only ever re-expressed, never edited."""
    N = len(estimates); valid = np.array([e.valid for e in estimates], bool)
    Ts = np.tile(np.eye(4), (N, 1, 1))
    for i, e in enumerate(estimates):
        if e.valid: Ts[i] = e.T_world_camera
    if not valid.any(): return Ts, valid, "none"
    anchor = None; how = "first_valid"
    if home_start is not None:
        m = valid & (t_ns >= home_start.t0_ns) & (t_ns <= home_start.t1_ns)
        if m.sum() >= 3: anchor = mean_pose(Ts[m]); how = "home_start_mean"
    if anchor is None: anchor = Ts[np.nonzero(valid)[0][0]]
    Ainv = inv_T(anchor)
    out = np.tile(np.eye(4), (N, 1, 1)); out[valid] = Ainv @ Ts[valid]
    return out, valid, how


def run_side(ep: RawEpisode, side: str, cfg: PoseCfg, *, backend_name: str | None = None, cal: SideCalibration | None = None,
             backend_options: dict | None = None, backend_factory=None) -> SideResult:
    t0 = time.time(); backend_name = backend_name or cfg.backend
    cal = cal or SideCalibration(side)
    stream = STREAM[side]
    if stream not in ep.frames: raise FileNotFoundError(f"{ep.path.name}: no {stream} stream in this episode")
    fm = ep.frames[stream]; t_frames = fm.capture_ns.astype(np.int64)
    imu = ep.imu.get(side); clock = None; imu_t = None; bias = None; home_start = home_end = None
    if imu is not None and len(imu) >= 2:
        clock = fit_device_to_host(imu.device_us, imu.host_ns)
        imu_t = imu_host_times_ns(imu, clock, cfg.camera_imu_offset_ns(side))
        leave = ep.protocol_events("home_leave"); ret = ep.protocol_events("home_return")
        home_start, home_end = detect_home_windows(imu_t, imu.gyro, imu.accel, cfg.static_window, t_start_ns=ep.t_start_ns, t_stop_ns=ep.t_stop_ns,
                                                   explicit_leave_ns=leave[0] if leave else None, explicit_return_ns=ret[-1] if ret else None)
        if home_start is not None and cfg.imu_bias.get("estimate_from_static_start", True): bias = home_start.gyro_bias
    opts = dict(cfg.backend_options.get(backend_name, {})); opts.update(backend_options or {})
    if backend_factory:                       # tests / benchmark: factory(side=...) when it accepts a side, else factory()
        try: be = backend_factory(side=side)
        except TypeError: be = backend_factory()
    else:
        be = make_backend(backend_name, downscale=cfg.image_downscale, **opts)
    if cfg.segmented.get("enabled", False) and imu is not None and imu_t is not None:
        from .backends.segmented import SegmentedEstimator
        if backend_factory:
            def inner_factory(_bf=backend_factory, _side=side):
                try: return _bf(side=_side)
                except TypeError: return _bf()
        else:
            inner_factory = lambda: make_backend(backend_name, downscale=cfg.image_downscale, **opts)
        be = SegmentedEstimator(inner_factory, imu_t_ns=imu_t, gyro=imu.gyro, accel=imu.accel, T_camera_imu=cal.T_camera_imu, cfg=cfg.segmented)
    error = None; estimates: list[PoseEstimate] = []
    try:
        be.initialize(intrinsics=cal.intrinsics, T_camera_imu=cal.T_camera_imu, imu_noise=cal.imu_noise, image_size=None)
        imu_idx_times = imu_t if (imu_t is not None and be.info.uses_imu) else np.zeros(0, np.int64)
        frames_iter = ep.iter_frames(stream, downscale=cfg.image_downscale)
        for kind, i in interleave(t_frames, imu_idx_times):
            if kind == "imu": be.push_imu(int(imu_idx_times[i]), imu.gyro[i], imu.accel[i]); continue
            # frames are decoded lazily in order; frame_meta order == capture order
            try: vf, fi, t_cap, img = next(frames_iter)
            except StopIteration: break
            try:
                est = be.push_image(int(t_cap), int(fi), img)
            except Exception as exc:                          # a backend crash mid-episode = LOST from here on, recorded, never fatal
                log.exception("%s %s backend crashed at frame %d", ep.path.name, side, fi)
                error = f"backend exception at frame {fi}: {exc!r}"; estimates.append(PoseEstimate(int(t_cap), int(fi), None, TrackingState.LOST)); break
            estimates.append(est)
        final = be.finish()
        if final is not None: estimates = final
    except BackendUnavailable as exc:
        error = f"backend unavailable: {exc}"; log.warning("%s %s: %s", ep.path.name, side, error)
    except Exception as exc:
        error = f"pipeline error: {exc!r}\n{traceback.format_exc()}"; log.exception("%s %s pipeline error", ep.path.name, side)
    # pad frames the backend never saw as LOST so array lengths always equal the raw frame count
    seen = {e.frame_index for e in estimates}
    for fi, t in zip(fm.frame_index, t_frames):
        if int(fi) not in seen: estimates.append(PoseEstimate(int(t), int(fi), None, TrackingState.LOST if error else TrackingState.UNINITIALIZED))
    estimates.sort(key=lambda e: e.timestamp_ns)
    t_ns = np.array([e.timestamp_ns for e in estimates], np.int64)
    Ts_cam, valid, how = canonicalize(estimates, t_ns, home_start)
    T_ct = cal.T_camera_tcp_or_identity()
    Ts_tcp = Ts_cam @ T_ct
    states = [e.tracking_state.value for e in estimates]
    qa = evaluate_side(side=side, backend=output_backend_name(cfg, backend_name), metric_scale=be.info.metric_scale, t_ns=t_ns, Ts=Ts_tcp, valid=valid, states=states,
                       home_start=home_start, home_end=home_end, imu_t_ns=imu_t, gyro=None if imu is None else imu.gyro, gyro_bias=bias,
                       R_camera_imu=cal.T_camera_imu, cfg=cfg, fps=cfg.fps)
    if not cal.tcp_calibrated: qa.flags.append("tcp_uncalibrated: T_camera_tcp = identity (camera pose reported as TCP)")
    if not cfg.offsets.get("measured", False): qa.flags.append("camera_imu_offset_unmeasured: pose.yaml offsets.measured=false (run tools.camera_imu_offset)")
    if error: qa.verdict = worst(qa.verdict, "REJECT"); qa.reasons.append(f"REJECT: {error.splitlines()[0]}")
    return SideResult(side, estimates, Ts_cam, Ts_tcp, valid, t_ns, how, clock, imu_t, bias, home_start, home_end, qa, time.time() - t0,
                      cal.summary(), error)


def build_canonical(ep: RawEpisode, results: dict[str, SideResult], cfg: PoseCfg) -> pd.DataFrame:
    """Head-timeline table: one row per head frame with per-side TCP pose7 (episode-local), validity, gap and grip."""
    head = ep.frames.get(cfg.master_timeline)
    if head is None: raise FileNotFoundError(f"master timeline stream {cfg.master_timeline!r} missing")
    t_head = head.capture_ns.astype(np.int64)
    df = pd.DataFrame(dict(head_video_frame=head.video_frame, head_frame_index=head.frame_index, t_ns=t_head, t_rel_s=(t_head - ep.t_start_ns) / 1e9))
    for side, r in results.items():
        T, ok, gap = resample_poses(t_head, r.t_ns, r.Ts_episode_tcp, r.valid, max_gap_ns=int(cfg.max_pose_gap_ms * 1e6))
        p7 = Ts_to_pose7(T)
        for k, nm in enumerate(("x", "y", "z", "qx", "qy", "qz", "qw")): df[f"{side}_tcp_{nm}"] = np.where(ok, p7[:, k], np.nan)
        df[f"{side}_tcp_valid"] = ok; df[f"{side}_pose_gap_ms"] = np.where(gap == np.iinfo(np.int64).max, np.nan, gap / 1e6)
        g = ep.grip.get(side)
        if g is not None:
            gv, gok = resample_scalar(t_head, g.t_ns, g.normalized, max_gap_ns=int(cfg.max_grip_gap_ms * 1e6))
            df[f"{side}_grip"] = gv; df[f"{side}_grip_valid"] = gok
        else:
            df[f"{side}_grip"] = np.nan; df[f"{side}_grip_valid"] = False
        hs, he = r.home_start, r.home_end
        df[f"{side}_segment"] = np.where(hs is not None and (t_head <= hs.t1_ns), "home_start", np.where(he is not None and (t_head >= he.t0_ns), "home_end", "manipulation"))
    return df


def process_episode(ep_path: str | Path, cfg: PoseCfg, *, backend_name: str | None = None, sides: list[str] | None = None,
                    cal_dir: Path | None = None, backend_factory=None, backend_options: dict | None = None) -> dict:
    ep = RawEpisode.load(ep_path); backend_name = backend_name or cfg.backend
    out = derived_dir(ep.path, output_backend_name(cfg, backend_name)); out.mkdir(parents=True, exist_ok=True)
    results: dict[str, SideResult] = {}
    for side in (sides or cfg.sides):
        cal = SideCalibration(side, cal_dir=cal_dir)
        try:
            r = run_side(ep, side, cfg, backend_name=backend_name, cal=cal, backend_factory=backend_factory, backend_options=backend_options)
        except Exception as exc:
            log.exception("%s %s failed", ep.path.name, side)
            qa = SideQA(side, backend_name, False); qa.verdict = "REJECT"; qa.reasons.append(f"REJECT: {exc!r}")
            r = SideResult(side, [], np.zeros((0, 4, 4)), np.zeros((0, 4, 4)), np.zeros(0, bool), np.zeros(0, np.int64), "none", None, None, None,
                           None, None, qa, 0.0, cal.summary(), repr(exc))
        results[side] = r
        rows = [e.to_row() for e in r.estimates]
        cam = pd.DataFrame(rows)
        if len(cam):
            p7 = Ts_to_pose7(r.Ts_episode_camera)
            for k, nm in enumerate(("x", "y", "z", "qx", "qy", "qz", "qw")): cam[f"ep_{nm}"] = np.where(r.valid, p7[:, k], np.nan)
            write_table(cam, out / f"{side}_camera_pose")
            tcp = cam[["t_ns", "frame_index", "valid", "tracking_state"]].copy(); p7t = Ts_to_pose7(r.Ts_episode_tcp)
            for k, nm in enumerate(("x", "y", "z", "qx", "qy", "qz", "qw")): tcp[nm] = np.where(r.valid, p7t[:, k], np.nan)
            write_table(tcp, out / f"{side}_tcp_pose")
    canonical = None
    try:
        canonical = build_canonical(ep, results, cfg); write_table(canonical, out / "canonical")
    except Exception as exc:
        log.exception("canonical build failed")
        for r in results.values(): r.qa.reasons.append(f"WARN: canonical timeline not built: {exc!r}")
    episode_verdict = worst(*[r.qa.verdict for r in results.values()])
    summary = dict(schema="handumi_pose_qa/v1", episode=ep.path.name, backend=backend_name, tracking_mode=cfg.tracking_mode,
                   verdict=episode_verdict, sides={s: dict(**r.qa.to_dict(), canonical_anchor=r.canonical_anchor, runtime_s=round(r.runtime_s, 2),
                                                          clock_fit=r.clock.to_dict() if r.clock else None, gyro_bias_dps=None if r.gyro_bias is None else np.degrees(r.gyro_bias).round(4).tolist(),
                                                          calibration=r.calibration, error=r.error) for s, r in results.items()},
                   canonical=dict(rows=0 if canonical is None else int(len(canonical)),
                                  **({f"{s}_tcp_valid_ratio": float(canonical[f"{s}_tcp_valid"].mean()) for s in results} if canonical is not None else {})),
                   config=dict(lead=cfg.lead, horizon=cfg.horizon, fps=cfg.fps, max_pose_gap_ms=cfg.max_pose_gap_ms, offsets=cfg.offsets,
                               home_return=cfg.home_return, image_downscale=cfg.image_downscale),
                   note="per-hand relative TCP motion only; no shared L/R world frame is produced (V1 design)")
    write_json(summary, out / "pose_qa.json")
    return summary
