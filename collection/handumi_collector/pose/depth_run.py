"""Offline single-side RGB-D rigid-body tracking run (Stage A / M2A).

    raw RGB-D episode  ->  DepthHandPoseTracker(mesh, init ROI)  ->  per-frame T_depthcam_body  ->  QA  ->
    <episode>/derived/depth_pose_<backend>/{<side>_body_pose.parquet, depth_track.json, report.txt}

Raw is never written to (§36): a tracker crash can lose the derived directory, never the episode. Failures are recorded,
not hidden — an uncertain frame is `tracking_valid=false` with no pose, and nothing here copies the last pose forward,
interpolates across a gap, or smooths (§17). Any such decision belongs to a later processing stage, on top of this raw
tracker output."""
from __future__ import annotations
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from .depth_config import DepthPoseCfg
from .depth_hand_tracker import DepthHandPoseTracker, RigidPoseEstimate, backend_info, make_tracker
from .depth_qa import DepthTrackQA, StaticSegment, evaluate, format_report
from .episode_io import write_json, write_table
from .rgbd_io import RgbdEpisode
from .se3 import pose7_to_T
from .tracking_mesh import TrackingMesh, load_tracking_mesh


def derived_depth_dir(ep_path: Path, backend: str) -> Path:
    return Path(ep_path) / "derived" / f"depth_pose_{backend}"


@dataclass
class TrackRunResult:
    side: str
    backend: str
    episode: Path
    estimates: list = field(default_factory=list)
    t_ns: np.ndarray | None = None
    Ts: np.ndarray | None = None            # (N,4,4) T_depthcam_body; identity where invalid (see `valid`)
    valid: np.ndarray | None = None
    qa: DepthTrackQA | None = None
    provenance: dict = field(default_factory=dict)
    out_dir: Path | None = None
    error: str | None = None

    @property
    def report(self) -> str:
        if self.qa is None:
            return f"{self.episode.name} [{self.side}] FAILED: {self.error}"
        return format_report(self.qa, episode=str(self.episode), mesh=str((self.provenance.get("mesh") or {}).get("path") or ""))


def _rows_to_arrays(estimates: list[RigidPoseEstimate]):
    n = len(estimates)
    Ts = np.tile(np.eye(4), (n, 1, 1))
    valid = np.array([e.tracking_valid for e in estimates], bool)
    for i, e in enumerate(estimates):
        T = e.T_depthcam_body
        if T is not None:
            Ts[i] = T
    t_ns = np.array([e.timestamp_ns for e in estimates], np.int64)
    return t_ns, Ts, valid


def static_segments_from_events(ep: RgbdEpisode, t_ns: np.ndarray, cfg: DepthPoseCfg) -> list[StaticSegment]:
    """Operator-marked still windows from the episode's events.json.

    Two mark vocabularies are understood, and the collector only emits the first:
      home_leave / home_return   (Session.mark(), the HOME protocol) -> still from episode start to home_leave, and from
                                 home_return to episode stop
      static_begin / static_end  explicit pairs, if a future UI writes them

    Preferred over auto-detection because an auto-detected still window is derived from the very pose it is judging: a
    frozen tracker reads as perfectly static. Returns segments in time order; the first and last are treated as the HOME
    start/end windows for the return-drift check."""
    p = ep.path / "events.json"
    if not p.exists():
        return []
    try:
        events = json.loads(p.read_text())
    except Exception:
        return []
    trim = int(float(cfg.edge_trim_s) * 1e9)
    min_n = 3

    def seg(t0: int, t1: int) -> StaticSegment | None:
        i0 = int(np.searchsorted(t_ns, t0, "left"))
        i1 = int(np.searchsorted(t_ns, t1, "right")) - 1
        return StaticSegment(i0, i1, int(t_ns[i0]), int(t_ns[i1]), "operator_event") if i1 - i0 >= min_n - 1 else None

    segs: list[StaticSegment] = []
    marks = sorted((int(e["t_ns"]), e.get("kind")) for e in events if e.get("kind") in
                   ("home_leave", "home_return", "static_begin", "static_end"))
    t_first, t_last = int(t_ns[0]), int(t_ns[-1])
    leave = next((t for t, k in marks if k == "home_leave"), None)
    ret = next((t for t, k in reversed(marks) if k == "home_return"), None)
    if leave is not None:
        s = seg(t_first + trim, leave)
        if s:
            s.role = "home_start"
            segs.append(s)
    if ret is not None:
        s = seg(ret, t_last - trim)
        if s:
            s.role = "home_end"
            segs.append(s)
    open_t = None
    for t, kind in marks:
        if kind == "static_begin":
            open_t = t
        elif kind == "static_end" and open_t is not None:
            s = seg(open_t, t)
            if s:
                segs.append(s)
            open_t = None
    return sorted(segs, key=lambda x: x.i0)


def run_side(episode: str | Path, side: str, *, cfg: DepthPoseCfg, mesh: str | Path | TrackingMesh | None = None,
             backend: str | None = None, initial_bbox=None, initial_mask=None, stream: str | None = None,
             start: int = 0, stop: int | None = None, step: int = 1, write: bool = True,
             tracker: DepthHandPoseTracker | None = None, backend_options: dict | None = None,
             progress=None) -> TrackRunResult:
    episode = Path(episode)
    backend = backend or cfg.backend
    res = TrackRunResult(side=side, backend=backend, episode=episode)
    t0 = time.time()
    try:
        ep = RgbdEpisode.load(episode, stream=stream or cfg.stream, fps_fallback=cfg.fps)
        info = backend_info(backend)
        tm = None
        if info.needs_mesh:
            src = mesh if mesh is not None else cfg.mesh_path(side)
            if src is None:
                raise FileNotFoundError(f"backend {backend!r} needs a tracking mesh; none given and depth_pose.yaml has "
                                        f"no mesh for side {side!r}")
            if not isinstance(src, TrackingMesh) and not Path(src).exists():
                raise FileNotFoundError(f"tracking mesh for side {side!r} not found: {src} — build one with "
                                        "`python -m handumi_collector.tools.make_tracking_mesh --template "
                                        f"{side}` (or point depth_pose.yaml `mesh.{side}` at an assembled CAD export)")
            tm = src if isinstance(src, TrackingMesh) else load_tracking_mesh(src, side=side)
            diag = tm.diagnostics()
            if diag["problems"]:
                raise ValueError(f"tracking mesh {src}: " + "; ".join(diag["problems"]))
        if tracker is None:
            opts = cfg.options_for(backend)
            opts.update(backend_options or {})
            if info.needs_mesh:
                opts.setdefault("mesh", tm)
            opts.setdefault("z_min_m", float(cfg.z_range_m[0]))
            opts.setdefault("z_max_m", float(cfg.z_range_m[1]))
            tracker = make_tracker(backend, **opts)
        if info.needs_init_roi and initial_bbox is None and initial_mask is None:
            raise ValueError(f"backend {backend!r} needs an initial bbox or mask on the first frame "
                             "(--bbox X,Y,W,H | --mask file.png | --pick)")

        estimates: list[RigidPoseEstimate] = []
        first = True
        for fr in ep.iter_frames(start, stop, step):
            if first:
                est = tracker.initialize(fr.rgb, fr.depth_m, ep.intrinsics, side, initial_mask=initial_mask,
                                         initial_bbox=initial_bbox, timestamp_ns=fr.t_ns, frame_index=fr.index)
                first = False
            else:
                est = tracker.track(fr.rgb, fr.depth_m, fr.t_ns, frame_index=fr.index)
            estimates.append(est)
            if progress is not None:
                progress(len(estimates), est)
        if not estimates:
            raise RuntimeError(f"{episode}: no frames in range [{start}, {stop})")
        runtime = time.time() - t0
        t_ns, Ts, valid = _rows_to_arrays(estimates)
        fps = ep.fps or float(cfg.fps)
        segs = static_segments_from_events(ep, t_ns, cfg)
        trust_mm = tracker.options.get("max_corr_track_m")
        trust_mm = float(trust_mm) * 1e3 if trust_mm else None
        qa = evaluate(side=side, backend=backend, t_ns=t_ns, Ts=Ts, valid=valid, trust_step_mm=trust_mm,
                      states=[e.tracking_state.value for e in estimates], cfg=cfg.qa, fps=fps,
                      depth_residual_mm=[e.depth_residual_mm for e in estimates],
                      confidence=[e.confidence for e in estimates], runtime_s=runtime, static_segments=segs)
        res.estimates, res.t_ns, res.Ts, res.valid, res.qa = estimates, t_ns, Ts, valid, qa
        res.provenance = dict(
            schema="handumi_depth_track/v1", episode=str(episode), side=side, backend=backend,
            backend_info=dict(name=info.name, runs_on=info.runs_on, metric_scale=info.metric_scale,
                              can_reacquire=info.can_reacquire, provides=list(info.provides), notes=info.notes),
            backend_options={k: (str(v) if isinstance(v, TrackingMesh) else v) for k, v in (tracker.options or {}).items()},
            backend_status=tracker.get_status(), episode_info=ep.describe(),
            frames=dict(start=start, stop=stop, step=step, n=len(estimates), fps_measured=round(fps, 3)),
            mesh=(tm.diagnostics() if tm is not None else None),
            init_roi=dict(bbox=list(initial_bbox) if initial_bbox is not None else None,
                          mask_pixels=(int(np.asarray(initial_mask).astype(bool).sum()) if initial_mask is not None else None)),
            config=dict(qa=cfg.qa, z_range_m=list(cfg.z_range_m)),
            runtime_s=round(runtime, 2), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    except Exception as exc:            # every failure is reported per side; a broken backend never costs the episode
        res.error = f"{type(exc).__name__}: {exc}"
        res.provenance = dict(schema="handumi_depth_track/v1", episode=str(episode), side=side, backend=backend,
                              error=res.error, traceback=traceback.format_exc())
    if write:
        res.out_dir = write_run(res)
    return res


def write_run(res: TrackRunResult) -> Path:
    out = derived_depth_dir(res.episode, res.backend)
    out.mkdir(parents=True, exist_ok=True)
    if res.estimates:
        df = pd.DataFrame([e.to_row() for e in res.estimates])
        write_table(df, out / f"{res.side}_body_pose")
    prov = dict(res.provenance)
    prov["qa"] = res.qa.to_dict() if res.qa else None
    write_json(prov, out / f"{res.side}_depth_track.json")
    (out / f"{res.side}_report.txt").write_text(res.report + "\n")
    return out


def load_run(episode: str | Path, side: str, backend: str):
    """Read back a written run: (DataFrame, provenance dict). Used by the inspector and by dual-hand/TCP stages."""
    from .episode_io import read_table
    out = derived_depth_dir(Path(episode), backend)
    df = read_table(out / f"{side}_body_pose")
    prov = json.loads((out / f"{side}_depth_track.json").read_text())
    return df, prov


def poses_from_table(df: pd.DataFrame):
    """(t_ns, Ts (N,4,4), valid) from a written *_body_pose table."""
    t_ns = df["t_ns"].to_numpy(np.int64)
    valid = df["tracking_valid"].to_numpy(bool)
    Ts = np.tile(np.eye(4), (len(df), 1, 1))
    p7 = df[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    for i in np.nonzero(valid)[0]:
        Ts[i] = pose7_to_T(p7[i])
    return t_ns, Ts, valid


# ---------------------------------------------------------------------------------------------- portable packet
def export_packet(episode: str | Path, out: str | Path, *, stream: str | None = None, start: int = 0, stop: int | None = None,
                  step: int = 1, mesh: str | Path | None = None, initial_bbox=None, initial_mask=None,
                  side: str = "left") -> Path:
    """Write a self-contained flat RGB-D directory (colour PNG + depth PNG + intrinsics + init ROI + mesh) so a GPU host
    can run a CUDA-only backend on exactly these frames and rsync `derived/` back. Nothing is resampled or rescaled."""
    import cv2
    episode, out = Path(episode), Path(out)
    ep = RgbdEpisode.load(episode, stream=stream)
    (out / "color").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    ts = []
    for k, fr in enumerate(ep.iter_frames(start, stop, step)):
        cv2.imwrite(str(out / "color" / f"frame_{k:06d}.png"), fr.rgb)
        cv2.imwrite(str(out / "depth" / f"frame_{k:06d}.png"), fr.depth_raw)
        ts.append((k, fr.index, fr.t_ns))
    (out / "timestamps.csv").write_text("index,source_frame_index,t_ns\n" + "\n".join(f"{a},{b},{c}" for a, b, c in ts) + "\n")
    (out / "intrinsics.json").write_text(json.dumps(ep.intrinsics.to_dict(), indent=1))
    if initial_mask is not None:
        cv2.imwrite(str(out / "init_mask.png"), (np.asarray(initial_mask).astype(np.uint8) * 255))
    mesh_out = None
    if mesh is not None:
        import shutil
        src = Path(mesh)
        if src.suffix.lower() in (".yaml", ".yml"):
            tm = load_tracking_mesh(src, side=side)
            mesh_out = out / "mesh.obj"
            tm.mesh.export(str(mesh_out))
        else:
            mesh_out = out / f"mesh{src.suffix}"
            shutil.copy2(src, mesh_out)
    (out / "packet_meta.json").write_text(json.dumps(dict(
        schema="handumi_rgbd_packet/v1", source_episode=str(episode), stream=ep.stream, side=side,
        frames=dict(start=start, stop=stop, step=step, n=len(ts)),
        init_bbox=list(initial_bbox) if initial_bbox is not None else None,
        init_mask="init_mask.png" if initial_mask is not None else None,
        mesh=(mesh_out.name if mesh_out else None), created=time.strftime("%Y-%m-%dT%H:%M:%S%z")), indent=1))
    return out
