"""QA for one side of an RGB-D rigid-body tracking run (§15/§16) and the §37 pilot report.

Metrics: tracking coverage, position/rotation continuity, static jitter, depth agreement, loss & reacquisition.
Thresholds come from depth_pose.yaml — they are ENGINEERING criteria, provisional until the first real pilot
distribution is measured, and every number below is also reported raw so they can be re-frozen without re-running.

Verdict vocabulary is the one the pilot report uses: PASS | NEEDS_WORK | FAIL."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.spatial.transform import Rotation
from .se3 import inv_T, mean_pose

ORDER = {"PASS": 0, "NEEDS_WORK": 1, "FAIL": 2}


def worst(*v: str) -> str:
    v = [x for x in v if x]
    return "PASS" if not v else max(v, key=lambda x: ORDER.get(x, 0))


@dataclass
class StaticSegment:
    i0: int
    i1: int                       # inclusive frame indices into the pose arrays
    t0_ns: int
    t1_ns: int
    source: str                   # "operator_event" | "auto_from_pose"
    role: str = "still"           # "home_start" | "home_end" | "still"
    n: int = 0
    translation_std_mm: float | None = None
    translation_p95_mm: float | None = None
    rotation_std_deg: float | None = None
    rotation_p95_deg: float | None = None

    @property
    def duration_s(self) -> float:
        return (self.t1_ns - self.t0_ns) / 1e9

    def to_dict(self) -> dict:
        d = asdict(self); d["duration_s"] = round(self.duration_s, 3); return d


@dataclass
class DepthTrackQA:
    side: str
    backend: str
    n_frames: int = 0
    n_valid: int = 0
    tracking_valid_ratio: float = 0.0
    tracking_states: dict = field(default_factory=dict)
    lost_frames: int = 0
    lost_events: int = 0
    longest_lost_s: float = 0.0
    reacquisitions: int = 0
    reacquired: bool | None = None
    # continuity (consecutive valid pairs only — a gap is never bridged)
    step_translation_mm_median: float | None = None
    step_translation_mm_p95: float | None = None
    step_translation_mm_max: float | None = None
    step_rotation_deg_median: float | None = None
    step_rotation_deg_p95: float | None = None
    step_rotation_deg_max: float | None = None
    frames_over_trust_radius: int | None = None      # steps larger than what the backend can resolve (see `trust_step_mm`)
    frac_over_trust_radius: float | None = None
    catastrophic_jumps_translation: int = 0
    catastrophic_jumps_rotation: int = 0
    # static jitter
    static_source: str | None = None
    static_segments: list = field(default_factory=list)
    static_translation_mm_median: float | None = None
    static_translation_mm_p95: float | None = None
    static_rotation_deg_median: float | None = None
    static_rotation_deg_p95: float | None = None
    # HOME return drift: inv(mean pose over the start still window) @ (mean pose over the end still window).
    # Without a mechanical dock the operator places the HandUMI back by hand, so this bounds TRACKER DRIFT AND
    # PLACEMENT REPEATABILITY TOGETHER — an upper bound on drift, never drift itself.
    return_translation_mm: float | None = None
    return_rotation_deg: float | None = None
    return_windows: str | None = None       # which two windows were compared
    # depth agreement / confidence
    depth_residual_mm_median: float | None = None
    depth_residual_mm_p95: float | None = None
    confidence_median: float | None = None
    # dual-hand only (§29); None for a single-hand pilot
    identity_swaps: int | None = None
    # runtime
    runtime_s: float | None = None
    fps: float | None = None
    verdict: str = "PASS"
    reasons: list = field(default_factory=list)
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["static_segments"] = [s.to_dict() if isinstance(s, StaticSegment) else s for s in self.static_segments]
        return d


def _flag(qa: DepthTrackQA, verdict: str, why: str) -> None:
    qa.verdict = worst(qa.verdict, verdict)
    qa.reasons.append(f"{verdict}: {why}")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if len(mask) == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.nonzero(d == 1)[0] + 1)
    ends = list(np.nonzero(d == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def detect_static_segments(t_ns: np.ndarray, Ts: np.ndarray, valid: np.ndarray, cfg: dict, fps: float) -> tuple[list[StaticSegment], str]:
    """Frames whose consecutive pose step is below the motion threshold, grouped into runs of at least min_duration_s.

    CAVEAT, recorded with the result: this is derived FROM the tracked pose, so a frozen/stuck tracker also reads as
    static. It measures jitter honestly only together with the overlay check; an operator-marked still segment
    (`static_begin`/`static_end` events) is always preferred and overrides this."""
    sd = cfg["static_detect"]
    n = len(t_ns)
    still = np.zeros(n, bool)
    vi = np.nonzero(valid)[0]
    for a, b in zip(vi[:-1], vi[1:]):
        if b - a != 1:
            continue
        dt = np.linalg.norm(Ts[b][:3, 3] - Ts[a][:3, 3]) * 1e3
        dr = np.degrees(Rotation.from_matrix(Ts[a][:3, :3].T @ Ts[b][:3, :3]).magnitude())
        if dt < float(sd["translation_mm_per_frame"]) and dr < float(sd["rotation_deg_per_frame"]):
            still[a] = still[b] = True
    min_n = max(int(float(sd["min_duration_s"]) * max(fps, 1.0)), 3)
    segs = [StaticSegment(a, b - 1, int(t_ns[a]), int(t_ns[b - 1]), "auto_from_pose", n=b - a)
            for a, b in _runs(still & valid) if b - a >= min_n]
    return segs, "auto_from_pose"


def segment_stats(Ts: np.ndarray, seg: StaticSegment, valid: np.ndarray | None = None) -> StaticSegment:
    T = Ts[seg.i0:seg.i1 + 1]
    if valid is not None:                       # invalid frames carry an identity placeholder, never a pose
        T = T[np.asarray(valid, bool)[seg.i0:seg.i1 + 1]]
    if len(T) < 3:
        seg.n = len(T)
        return seg
    Tm = mean_pose(T)
    dp = np.linalg.norm(T[:, :3, 3] - Tm[:3, 3], axis=1) * 1e3
    dr = np.degrees((Rotation.from_matrix(Tm[:3, :3]).inv() * Rotation.from_matrix(T[:, :3, :3])).magnitude())
    seg.n = len(T)
    seg.translation_std_mm = float(np.sqrt((dp ** 2).mean()))
    seg.translation_p95_mm = float(np.percentile(dp, 95))
    seg.rotation_std_deg = float(np.sqrt((dr ** 2).mean()))
    seg.rotation_p95_deg = float(np.percentile(dr, 95))
    return seg


def evaluate(*, side: str, backend: str, t_ns, Ts, valid, states: list[str], cfg: dict, fps: float,
             depth_residual_mm=None, confidence=None, runtime_s: float | None = None,
             static_segments: list[StaticSegment] | None = None, trust_step_mm: float | None = None) -> DepthTrackQA:
    qa = DepthTrackQA(side=side, backend=backend)
    t_ns = np.asarray(t_ns, np.int64); Ts = np.asarray(Ts, np.float64); valid = np.asarray(valid, bool)
    qa.n_frames = int(len(t_ns)); qa.n_valid = int(valid.sum())
    qa.tracking_valid_ratio = qa.n_valid / qa.n_frames if qa.n_frames else 0.0
    qa.tracking_states = {s: int(states.count(s)) for s in sorted(set(states))}
    qa.runtime_s = None if runtime_s is None else round(float(runtime_s), 2)
    qa.fps = None if not runtime_s else round(qa.n_frames / runtime_s, 2)

    # ---- coverage
    vr = cfg["valid_ratio"]
    if qa.n_frames == 0 or qa.n_valid == 0:
        _flag(qa, "FAIL", "no valid poses")
        return qa
    if qa.tracking_valid_ratio < float(vr["warn"]):
        _flag(qa, "FAIL", f"tracking_valid_ratio {qa.tracking_valid_ratio:.3f} < {vr['warn']}")
    elif qa.tracking_valid_ratio < float(vr["pass"]):
        _flag(qa, "NEEDS_WORK", f"tracking_valid_ratio {qa.tracking_valid_ratio:.3f} < {vr['pass']}")

    # ---- loss / reacquisition
    lost = ~valid
    qa.lost_frames = int(lost.sum())
    runs = _runs(lost)
    qa.lost_events = len(runs)
    if runs:
        qa.longest_lost_s = float(max((t_ns[b - 1] - t_ns[a]) / 1e9 + 1.0 / max(fps, 1.0) for a, b in runs))
        lim = float(cfg["lost"]["reject_if_lost_longer_than_s"])
        if qa.longest_lost_s > lim:
            _flag(qa, "FAIL", f"tracking lost for {qa.longest_lost_s:.2f}s > {lim}s")
        qa.reacquisitions = int(sum(1 for _a, b in runs if b < len(valid)))
        qa.reacquired = qa.reacquisitions > 0
    else:
        qa.reacquired = None

    # ---- continuity (consecutive valid pairs only; a gap is never bridged)
    vi = np.nonzero(valid)[0]
    a, b = vi[:-1], vi[1:]
    cons = (b - a) == 1
    if cons.any():
        Ta, Tb = Ts[a[cons]], Ts[b[cons]]
        dtr = np.linalg.norm(Tb[:, :3, 3] - Ta[:, :3, 3], axis=1) * 1e3
        drot = np.degrees((Rotation.from_matrix(Ta[:, :3, :3]).inv() * Rotation.from_matrix(Tb[:, :3, :3])).magnitude())
        qa.step_translation_mm_median = float(np.median(dtr)); qa.step_translation_mm_p95 = float(np.percentile(dtr, 95))
        qa.step_translation_mm_max = float(dtr.max())
        qa.step_rotation_deg_median = float(np.median(drot)); qa.step_rotation_deg_p95 = float(np.percentile(drot, 95))
        qa.step_rotation_deg_max = float(drot.max())
        if trust_step_mm:
            # A local tracker cannot follow a step larger than its own correspondence radius: it either loses the object
            # or converges on the wrong part of the model — and in the second case fitness, residual and model coverage
            # all still look healthy. Those frames are reported VALID by the backend, so the honest place to say
            # "do not trust these" is here, against the measured motion.
            over = dtr > float(trust_step_mm)
            qa.frames_over_trust_radius = int(over.sum())
            qa.frac_over_trust_radius = float(over.mean())
            if qa.frames_over_trust_radius:
                _flag(qa, "NEEDS_WORK", f"{qa.frames_over_trust_radius} frame step(s) ({qa.frac_over_trust_radius:.0%}) "
                                        f"exceed the backend's {trust_step_mm:.0f} mm trust radius — poses there are not "
                                        "trustworthy even where the backend marks them valid; check the overlay")
        j = cfg["jumps"]
        qa.catastrophic_jumps_translation = int((dtr > float(j["translation_mm"])).sum())
        qa.catastrophic_jumps_rotation = int((drot > float(j["rotation_deg"])).sum())
        if qa.catastrophic_jumps_translation or qa.catastrophic_jumps_rotation:
            _flag(qa, "FAIL", f"catastrophic jumps: {qa.catastrophic_jumps_translation} translation "
                              f"(> {j['translation_mm']} mm/frame), {qa.catastrophic_jumps_rotation} rotation "
                              f"(> {j['rotation_deg']} deg/frame)")

    # ---- static jitter
    if static_segments:
        segs, src = list(static_segments), "operator_event"
    else:
        segs, src = detect_static_segments(t_ns, Ts, valid, cfg, fps)
    qa.static_source = src
    if src == "auto_from_pose":
        qa.flags.append("static segments were auto-detected FROM the tracked pose — a frozen tracker also reads as "
                        "static; confirm on the overlay, or mark still segments during recording")
    segs = [segment_stats(Ts, s, valid) for s in segs]
    qa.static_segments = segs
    if segs:
        tv = np.array([s.translation_std_mm for s in segs if s.translation_std_mm is not None])
        rv = np.array([s.rotation_std_deg for s in segs if s.rotation_std_deg is not None])
        if len(tv):
            qa.static_translation_mm_median = float(np.median(tv)); qa.static_translation_mm_p95 = float(np.percentile(tv, 95))
            qa.static_rotation_deg_median = float(np.median(rv)); qa.static_rotation_deg_p95 = float(np.percentile(rv, 95))
            sj = cfg["static_jitter"]
            if qa.static_translation_mm_p95 > float(sj["warn_translation_mm"]):
                _flag(qa, "FAIL", f"static translation jitter p95 {qa.static_translation_mm_p95:.1f} mm > {sj['warn_translation_mm']}")
            elif qa.static_translation_mm_p95 > float(sj["pass_translation_mm"]):
                _flag(qa, "NEEDS_WORK", f"static translation jitter p95 {qa.static_translation_mm_p95:.1f} mm > {sj['pass_translation_mm']}")
            if qa.static_rotation_deg_p95 > float(sj["warn_rotation_deg"]):
                _flag(qa, "FAIL", f"static rotation jitter p95 {qa.static_rotation_deg_p95:.2f} deg > {sj['warn_rotation_deg']}")
            elif qa.static_rotation_deg_p95 > float(sj["pass_rotation_deg"]):
                _flag(qa, "NEEDS_WORK", f"static rotation jitter p95 {qa.static_rotation_deg_p95:.2f} deg > {sj['pass_rotation_deg']}")
    else:
        qa.flags.append("no still segment found — static jitter not measured (record a few seconds of a stationary HandUMI)")
    _home_return_drift(qa, Ts, valid, segs, cfg)

    # ---- depth agreement / confidence
    if depth_residual_mm is not None:
        r = np.asarray(depth_residual_mm, np.float64)
        r = r[np.isfinite(r)]
        if len(r):
            qa.depth_residual_mm_median = float(np.median(r)); qa.depth_residual_mm_p95 = float(np.percentile(r, 95))
            dr_cfg = cfg["depth_residual"]
            if qa.depth_residual_mm_p95 > float(dr_cfg["reject_mm"]):
                _flag(qa, "FAIL", f"model-vs-depth residual p95 {qa.depth_residual_mm_p95:.1f} mm > {dr_cfg['reject_mm']}")
            elif qa.depth_residual_mm_p95 > float(dr_cfg["warn_mm"]):
                _flag(qa, "NEEDS_WORK", f"model-vs-depth residual p95 {qa.depth_residual_mm_p95:.1f} mm > {dr_cfg['warn_mm']}")
    if confidence is not None:
        c = np.asarray(confidence, np.float64)
        c = c[np.isfinite(c)]
        if len(c):
            qa.confidence_median = float(np.median(c))
    return qa


def interhand_identity_swaps(Ts_left, valid_left, Ts_right, valid_right) -> int:
    """§29 dual-hand check: frames where the left body sits on the right body's side of the inter-hand axis.
    Sign is taken from the robust median inter-hand vector over the episode, so it needs no external convention."""
    Tl, Tr = np.asarray(Ts_left, np.float64), np.asarray(Ts_right, np.float64)
    m = np.asarray(valid_left, bool) & np.asarray(valid_right, bool)
    if m.sum() < 5:
        return 0
    d = Tr[m][:, :3, 3] - Tl[m][:, :3, 3]
    axis = np.median(d, axis=0)
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return 0
    return int((d @ (axis / n) < 0).sum())


def _home_return_drift(qa: DepthTrackQA, Ts, valid, segs, cfg: dict) -> None:
    """Compare the mean pose of the first and last still windows. Only meaningful when the operator actually put the
    HandUMI back in the same physical place — hence the explicit caveat carried with the number."""
    usable = [s for s in segs if s.n and s.n >= 3]
    if len(usable) < 2:
        return
    a, b = usable[0], usable[-1]
    Ta = _mean_valid(Ts, valid, a)
    Tb = _mean_valid(Ts, valid, b)
    if Ta is None or Tb is None:
        return
    E = inv_T(Ta) @ Tb
    qa.return_translation_mm = float(np.linalg.norm(E[:3, 3]) * 1e3)
    qa.return_rotation_deg = float(np.degrees(Rotation.from_matrix(E[:3, :3]).magnitude()))
    qa.return_windows = f"{a.role}[{a.i0}:{a.i1}] -> {b.role}[{b.i0}:{b.i1}] ({a.source})"
    hr = cfg.get("home_return")
    if not hr:
        return
    if a.source != "operator_event":
        qa.flags.append("HOME return drift computed over AUTO-DETECTED still windows — mark home_leave/home_return "
                        "during recording for a trustworthy number")
    qa.flags.append("HOME return drift includes hand-placement repeatability (no mechanical dock): upper bound on "
                    "tracker drift, not tracker drift")
    for name, val, key in (("translation", qa.return_translation_mm, "translation_mm"),
                           ("rotation", qa.return_rotation_deg, "rotation_deg")):
        if val > float(hr["warn"][key]):
            _flag(qa, "FAIL", f"HOME return {name} drift {val:.1f} > {hr['warn'][key]}")
        elif val > float(hr["pass"][key]):
            _flag(qa, "NEEDS_WORK", f"HOME return {name} drift {val:.1f} > {hr['pass'][key]}")


def _mean_valid(Ts, valid, seg: StaticSegment):
    m = np.asarray(valid, bool)[seg.i0:seg.i1 + 1]
    T = np.asarray(Ts)[seg.i0:seg.i1 + 1][m]
    return mean_pose(T) if len(T) >= 3 else None


# ---------------------------------------------------------------------------------------------- §37 pilot report
def format_report(qa: DepthTrackQA, *, episode: str, mesh: str = "", extra_failures: list | None = None) -> str:
    def f(v, unit="", nd=2):
        return "n/a" if v is None else f"{v:.{nd}f}{unit}"
    lines = [
        f"Episode:                 {episode}",
        f"Side:                    {qa.side}",
        f"Backend:                 {qa.backend}" + (f"   mesh: {mesh}" if mesh else ""),
        f"Frames:                  {qa.n_frames}",
        "",
        f"Tracking valid ratio:    {qa.tracking_valid_ratio:.4f}  ({qa.n_valid}/{qa.n_frames})",
        "",
        f"Static jitter            source: {qa.static_source or 'none'}  segments: {len(qa.static_segments)}",
        f"  translation med/p95:   {f(qa.static_translation_mm_median,' mm')} / {f(qa.static_translation_mm_p95,' mm')}",
        f"  rotation    med/p95:   {f(qa.static_rotation_deg_median,' deg')} / {f(qa.static_rotation_deg_p95,' deg')}",
        "",
        f"HOME return drift        {qa.return_windows or 'n/a (needs a still window at both ends)'}",
        f"  translation:           {f(qa.return_translation_mm,' mm')}",
        f"  rotation:              {f(qa.return_rotation_deg,' deg')}",
        "",
        "Motion:",
        f"  frame step transl med/p95/max:  {f(qa.step_translation_mm_median,' mm')} / {f(qa.step_translation_mm_p95,' mm')} / {f(qa.step_translation_mm_max,' mm')}",
        f"  frame step rot    med/p95/max:  {f(qa.step_rotation_deg_median,' deg')} / {f(qa.step_rotation_deg_p95,' deg')} / {f(qa.step_rotation_deg_max,' deg')}",
        f"  catastrophic jumps:             {qa.catastrophic_jumps_translation} translation, {qa.catastrophic_jumps_rotation} rotation",
        (f"  steps over the trust radius:    {qa.frames_over_trust_radius} ({(qa.frac_over_trust_radius or 0):.1%})"
         if qa.frames_over_trust_radius is not None else "  steps over the trust radius:    n/a"),
        "",
        "Occlusion:",
        f"  lost frames:           {qa.lost_frames}  ({qa.lost_events} event(s), longest {qa.longest_lost_s:.2f}s)",
        f"  reacquired:            {'n/a (never lost)' if qa.reacquired is None else ('yes' if qa.reacquired else 'no')}",
        "",
        "Depth agreement:",
        f"  residual med/p95:      {f(qa.depth_residual_mm_median,' mm')} / {f(qa.depth_residual_mm_p95,' mm')}",
        f"  confidence median:     {f(qa.confidence_median,'',3)}",
        "",
        "Runtime:",
        f"  FPS:                   {f(qa.fps)}   ({f(qa.runtime_s,' s')} for {qa.n_frames} frames)",
        "",
        "Observed failure modes:",
    ]
    fails = list(qa.reasons) + list(qa.flags) + list(extra_failures or [])
    lines += [f"  - {x}" for x in fails] or ["  - none"]
    if qa.identity_swaps is not None:
        lines += ["", f"Left/right identity swaps: {qa.identity_swaps}"]
    lines += ["", f"Verdict:                 {qa.verdict}"]
    return "\n".join(lines)
