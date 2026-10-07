"""The V0 metrics the plan asks for and the per-frame report did not yet compute, over one or more 60 s takes.

Everything here is post-hoc over `v0_palm_pose.parquet` — the columns the pose source already wrote (`x,y,z`,
`raw_q*`, `arm_pose_health`, `tracking_health`, `t_ns`). No take needs re-recording for a metric added here.

Definitions that are easy to get subtly wrong, and are therefore pinned in tests:

  * **A run ends at the next good frame.** A 10-frame outage at 30 Hz is 333 ms of robot-side hold, not 300: the arm
    is stuck until a pose arrives. Runs still in progress at the end of the take are included.
  * **A jump is never measured across a gap.** The hand disappearing at one place and reappearing at another is a
    tracking loss, already counted as one; charging it again as a 350 mm "motion" would double-count and would make
    the catastrophic-jump gate unreadable.
  * **Jitter is only reported where the hand was still.** Deviation about a window mean is a noise measure only if the
    window contains no motion; computed over a moving window it reports the motion.
  * **Translation leak is measured against its stimulus.** "The palm centroid wandered 2 cm" means one thing during a
    5 deg wobble and another during a 90 deg wrist rotation, so the rotation-only segment reports mm per 10 deg."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from ..transforms.frames import HumanRobotFrameMapper

ARM_OK, ARM_DEGRADED, ARM_LOST = "ARM_POSE_OK", "ARM_POSE_DEGRADED", "ARM_POSE_LOST"
TRACK_LOST = "TRACKING_LOST"
JUMP_MM = 30.0            # the provisional gate: one >30 mm/frame step is what endangers a real arm
LONG_LOSS_MS = 300.0
STILL_DRIFT_MM = 10.0     # a window is stationary when its net drift is below this (not its frame-to-frame speed)

# provisional V2 gate (engineering permission for the first low-speed real-arm test, not a paper benchmark)
GATE = (("arm_pose_valid_duty", "%", 0.95, 0.90, "min"),
        ("stationary_xyz_rms_mm", "mm", 5.0, 10.0, "max"),
        ("stationary_xyz_p95_mm", "mm", 10.0, 20.0, "max"),
        ("return_to_start_translation_mm", "mm", 15.0, 30.0, "max"),
        ("catastrophic_jump_count", "", 0, 1, "max"),
        ("long_tracking_loss_count", "", 0, 1, "max"))


def seconds(df: pd.DataFrame) -> np.ndarray:
    return (df["t_ns"].to_numpy(np.float64) - float(df["t_ns"].iloc[0])) / 1e9 if len(df) else np.zeros(0)


def window(df: pd.DataFrame, w) -> pd.DataFrame:
    s = seconds(df)
    return df[(s >= w[0]) & (s <= w[1])] if w else df.iloc[0:0]


def runs_ms(df: pd.DataFrame, mask: np.ndarray) -> list[float]:
    """Durations of each maximal True run, each measured to the NEXT non-True frame (or the end of the take)."""
    s = seconds(df)
    if not len(s) or not mask.any(): return []
    out, start = [], None
    for i, f in enumerate(mask):
        if f and start is None: start = i
        elif not f and start is not None:
            out.append(float(s[i] - s[start])); start = None
    if start is not None: out.append(float(s[-1] - s[start]))
    return [d * 1000.0 for d in out]


def dist_ms(v) -> dict:
    v = np.asarray([x for x in v if np.isfinite(x)], np.float64)
    if not v.size: return dict(n=0, median=float("nan"), p95=float("nan"), max=float("nan"))
    return dict(n=int(v.size), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()))


def _xyz(df: pd.DataFrame) -> np.ndarray:
    P = df[["x", "y", "z"]].to_numpy(np.float64)
    return P[np.isfinite(P).all(1)]


def _quat(df: pd.DataFrame) -> np.ndarray:
    if "raw_qx" not in df.columns: return np.zeros((0, 4))
    Q = df[["raw_qx", "raw_qy", "raw_qz", "raw_qw"]].to_numpy(np.float64)
    return Q[np.isfinite(Q).all(1)]


def tracked(df: pd.DataFrame) -> np.ndarray:
    """Frames whose pose the robot would actually have followed: not LOST and with a finite position."""
    P = df[["x", "y", "z"]].to_numpy(np.float64)
    return (df["tracking_health"].to_numpy() != TRACK_LOST) & np.isfinite(P).all(1)


def jumps(df: pd.DataFrame, limit_mm: float = JUMP_MM) -> dict:
    """Frame-to-frame steps, counted only between CONSECUTIVE tracked frames — never across a loss."""
    P = df[["x", "y", "z"]].to_numpy(np.float64)
    ok = tracked(df)
    pair = ok[:-1] & ok[1:]
    if not pair.any(): return dict(count=0, max_mm=float("nan"), over=[])
    d = np.linalg.norm(np.diff(P, axis=0)[pair], axis=1) * 1000.0
    s = seconds(df)[1:][pair]
    over = [dict(t_s=float(t), mm=float(v)) for t, v in zip(s[d > limit_mm], d[d > limit_mm])]
    return dict(count=int((d > limit_mm).sum()), max_mm=float(d.max()), over=over[:20])


def drift_mm(df: pd.DataFrame) -> float:
    P = _xyz(df[tracked(df)])
    if len(P) < 9: return float("nan")
    k = max(len(P) // 3, 3)
    return float(np.linalg.norm(P[-k:].mean(0) - P[:k].mean(0)) * 1000.0)


def jitter(df: pd.DataFrame) -> dict:
    """Scatter about the window mean. `still` says whether it is a noise measure or contaminated by motion."""
    P = _xyz(df[tracked(df)])
    d = drift_mm(df)
    out = dict(n=int(len(P)), still=bool(np.isfinite(d) and d <= STILL_DRIFT_MM), drift_mm=d,
               rms_mm=float("nan"), p95_mm=float("nan"), per_axis_std_mm={})
    if len(P) >= 5:
        r = np.linalg.norm(P - P.mean(0), axis=1) * 1000.0
        out.update(rms_mm=float(np.sqrt(np.mean(r ** 2))), p95_mm=float(np.percentile(r, 95)),
                   per_axis_std_mm={a: float(P[:, i].std() * 1000.0) for i, a in enumerate("xyz")})
    Q = _quat(df[tracked(df)])
    if len(Q) >= 5:
        R = Rotation.from_quat(Q)
        dev = np.degrees((R.mean().inv() * R).magnitude())
        out["orientation_rms_deg"] = float(np.sqrt(np.mean(dev ** 2)))
        out["orientation_p95_deg"] = float(np.percentile(dev, 95))
    else:
        out["orientation_rms_deg"] = out["orientation_p95_deg"] = float("nan")
    return out


def displacement(df: pd.DataFrame, mapper: HumanRobotFrameMapper | None = None) -> dict:
    """Net movement across a segment: the mean of its first fifth to the mean of its last fifth."""
    P = _xyz(df[tracked(df)])
    out = dict(camera_mm={}, robot_mm={}, path_mm=float("nan"), rotation_deg=float("nan"))
    if len(P) < 6: return out
    k = max(len(P) // 5, 2)
    d = P[-k:].mean(0) - P[:k].mean(0)
    out["camera_mm"] = {a: float(d[i] * 1000.0) for i, a in enumerate("xyz")}
    if mapper is not None:
        out["robot_mm"] = {a: float(v * 1000.0) for a, v in zip("xyz", mapper.map_translation(d))}
    out["path_mm"] = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() * 1000.0)
    Q = _quat(df[tracked(df)])
    if len(Q) >= 6:
        kq = max(len(Q) // 5, 2)
        R0 = Rotation.from_quat(Q[:kq]).mean(); R1 = Rotation.from_quat(Q[-kq:]).mean()
        out["rotation_deg"] = float(np.degrees((R0.inv() * R1).magnitude()))
        out["rotation_span_deg"] = float(np.degrees(np.percentile(
            (Rotation.from_quat(Q).mean().inv() * Rotation.from_quat(Q)).magnitude(), 95)) * 2.0)
    return out


def translation_leak(df: pd.DataFrame) -> dict:
    """Palm-origin excursion during a ROTATION-ONLY segment — the plan's dp_leak.

    Reported against the stimulus: 20 mm of wander under a 10 deg wobble is a broken palm origin, the same 20 mm
    under a 90 deg wrist rotation is ordinary geometry. If `mm_per_10deg` is large, the origin definition needs
    revisiting before any 6-DoF work, which is exactly the decision this segment exists to inform."""
    P = _xyz(df[tracked(df)])
    out = dict(n=int(len(P)), p95_mm=float("nan"), max_mm=float("nan"), net_mm=float("nan"),
               rotation_deg=float("nan"), mm_per_10deg=float("nan"))
    if len(P) < 6: return out
    r = np.linalg.norm(P - P.mean(0), axis=1) * 1000.0
    out.update(p95_mm=float(np.percentile(r, 95)), max_mm=float(r.max()))
    k = max(len(P) // 5, 2)
    out["net_mm"] = float(np.linalg.norm(P[-k:].mean(0) - P[:k].mean(0)) * 1000.0)
    Q = _quat(df[tracked(df)])
    if len(Q) >= 6:
        R = Rotation.from_quat(Q)
        span = float(np.degrees(np.percentile((R.mean().inv() * R).magnitude(), 95)) * 2.0)
        out["rotation_deg"] = span
        if span > 1.0: out["mm_per_10deg"] = float(out["p95_mm"] / span * 10.0)
    return out


def health_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    ah = df["arm_pose_health"].to_numpy()
    arm_lost = runs_ms(df, ah == ARM_LOST)
    trk_lost = runs_ms(df, df["tracking_health"].to_numpy() == TRACK_LOST)
    return dict(
        frames=n,
        arm_pose_counts={k: int(v) for k, v in df["arm_pose_health"].value_counts().items()},
        arm_pose_valid_duty=float((ah == ARM_OK).mean()) if n else float("nan"),
        arm_pose_usable_duty=float(np.isin(ah, [ARM_OK, ARM_DEGRADED]).mean()) if n else float("nan"),
        arm_pose_lost_run_ms=dist_ms(arm_lost),
        tracking_lost_run_ms=dist_ms(trk_lost),
        long_tracking_loss_count=int(sum(1 for d in trk_lost if d > LONG_LOSS_MS)),
        reasons={k: int(v) for k, v in df.loc[df["arm_pose_reason"].astype(bool), "arm_pose_reason"]
                 .value_counts().head(8).items()} if "arm_pose_reason" in df else {})


def return_to_start(df: pd.DataFrame, start_w, end_w) -> dict:
    if not start_w or not end_w:
        return dict(translation_mm=float("nan"), rotation_deg=float("nan"), note="needs a stationary and a return window")
    A, B = window(df, start_w), window(df, end_w)
    PA, PB = _xyz(A[tracked(A)]), _xyz(B[tracked(B)])
    if len(PA) < 3 or len(PB) < 3:
        return dict(translation_mm=float("nan"), rotation_deg=float("nan"), note="too few tracked frames in a window")
    d = PB.mean(0) - PA.mean(0)
    out = dict(translation_mm=float(np.linalg.norm(d) * 1000.0),
               per_axis_mm={a: float(d[i] * 1000.0) for i, a in enumerate("xyz")}, rotation_deg=float("nan"))
    QA, QB = _quat(A[tracked(A)]), _quat(B[tracked(B)])
    if len(QA) >= 3 and len(QB) >= 3:
        out["rotation_deg"] = float(np.degrees(
            (Rotation.from_quat(QA).mean().inv() * Rotation.from_quat(QB).mean()).magnitude()))
    return out


def take_metrics(df: pd.DataFrame, segments: dict, *, mapper: HumanRobotFrameMapper | None = None,
                 name: str = "") -> dict:
    """Every number the plan asks for, for one take."""
    m = dict(name=name, duration_s=float(seconds(df)[-1]) if len(df) else 0.0, **health_stats(df))
    starts = [n for n in segments if n.startswith(("stationary", "static", "still"))]
    ends = [n for n in segments if n.startswith("return")]
    st_w = segments[starts[0]] if starts else None
    j = jitter(window(df, st_w)) if st_w else jitter(df.iloc[0:0])
    m["stationary"] = j
    m["stationary_xyz_rms_mm"] = j["rms_mm"] if j["still"] else float("nan")
    m["stationary_xyz_p95_mm"] = j["p95_mm"] if j["still"] else float("nan")
    m["stationary_orientation_p95_deg"] = j["orientation_p95_deg"] if j["still"] else float("nan")
    r = return_to_start(df, st_w, segments[ends[-1]] if ends else None)
    m["return_to_start"] = r
    m["return_to_start_translation_mm"] = r["translation_mm"]
    m["return_to_start_rotation_deg"] = r["rotation_deg"]
    jm = jumps(df)
    m["jumps"] = jm
    m["catastrophic_jump_count"] = jm["count"]
    m["max_frame_step_mm"] = jm["max_mm"]
    m["long_tracking_loss_count"] = m["long_tracking_loss_count"]

    segs = {}
    for n, w in segments.items():
        d = window(df, w)
        s = dict(window_s=list(w), frames=int(len(d)), **{k: v for k, v in health_stats(d).items()
                                                          if k in ("arm_pose_valid_duty", "arm_pose_lost_run_ms")})
        s["jitter"] = jitter(d)
        s["displacement"] = displacement(d, mapper)
        s["jumps"] = jumps(d)["count"]
        if n.startswith("palm_rotation"): s["translation_leak"] = translation_leak(d)
        segs[n] = s
    m["segments"] = segs
    m["gate"] = gate_verdict(m)
    return m


def gate_verdict(m: dict) -> dict:
    """green / yellow / red per gate metric. Unmeasured reads `none` — never green by default."""
    out = {}
    for key, unit, green, yellow, sense in GATE:
        v = m.get(key, float("nan"))
        v = float(v) if v is not None else float("nan")
        if not np.isfinite(v):
            out[key] = dict(value=None, unit=unit, level="none", green=green, yellow=yellow, sense=sense)
            continue
        if sense == "min": level = "green" if v >= green else ("yellow" if v >= yellow else "red")
        else: level = "green" if v <= green else ("yellow" if v <= yellow else "red")
        out[key] = dict(value=v, unit=unit, level=level, green=green, yellow=yellow, sense=sense)
    return out


def aggregate(takes: list[dict]) -> dict:
    """Across takes, the gate is decided by the WORST take: three takes exist so one lucky one cannot pass it."""
    order = {"green": 0, "yellow": 1, "red": 2, "none": 3}
    agg = {"takes": len(takes), "gate": {}, "headline": {}}
    for key, unit, green, yellow, sense in GATE:
        vals = [t["gate"][key]["value"] for t in takes if t["gate"][key]["value"] is not None]
        levels = [t["gate"][key]["level"] for t in takes]
        worst = max(levels, key=lambda l: order[l]) if levels else "none"
        agg["gate"][key] = dict(unit=unit, green=green, yellow=yellow, sense=sense, level=worst,
                                values=vals, best=min(vals) if vals else None, worst_value=max(vals) if vals else None,
                                median=float(np.median(vals)) if vals else None, measured=len(vals))
    agg["verdict"] = max((g["level"] for g in agg["gate"].values()), key=lambda l: order[l]) if agg["gate"] else "none"
    agg["headline"] = {k: agg["gate"][k] for k in ("arm_pose_valid_duty", "stationary_xyz_rms_mm",
                                                   "return_to_start_translation_mm")}
    return agg
