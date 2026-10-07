"""M1 protocol metrics for one wrist trajectory (backend-independent), on top of the pose package's per-episode QA:

    static drift          : position/rotation spread + end-start drift inside a still window (first `static_s` seconds or HOME start)
    relative pose error   : over sliding windows of `rpe_window_s` — translation error (m) and rotation error (deg) vs ground truth,
                            scale ratio (est path length / gt path length) — catches non-metric or badly scaled VIO
    absolute error (ATE)  : after aligning the first valid poses (SE(3) anchor, no scale), RMSE of positions
    return-to-start       : inv(T_first_valid) · T_last_valid translation/rotation (with or without ground truth)
    initialization        : frames before the first valid pose (VIO needs a still window + motion to initialize; NOT a loss)
    lost / recovery       : LOST runs AFTER initialization (count, longest, time-to-recover), post-init valid ratio, largest gap
Ground truth is optional (synthetic episodes carry ground_truth.json); without it only self-consistency metrics are reported."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.spatial.transform import Rotation
from handumi_collector.pose.se3 import inv_T, pose7_to_T, interp_pose


def _rot_deg(R: np.ndarray) -> float: return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    out = []; m = np.asarray(mask, bool); i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]: j += 1
            out.append((i, j)); i = j
        else: i += 1
    return out


def static_window_stats(t_ns, Ts, valid, t0_ns, t1_ns) -> dict | None:
    m = valid & (t_ns >= t0_ns) & (t_ns <= t1_ns)
    if m.sum() < 5: return None
    P = Ts[m][:, :3, 3]; Rs = Ts[m][:, :3, :3]
    R0 = Rs[0]; angs = [_rot_deg(R0.T @ R) for R in Rs]
    return dict(n=int(m.sum()), duration_s=float((t_ns[m][-1] - t_ns[m][0]) / 1e9), pos_std_mm=float(np.linalg.norm(P.std(axis=0)) * 1e3),
                drift_mm=float(np.linalg.norm(P[-1] - P[0]) * 1e3), rot_std_deg=float(np.std(angs)), rot_drift_deg=float(angs[-1]))


def sample_gt(gt_t_ns: np.ndarray, gt_Ts: np.ndarray, t_ns: int) -> np.ndarray | None:
    k = int(np.searchsorted(gt_t_ns, t_ns))
    if k == 0: return gt_Ts[0] if abs(int(gt_t_ns[0] - t_ns)) < 50_000_000 else None
    if k >= len(gt_t_ns): return gt_Ts[-1] if abs(int(t_ns - gt_t_ns[-1])) < 50_000_000 else None
    a = (t_ns - gt_t_ns[k - 1]) / max(gt_t_ns[k] - gt_t_ns[k - 1], 1)
    return interp_pose(gt_Ts[k - 1], gt_Ts[k], float(a))


@dataclass
class M1Report:
    n_frames: int = 0
    n_valid: int = 0
    init_frames: int = 0            # frames before the first valid pose (initialization, not counted as lost)
    valid_ratio: float = 0.0        # over post-initialization frames
    largest_gap_ms: float = 0.0
    lost_runs: int = 0
    longest_lost_s: float = 0.0
    time_to_first_valid_s: float | None = None
    recovery_times_s: list[float] = field(default_factory=list)
    static: dict | None = None
    return_to_start_mm: float | None = None
    return_to_start_deg: float | None = None
    path_length_m: float | None = None
    # ground truth (optional)
    gt_available: bool = False
    ate_rmse_mm: float | None = None
    rpe_trans_rmse_mm: float | None = None
    rpe_rot_rmse_deg: float | None = None
    scale_ratio: float | None = None
    max_abs_error_mm: float | None = None
    verdict: str = "UNKNOWN"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict: return asdict(self)


def evaluate(t_ns: np.ndarray, Ts: np.ndarray, valid: np.ndarray, *, gt_t_ns=None, gt_Ts=None, static_s: float = 30.0, static_t0_ns: int | None = None,
             static_t1_ns: int | None = None, rpe_window_s: float = 1.0, thresholds: dict | None = None) -> M1Report:
    th = dict(static_drift_mm=20.0, static_rot_deg=2.0, return_mm=30.0, return_deg=5.0, rpe_mm=30.0, rpe_deg=3.0, scale_tol=0.10, valid_ratio=0.9, longest_lost_s=1.0)
    th.update(thresholds or {})
    t = np.asarray(t_ns, np.int64); Ts = np.asarray(Ts, np.float64).reshape(-1, 4, 4); v = np.asarray(valid, bool); r = M1Report(int(len(t)), int(v.sum()))
    if len(t) == 0: r.verdict = "FAIL"; r.reasons.append("no frames"); return r
    r.largest_gap_ms = float(np.max(np.diff(t)) / 1e6) if len(t) > 1 else 0.0
    vi = np.nonzero(v)[0]
    if len(vi) == 0: r.verdict = "FAIL"; r.reasons.append("never valid"); r.init_frames = int(len(t)); return r
    r.init_frames = int(vi[0]); r.time_to_first_valid_s = float((t[vi[0]] - t[0]) / 1e9)
    post = v[vi[0]:]; r.valid_ratio = float(post.mean())
    lost = [(a + vi[0], b + vi[0]) for a, b in runs(~post)]; r.lost_runs = len(lost)          # initialization is not a loss
    if lost: r.longest_lost_s = float(max((t[min(b, len(t) - 1)] - t[a]) / 1e9 for a, b in lost))
    for a, b in lost:
        if b < len(t): r.recovery_times_s.append(float((t[b] - t[a]) / 1e9))
    # static window: explicit (HOME) or the first static_s seconds
    s0 = t[0] if static_t0_ns is None else static_t0_ns; s1 = (t[0] + int(static_s * 1e9)) if static_t1_ns is None else static_t1_ns
    r.static = static_window_stats(t, Ts, v, s0, s1)
    Tv = Ts[vi]; P = Tv[:, :3, 3]; r.path_length_m = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    d = inv_T(Tv[0]) @ Tv[-1]; r.return_to_start_mm = float(np.linalg.norm(d[:3, 3]) * 1e3); r.return_to_start_deg = _rot_deg(d[:3, :3])
    if gt_t_ns is not None and gt_Ts is not None:
        gt_t = np.asarray(gt_t_ns, np.int64); gt = np.asarray(gt_Ts, np.float64).reshape(-1, 4, 4); r.gt_available = True
        pairs = [(i, sample_gt(gt_t, gt, int(t[i]))) for i in vi]; pairs = [(i, G) for i, G in pairs if G is not None]
        if len(pairs) >= 3:
            i0, G0 = pairs[0]; A = G0 @ inv_T(Ts[i0])                     # anchor: align first valid pose to GT (SE(3), no scale)
            err = np.array([np.linalg.norm((A @ Ts[i])[:3, 3] - G[:3, 3]) for i, G in pairs])
            r.ate_rmse_mm = float(np.sqrt(np.mean(err ** 2)) * 1e3); r.max_abs_error_mm = float(err.max() * 1e3)
            gP = np.array([G[:3, 3] for _, G in pairs]); gt_len = float(np.sum(np.linalg.norm(np.diff(gP, axis=0), axis=1)))
            eP = np.array([Ts[i][:3, 3] for i, _ in pairs]); est_len = float(np.sum(np.linalg.norm(np.diff(eP, axis=0), axis=1)))
            r.scale_ratio = est_len / gt_len if gt_len > 1e-6 else None
            w = int(rpe_window_s * 1e9); te = []; re = []
            idx = [i for i, _ in pairs]; Gs = {i: G for i, G in pairs}; ti = t[idx]
            for a_k, i in enumerate(idx):
                j_k = int(np.searchsorted(ti, ti[a_k] + w))
                if j_k >= len(idx): break
                j = idx[j_k]; dE = inv_T(Ts[i]) @ Ts[j]; dG = inv_T(Gs[i]) @ Gs[j]
                te.append(np.linalg.norm(dE[:3, 3] - dG[:3, 3])); re.append(_rot_deg(dE[:3, :3].T @ dG[:3, :3]))
            if te: r.rpe_trans_rmse_mm = float(np.sqrt(np.mean(np.square(te))) * 1e3); r.rpe_rot_rmse_deg = float(np.sqrt(np.mean(np.square(re))))
    # verdict
    fails, warns = [], []
    if r.valid_ratio < th["valid_ratio"]: fails.append(f"valid_ratio {r.valid_ratio:.2f} < {th['valid_ratio']}")
    if r.longest_lost_s > th["longest_lost_s"]: warns.append(f"longest_lost {r.longest_lost_s:.2f}s")
    if r.static and r.static["drift_mm"] > th["static_drift_mm"]: fails.append(f"static drift {r.static['drift_mm']:.1f}mm")
    if r.static and r.static["rot_drift_deg"] > th["static_rot_deg"]: fails.append(f"static rot drift {r.static['rot_drift_deg']:.2f}deg")
    if r.gt_available:
        if r.rpe_trans_rmse_mm is not None and r.rpe_trans_rmse_mm > th["rpe_mm"]: fails.append(f"RPE {r.rpe_trans_rmse_mm:.1f}mm/{rpe_window_s}s")
        if r.rpe_rot_rmse_deg is not None and r.rpe_rot_rmse_deg > th["rpe_deg"]: fails.append(f"RPE rot {r.rpe_rot_rmse_deg:.2f}deg")
        if r.scale_ratio is not None and abs(r.scale_ratio - 1) > th["scale_tol"]: fails.append(f"scale ratio {r.scale_ratio:.3f} (non-metric or mis-scaled)")
    else:
        if r.return_to_start_mm is not None and r.return_to_start_mm > th["return_mm"]: warns.append(f"return-to-start {r.return_to_start_mm:.1f}mm (only meaningful if the protocol returned to start)")
    r.verdict = "FAIL" if fails else ("WARN" if warns else "PASS"); r.reasons = fails + warns
    return r


# ----------------------------------------------------------------------------------------------------------------------------
# Protocol segments (M1 real-hardware validation): per-segment metrics + metric scale from KNOWN PHYSICAL DISPLACEMENT.
# Segments come from events.json (`kind: "segment"`, detail.name) or from a `--segments "S0:0-10,S1:10-25,..."` string
# (relative seconds). The protocol file (configs/ego_teleop/m1_protocol.yaml) says what each segment is and, for translation
# segments, the physical peak-to-peak distance moved (fixture: two stops / a ruler), so scale = VIO peak-to-peak / physical.
# ----------------------------------------------------------------------------------------------------------------------------

def parse_segments(spec: str, t0_ns: int) -> list[dict]:
    """'S0:0-10,S1:10-25' (relative seconds) -> [{name, t0_ns, t1_ns}]"""
    out = []
    for item in [x for x in spec.split(",") if x.strip()]:
        name, rng = item.split(":"); a, b = rng.split("-")
        out.append(dict(name=name.strip(), t0_ns=int(t0_ns + float(a) * 1e9), t1_ns=int(t0_ns + float(b) * 1e9)))
    return out


def segments_from_events(events: list[dict], t_end_ns: int) -> list[dict]:
    """events with kind == 'segment' (detail.name) mark segment starts; each runs to the next mark (or t_end)."""
    marks = sorted([(int(e["t_ns"]), (e.get("detail") or {}).get("name", f"seg{i}")) for i, e in enumerate(events) if e.get("kind") == "segment"])
    return [dict(name=n, t0_ns=t, t1_ns=(marks[i + 1][0] if i + 1 < len(marks) else int(t_end_ns))) for i, (t, n) in enumerate(marks)]


def _principal_pp(P: np.ndarray) -> tuple[float, np.ndarray]:
    """Peak-to-peak displacement along the principal motion axis (m) and that axis."""
    if len(P) < 2: return 0.0, np.array([1.0, 0, 0])
    C = P - P.mean(axis=0); _, _, Vt = np.linalg.svd(C, full_matrices=False); ax = Vt[0]
    proj = C @ ax; return float(proj.max() - proj.min()), ax


def segment_metrics(t_ns, Ts, valid, segments: list[dict], protocol: dict | None = None, *, fps: float = 30.0, jump_m: float = 0.15, jump_deg: float = 20.0) -> list[dict]:
    t = np.asarray(t_ns, np.int64); Ts = np.asarray(Ts, np.float64).reshape(-1, 4, 4); v = np.asarray(valid, bool); proto = (protocol or {}).get("segments", {})
    out = []
    for seg in segments:
        m = (t >= seg["t0_ns"]) & (t < seg["t1_ns"]); mv = m & v; spec = proto.get(seg["name"], {})
        r = dict(name=seg["name"], kind=spec.get("kind", "unknown"), duration_s=float((seg["t1_ns"] - seg["t0_ns"]) / 1e9), n=int(m.sum()), valid_ratio=float(mv.sum() / max(m.sum(), 1)),
                 lost_runs=len(runs(~v[m])) if m.sum() else 0)
        if mv.sum() >= 3:
            P = Ts[mv][:, :3, 3]; Rs = Ts[mv][:, :3, :3]
            steps = np.linalg.norm(np.diff(P, axis=0), axis=1); rsteps = np.array([_rot_deg(Rs[i].T @ Rs[i + 1]) for i in range(len(Rs) - 1)])
            r.update(pp_m=_principal_pp(P)[0], path_m=float(steps.sum()), max_speed_m_s=float(steps.max() * fps) if len(steps) else 0.0,
                     max_rot_deg_from_start=float(max(_rot_deg(Rs[0].T @ R) for R in Rs)), jumps=int(np.sum(steps > jump_m) + np.sum(rsteps > jump_deg)),
                     drift_mm=float(np.linalg.norm(P[-1] - P[0]) * 1e3), pos_std_mm=float(np.linalg.norm(P.std(axis=0)) * 1e3))
            if spec.get("kind") == "translation" and spec.get("physical_pp_m"):
                r["physical_pp_m"] = float(spec["physical_pp_m"]); r["scale_ratio"] = r["pp_m"] / float(spec["physical_pp_m"])
            if spec.get("kind") == "rotation" and spec.get("physical_deg"):
                r["physical_deg"] = float(spec["physical_deg"]); r["rotation_ratio"] = r["max_rot_deg_from_start"] / float(spec["physical_deg"])
            if spec.get("kind") == "fast": r["runaway"] = bool(r["max_speed_m_s"] > float(spec.get("runaway_speed_m_s", 5.0)) or r["jumps"] > 0)
        out.append(r)
    return out


def segment_verdict(segs: list[dict], protocol: dict | None = None) -> tuple[str, list[str]]:
    g = dict(scale_min=0.9, scale_max=1.1, stationary_drift_mm=20.0, post_init_valid=0.95, rotation_ratio_min=0.7, rotation_ratio_max=1.3)
    g.update((protocol or {}).get("gates", {}))
    fails, warns = [], []
    for s in segs:
        k = s.get("kind")
        if k == "stationary" and s.get("drift_mm") is not None and s["drift_mm"] > g["stationary_drift_mm"]: fails.append(f"{s['name']} stationary drift {s['drift_mm']:.0f}mm")
        if "scale_ratio" in s and not (g["scale_min"] <= s["scale_ratio"] <= g["scale_max"]): fails.append(f"{s['name']} scale {s['scale_ratio']:.3f}")
        if "rotation_ratio" in s and not (g["rotation_ratio_min"] <= s["rotation_ratio"] <= g["rotation_ratio_max"]): fails.append(f"{s['name']} rotation ratio {s['rotation_ratio']:.2f}")
        if k == "fast":
            if s.get("runaway"): fails.append(f"{s['name']} runaway trajectory")   # loss is allowed in the fast segment, runaway is not
        elif k not in ("unknown", None) and s["valid_ratio"] < g["post_init_valid"]: warns.append(f"{s['name']} valid {s['valid_ratio']:.2f}")
    return ("FAIL" if fails else "WARN" if warns else "PASS"), fails + warns


# ---------------------------------------------------------------------------------------------------------------------
# Relative metrics, no ground truth (2026-09-16). The wrist has no external tracker, so nothing here compares against an
# absolute trajectory: HOME closure and ATE are deliberately absent. Every quantity below is either (a) measured inside a
# window where the IMU says the unit did not move, or (b) a comparison between VIO and an INDEPENDENT sensor (the gyro),
# or (c) VIO against itself over different horizons. None of these can prove the translation scale is right -- only that
# it is self-consistent and free of jumps. Say so wherever they are reported.

def still_windows_from_imu(imu_t_ns, gyro, accel, *, min_s: float = 0.4, gyro_deg_s: float = 4.0, accel_std: float = 0.25) -> list[tuple[int, int]]:
    """Intervals where the IMU says the unit is stationary: low angular rate AND low accelerometer variation."""
    if imu_t_ns is None or gyro is None or len(imu_t_ns) < 10: return []
    g = np.degrees(np.linalg.norm(np.asarray(gyro), axis=1)); a = np.linalg.norm(np.asarray(accel), axis=1)
    hz = len(imu_t_ns) / max((imu_t_ns[-1] - imu_t_ns[0]) / 1e9, 1e-9); w = max(int(hz * 0.2), 3)
    k = np.ones(w) / w
    gm = np.convolve(g, k, "same")
    am = np.sqrt(np.maximum(np.convolve(a * a, k, "same") - np.convolve(a, k, "same") ** 2, 0.0))
    still = (gm < gyro_deg_s) & (am < accel_std)
    out = []
    for i0, i1 in runs(still):
        if (imu_t_ns[i1 - 1] - imu_t_ns[i0]) / 1e9 >= min_s: out.append((int(imu_t_ns[i0]), int(imu_t_ns[i1 - 1])))
    return out


def stationary_jitter(t_ns, Ts, valid, windows: list[tuple[int, int]]) -> dict:
    """What VIO reports while the IMU says nothing moved. This is the cleanest bound on usable delta-p: whatever appears
    here is noise, and a delta-p smaller than it cannot be trusted."""
    pp = []; drift = []; rot = []; used = 0
    for t0, t1 in windows:
        m = valid & (t_ns >= t0) & (t_ns <= t1)
        if m.sum() < 5: continue
        P = Ts[m][:, :3, 3]; used += 1
        pp.append(float(np.linalg.norm(P.max(0) - P.min(0)) * 1e3))                       # peak-to-peak, mm
        drift.append(float(np.linalg.norm(P[-1] - P[0]) / max((t1 - t0) / 1e9, 1e-9) * 1e3))  # mm/s
        R = Ts[m][:, :3, :3]; rot.append(_rot_deg(R[0].T @ R[-1]))
    if not pp: return dict(n_windows=0)
    return dict(n_windows=used, total_windows=len(windows), p2p_mm_median=float(np.median(pp)), p2p_mm_p95=float(np.percentile(pp, 95)),
                p2p_mm_max=float(np.max(pp)), drift_mm_s_median=float(np.median(drift)), rot_deg_median=float(np.median(rot)))


def horizon_consistency(t_ns, Ts, valid, imu_t_ns=None, gyro=None, gyro_bias=None, R_camera_imu=None,
                        horizons=(5, 10, 15)) -> dict:
    """VIO over 5/10/15-frame horizons. Rotation is checked against the gyro (an INDEPENDENT sensor). Translation can only
    be checked against itself: `chain_ratio` is |p(t+k)-p(t)| divided by the path actually walked in between -- it must be
    <= 1, and a value far below 1 means the short steps are dominated by noise rather than motion. `scale_drift` compares
    the median step size in the first and last third of the episode; a metric filter that is slipping shows up as a trend."""
    from handumi_collector.pose.imu import integrate_gyro
    from handumi_collector.pose.se3 import rotation_angle_deg
    out = {}
    vi = np.nonzero(valid)[0]
    Rci = np.eye(3) if R_camera_imu is None else np.asarray(R_camera_imu)[:3, :3]
    for k in horizons:
        dp = []; ratio = []; rres = []; idx = []
        for a in range(0, len(vi) - k):
            i, j = vi[a], vi[a + k]
            if j - i != k: continue                                  # contiguous valid only
            step = float(np.linalg.norm(Ts[j][:3, 3] - Ts[i][:3, 3]))
            path = float(np.linalg.norm(np.diff(Ts[i:j + 1][:, :3, 3], axis=0), axis=1).sum())
            dp.append(step * 1e3); idx.append(i)
            if path > 1e-6: ratio.append(step / path)
            if imu_t_ns is not None and gyro is not None:
                dR_v = Ts[i][:3, :3].T @ Ts[j][:3, :3]
                dR_g = Rci @ integrate_gyro(imu_t_ns, gyro, int(t_ns[i]), int(t_ns[j]), gyro_bias) @ Rci.T
                rres.append(rotation_angle_deg(dR_v, dR_g))
        if not dp: out[f"k{k}"] = dict(n=0); continue
        dp = np.asarray(dp); idx = np.asarray(idx); r = dict(n=int(len(dp)), dp_mm_median=float(np.median(dp)), dp_mm_p95=float(np.percentile(dp, 95)))
        if ratio: r["chain_ratio_median"] = float(np.median(ratio))
        if rres: r.update(rot_vs_gyro_deg_median=float(np.median(rres)), rot_vs_gyro_deg_p95=float(np.percentile(rres, 95)))
        n3 = max(len(dp) // 3, 1)
        first, last = np.median(dp[:n3]), np.median(dp[-n3:])
        r["scale_drift_last_over_first"] = float(last / first) if first > 1e-9 else None
        out[f"k{k}"] = r
    # a metric VIO should walk ~k times as far over k frames as over 1; badly scaled or noise-dominated motion does not
    if all(out.get(f"k{k}", {}).get("dp_mm_median") for k in horizons):
        m5, m15 = out["k5"]["dp_mm_median"], out["k15"]["dp_mm_median"]
        out["growth_15_over_5"] = float(m15 / m5) if m5 > 1e-9 else None      # 3.0 = perfectly straight motion, 1.0 = noise
    return out
