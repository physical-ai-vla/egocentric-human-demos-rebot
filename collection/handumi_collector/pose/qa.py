"""Episode-level pose QA (fiducial-free): HOME-return drift, static HOME stability, catastrophic jumps, tracking validity /
lost events, IMU-vs-VIO rotation consistency. Thresholds come from pose.yaml — nothing is hard-coded. Verdicts:
PASS | WARN | REJECT (plus REVIEW flags that do not by themselves reject)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from .imu import StaticWindow, integrate_gyro
from .se3 import inv_T, mean_pose, rotation_angle_deg

ORDER = {"PASS": 0, "WARN": 1, "REVIEW": 1, "REJECT": 2}


def worst(*verdicts: str) -> str:
    v = [x for x in verdicts if x]
    return "PASS" if not v else max(v, key=lambda x: ORDER.get(x, 0))


# These verdicts judge an ABSOLUTE trajectory: how much of it was solved at all, and how far it drifted by the time the
# hand came home. They belong to the absolute backends and must not be used as acceptance for the relative pipeline,
# which consumes inv(T(t)) T(t+k) and object-relative geometry and never integrates a path for them to describe.
#
# The distinction is not pedantic. On the 2026-09-15 pilot `opencv_vo` was first rejected on these -- valid ratio 0.29
# and 48 m of HOME drift -- which said nothing about whether its SHORT-WINDOW relative motion was usable. Measured
# properly afterwards it was not (3-10 deg median against the gyro), so the conclusion survived; the reasoning did
# not. A relative estimator judged by absolute criteria can be discarded for the wrong reason, or kept for one.
JUDGES = "absolute_trajectory"
ABSOLUTE_ONLY_METRICS = ("valid_ratio", "return_translation_mm", "return_rotation_deg", "home_start", "home_end")

@dataclass
class SideQA:
    side: str
    backend: str
    metric_scale: bool
    n_frames: int = 0
    n_valid: int = 0
    valid_ratio: float = 0.0
    tracking_states: dict = field(default_factory=dict)
    lost_events: int = 0
    longest_lost_s: float = 0.0
    home_start: dict | None = None
    home_end: dict | None = None
    return_translation_mm: float | None = None
    return_rotation_deg: float | None = None
    home_start_pos_std_mm: float | None = None
    home_start_rot_std_deg: float | None = None
    home_end_pos_std_mm: float | None = None
    home_end_rot_std_deg: float | None = None
    jumps_translation: int = 0
    jumps_rotation: int = 0
    max_step_translation_m: float | None = None
    max_step_rotation_deg: float | None = None
    imu_residual_deg_median: float | None = None
    imu_residual_deg_p95: float | None = None
    imu_residual_frac_over_warn: float | None = None
    flags: list[str] = field(default_factory=list)
    verdict: str = "PASS"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__); return d


def _flag(qa: SideQA, v: str, why: str) -> None:
    qa.verdict = worst(qa.verdict, v if v != "REVIEW" else "WARN"); qa.reasons.append(f"{v}: {why}")
    if v == "REVIEW": qa.flags.append(why)


def pose_stats_in_window(t_ns: np.ndarray, Ts: np.ndarray, valid: np.ndarray, w: StaticWindow):
    m = valid & (t_ns >= w.t0_ns) & (t_ns <= w.t1_ns)
    if m.sum() < 3: return None, None, None
    T = Ts[m]; Tm = mean_pose(T)
    pos_std = float(np.sqrt(((T[:, :3, 3] - Tm[:3, 3]) ** 2).sum(axis=1).mean()) * 1e3)
    rel = Rotation.from_matrix(Tm[:3, :3]).inv() * Rotation.from_matrix(T[:, :3, :3])
    rot_std = float(np.degrees(np.sqrt((rel.magnitude() ** 2).mean())))
    return Tm, pos_std, rot_std


def evaluate_side(*, side: str, backend: str, metric_scale: bool, t_ns: np.ndarray, Ts: np.ndarray, valid: np.ndarray, states: list[str],
                  home_start: StaticWindow | None, home_end: StaticWindow | None, imu_t_ns: np.ndarray | None, gyro: np.ndarray | None,
                  gyro_bias: np.ndarray | None, R_camera_imu: np.ndarray | None, cfg, fps: float) -> SideQA:
    qa = SideQA(side, backend, metric_scale)
    t_ns = np.asarray(t_ns, np.int64); Ts = np.asarray(Ts, np.float64); valid = np.asarray(valid, bool)
    qa.n_frames = int(len(t_ns)); qa.n_valid = int(valid.sum()); qa.valid_ratio = qa.n_valid / qa.n_frames if qa.n_frames else 0.0
    qa.tracking_states = {s: int(states.count(s)) for s in sorted(set(states))}
    # ---- validity / lost
    vr = cfg.valid_ratio
    if qa.n_frames == 0 or qa.n_valid == 0: _flag(qa, "REJECT", "no valid poses")
    elif qa.valid_ratio < float(vr["reject"]): _flag(qa, "REJECT", f"valid ratio {qa.valid_ratio:.3f} < {vr['reject']}")
    elif qa.valid_ratio < float(vr["warn"]): _flag(qa, "WARN", f"valid ratio {qa.valid_ratio:.3f} < {vr['warn']}")
    lost_runs = _runs(np.array([s == "lost" for s in states], bool))
    qa.lost_events = len(lost_runs)
    if lost_runs:
        qa.longest_lost_s = max((t_ns[b - 1] - t_ns[a]) / 1e9 + 1.0 / fps for a, b in lost_runs)
        lim = float(cfg.lost_events["reject_if_lost_longer_than_s"])
        if qa.longest_lost_s > lim: _flag(qa, "REJECT", f"tracking lost for {qa.longest_lost_s:.2f}s > {lim}s")
        else: _flag(qa, "WARN", f"{qa.lost_events} lost event(s), longest {qa.longest_lost_s:.2f}s")
    # ---- HOME windows
    qa.home_start = home_start.to_dict() if home_start else None; qa.home_end = home_end.to_dict() if home_end else None
    Ts_start = Ts_end = None
    if home_start is None: _flag(qa, "REVIEW", "no still HOME window at episode start")
    else:
        Ts_start, qa.home_start_pos_std_mm, qa.home_start_rot_std_deg = pose_stats_in_window(t_ns, Ts, valid, home_start)
        if Ts_start is None: _flag(qa, "REVIEW", "no valid poses inside the HOME start window (VIO not yet initialised?)")
    if home_end is None: _flag(qa, "REVIEW", "no still HOME window at episode end")
    else:
        Ts_end, qa.home_end_pos_std_mm, qa.home_end_rot_std_deg = pose_stats_in_window(t_ns, Ts, valid, home_end)
        if Ts_end is None: _flag(qa, "REVIEW", "no valid poses inside the HOME end window")
    sj = cfg.static_jitter["review"]
    for nm, ps, rs in (("start", qa.home_start_pos_std_mm, qa.home_start_rot_std_deg), ("end", qa.home_end_pos_std_mm, qa.home_end_rot_std_deg)):
        if rs is not None and rs > float(sj["rot_std_deg"]): _flag(qa, "REVIEW", f"HOME {nm} rotation jitter {rs:.2f}deg > {sj['rot_std_deg']}")
        if ps is not None and metric_scale and ps > float(sj["pos_std_mm"]): _flag(qa, "REVIEW", f"HOME {nm} position jitter {ps:.1f}mm > {sj['pos_std_mm']}")
    # ---- HOME return drift
    if Ts_start is not None and Ts_end is not None:
        E = inv_T(Ts_start) @ Ts_end
        qa.return_translation_mm = float(np.linalg.norm(E[:3, 3]) * 1e3); qa.return_rotation_deg = rotation_angle_deg(E[:3, :3])
        hr = cfg.home_return
        rot_v = "PASS" if qa.return_rotation_deg < float(hr["pass"]["rotation_deg"]) else "WARN" if qa.return_rotation_deg < float(hr["warn"]["rotation_deg"]) else "REJECT"
        if rot_v != "PASS": _flag(qa, rot_v, f"HOME return rotation drift {qa.return_rotation_deg:.2f}deg")
        if metric_scale:
            tr_v = "PASS" if qa.return_translation_mm < float(hr["pass"]["translation_mm"]) else "WARN" if qa.return_translation_mm < float(hr["warn"]["translation_mm"]) else "REJECT"
            if tr_v != "PASS": _flag(qa, tr_v, f"HOME return translation drift {qa.return_translation_mm:.1f}mm")
        else:
            qa.flags.append("translation drift not metric (visual-only backend) — reported in backend units")
    # ---- jumps (consecutive valid frames only)
    vi = np.nonzero(valid)[0]
    if len(vi) > 1:
        a, b = vi[:-1], vi[1:]; consecutive = (b - a) == 1
        if consecutive.any():
            Ta, Tb = Ts[a[consecutive]], Ts[b[consecutive]]
            dtr = np.linalg.norm(Tb[:, :3, 3] - Ta[:, :3, 3], axis=1)
            drot = np.degrees((Rotation.from_matrix(Ta[:, :3, :3]).inv() * Rotation.from_matrix(Tb[:, :3, :3])).magnitude())
            qa.max_step_translation_m = float(dtr.max()); qa.max_step_rotation_deg = float(drot.max())
            qa.jumps_rotation = int((drot > float(cfg.jumps["rotation_deg"])).sum())
            qa.jumps_translation = int((dtr > float(cfg.jumps["translation_m"])).sum()) if metric_scale else 0
            if qa.jumps_rotation or qa.jumps_translation: _flag(qa, "REJECT", f"catastrophic pose jumps: {qa.jumps_translation} translation, {qa.jumps_rotation} rotation")
    # ---- IMU consistency (independent gyro integration vs VIO delta rotation)
    if imu_t_ns is not None and gyro is not None and len(imu_t_ns) > 10 and len(vi) > 2:
        ic = cfg.imu_consistency; W = int(ic["window_frames"]); Rci = np.eye(3) if R_camera_imu is None else np.asarray(R_camera_imu)[:3, :3]
        res = []
        for k in range(0, len(vi) - W, max(W // 2, 1)):
            i, j = vi[k], vi[k + W]
            if j - i != W: continue                      # window must be contiguous valid
            dR_vio = Ts[i][:3, :3].T @ Ts[j][:3, :3]
            dR_gyro_imu = integrate_gyro(imu_t_ns, gyro, int(t_ns[i]), int(t_ns[j]), gyro_bias)
            dR_gyro_cam = Rci @ dR_gyro_imu @ Rci.T
            res.append(rotation_angle_deg(dR_vio, dR_gyro_cam))
        if res:
            r = np.array(res); qa.imu_residual_deg_median = float(np.median(r)); qa.imu_residual_deg_p95 = float(np.percentile(r, 95))
            qa.imu_residual_frac_over_warn = float((r > float(ic["warn_deg"])).mean())
            if R_camera_imu is None: qa.flags.append("imu_consistency computed with identity T_camera_imu (uncalibrated) — magnitudes only meaningful after calibration")
            if qa.imu_residual_deg_p95 > float(ic["reject_deg"]): _flag(qa, "REJECT", f"IMU/VIO rotation residual p95 {qa.imu_residual_deg_p95:.1f}deg > {ic['reject_deg']}")
            elif qa.imu_residual_frac_over_warn > float(ic["max_frac_over_warn"]): _flag(qa, "WARN", f"{qa.imu_residual_frac_over_warn:.0%} of windows have IMU/VIO residual > {ic['warn_deg']}deg")
    return qa


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if len(mask) == 0: return []
    d = np.diff(mask.astype(np.int8)); starts = list(np.nonzero(d == 1)[0] + 1); ends = list(np.nonzero(d == -1)[0] + 1)
    if mask[0]: starts.insert(0, 0)
    if mask[-1]: ends.append(len(mask))
    return list(zip(starts, ends))
