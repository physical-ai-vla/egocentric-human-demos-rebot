"""Metric UMI position from a FIXED monocular camera (C922) + the wrist IMU orientation (2026-10-15, user: depth cam out of teleop).

Known rigid UMI keypoints P_i (UMI/TCP frame, metres, from CAD or a caliper-scaled self-calibration), the IMU orientation of the
UMI in the camera frame R_CH (= R_CW . R_WH, after the INIT yaw alignment), and detected pixels u_i give the translation t_CH:

    u_i ~ K (R_CH P_i + t)      ->  for each point, x_i x (R_CH P_i + t) = 0 with x_i = K^-1 [u_i, 1]  (2 independent rows, LINEAR in t)

so >= 2 visible keypoints give a metric t (the known spacing fixes the depth); 1 keypoint leaves depth along its ray unobservable
(returned with ok=False, a caller may hold the last depth). The linear solve is refined by Gauss-Newton on the pixel reprojection
error and one outlier pass (a keypoint whose residual > max(3 * median, outlier_px) is dropped and t re-solved).
Depth accuracy ~ z^2 / (f * span) * pixel noise: 1 m, f = 1400 px (1080p), 15 cm span, 1.5 px -> ~7 mm."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class KptSolution:
    ok: bool
    t_CH: np.ndarray | None          # UMI origin (TCP) in the camera frame, metres
    rms_px: float
    used: list                       # keypoint names kept
    dropped: list                    # names rejected as outliers
    reason: str = ""


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])


def project(K: np.ndarray, R_CH: np.ndarray, t: np.ndarray, P: np.ndarray) -> np.ndarray:
    X = P @ R_CH.T + t; return (X[:, :2] / X[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def solve_translation(K: np.ndarray, R_CH: np.ndarray, P: np.ndarray, uv: np.ndarray, names=None, *, w=None,
                      outlier_px: float = 6.0, gn_iters: int = 5, min_z: float = 0.1) -> KptSolution:
    """P (N,3) keypoints in the UMI frame, uv (N,2) pixels (NaN = not detected), w (N,) optional confidences."""
    P, uv = np.asarray(P, np.float64), np.asarray(uv, np.float64); names = list(names or range(len(P)))
    w = np.ones(len(P)) if w is None else np.asarray(w, np.float64)
    keep = np.all(np.isfinite(uv), 1) & (w > 0); dropped = []
    for _ in range(2):                                                     # solve, drop the worst outlier once, re-solve
        if keep.sum() < 2: return KptSolution(False, None, np.nan, [names[i] for i in np.flatnonzero(keep)], dropped, "need >= 2 keypoints for metric depth")
        idx = np.flatnonzero(keep); Kinv = np.linalg.inv(K); A, b = [], []
        for i in idx:
            x = Kinv @ np.r_[uv[i], 1.0]; S = _skew(x) * np.sqrt(w[i]); A.append(S); b.append(-S @ (R_CH @ P[i]))
        t, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
        for _ in range(gn_iters):                                          # Gauss-Newton on pixel residuals
            X = P[idx] @ R_CH.T + t
            if np.any(X[:, 2] <= min_z * 0.5): break
            r = (project(K, R_CH, t, P[idx]) - uv[idx]) * np.sqrt(w[idx])[:, None]
            J = np.zeros((2 * len(idx), 3))
            for k, (Xk, wk) in enumerate(zip(X, np.sqrt(w[idx]))):
                z = Xk[2]; J[2 * k] = wk * K[0, 0] * np.array([1 / z, 0, -Xk[0] / z ** 2]); J[2 * k + 1] = wk * K[1, 1] * np.array([0, 1 / z, -Xk[1] / z ** 2])
            dt, *_ = np.linalg.lstsq(J, -r.reshape(-1), rcond=None); t = t + dt
            if np.linalg.norm(dt) < 1e-7: break
        res = np.linalg.norm(project(K, R_CH, t, P[idx]) - uv[idx], axis=1)
        worst = int(np.argmax(res))
        if len(idx) >= 3 and res[worst] > max(3 * np.median(res), outlier_px) and not dropped:
            keep[idx[worst]] = False; dropped.append(names[idx[worst]]); continue
        break
    z_ok = np.all((P[idx] @ R_CH.T + t)[:, 2] > min_z)
    return KptSolution(bool(z_ok), t, float(np.sqrt(np.mean(res ** 2))), [names[i] for i in idx], dropped, "" if z_ok else "behind / too close to the camera")
