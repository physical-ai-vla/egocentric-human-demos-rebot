"""Put a scale-free wrist VO trajectory into metres, using sparse cross-view anchors.

    monocular wrist VO  (shape correct, scale unknown, own arbitrary world)
              +
    cross-view anchors  (sparse, metric, in the head camera frame)
              |
         sim(3) fit  ->  s, R, t
              |
    metric wrist trajectory, with the residual at every anchor reported

This is the simplest thing that can work and is deliberately NOT an EKF or a factor graph (those come only if the
measured residuals demand them). It is offline post-processing on top of the existing pipeline: the VO trajectory comes
from handumi_collector.pose.run via the existing backends, the anchors from cross_view.py, and nothing here estimates
motion of its own.

A single global sim(3) assumes the VO scale is CONSTANT over the episode. That is the assumption the residuals test:
`scale_consistency` compares the scale implied by consecutive anchor pairs against the VO distance between the same
two times, so a drifting scale shows up as spread rather than hiding in one averaged number."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from .se3 import inv_T, make_T


def umeyama_sim3(src: np.ndarray, dst: np.ndarray, *, with_scale: bool = True):
    """Least-squares similarity transform mapping `src` onto `dst` (Umeyama 1991).

    Returns (scale, R (3x3), t (3,), rmse). `dst ~= scale * R @ src + t`."""
    S = np.asarray(src, np.float64).reshape(-1, 3)
    D = np.asarray(dst, np.float64).reshape(-1, 3)
    if len(S) != len(D) or len(S) < 3:
        raise ValueError(f"need >= 3 matched points, got {len(S)} and {len(D)}")
    mu_s, mu_d = S.mean(axis=0), D.mean(axis=0)
    Sc, Dc = S - mu_s, D - mu_d
    C = Dc.T @ Sc / len(S)
    U, sig, Vt = np.linalg.svd(C)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:      # keep it a rotation, never a reflection
        W[2, 2] = -1.0
    R = U @ W @ Vt
    var_s = (Sc ** 2).sum() / len(S)
    s = float((sig * np.diag(W)).sum() / var_s) if (with_scale and var_s > 1e-18) else 1.0
    t = mu_d - s * R @ mu_s
    res = D - (s * (R @ S.T).T + t)
    return s, R, t, float(np.sqrt((res ** 2).sum(axis=1).mean()))


@dataclass
class AlignmentResult:
    n_anchors: int = 0
    scale: float | None = None                 # VO units -> metres
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    position_rmse_mm: float | None = None
    position_residuals_mm: np.ndarray | None = None
    rotation_residuals_deg: np.ndarray | None = None
    rotation_rmse_deg: float | None = None
    scale_consistency: dict = field(default_factory=dict)
    anchor_geometry: dict = field(default_factory=dict)   # what the sim(3) rotation was actually constrained by
    ok: bool = False
    reason: str = ""

    def T_align(self) -> np.ndarray | None:
        """The rigid part as a 4x4 (the scale is applied separately — a 4x4 cannot carry it)."""
        return None if self.R is None else make_T(self.R, self.t)

    def apply(self, Ts_vo: np.ndarray) -> np.ndarray:
        """Map a VO trajectory into the metric frame: positions scaled and rotated, orientations rotated."""
        if self.R is None:
            raise RuntimeError("alignment did not succeed")
        Ts = np.asarray(Ts_vo, np.float64).reshape(-1, 4, 4).copy()
        out = np.tile(np.eye(4), (len(Ts), 1, 1))
        out[:, :3, 3] = self.scale * (self.R @ Ts[:, :3, 3].T).T + self.t
        out[:, :3, :3] = self.R @ Ts[:, :3, :3]
        return out

    def to_dict(self) -> dict:
        return dict(n_anchors=self.n_anchors, scale=self.scale, ok=self.ok, reason=self.reason,
                    position_rmse_mm=self.position_rmse_mm, rotation_rmse_deg=self.rotation_rmse_deg,
                    position_residual_p95_mm=(None if self.position_residuals_mm is None or not len(self.position_residuals_mm)
                                              else float(np.percentile(self.position_residuals_mm, 95))),
                    scale_consistency=self.scale_consistency,
                    R=(None if self.R is None else self.R.tolist()), t=(None if self.t is None else self.t.tolist()))


def scale_consistency(t_ns: np.ndarray, Ts_vo: np.ndarray, anchor_idx: np.ndarray, anchor_pos: np.ndarray,
                      *, min_baseline_m: float = 0.03) -> dict:
    """Scale implied by each consecutive anchor PAIR, against the VO distance over the same interval.

    A single global scale is only meaningful if these agree. Pairs closer than `min_baseline_m` in metric space are
    skipped: dividing two small noisy distances produces a scale estimate that says nothing."""
    idx = np.asarray(anchor_idx, int)
    A = np.asarray(anchor_pos, np.float64)
    V = np.asarray(Ts_vo, np.float64)[idx][:, :3, 3]
    if len(idx) < 2:
        return dict(n_pairs=0)
    d_metric = np.linalg.norm(np.diff(A, axis=0), axis=1)
    d_vo = np.linalg.norm(np.diff(V, axis=0), axis=1)
    use = (d_metric > min_baseline_m) & (d_vo > 1e-9)
    if not use.any():
        return dict(n_pairs=0, note=f"no anchor pair separated by more than {min_baseline_m*1e3:.0f} mm")
    s = d_metric[use] / d_vo[use]
    return dict(n_pairs=int(use.sum()), median=float(np.median(s)), p05=float(np.percentile(s, 5)),
                p95=float(np.percentile(s, 95)),
                spread_pct=float(100.0 * (np.percentile(s, 95) - np.percentile(s, 5)) / max(np.median(s), 1e-12)))


def align_to_anchors(t_ns: np.ndarray, Ts_vo: np.ndarray, valid_vo: np.ndarray, anchors, *,
                     max_dt_ms: float = 20.0, min_anchors: int = 3, min_span_m: float = 0.05) -> AlignmentResult:
    """Fit one sim(3) taking the VO trajectory into the metric anchor frame.

    `anchors` is any iterable of objects with `.t_ns`, `.valid` and `.T_head_wrist` (cross_view.CrossViewAnchor).
    Anchors are matched to the nearest VO sample within `max_dt_ms`; a VO sample that is not valid is never used."""
    t_ns = np.asarray(t_ns, np.int64)
    Ts_vo = np.asarray(Ts_vo, np.float64).reshape(-1, 4, 4)
    valid_vo = np.asarray(valid_vo, bool)
    r = AlignmentResult()
    good = [a for a in anchors if getattr(a, "valid", False) and getattr(a, "T_head_wrist", None) is not None]
    if len(good) < min_anchors:
        r.reason = f"{len(good)} valid anchors (< {min_anchors})"
        return r
    idx, dst, src, rot_pairs = [], [], [], []
    for a in good:
        j = int(np.argmin(np.abs(t_ns - a.t_ns)))
        if abs(int(t_ns[j]) - int(a.t_ns)) > max_dt_ms * 1e6 or not valid_vo[j]:
            continue
        idx.append(j); dst.append(a.T_head_wrist[:3, 3]); src.append(Ts_vo[j][:3, 3])
        rot_pairs.append((Ts_vo[j][:3, :3], a.T_head_wrist[:3, :3]))
    r.n_anchors = len(idx)
    if r.n_anchors < min_anchors:
        r.reason = f"{r.n_anchors} anchors line up with a valid VO sample within {max_dt_ms:.0f} ms (< {min_anchors})"
        return r
    D = np.asarray(dst); S = np.asarray(src)
    span = float(np.linalg.norm(D - D.mean(axis=0), axis=1).max())
    if span < min_span_m:
        r.reason = (f"anchors span only {span*1e3:.0f} mm (< {min_span_m*1e3:.0f}) — scale is not observable from a "
                    "set of points this close together")
        return r
    # Umeyama fits R from POSITIONS ONLY. Anchors strung along a line leave the rotation about that line completely
    # unconstrained, and anchors on a plane leave its sign poorly constrained -- yet the position residual comes out
    # beautiful either way, so a degenerate take would otherwise pass with a garbage orientation. Report the geometry
    # that the fit actually had to work with.
    C = D - D.mean(axis=0)
    sv = np.linalg.svd(C, compute_uv=False) / max(np.sqrt(len(C)), 1.0)
    r.anchor_geometry = dict(extent_m=[float(x) for x in sv],
                             collinearity=float(sv[1] / sv[0]) if sv[0] > 0 else 0.0,
                             planarity=float(sv[2] / sv[0]) if sv[0] > 0 else 0.0)
    if sv[0] > 0 and sv[1] / sv[0] < 0.05:
        r.anchor_geometry["warning"] = ("anchors are nearly COLLINEAR — the rotation about that line is not observable "
                                        "from them; the rotation residual below is meaningless, move in more than one "
                                        "direction")
    elif sv[0] > 0 and sv[2] / sv[0] < 0.02:
        r.anchor_geometry["warning"] = ("anchors are nearly COPLANAR — the rotation is weakly constrained out of that "
                                        "plane; treat the rotation residual as a lower bound")
    s, R, t, rmse = umeyama_sim3(S, D)
    r.scale, r.R, r.t = s, R, t
    res = D - (s * (R @ S.T).T + t)
    r.position_residuals_mm = np.linalg.norm(res, axis=1) * 1e3
    r.position_rmse_mm = float(rmse * 1e3)
    dr = [np.degrees(Rotation.from_matrix(inv_T(make_T(R @ Rv))[:3, :3] @ Ra).magnitude()) for Rv, Ra in rot_pairs]
    r.rotation_residuals_deg = np.asarray(dr)
    r.rotation_rmse_deg = float(np.sqrt((r.rotation_residuals_deg ** 2).mean()))
    r.scale_consistency = scale_consistency(t_ns, Ts_vo, np.asarray(idx), D)
    r.ok = True
    return r
