"""[2026-10-12] Per-side UMI pose fusion v1: MASt3R segments (precise local 6DoF + trajectory shape) + Orbbec rigid body point
(metric position, shared L/R world) + wrist IMU gyro (ROTATION ONLY, short gaps).  IMU never makes position.

Model (one wrist, camera frame c, MASt3R segment s with its own Sim3 frame):
    p_W_body(t) = R_s (sigma_s p_s(t) + R_s_c(t) r_c) + t_s          r_c = rigid body point in the CAMERA frame, shared by all segments
    R_W_c(t)    = R_s R_s_c(t)                                     p_W_c(t) = R_s sigma_s p_s(t) + t_s
Per segment (sigma_s, R_s, t_s) and the shared r_c are fit jointly to the Orbbec body point (robust soft-L1 loss).
r_c may be FIXED (r_c_prior, e.g. the robust median over earlier episodes) -- recommended once a few episodes exist.
Gaps (frames with no valid MASt3R pose): rotation = gyro integrated FORWARD from the last aligned frame and BACKWARD from the
next one, blended linearly over the gap (continuity at both ends); position = p_W_body - R_W_c r_c (needs Orbbec).
TCP = T_W_c @ X_cam_tcp (camera_tcp_v2) -- applied by the caller.

Provenance per frame (spec): pose_source  MAST3R_ORBBEC | ORBBEC_IMU | INVALID   (MAST3R alone has no metric world here)
                             position_source MAST3R | ORBBEC | NONE ;  rotation_source MAST3R | IMU | NONE
                             confidence_position / confidence_rotation in [0, 1];  imu_gap_s (length of the gap a frame sits in)
IMU gap policy (measured 2026-10-12, 14:31 ep, Kalibr R_cb): the error grows with the integration time from an anchor, and a
gap is bridged from BOTH ends, so a frame is judged by its distance to the NEAREST aligned MASt3R anchor:
<= 0.5 s PASS (conf 1.0), <= 3 s WARN (conf 0.5), > 3 s INVALID; and a whole gap longer than IMU_MAX_GAP_S is INVALID.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

IMU_PASS_S, IMU_WARN_S, IMU_MAX_GAP_S = 0.5, 3.0, 6.0


@dataclass
class SideInputs:
    t: np.ndarray                 # (n,) wrist frame times, s (host clock)
    seg: np.ndarray               # (n,) MASt3R segment id, -1 = no valid pose
    p_seg: np.ndarray             # (n,3) MASt3R camera position in its segment frame
    R_seg: np.ndarray             # (n,3,3) MASt3R camera rotation in its segment frame
    body_t: np.ndarray            # (m,) Orbbec times, s (same host clock)
    body_p: np.ndarray            # (m,3) Orbbec rigid body point, world (table) frame, m
    body_ok: np.ndarray           # (m,)
    gyro_t: np.ndarray            # (k,) s, host clock
    gyro_w: np.ndarray            # (k,3) rad/s, IMU body frame, bias removed
    R_cb: np.ndarray              # (3,3) camera <- IMU-body rotation (Kalibr)
    r_c_prior: np.ndarray | None = None
    min_seg_frames: int = 30


@dataclass
class SideFused:
    T_W_c: np.ndarray
    pose_source: np.ndarray
    position_source: np.ndarray
    rotation_source: np.ndarray
    confidence_position: np.ndarray
    confidence_rotation: np.ndarray
    imu_gap_s: np.ndarray
    segments: list = field(default_factory=list)
    r_c: np.ndarray | None = None
    fit: dict = field(default_factory=dict)


def _interp_body(inp, t):
    ok_t = inp.body_t[inp.body_ok]; p = inp.body_p[inp.body_ok]
    if len(ok_t) < 2: return np.full((len(t), 3), np.nan), np.zeros(len(t), bool)
    j = np.clip(np.searchsorted(ok_t, t) - 1, 0, len(ok_t) - 2); t0, t1 = ok_t[j], ok_t[j + 1]
    w = np.clip((t - t0) / np.maximum(t1 - t0, 1e-9), 0, 1); out = (1 - w)[:, None] * p[j] + w[:, None] * p[j + 1]
    near = (t1 - t0 <= 0.1) & (t >= ok_t[0]) & (t <= ok_t[-1]); return out, near


def _gyro_delta(inp, ta, tb):
    """camera rotation increment R_c(ta)^T R_c(tb) from the gyro (ta < tb or ta > tb)."""
    sgn = 1.0 if tb >= ta else -1.0; lo, hi = (ta, tb) if tb >= ta else (tb, ta)
    sel = (inp.gyro_t >= lo) & (inp.gyro_t < hi); R = np.eye(3)
    if sel.any():
        ts = np.r_[lo, inp.gyro_t[sel], hi]; ws = np.r_[inp.gyro_w[sel][:1], inp.gyro_w[sel]]
        for k in range(len(ts) - 1): R = R @ Rot.from_rotvec(ws[k] * (ts[k + 1] - ts[k])).as_matrix()
    Rc = inp.R_cb @ R @ inp.R_cb.T
    return Rc if sgn > 0 else Rc.T


def fit_segments(inp):
    """joint robust fit of per-segment Sim3 + shared r_c to the Orbbec body point. -> dict seg -> (sigma, R, t), r_c, report"""
    pb, okb = _interp_body(inp, inp.t)
    segs = [s for s in np.unique(inp.seg[inp.seg >= 0]) if ((inp.seg == s) & okb).sum() >= inp.min_seg_frames]
    if not segs: return {}, inp.r_c_prior, dict(reason="no segment overlaps Orbbec")
    fix_r = inp.r_c_prior is not None; init = []
    for s in segs:                                                       # Umeyama init with r_c = 0
        m = (inp.seg == s) & okb; X, Y = inp.p_seg[m], pb[m]; mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
        U, S, Vt = np.linalg.svd(Yc.T @ Xc); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt
        sig = float((S * np.diag(D)).sum() / max((Xc ** 2).sum(), 1e-12)); t = my - sig * R @ mx
        init += list(Rot.from_matrix(R).as_rotvec()) + list(t) + [np.log(max(sig, 1e-6))]
    x0 = np.array(init + ([] if fix_r else [0.0, 0.0, 0.0]))

    def unpack(x):
        par = {}
        for k, s in enumerate(segs):
            v = x[7 * k:7 * k + 7]; par[s] = (np.exp(v[6]), Rot.from_rotvec(v[:3]).as_matrix(), v[3:6])
        return par, (inp.r_c_prior if fix_r else x[7 * len(segs):])

    def resid(x):
        par, rc = unpack(x); r = []
        for s in segs:
            m = (inp.seg == s) & okb; sig, R, t = par[s]
            pred = (R @ (sig * inp.p_seg[m] + np.einsum("nij,j->ni", inp.R_seg[m], rc)).T).T + t
            r.append((pred - pb[m]).ravel())
        return np.concatenate(r)
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=0.01)
    par, rc = unpack(sol.x); res = np.abs(resid(sol.x)).reshape(-1, 3)
    rep = dict(segments=[dict(segment=int(s), frames=int(((inp.seg == s) & okb).sum()), sigma=float(par[s][0])) for s in segs],
               r_c=np.asarray(rc).tolist(), r_c_fixed=fix_r, resid_mm_p50=float(np.median(np.linalg.norm(res, axis=1)) * 1e3),
               resid_mm_p90=float(np.percentile(np.linalg.norm(res, axis=1), 90) * 1e3))
    return par, np.asarray(rc), rep


def fuse_side(inp):
    n = len(inp.t); par, r_c, rep = fit_segments(inp)
    T = np.tile(np.eye(4), (n, 1, 1)); have_R = np.zeros(n, bool); have_p = np.zeros(n, bool)
    rot_src = np.full(n, "NONE", object); pos_src = np.full(n, "NONE", object); conf_r = np.zeros(n); conf_p = np.zeros(n); gap = np.zeros(n)
    for s, (sig, R, t) in par.items():                                  # aligned MASt3R frames
        m = inp.seg == s
        T[m, :3, :3] = R @ inp.R_seg[m]; T[m, :3, 3] = (R @ (sig * inp.p_seg[m]).T).T + t
        have_R[m] = have_p[m] = True; rot_src[m] = "MAST3R"; pos_src[m] = "MAST3R"; conf_r[m] = 1.0; conf_p[m] = 1.0
    # rotation bridge over every run of frames without an aligned MASt3R rotation
    i = 0
    while i < n:
        if have_R[i] or rot_src[i] == "IMU": i += 1; continue
        j = i
        while j < n and not have_R[j]: j += 1                           # gap = [i, j)
        a, b = i - 1, j                                                  # anchors (may be out of range)
        L = (inp.t[min(b, n - 1)] - inp.t[max(a, 0)]) if (a >= 0 and b < n) else np.inf
        for k in range(i, j):
            Rf = T[a, :3, :3] @ _gyro_delta(inp, inp.t[a], inp.t[k]) if a >= 0 else None
            Rb = T[b, :3, :3] @ _gyro_delta(inp, inp.t[b], inp.t[k]) if b < n else None
            if Rf is not None and Rb is not None:
                w = (inp.t[k] - inp.t[a]) / max(inp.t[b] - inp.t[a], 1e-9)
                d = Rot.from_matrix(Rf.T @ Rb).as_rotvec(); Rk = Rf @ Rot.from_rotvec(w * d).as_matrix(); dist = min(inp.t[k] - inp.t[a], inp.t[b] - inp.t[k])
            else:
                Rk = Rf if Rf is not None else Rb; dist = abs(inp.t[k] - inp.t[a if Rf is not None else b]) if Rk is not None else np.inf
            gap[k] = L if np.isfinite(L) else dist
            if Rk is None or dist > IMU_WARN_S or gap[k] > IMU_MAX_GAP_S: continue
            T[k, :3, :3] = Rk; rot_src[k] = "IMU"; conf_r[k] = 1.0 if dist <= IMU_PASS_S else 0.5
        i = j
    # position on non-MASt3R frames: Orbbec body point - R r_c
    pb, okb = _interp_body(inp, inp.t)
    if r_c is not None:
        m = (rot_src == "IMU") & okb
        T[m, :3, 3] = pb[m] - np.einsum("nij,j->ni", T[m, :3, :3], r_c); pos_src[m] = "ORBBEC"; conf_p[m] = 0.8
    pose_src = np.full(n, "INVALID", object)
    pose_src[(rot_src == "MAST3R") & (pos_src == "MAST3R")] = "MAST3R_ORBBEC"
    pose_src[(rot_src == "IMU") & (pos_src == "ORBBEC")] = "ORBBEC_IMU"
    return SideFused(T, pose_src, pos_src, rot_src, conf_p, conf_r, gap, rep.get("segments", []), r_c, rep)
