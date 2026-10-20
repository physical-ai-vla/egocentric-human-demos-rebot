"""[2026-10-13] Offline batch smoother (factor graph, robust least squares) for ONE wrist: training-label pose, so future frames
are used. Replaces the per-segment Sim3 of fusion.py after 10-12/13 measurements showed MASt3R drifts within seconds
(position Sim3 residual p50 69 mm per segment vs 9 mm on 2 s windows; tilt vs gravity 5-12 deg at 10 s).

Variables (frame k on the wrist timeline): d_k (3) rotation correction, R_k = Exp(d_k) R0_k (world <- camera);
p_k (3) camera centre in the table world W; log-scale of each MASt3R scale window (segment x SCALE_WIN_S); r_c (3) = canonical
body point in the camera frame (shared); b (3) gyro bias residual.
Factors (all whitened, robust soft-L1 where outliers are expected):
  GYRO   Log(dR_gyro^T R_k^T R_k+1)                                 sigma_gyro  per frame (rotation only, IMU never makes position)
  M3R    Log(dR_m3^T R_k^T R_k+h)          h in M3_HORIZONS, same segment      sigma_m3r(h)
  M3T    R_k^T (p_k+h - p_k) - exp(s) dp_m3(k->k+h, in camera k)             sigma_m3t(h)  (short horizons only: MASt3R is trusted locally)
  GRAV   R_k^T e_z - up_meas_k  (accel / |a|, camera frame) when ||a|-g| small, every GRAV_STRIDE frames      sigma_grav, robust
  ORB    p_k + R_k r_c - z_k   (Orbbec canonical, wrist clock)          sigma_orb, robust; pass 2 drops |res| > GATE_MM (LEFT_RIGHT_MISMATCH)
  ACC    (p_k+1 - 2 p_k + p_k-1) / (sigma_acc dt^2), robust  -- continuity across MASt3R segment starts / Orbbec gaps
  priors: r_c ~ r_c0 (5 cm), scale windows smooth within a segment and ~ median of Orbbec-supported windows, bias ~ 0.
Yaw has no absolute factor: it is carried by GYRO/M3R and corrected only through ORB geometry when the motion makes it observable.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot

G = 9.81


@dataclass
class SmootherCfg:
    sigma_gyro_deg: float = 0.25          # [2026-10-13 user] IMU = authoritative rotation, MASt3R rotation = secondary
    m3_horizons: tuple = (1, 5, 15)
    sigma_m3r_deg: tuple = (2.0, 4.0, 8.0)
    sigma_m3t_mm: tuple = (2.0, 6.0, 15.0)
    sigma_grav_deg: float = 3.0
    grav_stride: int = 3
    grav_acc_tol: float = 0.8            # m/s^2 around g
    sigma_orb_mm: float = 6.0
    gate_mm: float = 40.0
    m3_vmax: float = 2.0                 # m/s: MASt3R pairs faster than this (teleport spikes) are dropped
    m3_gate_sigma: float = 6.0           # pass-2: MASt3R pairs whose translation residual exceeds this many sigma are dropped
    scale_win_s: float = 2.0
    sigma_scale_step: float = 0.05       # log-scale change between consecutive windows
    sigma_scale_prior: float = 0.4       # every window pulled to the median of Orbbec-supported windows (unsupported segments)
    sigma_acc: float = 6.0               # m/s^2, position smoothness p_k+1 - 2p_k + p_k-1 (keeps segments / gaps continuous)
    r_c0: tuple = (0.0, 0.06, 0.10)
    sigma_rc_m: float = 0.05
    max_nfev: int = 30


def _so3_log(Rm):
    return Rot.from_matrix(Rm).as_rotvec()


def smooth_side(t, R0, p0, seg, R_m3, p_m3, gyro_t, gyro_w, R_cb, acc_t, acc, orb_z, orb_ok, cfg=SmootherCfg(), orb_keep=None, pair_drop=None, scale_init=None):
    """t (n,) s; R0 (n,3,3) initial world<-camera; p0 (n,3); seg (n,) MASt3R segment (-1 none); R_m3/p_m3 MASt3R camera pose in its
    segment frame; gyro_t/gyro_w (rad/s, body frame, raw) ; R_cb camera<-body; acc_t/acc (m/s^2 body); orb_z (n,3) Orbbec canonical
    point on the wrist clock, orb_ok (n,). Returns dict(R, p, r_c, scale_log, info, orb_res_mm, orb_used)."""
    n = len(t)
    # gyro increments between consecutive frames (camera frame), raw (bias handled as a variable, linearised)
    dRg = np.zeros((n - 1, 3, 3)); Jb = np.zeros((n - 1, 3, 3))
    for k in range(n - 1):
        sel = (gyro_t >= t[k]) & (gyro_t < t[k + 1]); dt = t[k + 1] - t[k]
        w = gyro_w[sel].mean(0) if sel.any() else (gyro_w[np.argmin(np.abs(gyro_t - t[k]))])
        dRg[k] = R_cb @ Rot.from_rotvec(w * dt).as_matrix() @ R_cb.T; Jb[k] = -R_cb * dt         # d(rotvec)/d(bias) ~ -R_cb dt
    # gravity measurements (camera frame "up" = +a / |a| when quasi-static)
    up = np.full((n, 3), np.nan)
    for k in range(0, n, cfg.grav_stride):
        sel = np.abs(acc_t - t[k]) <= 0.5 / 30
        if not sel.any(): continue
        a = acc[sel].mean(0)
        if abs(np.linalg.norm(a) - G) < cfg.grav_acc_tol: up[k] = R_cb @ (a / np.linalg.norm(a))
    # scale windows
    sw = np.full(n, -1)
    wid = 0; wmap = {}
    for s in np.unique(seg[seg >= 0]):
        m = np.flatnonzero(seg == s); t0 = t[m[0]]
        for k in m:
            key = (s, int((t[k] - t0) // cfg.scale_win_s))
            if key not in wmap: wmap[key] = wid; wid += 1
            sw[k] = wmap[key]
    nW = wid
    # MASt3R pairs
    pairs = []
    for hi, h in enumerate(cfg.m3_horizons):
        for k in range(n - h):
            if seg[k] >= 0 and seg[k] == seg[k + h]:
                Rk = R_m3[k]; dR = Rk.T @ R_m3[k + h]; dp = Rk.T @ (p_m3[k + h] - p_m3[k])
                pairs.append((k, k + h, hi, dR, dp))
    # initial scale per window: median ratio of |Orbbec displacement| to |MASt3R displacement| over 15-frame pairs
    s0 = np.zeros(nW)
    for key, j in wmap.items():
        r = [np.linalg.norm(orb_z[b] - orb_z[a]) / max(np.linalg.norm(dp), 1e-9) for a, b, hi, dR, dp in pairs
             if hi == len(cfg.m3_horizons) - 1 and sw[a] == j and orb_ok[a] and orb_ok[b] and np.linalg.norm(orb_z[b] - orb_z[a]) > 0.02]
        s0[j] = np.log(np.median(r)) if len(r) >= 3 else np.nan
    fill = np.nanmedian(s0) if np.isfinite(s0).any() else 0.0; s0 = np.where(np.isfinite(s0), s0, fill); s_med = fill
    if scale_init is not None: s0 = scale_init.copy()
    # [teleport guard] drop MASt3R pairs whose scaled speed is implausible, plus pass-2 rejects
    n_all = len(pairs)
    pairs = [q for q in pairs if np.exp(s0[sw[q[0]]]) * np.linalg.norm(q[4]) / max(t[q[1]] - t[q[0]], 1e-6) <= cfg.m3_vmax]
    n_fast = n_all - len(pairs)
    if pair_drop: pairs = [q for q in pairs if (q[0], q[1]) not in pair_drop]
    keep = orb_ok.copy() if orb_keep is None else (orb_ok & orb_keep)
    # parameter layout
    iD, iP = 0, 3 * n; iS = 6 * n; iR = iS + nW; iB = iR + 3; nx = iB + 3
    x0 = np.zeros(nx); x0[iP:iS] = p0.ravel(); x0[iS:iR] = s0; x0[iR:iB] = cfg.r_c0
    sg = np.radians(cfg.sigma_gyro_deg); sgr = np.radians(cfg.sigma_grav_deg); so = cfg.sigma_orb_mm / 1e3
    smr = [np.radians(v) for v in cfg.sigma_m3r_deg]; smt = [v / 1e3 for v in cfg.sigma_m3t_mm]
    grav_idx = np.flatnonzero(np.isfinite(up[:, 0])); orb_idx = np.flatnonzero(keep)
    scale_pairs = [(wmap[(s, w)], wmap[(s, w + 1)]) for (s, w) in wmap if (s, w + 1) in wmap]
    ez = np.array([0, 0, 1.0]); dt2 = (0.5 * (t[2:] - t[:-2])) ** 2

    def rot(x):
        return Rot.from_rotvec(x[iD:iP].reshape(n, 3)).as_matrix() @ R0

    def resid(x):
        R = rot(x); p = x[iP:iS].reshape(n, 3); s = x[iS:iR]; rc = x[iR:iB]; b = x[iB:]
        out = []
        rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
        corr = np.einsum("nij,j->ni", Jb, b)                                  # first-order bias correction
        dRgb = np.einsum("nij,njk->nik", dRg, Rot.from_rotvec(corr).as_matrix())
        out.append((_so3_log(np.einsum("nji,njk->nik", dRgb, rel)) / sg).ravel())
        if pairs:
            a = np.array([q[0] for q in pairs]); bb = np.array([q[1] for q in pairs]); hi = np.array([q[2] for q in pairs])
            dR = np.array([q[3] for q in pairs]); dp = np.array([q[4] for q in pairs])
            er = _so3_log(np.einsum("nji,njk->nik", dR, np.einsum("nji,njk->nik", R[a], R[bb]))) / np.array(smr)[hi][:, None]
            et = (np.einsum("nji,nj->ni", R[a], p[bb] - p[a]) - np.exp(s[sw[a]])[:, None] * dp) / np.array(smt)[hi][:, None]
            out += [er.ravel(), et.ravel()]
        if len(grav_idx):
            out.append(((np.einsum("nji,j->ni", R[grav_idx], ez) - up[grav_idx]) / sgr).ravel())
        if len(orb_idx):
            out.append(((p[orb_idx] + np.einsum("nij,j->ni", R[orb_idx], rc) - orb_z[orb_idx]) / so).ravel())
        if scale_pairs:
            out.append(np.array([(s[j2] - s[j1]) / cfg.sigma_scale_step for j1, j2 in scale_pairs]))
        out.append((s - s_med) / cfg.sigma_scale_prior)
        out.append(((p[2:] - 2 * p[1:-1] + p[:-2]) / (cfg.sigma_acc * dt2[:, None])).ravel())
        out.append((rc - np.array(cfg.r_c0)) / cfg.sigma_rc_m); out.append(b / 0.01)
        return np.concatenate(out)

    # sparsity pattern (same row order as resid)
    rows = []
    def blk(cols): rows.append(cols)
    for k in range(n - 1):
        for _ in range(3): blk([*range(3 * k, 3 * k + 6), iB, iB + 1, iB + 2])
    for a, bb, hi, dR, dp in pairs:
        for _ in range(3): blk([*range(3 * a, 3 * a + 3), *range(3 * bb, 3 * bb + 3)])
    for a, bb, hi, dR, dp in pairs:
        for _ in range(3): blk([*range(3 * a, 3 * a + 3), *range(iP + 3 * a, iP + 3 * a + 3), *range(iP + 3 * bb, iP + 3 * bb + 3), iS + sw[a]])
    for k in grav_idx:
        for _ in range(3): blk([*range(3 * k, 3 * k + 3)])
    for k in orb_idx:
        for _ in range(3): blk([*range(3 * k, 3 * k + 3), *range(iP + 3 * k, iP + 3 * k + 3), iR, iR + 1, iR + 2])
    for j1, j2 in scale_pairs: blk([iS + j1, iS + j2])
    for j in range(nW): blk([iS + j])
    for k in range(1, n - 1):
        for _ in range(3): blk([*range(iP + 3 * (k - 1), iP + 3 * (k + 2))])
    for j in range(3): blk([iR + j])
    for j in range(3): blk([iB + j])
    S = lil_matrix((len(rows), nx), dtype=np.int8)
    for r, cols in enumerate(rows): S[r, cols] = 1
    nres = len(resid(x0)); assert nres == len(rows), (nres, len(rows))
    sol = least_squares(resid, x0, jac_sparsity=S, loss="soft_l1", f_scale=3.0, max_nfev=cfg.max_nfev, x_scale="jac", method="trf")
    R = rot(sol.x); p = sol.x[iP:iS].reshape(n, 3); rc = sol.x[iR:iB]
    orb_res = np.full(n, np.nan); m = orb_ok
    orb_res[m] = np.linalg.norm(p[m] + np.einsum("nij,j->ni", R[m], rc) - orb_z[m], axis=1) * 1e3
    return dict(R=R, p=p, r_c=rc, scale_log=sol.x[iS:iR], bias=sol.x[iB:], sw=sw, wmap=wmap, orb_res_mm=orb_res, orb_used=keep,
                pair_res=[(q[0], q[1], float(np.linalg.norm(R[q[0]].T @ (p[q[1]] - p[q[0]]) - np.exp(sol.x[iS + sw[q[0]]]) * q[4]) / smt[q[2]])) for q in pairs],
                info=dict(cost=float(sol.cost), nfev=int(sol.nfev), status=int(sol.status), n_pairs=len(pairs), n_pairs_fast_dropped=int(n_fast),
                          n_pairs_gate_dropped=int(len(pair_drop or ())), n_grav=int(len(grav_idx)),
                          n_orb=int(len(orb_idx)), n_scale_windows=nW))
