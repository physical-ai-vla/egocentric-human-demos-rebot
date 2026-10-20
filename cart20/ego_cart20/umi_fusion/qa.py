"""[2026-10-12] Frame-level fusion QA (per side, on the wrist-frame timeline).

coverage(): mast3r_valid / orbbec_valid / both_valid / either_valid / full_6dof_valid fractions + LONGEST gap (s) of each --
coverage is computed per frame (overlap/union), never by adding averages. A 90 % coverage with one 8 s hole is reported as such.
overlap_error(): on frames where BOTH sources give a TCP position, e_p = |p_mast3r - p_orbbec| p50/p90/p95 + its trend over time
(drift) and its correlation with attitude (canonical-point / TCP-offset error), the two failure signatures to tell apart.
episode_gate(): PASS / WARN / FAIL from full_6dof coverage, longest full-6DoF gap and e_p (thresholds are arguments).
"""
import numpy as np


def longest_run_s(mask, t):
    """longest run of False (a gap) in seconds."""
    mask = np.asarray(mask, bool); t = np.asarray(t, float)
    if mask.all() or len(mask) == 0: return 0.0
    d = np.diff(np.r_[1, mask.astype(int), 1]); st = np.flatnonzero(d == -1); en = np.flatnonzero(d == 1)
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.0
    return float(max(t[min(e, len(t)) - 1] - t[s] + dt for s, e in zip(st, en)))


def coverage(t, mast3r_valid, orbbec_valid, full_6dof_valid):
    m = np.asarray(mast3r_valid, bool); o = np.asarray(orbbec_valid, bool); f = np.asarray(full_6dof_valid, bool)
    out = {}
    for k, v in (("mast3r_valid", m), ("orbbec_valid", o), ("both_valid", m & o), ("either_valid", m | o), ("full_6dof_valid", f)):
        out[k] = dict(frac=round(float(v.mean()), 4), longest_gap_s=round(longest_run_s(v, t), 2))
    return out


def overlap_error(t, p_mast3r, p_orbbec, both_valid, R_W=None):
    b = np.asarray(both_valid, bool)
    if b.sum() < 10: return dict(n=int(b.sum()))
    e = np.linalg.norm(np.asarray(p_mast3r)[b] - np.asarray(p_orbbec)[b], axis=1) * 1e3; tb = np.asarray(t)[b]
    r = dict(n=int(b.sum()), p50_mm=round(float(np.percentile(e, 50)), 1), p90_mm=round(float(np.percentile(e, 90)), 1),
             p95_mm=round(float(np.percentile(e, 95)), 1),
             drift_mm_per_s=round(float(np.polyfit(tb - tb[0], e, 1)[0]), 2) if np.ptp(tb) > 1 else None)
    if R_W is not None:                                   # attitude dependence: |corr| of e with each column of the approach axis
        z = np.asarray(R_W)[b][:, :, 2]
        r["attitude_corr_max"] = round(float(max(abs(np.corrcoef(e, z[:, k])[0, 1]) for k in range(3) if np.std(z[:, k]) > 1e-6)), 3) \
            if any(np.std(z[:, k]) > 1e-6 for k in range(3)) else None
    return r


def episode_gate(cov, err=None, pass_cov=0.9, warn_cov=0.8, max_gap_s=3.0, ep_pass_mm=10.0, ep_warn_mm=20.0):
    f = cov["full_6dof_valid"]; reasons = []; lvl = 0
    if f["frac"] < warn_cov: lvl = 2; reasons.append(f"full_6dof {f['frac']:.2f} < {warn_cov}")
    elif f["frac"] < pass_cov: lvl = max(lvl, 1); reasons.append(f"full_6dof {f['frac']:.2f} < {pass_cov}")
    if f["longest_gap_s"] > max_gap_s: lvl = max(lvl, 1); reasons.append(f"longest full-6DoF gap {f['longest_gap_s']} s > {max_gap_s}")
    if err and err.get("p90_mm") is not None:
        if err["p90_mm"] > ep_warn_mm: lvl = 2; reasons.append(f"e_p p90 {err['p90_mm']} mm > {ep_warn_mm}")
        elif err["p90_mm"] > ep_pass_mm: lvl = max(lvl, 1); reasons.append(f"e_p p90 {err['p90_mm']} mm > {ep_pass_mm}")
    return dict(status=["PASS", "WARN", "FAIL"][lvl], reasons=reasons)
