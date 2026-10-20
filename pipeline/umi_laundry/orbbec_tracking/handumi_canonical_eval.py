#!/usr/bin/env python3
"""[2026-10-12] Score laundry_v1 detector + canonical tracker against hand labels, and test canonical RIGIDITY.

Label file (json): {"episode": "<session>/<episode>", "labels": {"<orbbec frame>": [[u,v] | null, [u,v] | null]}}
(left first, right second; null = that unit not visible).  Matching tolerance: 60 px between the projected track
point and the labelled unit centre.

Per side and method (det = laundry_v1 trimmed-median centroid, canon = canonical point, final validity):
  TP: valid and within TOL of its own label; SWAP: valid and within TOL of the OTHER unit's label; WRONG: valid
  but on neither; FN: label visible but track invalid; FP_invis: valid while the unit is labelled not visible.
  precision = TP / (TP + SWAP + WRONG + FP_invis)   recall = TP / (#labels visible)
Track-level: coverage, simultaneous coverage, longest missing interval (s), >50 mm adjacent-frame jump rate,
image-order inversion events (sustained >= 10 frames).

Rigidity (canonical vs centroid):
  * MASt3R: on frames with MASt3R TRACKING, fit  p(t) = R_s (s c_cam(t) + R_cam(t) l) + t_s  (Sim3 + a constant
    lever l in the camera frame) to each Orbbec point on the SAME frames; a rigid point fits with a small residual,
    a sliding centroid cannot. Also plain Sim3 (no lever) for reference.
  * viewpoint slide: s = centroid - canonical, in the body frame, regressed on the camera viewing direction in the
    body frame (5-fold CV R^2): the share of the difference explained by viewpoint.
  * jitter: |p - SavGol(p)| on gap-free runs (high-frequency wobble), and corr(registration residual, attitude).

    .venv/bin/python scripts/handumi_canonical_eval.py --episode <ep> --labels <labels.json> [--json out.json]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handumi_collector.pose.rgbd_io import RgbdEpisode                              # noqa: E402
from handumi_collector.pose.episode_io import RawEpisode                            # noqa: E402
from scripts.handumi_canonical_track import interp_points, fib_sphere              # noqa: E402

SIDES = ("left", "right")
TOL_PX = 60.0
M3_RUNS = Path.home() / "c8" / "robotlike" / "runs"
M3_WIN_S = 5.0          # windowed Sim3 (MASt3R scale/drift absorbed per window); the lever stays ONE constant
SCALE_BOUND = 0.5       # per-window log-scale within +-0.5 of the global Umeyama scale (lever/scale degeneracy guard)
LEVER_BOUND_M = 0.25    # the wrist camera is within 25 cm of any point on its own unit


def uv(p, K):
    return np.stack([K[0, 0] * p[..., 0] / p[..., 2] + K[0, 2], K[1, 1] * p[..., 1] / p[..., 2] + K[1, 2]], -1)


def longest_gap(v, t_s):
    best, i, n = 0.0, 0, len(v)
    while i < n:
        if not v[i]:
            j = i
            while j < n and not v[j]:
                j += 1
            a = t_s[i - 1] if i > 0 else t_s[0]
            b = t_s[j] if j < n else t_s[-1]
            best = max(best, float(b - a)); i = j
        else:
            i += 1
    return best


def order_events(pl, pr, vl, vr, K):
    both = vl & vr
    if not both.any():
        return 0, 0
    ok = uv(pl[both], K)[:, 0] < uv(pr[both], K)[:, 0]
    ev = run = 0
    for o in ok:
        run = 0 if o else run + 1
        ev += run == 10
    return int(ev), int((~ok).sum())


def score_labels(labels, P, V, K):
    out = {}
    for k, s in enumerate(SIDES):
        c = dict(TP=0, SWAP=0, WRONG=0, FN=0, FP_invis=0, n_vis=0, n_invis=0)
        for fi, lab in labels.items():
            fi = int(fi)
            g, go = lab[k], lab[1 - k]
            val = bool(V[s][fi]) and np.all(np.isfinite(P[s][fi]))
            if val:
                u = uv(P[s][fi], K)
            if g is None:
                c["n_invis"] += 1
                c["FP_invis"] += int(val)
                continue
            c["n_vis"] += 1
            if not val:
                c["FN"] += 1
            elif np.hypot(*(u - g)) < TOL_PX:
                c["TP"] += 1
            elif go is not None and np.hypot(*(u - go)) < TOL_PX:
                c["SWAP"] += 1
            else:
                c["WRONG"] += 1
        npred = c["TP"] + c["SWAP"] + c["WRONG"] + c["FP_invis"]
        c["precision"] = c["TP"] / npred if npred else float("nan")
        c["recall"] = c["TP"] / c["n_vis"] if c["n_vis"] else float("nan")
        out[s] = c
    return out


def track_metrics(P, V, t_s, K):
    r = {}
    for s in SIDES:
        v = V[s]
        adj = v[1:] & v[:-1]
        st = np.linalg.norm(P[s][1:][adj] - P[s][:-1][adj], axis=1) * 1e3 if adj.any() else np.zeros(0)
        r[s] = dict(coverage=float(v.mean()), longest_gap_s=longest_gap(v, t_s),
                    jump50_adjacent=int((st > 50).sum()), jump50_rate=float((st > 50).mean()) if len(st) else float("nan"),
                    step_p95_mm=float(np.percentile(st, 95)) if len(st) else float("nan"))
    r["both_coverage"] = float((V["left"] & V["right"]).mean())
    r["order_inversion_events"], r["order_inverted_frames"] = order_events(P["left"], P["right"], V["left"], V["right"], K)
    return r


# ----------------------------------------------------------------------------------------- MASt3R rigidity
def load_ss1(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        if r["state"] != "TRACKING" or r["is_lost"] != "false":
            continue
        q = np.array([float(r[k]) for k in ("q_x", "q_y", "q_z", "q_w")])
        x = np.array([float(r[k]) for k in ("x", "y", "z")])
        if np.all(np.isfinite(q)) and np.linalg.norm(q) > 0.5 and np.all(np.isfinite(x)):
            out[int(r["frame_idx"])] = (x, Rotation.from_quat(q).as_matrix())
    return out


def _umeyama(c, p):
    mc, mp = c.mean(0), p.mean(0)
    C, Pp = c - mc, p - mp
    U, S, Vt = np.linalg.svd(Pp.T @ C / len(c))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R0 = U @ D @ Vt
    s0 = np.trace(np.diag(S) @ D) / max((C ** 2).sum() / len(c), 1e-12)
    return R0, s0, mp - s0 * R0 @ mc


def fit_sim3_lever(c, Rc, p, lever=True, win=None):
    """p_i ~ R_w (s_w c_i + Rc_i l) + t_w with ONE lever l (camera frame, metres) and a Sim3 per window w (win: window
    id per sample; None = one window). Robust (soft-L1, 10 mm). The lever fit starts from the no-lever solution, so
    it can never end worse than it in cost. Returns residual norms (m) and params."""
    win = np.zeros(len(c), int) if win is None else np.asarray(win)
    ids = np.unique(win)
    Rg_, sg, tg = _umeyama(c, p)                       # global solution: init + scale prior for every window
    lsg = np.log(max(sg, 1e-6))
    x0, lo, hi = [], [], []
    for w in ids:
        x0 += [*Rotation.from_matrix(Rg_).as_rotvec(), lsg, *tg]
        lo += [-7, -7, -7, lsg - SCALE_BOUND, -np.inf, -np.inf, -np.inf]
        hi += [7, 7, 7, lsg + SCALE_BOUND, np.inf, np.inf, np.inf]
    x0 = np.array(x0 + [0.0, 0.0, 0.0])
    lo = np.array(lo + [-LEVER_BOUND_M] * 3); hi = np.array(hi + [LEVER_BOUND_M] * 3)
    W = {w: k for k, w in enumerate(ids)}
    wi = np.array([W[w] for w in win])

    def f(x, use_lever):
        l = x[-3:] if use_lever else np.zeros(3)
        out = np.empty((len(c), 3))
        for k in range(len(ids)):
            m = wi == k
            q = x[7 * k: 7 * k + 7]
            R = Rotation.from_rotvec(q[:3]).as_matrix()
            out[m] = (np.exp(q[3]) * c[m] + Rc[m] @ l) @ R.T + q[4:7] - p[m]
        return out.ravel()
    sol0 = least_squares(lambda x: f(x, False), x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.01, max_nfev=400)
    sol = sol0
    if lever:
        sol = least_squares(lambda x: f(x, True), sol0.x, bounds=(lo, hi), loss="soft_l1", f_scale=0.01, max_nfev=800)
    r = np.linalg.norm(sol.fun.reshape(-1, 3), axis=1)
    return r, dict(n_windows=int(len(ids)), scale_p50=float(np.median(np.exp(sol.x[3:-3:7]))),
                   lever_mm=(1e3 * sol.x[-3:]).tolist() if lever else None)


def mast3r_rigidity(ep_path, raw, can, t_orb):
    tag = f"{ep_path.parent.name.split('_', 1)[1]}_{int(ep_path.name.split('_')[1]):06d}"
    res = {}
    for s in SIDES:
        f = M3_RUNS / f"{tag}_{s}" / "ss1.csv"
        if not f.exists():
            continue
        m3 = load_ss1(f)
        fm = raw.frames[f"{s}_wrist"]
        vf2t = dict(zip(fm.video_frame.tolist(), fm.capture_ns.tolist()))
        idx = sorted(i for i in m3 if i in vf2t)
        if len(idx) < 60:
            res[s] = dict(n=len(idx), note="too few MASt3R TRACKING frames"); continue
        # lag check: |omega| of MASt3R vs IMU gyro magnitude (rotation-invariant: no extrinsic needed)
        tq = np.array([vf2t[i] for i in idx], np.int64)
        Rm = np.stack([m3[i][1] for i in idx]); cm = np.stack([m3[i][0] for i in idx])
        consec = np.diff(idx) == 1
        wm = np.full(len(idx), np.nan)
        dts = np.diff(tq) / 1e9
        ang = Rotation.from_matrix(np.einsum("nji,njk->nik", Rm[:-1], Rm[1:])).magnitude() / np.maximum(dts, 1e-3)
        wm[:-1][consec] = ang[consec]
        im = raw.imu[s]
        gm = np.linalg.norm(im.gyro, axis=1)
        lags = range(-6, 7)
        cors = []
        for lag in lags:
            ti = tq + int(lag * 33.3e6)
            gi = np.interp(ti + 16.7e6, im.host_ns, gm)
            ok = np.isfinite(wm)
            cors.append(float(np.corrcoef(wm[ok], gi[ok])[0, 1]))
        lag_best = list(lags)[int(np.argmax(cors))]
        entry = dict(n_m3_frames=len(idx), m3_span_s=float((tq[-1] - tq[0]) / 1e9),
                     gyro_lag_frames=lag_best, gyro_corr=float(max(cors)), gyro_corr_lag0=cors[list(lags).index(0)])
        tqc = tq + int(lag_best * 33.3e6)          # MASt3R frame time on the IMU/host clock (measured lag)
        rig = can.get(f"rigid_ok_{s}", can[f"valid_{s}"])
        for subset, vmask in (("all_valid", can[f"valid_{s}"]), ("rigid_ok", rig)):
            pts = {"canonical": interp_points(t_orb, can[f"p_W_{s}"], vmask, tqc, 40e6),
                   "centroid": interp_points(t_orb, can[f"centroid_W_{s}"], vmask & np.isfinite(can[f"centroid_W_{s}"]).all(1), tqc, 40e6)}
            common = np.isfinite(pts["canonical"]).all(1) & np.isfinite(pts["centroid"]).all(1)
            e2 = dict(n_common=int(common.sum()))
            if common.sum() >= 2:
                pc_ = pts["canonical"][common]
                e2["motion"] = dict(path_mm=float(np.linalg.norm(np.diff(pc_, axis=0), axis=1).sum() * 1e3),
                                    extent_mm=float(np.linalg.norm(pc_.max(0) - pc_.min(0)) * 1e3),
                                    cam_rot_range_deg=float(np.degrees(np.max(Rotation.from_matrix(
                                        np.einsum("ji,njk->nik", Rm[common][0], Rm[common])).magnitude()))))
            if common.sum() < 40:
                e2["note"] = "too few common frames"; entry[subset] = e2; continue
            tw = (tqc[common] - tqc[common][0]) / 1e9
            win = np.floor(tw / M3_WIN_S).astype(int)
            # merge a short last window into the previous one
            for w in np.unique(win)[::-1]:
                if (win == w).sum() < 30 and w > 0:
                    win[win == w] = w - 1
            for name, pp in pts.items():
                for lev in (True, False):
                    for wmode, wv in (("global", None), (f"win{M3_WIN_S:.0f}s", win)):
                        r, info = fit_sim3_lever(cm[common], Rm[common], pp[common], lever=lev, win=wv)
                        e2[f"{name}_{'sim3lever' if lev else 'sim3'}_{wmode}"] = dict(
                            p50_mm=float(np.median(r) * 1e3), p90_mm=float(np.percentile(r, 90) * 1e3), **info)
            entry[subset] = e2
        res[s] = entry
    return res


# ----------------------------------------------------------------------------------------- viewpoint slide
def viewpoint_slide(can, valid_det):
    out = {}
    cam = can["t_WC"]
    rng = np.random.default_rng(0)
    for s in SIDES:
        if f"R_world_body_{s}" not in can:
            continue
        v = can.get(f"rigid_ok_{s}", can[f"valid_{s}"]) & valid_det[s]
        if v.sum() < 100:
            continue
        pc, pz = can[f"p_W_{s}"][v], can[f"centroid_W_{s}"][v]
        Rb = can[f"R_world_body_{s}"][v]
        sl = np.einsum("nji,nj->ni", Rb, pz - pc)                        # slide in the body frame
        vb = np.einsum("nji,nj->ni", Rb, cam - pc); vb /= np.linalg.norm(vb, axis=1, keepdims=True)
        X = np.c_[vb, np.ones(len(vb))]
        fold = rng.integers(0, 5, len(vb))
        pred = np.zeros_like(sl)
        for f in range(5):
            tr, te = fold != f, fold == f
            B, *_ = np.linalg.lstsq(X[tr], sl[tr], rcond=None)
            pred[te] = X[te] @ B
        ss_res = ((sl - pred) ** 2).sum(); ss_tot = ((sl - sl.mean(0)) ** 2).sum()
        # residual-vs-attitude correlation (memory note): angle from the most populated attitude
        bins = fib_sphere(42)
        b = np.argmax(vb @ bins.T, 1)
        dom = bins[np.bincount(b).argmax()]
        att = np.degrees(np.arccos(np.clip(vb @ dom, -1, 1)))
        res = can[f"resid_mm_{s}"][v]
        # jitter on gap-free runs: deviation from a 9-frame quadratic SavGol fit
        jit = {}
        for name, key in (("canonical", f"p_W_{s}"), ("centroid", f"centroid_W_{s}")):
            P_ = can[key]; vv = can.get(f"rigid_ok_{s}", can[f"valid_{s}"]) & valid_det[s]
            d = []
            i = 0; n = len(vv)
            while i < n:
                if vv[i]:
                    j = i
                    while j < n and vv[j]:
                        j += 1
                    if j - i >= 15:
                        seg = P_[i:j]
                        d.append(np.linalg.norm(seg - savgol_filter(seg, 9, 2, axis=0), axis=1))
                    i = j
                else:
                    i += 1
            d = np.concatenate(d) * 1e3 if d else np.zeros(1)
            jit[name] = dict(p50_mm=float(np.median(d)), p95_mm=float(np.percentile(d, 95)))
        out[s] = dict(n=int(v.sum()), slide_norm_mm_p50=float(np.median(np.linalg.norm(sl, axis=1)) * 1e3),
                      slide_norm_mm_p90=float(np.percentile(np.linalg.norm(sl, axis=1), 90) * 1e3),
                      slide_cv_r2_on_viewdir=float(1 - ss_res / ss_tot),
                      corr_resid_attitude=float(np.corrcoef(res, att)[0, 1]),
                      corr_slide_attitude=float(np.corrcoef(np.linalg.norm(sl - np.median(sl, 0), axis=1), att)[0, 1]),
                      jitter=jit)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--set", default="", help="label for the report (tuned / held-out)")
    a = ap.parse_args()
    bdir = a.episode / "derived" / "body_track"
    det = np.load(bdir / "track.npz")
    can = dict(np.load(bdir / "canonical.npz"))
    lab = json.loads(a.labels.read_text())["labels"]
    K = RgbdEpisode.load(a.episode).intrinsics.K
    t_ns = det["t_ns"]; t_s = (t_ns - t_ns[0]) / 1e9
    Pd = {s: det[f"xyz_{s}"] for s in SIDES}; Vd = {s: det[f"valid_{s}"] for s in SIDES}
    Pc = {s: can[f"p_C_{s}"] for s in SIDES}; Vc = {s: can[f"valid_{s}"] for s in SIDES}
    R = dict(episode=f"{a.episode.parent.name}/{a.episode.name}", set=a.set, n_labels=len(lab), tol_px=TOL_PX,
             duration_s=float(t_s[-1]))
    R["det"] = dict(labels=score_labels(lab, Pd, Vd, K), track=track_metrics(Pd, Vd, t_s, K))
    R["canon"] = dict(labels=score_labels(lab, Pc, Vc, K), track=track_metrics(Pc, Vc, t_s, K))
    # canonical before the temporal pass (fitness only) -- shows what the gate removed
    Vp = {s: can[f"valid_prepass_{s}"] for s in SIDES}
    R["canon_prepass"] = dict(labels=score_labels(lab, Pc, Vp, K), track=track_metrics(Pc, Vp, t_s, K))
    raw = RawEpisode.load(a.episode)
    R["rigidity_mast3r"] = mast3r_rigidity(a.episode, raw, can, t_ns)
    R["rigidity_viewpoint"] = viewpoint_slide(can, Vd)
    for m in ("det", "canon_prepass", "canon"):
        L, T = R[m]["labels"], R[m]["track"]
        print(f"{R['episode']:>40s} [{a.set}] {m:>13s}: " + "  ".join(
            f"{s[0].upper()} P{L[s]['precision']:.2f} R{L[s]['recall']:.2f} sw{L[s]['SWAP']} cov{T[s]['coverage']*100:.0f}% "
            f"gap{T[s]['longest_gap_s']:.1f}s j50 {T[s]['jump50_rate']*100:.2f}%" for s in SIDES)
            + f"  both {T['both_coverage']*100:.0f}%  inv {T['order_inversion_events']}")
    for s, e in R["rigidity_mast3r"].items():
        for subset in ("all_valid", "rigid_ok"):
            e2 = e.get(subset)
            if not e2 or "note" in e2:
                print(f"   MASt3R {s} {subset}: {e2 or e}"); continue
            mo = e2["motion"]
            line = (f"   MASt3R {s} {subset:9s} n={e2['n_common']:4d} lag {e['gyro_lag_frames']:+d} (r {e['gyro_corr']:.2f}) "
                    f"[extent {mo['extent_mm']:.0f} mm, cam rot {mo['cam_rot_range_deg']:.0f} deg] ")
            for wm in ("global", f"win{M3_WIN_S:.0f}s"):
                a_, b_ = e2[f"canonical_sim3lever_{wm}"], e2[f"centroid_sim3lever_{wm}"]
                line += (f"| {wm} sim3+lever p50/p90 canon {a_['p50_mm']:.1f}/{a_['p90_mm']:.1f} vs centroid "
                         f"{b_['p50_mm']:.1f}/{b_['p90_mm']:.1f} (no-lever {e2[f'canonical_sim3_{wm}']['p50_mm']:.1f} vs "
                         f"{e2[f'centroid_sim3_{wm}']['p50_mm']:.1f}) ")
            print(line)
    for s, e in R["rigidity_viewpoint"].items():
        print(f"   view {s}: slide p50 {e['slide_norm_mm_p50']:.1f} mm, CV R2(slide~viewdir) {e['slide_cv_r2_on_viewdir']:.2f}, "
              f"corr(resid,att) {e['corr_resid_attitude']:+.2f}, jitter p95 canon {e['jitter']['canonical']['p95_mm']:.2f} "
              f"vs centroid {e['jitter']['centroid']['p95_mm']:.2f} mm")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(R, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
