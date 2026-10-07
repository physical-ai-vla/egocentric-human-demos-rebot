#!/usr/bin/env python3
"""[2026-09-23] C-P6: metric scale source for the (frozen) MASt3R-SLAM RGB-only trajectories.

GT = ArUco long-baseline scale (qa_all.json lb_scale, marker_flip_reject_v1) -- VALIDATION ONLY, never a production input.
Split: FIT = valB (9 demos, 18 sides) chooses every constant / hyperparameter; EVAL = valC (5 demos, 10 sides).

Methods
  global      one constant = median GT scale over FIT sides
  lr_const    per camera (left / right) constant = median GT scale over FIT sides of that camera
  imu_vi      per EPISODE from the wrist IMU, no double integration:
                R_wb f_b - a_lever = s * a_cam - g + R_wb b_a            (per camera frame, 3 rows)
              a_cam  = 2nd derivative of the MASt3R camera position (world units), Savitzky-Golay window W
              a_lever= 2nd derivative of R_wc t_cb (metric lever arm camera->IMU, from IMU.T_b_c1)
              f_b    = accelerometer specific force, box-averaged to camera times, rotated by R_wb = R_wc R_cb,
                       then the SAME SG filter (deriv 0) so both sides carry the same band limit
              unknowns s, g (3), b_a (3, optional); robust LS (Huber IRLS); |g| reported as a sanity check
              hyperparameters (W, bias on/off) chosen on FIT by median |log(s/s_gt)|
Downstream (per side; TCP = camera_tcp_v2 on the metric-scaled trajectory; rotation is the same in every method, so the
difference is scale-induced only):
  scale error        |s_est / s_gt - 1|
  UMI displacement   d_k = translation of inv(T_tcp(t)) @ T_tcp(t + k) on the 15 Hz grid, k = 1, 4, 8, 16, 32;
                     error |d_k(s_est) - d_k(s_gt)| in mm and relative to |d_k(s_gt)|, p50 / p90
"""
import csv, json, os, pathlib, re
import numpy as np, yaml
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
BASE = pathlib.Path(os.environ.get("M3_BASE", "/srv/data/johann/trackA_mast3r_pose_v1"))
VAR = os.environ.get("M3_VARIANT", "ss1")   # [2026-09-23] C8: ss1_5080 = the same benchmark re-run on the RTX 5080
QA = json.load(open(BASE / "qa/qa_all.json"))
CT = yaml.safe_load(open(BASE / "handumi_camera_tcp_v2.yaml"))
X = np.eye(4); X[:3, :3] = np.array(CT["rotation"]["R_camera_tcp"], float); X[:3, 3] = CT["translation"]["t_camera_tcp_m"]
KS = (1, 4, 8, 16, 32); G = 9.81


def load_traj(p, n):
    T = np.tile(np.eye(4), (n, 1, 1)); ok = np.zeros(n, bool)
    for r in csv.DictReader(open(p)):
        i = int(r["frame_idx"])
        if i < n and r["is_lost"] == "false":
            T[i, :3, :3] = Rot.from_quat([float(r[k]) for k in ("q_x", "q_y", "q_z", "q_w")]).as_matrix()
            T[i, :3, 3] = [float(r["x"]), float(r["y"]), float(r["z"])]; ok[i] = True
    return T, ok


def T_b_c(setting):
    txt = setting.read_text(); m = re.search(r"IMU\.T_b_c1:.*?data:\s*\[([^\]]+)\]", txt, re.S)
    return np.array([float(v) for v in m.group(1).split(",")]).reshape(4, 4)


def imu_vi(tag, W, use_bias):
    root = BASE / "val" / tag.split("_", 1)[0] / tag.split("_", 1)[1]
    d = json.load(open(root / "imu_data.json"))["1"]["streams"]
    t_cam = np.array([s["cts"] for s in d["CORI"]["samples"]]) / 1e3
    t_imu = np.array([s["cts"] for s in d["ACCL"]["samples"]]) / 1e3
    f = np.array([s["value"] for s in d["ACCL"]["samples"]], float)
    n = len(t_cam)
    T, ok = load_traj(BASE / f"runs/{tag}/{VAR}.csv", n)
    Tcb = np.linalg.inv(T_b_c(root / "orbslam_setting.yaml")); R_cb, t_cb = Tcb[:3, :3], Tcb[:3, 3]
    # accelerometer box-averaged over each camera frame period
    fb = np.full((n, 3), np.nan)
    for i, t in enumerate(t_cam):
        sel = np.abs(t_imu - t) <= 0.5 / 30
        if sel.any(): fb[i] = f[sel].mean(0)
    good = ok & np.isfinite(fb).all(1)
    idx = np.flatnonzero(good)
    if len(idx) < 3 * W: return None
    # use the longest contiguous valid run (MASt3R coverage is 1.0 on almost every side)
    br = np.flatnonzero(np.diff(idx) != 1); runs = np.split(idx, br + 1); r = max(runs, key=len)
    dt = float(np.median(np.diff(t_cam[r])))
    Rwc = T[r, :3, :3]; pc = T[r, :3, 3]; Rwb = Rwc @ R_cb
    a_cam = savgol_filter(pc, W, 3, deriv=2, delta=dt, axis=0)
    a_lev = savgol_filter(np.einsum("nij,j->ni", Rwc, t_cb), W, 3, deriv=2, delta=dt, axis=0)
    Fw = savgol_filter(np.einsum("nij,nj->ni", Rwb, fb[r]), W, 3, deriv=0, axis=0)
    h = W // 2; sl = slice(h, len(r) - h)                                     # drop SG edges
    y = (Fw - a_lev)[sl].reshape(-1)
    cols = [a_cam[sl].reshape(-1)]
    for k in range(3):
        e = np.zeros((len(r), 3)); e[:, k] = -1.0; cols.append(e[sl].reshape(-1))   # -g
    if use_bias:
        for k in range(3):
            cols.append(Rwb[sl][:, :, k].reshape(-1))                                # R_wb b_a
    A = np.stack(cols, 1); w = np.ones(len(y))
    for _ in range(10):
        xs = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        res = A @ xs - y; sc = 1.4826 * np.median(np.abs(res)) + 1e-9
        w = np.clip(1.345 * sc / np.maximum(np.abs(res), 1e-9), 0, 1)
    return dict(s=float(xs[0]), g_norm=float(np.linalg.norm(xs[1:4])), n=int(len(y) // 3),
                resid_rms=float(np.sqrt(np.mean(res ** 2))))


def umi_err(tag, s_est, s_gt):
    n = QA[tag]["n_src_frames"]; T, ok = load_traj(BASE / f"runs/{tag}/{VAR}.csv", n)
    out = {}
    for k in KS:
        e, rel = [], []
        for t in range(0, n - 2 * k, 2):
            if not (ok[t] and ok[t + 2 * k]): continue
            def dk(s):
                A_ = T[t].copy(); B_ = T[t + 2 * k].copy(); A_[:3, 3] *= s; B_[:3, 3] *= s
                return (np.linalg.inv(A_ @ X) @ (B_ @ X))[:3, 3]
            dg, de = dk(s_gt), dk(s_est)
            e.append(np.linalg.norm(de - dg) * 1e3)
            if np.linalg.norm(dg) > 0.005: rel.append(np.linalg.norm(de - dg) / np.linalg.norm(dg))
        out[k] = dict(mm_p50=float(np.median(e)), mm_p90=float(np.percentile(e, 90)),
                      rel_p50=float(np.median(rel)) if rel else np.nan, rel_p90=float(np.percentile(rel, 90)) if rel else np.nan)
    return out


tags = [t for t in QA if "lb_scale" in QA[t].get(f"M3_{VAR}", {}).get("marker", {})]
gt = {t: QA[t][f"M3_{VAR}"]["marker"]["lb_scale"] for t in tags}
fit = [t for t in tags if t.startswith("valB")]; ev = [t for t in tags if t.startswith("valC")]
side = lambda t: "left" if t.endswith("_left") else "right"

# ---- constants fitted on FIT only
c_global = float(np.median([gt[t] for t in fit]))
c_lr = {s: float(np.median([gt[t] for t in fit if side(t) == s])) for s in ("left", "right")}

# ---- IMU-VI hyperparameters chosen on FIT only
grid = [(W, b) for W in (9, 15, 21, 31, 45) for b in (False, True)]
cache = {}
for (W, b) in grid:
    for t in tags: cache[(t, W, b)] = imu_vi(t, W, b)
def fit_score(W, b):
    v = [abs(np.log(cache[(t, W, b)]["s"] / gt[t])) for t in fit if cache[(t, W, b)] and cache[(t, W, b)]["s"] > 0]
    return float(np.median(v)) if len(v) >= len(fit) // 2 else np.inf
scores = {(W, b): fit_score(W, b) for (W, b) in grid}
Wb = min(scores, key=scores.get)
print("IMU-VI hyperparameter search on FIT (median |log s/s_gt|):")
for (W, b), v in scores.items(): print(f"   W {W:>2} bias {'on ' if b else 'off'}  {v:.3f}{'   <- chosen' if (W, b) == Wb else ''}")

def est(method, t):
    if method == "global": return c_global
    if method == "lr_const": return c_lr[side(t)]
    r = cache[(t, *Wb)]; return r["s"] if r and r["s"] > 0 else np.nan

rows = []
for t in tags:
    for m in ("global", "lr_const", "imu_vi"):
        s = est(m, t)
        if not s == s: rows.append(dict(tag=t, split="fit" if t in fit else "eval", side=side(t), method=m, s=np.nan)); continue
        rows.append(dict(tag=t, split="fit" if t in fit else "eval", side=side(t), method=m, s=s, s_gt=gt[t],
                         scale_err=abs(s / gt[t] - 1), umi=umi_err(t, s, gt[t]),
                         g_norm=cache[(t, *Wb)]["g_norm"] if m == "imu_vi" and cache[(t, *Wb)] else None))

print(f"\nconstants (FIT): global {c_global:.3f}  left {c_lr['left']:.3f}  right {c_lr['right']:.3f}   IMU-VI W {Wb[0]} bias {Wb[1]}")
gn = [r["g_norm"] for r in rows if r["method"] == "imu_vi" and r.get("g_norm")]
print(f"IMU-VI |g| estimate: median {np.median(gn):.2f} (p10 {np.percentile(gn,10):.2f}, p90 {np.percentile(gn,90):.2f}) m/s2  -- 9.81 expected")
for split in ("eval", "fit"):
    print(f"\n=== {split.upper()} ({'valC, held out' if split == 'eval' else 'valB, used for fitting'}) ===")
    hdr = "  method    side   n  scale_err p50/p90 | " + " | ".join(f"k{k} mm p50/p90 (rel p90)" for k in KS)
    print(hdr)
    for m in ("global", "lr_const", "imu_vi"):
        for sd in ("left", "right", "all"):
            v = [r for r in rows if r["split"] == split and r["method"] == m and (sd == "all" or r["side"] == sd) and r.get("s") == r.get("s")]
            if not v: continue
            se = np.array([r["scale_err"] for r in v])
            cells = []
            for k in KS:
                a = np.array([r["umi"][k]["mm_p50"] for r in v]); b = np.array([r["umi"][k]["mm_p90"] for r in v])
                c = np.array([r["umi"][k]["rel_p90"] for r in v])
                cells.append(f"{np.median(a):5.1f}/{np.median(b):5.1f} ({np.nanmedian(c)*100:4.1f}%)")
            print(f"  {m:<9} {sd:<5} {len(v):>2}  {np.median(se)*100:5.1f}%/{np.percentile(se,90)*100:5.1f}%  | " + " | ".join(cells))
json.dump(dict(constants=dict(global_=c_global, **c_lr), imu_vi_choice=dict(W=Wb[0], bias=Wb[1]), fit_scores={f"{k[0]}_{k[1]}": v for k, v in scores.items()},
               rows=rows), open(BASE / ("qa/cp6_scale.json" if VAR == "ss1" else f"qa/cp6_scale_{VAR}.json"), "w"), indent=1, default=float)
