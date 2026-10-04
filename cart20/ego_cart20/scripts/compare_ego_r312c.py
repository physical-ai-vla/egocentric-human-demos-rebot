#!/usr/bin/env python3
"""READ-ONLY comparison of the ego CART20 pretrain set and the R312c RELCART20 fine-tune set, in the exact space X-VLA sees.
Both are the LeRobot datasets the trainer reads (parquet state / action / task; videos for the camera section).

Rules (user guide 2026-10-01): horizon-specific (k1/4/8/16 = 50/200/400/800 ms), raw physical units for the primary numbers,
ONE common scaler fitted on the union for every multivariate analysis (never per-dataset normalization), equal-size
deterministic subsamples (seed 1000) for joint analyses, nearest-neighbour coverage in BOTH directions with in-domain
baselines (robot->robot, ego->ego split halves) so a distance has a reference scale, nothing is modified or remapped.

usage: compare_ego_r312c.py --ego <ego_cart20_v2_train> --robot <r312c_relcart20_rel16_v4> --output <dir> [--seed 1000]
       [--n-joint 20000] [--n-images 5000]   (env ~/xvla-mac/bin/python)"""
import argparse
import collections
import csv
import glob
import json
import pathlib
import re
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ego_cart20.geometry.rotation6d import rotation_6d_to_matrix, rotation_angle  # noqa: E402

KS = (1, 4, 8, 16); ARMS = ("L", "R")
ST_POS = {"L": slice(0, 3), "R": slice(9, 12)}; ST_ROT = {"L": slice(3, 9), "R": slice(12, 18)}; ST_G = {"L": 18, "R": 19}
AC_POS = {"L": slice(0, 3), "R": slice(10, 13)}; AC_ROT = {"L": slice(3, 9), "R": slice(13, 19)}; AC_G = {"L": 9, "R": 19}
PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)
COLOR = {"ego": "#d95f02", "robot": "#1b9e77"}


# ----------------------------------------------------------------------------------------------------------------- loading
def _flat(col, shape):
    a = col.combine_chunks()
    if isinstance(a, pa.ExtensionArray): a = a.storage
    while pa.types.is_list(a.type) or pa.types.is_fixed_size_list(a.type) or pa.types.is_large_list(a.type): a = a.flatten()
    return a.to_numpy(zero_copy_only=False).reshape((-1,) + shape)


def order_of(instr):
    """stack order from the instruction text (bottom -> top), e.g. 'Stack the red cube on the bottom, blue ... purple ...' -> RBP"""
    cols = re.findall(r"\b(red|blue|purple)\b", instr.lower()); return "".join(c[0].upper() for c in cols[:3]) if len(cols) >= 3 else "?"


def load(root):
    root = pathlib.Path(root); S, A, E, TI, Q = [], [], [], [], []
    for f in sorted(glob.glob(str(root / "data/*/*.parquet"))):
        names = pq.read_schema(f).names
        t = pq.read_table(f, columns=[c for c in ("observation.state", "action", "episode_index", "task_index", "aux.q_t") if c in names])
        S.append(_flat(t.column("observation.state"), (20,))); A.append(_flat(t.column("action"), (16, 32)))
        E.append(t.column("episode_index").to_numpy()); TI.append(t.column("task_index").to_numpy())
        if "aux.q_t" in names: Q.append(_flat(t.column("aux.q_t"), (12,)))
    import pandas as pd
    tasks = pd.read_parquet(root / "meta/tasks.parquet"); tstr = {int(v): k for k, v in tasks["task_index"].items()}
    d = dict(state=np.concatenate(S).astype(np.float64), action=np.concatenate(A).astype(np.float64), ep=np.concatenate(E), task=np.concatenate(TI),
             q=np.concatenate(Q).astype(np.float64) if Q else None, tasks=tstr, info=json.load(open(root / "meta/info.json")), root=str(root))
    d["order"] = np.array([order_of(tstr[int(i)]) for i in d["task"]])
    return d


# ------------------------------------------------------------------------------------------------------------- primitives
def pct(x, ps=PCTS):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    out = {f"p{p:02d}": float(np.percentile(x, p)) for p in ps}
    out.update(mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max()), n=int(len(x))); return out


def rot_deg(d6):
    return np.degrees(rotation_angle(rotation_6d_to_matrix(d6)))


def rot_axis(d6):
    R = rotation_6d_to_matrix(d6); v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    n = np.linalg.norm(v, axis=-1, keepdims=True); return np.where(n > 1e-9, v / np.maximum(n, 1e-12), np.nan)


def sub(n_total, n, rng):
    return np.sort(rng.choice(n_total, min(n, n_total), replace=False))


class Scaler:
    """ONE scaler fitted on concat(ego, robot) -- never per dataset"""
    def __init__(self, *Xs):
        X = np.concatenate(Xs); self.m = X.mean(0); self.s = X.std(0); self.s[self.s < 1e-12] = 1.0
    def __call__(self, X): return (X - self.m) / self.s


def pca(X, k=3):
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k], (S ** 2 / (S ** 2).sum())[:k]


def nn_dist(query, ref):
    return cKDTree(ref).query(query, k=1, workers=-1)[0]


def mmd_rbf(X, Y, rng, n=2000):
    X = X[sub(len(X), n, rng)]; Y = Y[sub(len(Y), n, rng)]; Z = np.concatenate([X, Y])
    d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1); sig2 = np.median(d2[d2 > 0]); K = np.exp(-d2 / sig2); n1 = len(X)
    return float(K[:n1, :n1].mean() + K[n1:, n1:].mean() - 2 * K[:n1, n1:].mean()), float(sig2)


# ------------------------------------------------------------------------------------------------------------------- main
def main(a):
    t_start = time.time(); OUT = pathlib.Path(a.output).expanduser(); (OUT / "plots").mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(a.seed)
    D = {"ego": load(a.ego), "robot": load(a.robot)}; R = collections.OrderedDict(); md = []
    names = {"ego": "Ego (ego_cart20_v2_train)", "robot": "R312c (r312c_relcart20_rel16_v4)"}

    # ---------------------------------------------------------------- 1 contract parity (abort on a semantic mismatch)
    par = []
    for k_, f in (("state dim", lambda d: d["state"].shape[1]), ("action shape", lambda d: list(d["action"].shape[1:])),
                  ("fps (rows)", lambda d: d["info"]["fps"]), ("camera keys", lambda d: sorted(k for k in d["info"]["features"] if "images" in k))):
        par.append((k_, f(D["ego"]), f(D["robot"])))
    aux = {n: float(np.abs(d["action"][..., 20:]).max()) for n, d in D.items()}
    anchor = {}
    for n, d in D.items():
        first = np.r_[True, np.diff(d["ep"]) != 0]
        anchor[n] = float(np.abs(d["state"][first][:, :18] - np.array([0, 0, 0, 1, 0, 0, 0, 1, 0] * 2)).max())
    rawrot = {}
    for n, d in D.items():
        x = np.concatenate([d["state"][:, 3:9], d["state"][:, 12:18], d["action"][:, :, 3:9].reshape(-1, 6), d["action"][:, :, 13:19].reshape(-1, 6)])
        a_, b_ = x[:, :3], x[:, 3:]
        rawrot[n] = float(np.max([np.abs(np.linalg.norm(a_, axis=1) - 1).max(), np.abs(np.linalg.norm(b_, axis=1) - 1).max(), np.abs((a_ * b_).sum(1)).max()]))
    grng = {n: [float(min(d["state"][:, 18:].min(), d["action"][..., [9, 19]].min())), float(max(d["state"][:, 18:].max(), d["action"][..., [9, 19]].max()))] for n, d in D.items()}
    ego_first = []
    for e in np.unique(D["ego"]["ep"]):
        i0 = np.flatnonzero(D["ego"]["ep"] == e)[0]; s0 = D["ego"]["state"][i0]
        ego_first.append(max(np.linalg.norm(s0[0:3]), np.linalg.norm(s0[9:12])))
    ego_first = np.array(ego_first)
    contract_first = dict(episodes=int(len(ego_first)), first_frame_identity=int((ego_first < 1e-6).sum()), first_frame_pos_offset_cm=pct(100 * ego_first, (50, 90, 99)),
                          note="ego anchor = task start (row 0 of the 15 Hz grid); LeRobot frames = TRAINING rows only, so an episode whose first rows "
                               "lack 16 valid targets / a head frame starts later than its anchor (report-only, not a contract violation)")
    par += [("action[..., 20:32] max |x| in the DATASET", aux["ego"], aux["robot"]),
            ("state at episode row 0 == identity (max err)", anchor["ego"], anchor["robot"]),
            ("stored rot6d rows orthonormal (max err)", rawrot["ego"], rawrot["robot"]),
            ("gripper value range (state + action)", grng["ego"], grng["robot"])]
    contract = dict(
        rows=[dict(item=i, ego=e, robot=r) for i, e, r in par],
        documented=dict(
            UMI_DT_ms=dict(ego="50.05 (ego_cart20.config.UMI_DT, raw-track interpolation)", robot="50.05 (umi76_to_lerobot UMI_DT, UMI76_CONTRACT 2b)", match=True),
            rot6d=dict(ego="first two ROWS", robot="first two ROWS (umi pose_util.mat_to_rot6d)", match=True),
            state=dict(ego="RELCART20 inv(T_task_start) T(t), [L9|R9|gL gR]", robot="RELCART20 inv(T_ep_row0) T(t), [L9|R9|gL gR]", match=True),
            arm_order=dict(ego="LEFT first", robot="LEFT first (robot0 = LEFT)", match=True),
            gripper_polarity=dict(ego="0 closed / 1 open, caliper mm / 80 (state and action SAME signal)",
                                  robot="0 closed / 1 open; state = FOLLOWER width / 0.11441 m, action = LEADER cmd / 45 (different signals)", match=True),
            aux12=dict(ego="0 in the dataset", robot="dq12 in the dataset; the REL-only trainer hard-zeroes 20:32 at model input AND output, so the model sees 0", match=True)))
    sem_ok = (D["ego"]["state"].shape[1] == D["robot"]["state"].shape[1] == 20 and D["ego"]["action"].shape[1:] == D["robot"]["action"].shape[1:] == (16, 32)
              and aux["ego"] == 0 and anchor["robot"] < 1e-5 and rawrot["ego"] < 1e-4 and rawrot["robot"] < 1e-4
              and grng["ego"][0] >= 0 and grng["ego"][1] <= 1 and grng["robot"][0] >= 0 and grng["robot"][1] <= 1)
    contract["semantic_parity_pass"] = bool(sem_ok); contract["ego_first_exported_frame_vs_anchor"] = contract_first; R["contract_parity"] = contract
    if not sem_ok: json.dump(R, open(OUT / "summary.json", "w"), indent=1); raise SystemExit("contract parity FAILED -- comparison aborted (summary.json has the table)")

    # ---------------------------------------------------------------- 2 size
    size = {}
    for n, d in D.items():
        eps, cnt = np.unique(d["ep"], return_counts=True)
        size[n] = dict(episodes=int(len(eps)), training_rows=int(len(d["state"])), action_targets=int(len(d["state"]) * 16),
                       row_time_min=round(len(d["state"]) / 15 / 60, 1), rows_per_episode=pct(cnt, (5, 50, 95)))
    R["size"] = size

    # convenient per-arm arrays
    F = {}
    for n, d in D.items():
        S, A = d["state"], d["action"]; f = {}
        for arm in ARMS:
            f[f"{arm}_sxyz"] = S[:, ST_POS[arm]]; f[f"{arm}_srot"] = rot_deg(S[:, ST_ROT[arm]]); f[f"{arm}_g"] = S[:, ST_G[arm]]
            for k in KS:
                f[f"{arm}_axyz_k{k}"] = A[:, k - 1, AC_POS[arm]]; f[f"{arm}_arot_k{k}"] = rot_deg(A[:, k - 1, AC_ROT[arm]]); f[f"{arm}_ag_k{k}"] = A[:, k - 1, AC_G[arm]]
        F[n] = f

    # ---------------------------------------------------------------- 3 state distribution
    st_rows = []; st = {}
    for arm in ARMS:
        for ax, axn in enumerate("xyz"):
            for n in D:
                p = pct(F[n][f"{arm}_sxyz"][:, ax]); st_rows.append(dict(dataset=n, arm=arm, feature=f"state_{axn}_m", **p))
        for n in D:
            st_rows.append(dict(dataset=n, arm=arm, feature="state_trans_norm_m", **pct(np.linalg.norm(F[n][f"{arm}_sxyz"], axis=1))))
            st_rows.append(dict(dataset=n, arm=arm, feature="state_rot_deg", **pct(F[n][f"{arm}_srot"])))
        st[arm] = dict(axis_hist_cos_to_z={n: pct(np.abs(rot_axis(D[n]["state"][:, ST_ROT[arm]])[:, 2]), (10, 50, 90)) for n in D})
    write_csv(OUT / "state_distribution.csv", st_rows); R["state_rotation_axis_abs_z_component"] = st

    # ---------------------------------------------------------------- 4/5/6 action per k
    ac_rows = []; hz_rows = []
    for arm in ARMS:
        for k in KS:
            for n in D:
                xyz = F[n][f"{arm}_axyz_k{k}"]; tn = np.linalg.norm(xyz, axis=1)
                for ax, axn in enumerate("xyz"): ac_rows.append(dict(dataset=n, arm=arm, k=k, feature=f"d{axn}_m", **pct(xyz[:, ax])))
                ac_rows.append(dict(dataset=n, arm=arm, k=k, feature="trans_norm_m", **pct(tn)))
                ac_rows.append(dict(dataset=n, arm=arm, k=k, feature="rot_deg", **pct(F[n][f"{arm}_arot_k{k}"])))
                ac_rows.append(dict(dataset=n, arm=arm, k=k, feature="gripper_target", **pct(F[n][f"{arm}_ag_k{k}"])))
                hz_rows.append(dict(dataset=n, arm=arm, k=k, ms=round(k * 50.05, 1), trans_p50_mm=1e3 * np.median(tn), trans_p95_mm=1e3 * np.percentile(tn, 95),
                                    trans_p99_mm=1e3 * np.percentile(tn, 99), rot_p50_deg=np.median(F[n][f"{arm}_arot_k{k}"]), rot_p95_deg=np.percentile(F[n][f"{arm}_arot_k{k}"], 95),
                                    rot_p99_deg=np.percentile(F[n][f"{arm}_arot_k{k}"], 99), mean_dx_mm=1e3 * xyz[:, 0].mean(), mean_dy_mm=1e3 * xyz[:, 1].mean(), mean_dz_mm=1e3 * xyz[:, 2].mean()))
    write_csv(OUT / "action_distribution.csv", ac_rows); write_csv(OUT / "horizon_distribution.csv", hz_rows)
    # all-horizon (secondary) aggregate
    R["action_all_horizons_secondary"] = {n: {arm: dict(trans_norm_m=pct(np.linalg.norm(D[n]["action"][:, :, AC_POS[arm]], axis=-1).ravel(), (50, 95, 99)))
                                              for arm in ARMS} for n in D}

    # ---------------------------------------------------------------- 7 gripper
    g_rows = []; gsum = {}
    edges = [(-1, 0.1, "g<0.1"), (-1, 0.25, "g<0.25"), (0.25, 0.5, "0.25<=g<0.5"), (0.5, 0.75, "0.5<=g<0.75"), (0.75, 2, "g>=0.75")]
    for arm in ARMS:
        for n in D:
            g = F[n][f"{arm}_g"]; row = dict(dataset=n, arm=arm, signal="state", **pct(g))
            for lo, hi, nm in edges: row[f"frac_{nm}"] = float(((g >= lo) & (g < hi)).mean()) if lo >= 0 else float((g < hi).mean())
            g_rows.append(row)
            for k in KS:
                ga = F[n][f"{arm}_ag_k{k}"]; dg = ga - g
                r2 = dict(dataset=n, arm=arm, signal=f"action_k{k}", **pct(ga))
                for lo, hi, nm in edges: r2[f"frac_{nm}"] = float(((ga >= lo) & (ga < hi)).mean()) if lo >= 0 else float((ga < hi).mean())
                r2.update(dg_mean=float(dg.mean()), dg_abs_p50=float(np.median(np.abs(dg))), dg_abs_p95=float(np.percentile(np.abs(dg), 95)),
                          frac_abs_dg_gt_0p1=float((np.abs(dg) > 0.1).mean()), frac_abs_dg_gt_0p3=float((np.abs(dg) > 0.3).mean()),
                          corr_dg_vs_trans=float(np.corrcoef(np.abs(dg), np.linalg.norm(F[n][f"{arm}_axyz_k{k}"], axis=1))[0, 1]))
                ga1 = F[n][f"{arm}_ag_k1"]; dga = ga - ga1                       # SAME-signal change inside the action chunk
                r2.update(dg_action_internal_abs_p95=float(np.percentile(np.abs(dga), 95)), frac_abs_dg_action_internal_gt_0p1=float((np.abs(dga) > 0.1).mean()))
                g_rows.append(r2)
            # transitions per episode on the state signal (crossing 0.5 with 0.1 hysteresis)
            tr = []
            for e in np.unique(D[n]["ep"]):
                x = g[D[n]["ep"] == e]; s = x[0] > 0.5; c = 0
                for v in x:
                    if s and v < 0.4: s = False; c += 1
                    elif not s and v > 0.6: s = True; c += 1
                tr.append(c)
            gsum[f"{n}_{arm}_state_transitions_per_episode_hyst_0.4_0.6"] = pct(tr, (25, 50, 75))
    write_csv(OUT / "gripper_distribution.csv", g_rows); R["gripper_transitions"] = gsum
    # within-robot state vs action gripper (different signals by contract)
    R["robot_gripper_state_vs_action_k1"] = {arm: dict(mean_abs_diff=float(np.abs(F["robot"][f"{arm}_ag_k1"] - F["robot"][f"{arm}_g"]).mean()),
                                                       corr=float(np.corrcoef(F["robot"][f"{arm}_ag_k1"], F["robot"][f"{arm}_g"])[0, 1])) for arm in ARMS}
    R["ego_gripper_state_vs_action_k1"] = {arm: dict(mean_abs_diff=float(np.abs(F["ego"][f"{arm}_ag_k1"] - F["ego"][f"{arm}_g"]).mean()),
                                                     corr=float(np.corrcoef(F["ego"][f"{arm}_ag_k1"], F["ego"][f"{arm}_g"])[0, 1])) for arm in ARMS}

    # ---------------------------------------------------------------- 8 L/R asymmetry, 9 direction
    asym = {}; dirn = {}
    for n in D:
        for k in KS:
            L, Rr = (np.linalg.norm(F[n][f"{a_}_axyz_k{k}"], axis=1) for a_ in ARMS)
            asym[f"{n}_k{k}"] = dict(trans_p50_mm_L=1e3 * float(np.median(L)), trans_p50_mm_R=1e3 * float(np.median(Rr)), trans_ratio_R_over_L_p50=float(np.median(Rr) / max(np.median(L), 1e-12)),
                                     trans_p95_mm_L=1e3 * float(np.percentile(L, 95)), trans_p95_mm_R=1e3 * float(np.percentile(Rr, 95)),
                                     rot_p95_deg_L=float(np.percentile(F[n][f"L_arot_k{k}"], 95)), rot_p95_deg_R=float(np.percentile(F[n][f"R_arot_k{k}"], 95)),
                                     g_mean_L=float(F[n]["L_g"].mean()), g_mean_R=float(F[n]["R_g"].mean()),
                                     mean_dxyz_mm_L=(1e3 * F[n][f"L_axyz_k{k}"].mean(0)).round(2).tolist(), mean_dxyz_mm_R=(1e3 * F[n][f"R_axyz_k{k}"].mean(0)).round(2).tolist())
    R["left_right_asymmetry"] = asym
    for arm in ARMS:
        for k in (4, 8, 16):
            H = {}
            for n in D:
                v = F[n][f"{arm}_axyz_k{k}"]; m = np.linalg.norm(v, axis=1) > 0.005; u = v[m] / np.linalg.norm(v[m], axis=1, keepdims=True)
                az = np.arctan2(u[:, 1], u[:, 0]); el = np.arcsin(np.clip(u[:, 2], -1, 1))
                h, _, _ = np.histogram2d(az, el, bins=[12, 6], range=[[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]]); H[n] = (h / h.sum(), u)
            p, q = H["ego"][0].ravel() + 1e-12, H["robot"][0].ravel() + 1e-12; mm = 0.5 * (p + q)
            js = float(0.5 * (p * np.log2(p / mm)).sum() + 0.5 * (q * np.log2(q / mm)).sum())
            dirn[f"{arm}_k{k}"] = dict(js_divergence_bits_12x6_bins=js, mean_unit_ego=H["ego"][1].mean(0).round(3).tolist(), mean_unit_robot=H["robot"][1].mean(0).round(3).tolist(),
                                       robot_mass_in_bins_ego_lacks=float(q[(p < 1e-4)].sum()), ego_mass_in_bins_robot_lacks=float(p[(q < 1e-4)].sum()),
                                       note="moving samples only (|dxyz| > 5 mm)")
    R["direction"] = dirn

    # ---------------------------------------------------------------- 10 near-zero
    nz = {}
    for n in D:
        for k in KS:
            for arm in ARMS:
                tn = np.linalg.norm(F[n][f"{arm}_axyz_k{k}"], axis=1); rd = F[n][f"{arm}_arot_k{k}"]
                nz[f"{n}_{arm}_k{k}"] = {**{f"trans_lt_{t}mm": float((tn < t / 1e3).mean()) for t in (1, 2, 5, 10)}, **{f"rot_lt_{t}deg": float((rd < t).mean()) for t in (1, 2, 5)},
                                        "both_arms_trans_lt_2mm": float(((np.linalg.norm(F[n][f"L_axyz_k{k}"], axis=1) < 2e-3) & (np.linalg.norm(F[n][f"R_axyz_k{k}"], axis=1) < 2e-3)).mean())}
    R["near_zero"] = nz

    # ---------------------------------------------------------------- 11 joint CART20: PCA (raw + common-standardized), NN coverage per block
    BL = {"position": [0, 1, 2, 10, 11, 12], "rotation": list(range(3, 9)) + list(range(13, 19)), "gripper": [9, 19], "full_CART20": list(range(20))}
    ie = sub(len(D["ego"]["state"]), a.n_joint, rng); ir = sub(len(D["robot"]["state"]), a.n_joint, rng)
    cov = {}; pcs = {}
    for k in (4, 8, 16):
        Xe = D["ego"]["action"][ie, k - 1, :20]; Xr = D["robot"]["action"][ir, k - 1, :20]
        for bn, cols in BL.items():
            sc = Scaler(Xe[:, cols], Xr[:, cols]); E_, Rr_ = sc(Xe[:, cols]), sc(Xr[:, cols])
            h = len(Rr_) // 2; hr = len(E_) // 2
            r2e = nn_dist(Rr_, E_); e2r = nn_dist(E_, Rr_)
            r2r = nn_dist(Rr_[:h], Rr_[h:]); e2e = nn_dist(E_[:hr], E_[hr:])
            # equal reference size for the cross distances (half-size refs) so the baselines are like-for-like
            r2e_h = nn_dist(Rr_[:h], E_[hr:]); e2r_h = nn_dist(E_[:hr], Rr_[h:])
            cov[f"k{k}_{bn}"] = dict(dims=len(cols), robot_to_ego=pct(r2e_h, (50, 90, 95, 99)), robot_to_robot_baseline=pct(r2r, (50, 90, 95, 99)),
                                     ego_to_robot=pct(e2r_h, (50, 90, 95, 99)), ego_to_ego_baseline=pct(e2e, (50, 90, 95, 99)),
                                     robot_to_ego_over_baseline_p50=float(np.median(r2e_h) / max(np.median(r2r), 1e-12)),
                                     robot_to_ego_over_baseline_p95=float(np.percentile(r2e_h, 95) / max(np.percentile(r2r, 95), 1e-12)),
                                     frac_robot_covered_within_robot_p95=float((r2e_h <= np.percentile(r2r, 95)).mean()),
                                     frac_ego_covered_within_ego_p95=float((e2r_h <= np.percentile(e2e, 95)).mean()),
                                     note="common scaler on the union; half-size references for all four distances")
            if bn == "full_CART20": pcs[k] = (Xe, Xr, r2e, e2r)
        cov[f"k{k}_full_CART20"]["mmd_rbf_secondary"], cov[f"k{k}_full_CART20"]["mmd_sigma2"] = mmd_rbf(Scaler(Xe, Xr)(Xe), Scaler(Xe, Xr)(Xr), rng)
    R["coverage_nn"] = cov
    # state coverage (pose part of the state)
    Se = D["ego"]["state"][ie][:, list(range(18))]; Sr = D["robot"]["state"][ir][:, list(range(18))]; sc = Scaler(Se, Sr); h = len(Sr) // 2; hr = len(Se) // 2
    r2e = nn_dist(sc(Sr)[:h], sc(Se)[hr:]); r2r = nn_dist(sc(Sr)[:h], sc(Sr)[h:]); e2r = nn_dist(sc(Se)[:hr], sc(Sr)[h:]); e2e = nn_dist(sc(Se)[:hr], sc(Se)[hr:])
    R["coverage_nn"]["state_pose18"] = dict(robot_to_ego=pct(r2e, (50, 90, 95, 99)), robot_to_robot_baseline=pct(r2r, (50, 90, 95, 99)), ego_to_robot=pct(e2r, (50, 90, 95, 99)),
                                            ego_to_ego_baseline=pct(e2e, (50, 90, 95, 99)), frac_robot_covered_within_robot_p95=float((r2e <= np.percentile(r2r, 95)).mean()))

    # ---------------------------------------------------------------- 12 Wasserstein (raw + pooled-std normalised)
    W = {}
    def wd(name, x, y):
        s = np.concatenate([x, y]).std(); W[name] = dict(raw=float(wasserstein_distance(x, y)), normalized=float(wasserstein_distance(x, y) / max(s, 1e-12)))
    for arm in ARMS:
        for ax, axn in enumerate("xyz"): wd(f"state_{arm}_{axn}", F["ego"][f"{arm}_sxyz"][:, ax], F["robot"][f"{arm}_sxyz"][:, ax])
        wd(f"state_{arm}_rot_deg", F["ego"][f"{arm}_srot"], F["robot"][f"{arm}_srot"]); wd(f"state_{arm}_g", F["ego"][f"{arm}_g"], F["robot"][f"{arm}_g"])
        for k in KS:
            for ax, axn in enumerate("xyz"): wd(f"action_k{k}_{arm}_d{axn}", F["ego"][f"{arm}_axyz_k{k}"][:, ax], F["robot"][f"{arm}_axyz_k{k}"][:, ax])
            wd(f"action_k{k}_{arm}_trans_norm", np.linalg.norm(F["ego"][f"{arm}_axyz_k{k}"], axis=1), np.linalg.norm(F["robot"][f"{arm}_axyz_k{k}"], axis=1))
            wd(f"action_k{k}_{arm}_rot_deg", F["ego"][f"{arm}_arot_k{k}"], F["robot"][f"{arm}_arot_k{k}"]); wd(f"action_k{k}_{arm}_g", F["ego"][f"{arm}_ag_k{k}"], F["robot"][f"{arm}_ag_k{k}"])
    R["wasserstein"] = W

    # ---------------------------------------------------------------- 13 workspace + action occupancy (common grid)
    occ = {}
    def occupancy(name, Xe, Xr, res, min_count):
        lo = np.minimum(Xe.min(0), Xr.min(0)); ke = np.floor((Xe - lo) / res).astype(int); kr = np.floor((Xr - lo) / res).astype(int)
        ce = collections.Counter(map(tuple, ke)); cr = collections.Counter(map(tuple, kr))
        Be = {b for b, c in ce.items() if c >= min_count}; Br = {b for b, c in cr.items() if c >= min_count}; sh = Be & Br
        rmass = sum(cr[b] for b in Br & set(ce)) / max(sum(cr[b] for b in Br), 1)
        occ[name] = dict(res_m=res, min_count=min_count, ego_bins=len(Be), robot_bins=len(Br), shared=len(sh), ego_only=len(Be - Br), robot_only=len(Br - Be),
                         robot_bins_covered_by_ego=len(sh) / max(len(Br), 1), ego_bins_covered_by_robot=len(sh) / max(len(Be), 1),
                         robot_SAMPLES_in_bins_ego_visits=float(rmass))
    for arm in ARMS:
        for mc in (1, 5): occupancy(f"state_{arm}_xyz_2cm_min{mc}", F["ego"][f"{arm}_sxyz"], F["robot"][f"{arm}_sxyz"], 0.02, mc)
        for k in (4, 8, 16):
            for mc in (1, 5): occupancy(f"action_k{k}_{arm}_dxyz_1cm_min{mc}", F["ego"][f"{arm}_axyz_k{k}"], F["robot"][f"{arm}_axyz_k{k}"], 0.01, mc)
            for pl, (i0, i1) in (("dxdy", (0, 1)), ("dxdz", (0, 2)), ("dydz", (1, 2))):
                occupancy(f"action_k{k}_{arm}_{pl}_5mm_min1", F["ego"][f"{arm}_axyz_k{k}"][:, [i0, i1]], F["robot"][f"{arm}_axyz_k{k}"][:, [i0, i1]], 0.005, 1)
    R["occupancy"] = occ

    # ---------------------------------------------------------------- 14 order composition
    R["orders"] = {n: dict(episodes=dict(collections.Counter(d["order"][np.r_[True, np.diff(d["ep"]) != 0]].tolist())),
                           rows=dict(collections.Counter(d["order"].tolist())), source="instruction text (both datasets use the same 6 templates)") for n, d in D.items()}

    # ---------------------------------------------------------------- 15 right-arm low-x boundary (robot base frame via FK)
    R["right_arm_boundary"] = right_boundary(D, F, rng)

    # ---------------------------------------------------------------- 16 state-action relationship (binned medians)
    sa = {}
    for arm in ARMS:
        for k in (4, 8, 16):
            row = {}
            for n in D:
                x = np.linalg.norm(F[n][f"{arm}_sxyz"], axis=1); y = np.linalg.norm(F[n][f"{arm}_axyz_k{k}"], axis=1); bins = np.array([0, .02, .05, .1, .15, .2, .3, .5, 2])
                b = np.digitize(x, bins) - 1
                row[n] = [dict(state_r_bin_m=[float(bins[i]), float(bins[i + 1])], n=int((b == i).sum()), action_p50_mm=float(1e3 * np.median(y[b == i])) if (b == i).sum() > 20 else None,
                               action_p90_mm=float(1e3 * np.percentile(y[b == i], 90)) if (b == i).sum() > 20 else None) for i in range(len(bins) - 1)]
            sa[f"{arm}_k{k}"] = row
    R["state_action_relationship"] = sa

    # ---------------------------------------------------------------- 17 camera domain (low-level)
    R["camera"] = camera_stats(a, OUT, rng) if a.n_images > 0 else "skipped"

    # ---------------------------------------------------------------- plots
    plots(D, F, R, pcs, OUT / "plots", rng)
    R["runtime_s"] = round(time.time() - t_start, 1); R["inputs"] = dict(ego=a.ego, robot=a.robot, seed=a.seed, n_joint=a.n_joint, n_images=a.n_images)
    json.dump(R, open(OUT / "summary.json", "w"), indent=1, default=float)
    write_md(R, OUT / "summary.md", names)
    print("COMPARE DONE", OUT, f"{R['runtime_s']} s")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}) for r in rows]


def right_boundary(D, F, rng):
    """R312c right-arm absolute x (robot base frame) from FK(aux.q_t); the live failure drove R to x 58-160 mm while training has
    x >= 172 mm for 99 % (boundary_bias_eval.py).  Ego has NO robot base frame: the comparison is in TASK-RELATIVE state xyz only."""
    d = D["robot"]
    if d["q"] is None: return "robot aux.q_t missing"
    import os; os.environ.setdefault("REBOT_URDF", str(pathlib.Path.home() / "c8/rel16ego/rebot_ee/reBot_B601_DM_dualarm.urdf"))
    sys.path[:0] = [str(pathlib.Path.home() / "c8/c8old"), str(pathlib.Path.home() / "holobrain-mac-model")]; import torch, rebot_fk_torch
    fk = rebot_fk_torch.ReBotFKTorch(device="cpu", dtype=torch.float64); Ts = []
    for i in range(0, len(d["q"]), 20000): Ts.append(fk.tcp(torch.tensor(d["q"][i:i + 20000])).numpy())
    T = np.concatenate(Ts); xr = T[:, 1, 0, 3] * 1e3
    bins = [(250, 1e9), (220, 250), (190, 220), (172, 190), (-1e9, 172)]; out = dict(robot_abs_right_x_mm=pct(xr, (1, 5, 50, 95, 99)), bins=[])
    Se = F["ego"]["R_sxyz"]; tree = cKDTree(Se)
    for lo, hi in bins:
        m = (xr >= lo) & (xr < hi); n = int(m.sum())
        if n < 20: out["bins"].append(dict(x_mm=[lo, hi], robot_rows=n)); continue
        rel = F["robot"]["R_sxyz"][m]; dist, j = tree.query(rel, k=1)
        near = dist < 0.02; ej = np.unique(j[near])
        u = lambda v: (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)).mean(0).round(3).tolist()
        out["bins"].append(dict(x_mm=[lo, hi], robot_rows=n, robot_task_rel_state_xyz_mean_m=rel.mean(0).round(3).tolist(),
                                frac_robot_rows_with_ego_within_2cm_task_rel=float(near.mean()), distinct_ego_rows_matched=int(len(ej)),
                                robot_k8_mean_unit_dir=u(F["robot"]["R_axyz_k8"][m]), robot_k8_mean_dxyz_mm=(1e3 * F["robot"]["R_axyz_k8"][m].mean(0)).round(2).tolist(),
                                ego_matched_k8_mean_unit_dir=u(F["ego"]["R_axyz_k8"][ej]) if len(ej) else None,
                                ego_matched_k8_mean_dxyz_mm=(1e3 * F["ego"]["R_axyz_k8"][ej].mean(0)).round(2).tolist() if len(ej) else None))
    out["caveat"] = ("robot x is the ABSOLUTE base-frame TCP (FK aux.q_t). Ego has no robot base frame; ego rows are matched by TASK-RELATIVE right-arm "
                     "state xyz (inv(T_start) T(t)) within 2 cm of the robot rows' task-relative xyz. That is a like-for-like comparison only if the "
                     "robot's task-start pose is consistent across episodes; it says nothing about the absolute workspace edge for humans.")
    return out


def camera_stats(a, OUT, rng):
    import av, cv2
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    res = {}; sheets = {}
    for n, root in (("ego", a.ego), ("robot", a.robot)):
        for key in ("observation.images.global", "observation.images.left_wrist", "observation.images.right_wrist"):
            files = sorted(glob.glob(f"{root}/videos/{key}/*/*.mp4")); tot = 0; cnt = []
            for f in files:
                with av.open(f) as c: s = c.streams.video[0]; cnt.append(s.frames or 0)
            tot = sum(cnt); stride = max(1, tot // a.n_images); imgs = []; gi = 0
            for f in files:
                with av.open(f) as c:
                    for fr in c.decode(video=0):
                        if gi % stride == 0 and len(imgs) < a.n_images: imgs.append(fr.to_ndarray(format="rgb24"))
                        gi += 1
                if len(imgs) >= a.n_images: break
            cm, cs, br, ct, sh = [], [], [], [], []
            for im in imgs:                                                   # per-image float64 stats (no big float32 accumulation)
                x = im.astype(np.float64); g = x.mean(-1)
                cm.append(x.reshape(-1, 3).mean(0)); cs.append(x.reshape(-1, 3).std(0)); br.append(g.mean()); ct.append(g.std())
                sh.append(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).var())
            res[f"{n}_{key.split('.')[-1]}"] = dict(n=len(imgs), rgb_mean=np.mean(cm, 0).round(2).tolist(), rgb_std_within_image=np.mean(cs, 0).round(2).tolist(),
                                                     brightness=pct(br, (5, 50, 95)), contrast=pct(ct, (5, 50, 95)), sharpness_laplacian_var=pct(sh, (5, 50, 95)))
            sheets[(n, key)] = np.stack([imgs[i] for i in np.linspace(0, len(imgs) - 1, 24).astype(int)])
    fig, axs = plt.subplots(6, 1, figsize=(16, 20))
    for ax, ((n, key), S) in zip(axs, sheets.items()):
        ax.imshow(np.concatenate([np.concatenate(list(S[r * 12:(r + 1) * 12]), 1) for r in range(2)], 0)); ax.set_title(f"{n} {key}"); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "plots" / "camera_contact_sheets.png", dpi=70); plt.close(fig)
    res["note"] = "low-level RGB statistics only; no semantic similarity is claimed. X-VLA encoder embeddings not computed (optional section 25)."
    return res


def plots(D, F, R, pcs, P, rng):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def hist2(ax, xs, title, bins=80, rng_=None, log=False):
        lo = min(np.percentile(x, 0.5) for x in xs.values()) if rng_ is None else rng_[0]; hi = max(np.percentile(x, 99.5) for x in xs.values()) if rng_ is None else rng_[1]
        for n, x in xs.items(): ax.hist(x, bins=bins, range=(lo, hi), density=True, histtype="step", lw=1.5, color=COLOR[n], label=n, log=log)
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7)
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for i, arm in enumerate(ARMS):
        for ax_i, axn in enumerate("xyz"): hist2(axs[i, ax_i], {n: F[n][f"{arm}_sxyz"][:, ax_i] for n in D}, f"state {arm} {axn} (m, task-relative)")
    fig.tight_layout(); fig.savefig(P / "state_xyz_hist.png", dpi=90); plt.close(fig)
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for i, arm in enumerate(ARMS):
        hist2(axs[i], {n: np.linalg.norm(F[n][f"{arm}_sxyz"], axis=1) for n in D}, f"state {arm} |xyz| (m)")
        hist2(axs[i + 2], {n: F[n][f"{arm}_srot"] for n in D}, f"state {arm} rotation from task start (deg)")
    fig.tight_layout(); fig.savefig(P / "state_translation_norm.png", dpi=90); plt.close(fig)
    for k in KS:
        fig, axs = plt.subplots(2, 3, figsize=(14, 7))
        for i, arm in enumerate(ARMS):
            for ax_i, axn in enumerate("xyz"): hist2(axs[i, ax_i], {n: 1e3 * F[n][f"{arm}_axyz_k{k}"][:, ax_i] for n in D}, f"action k{k} ({k * 50.05:.0f} ms) {arm} d{axn} (mm)", log=True)
        fig.tight_layout(); fig.savefig(P / f"action_xyz_hist_k{k}.png", dpi=90); plt.close(fig)
    for feat, fn, unit in (("translation", lambda n, arm, k: 1e3 * np.linalg.norm(F[n][f"{arm}_axyz_k{k}"], axis=1), "mm"), ("rotation", lambda n, arm, k: F[n][f"{arm}_arot_k{k}"], "deg")):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for i, arm in enumerate(ARMS):
            for n in D:
                for q, ls in ((50, "-"), (90, "--"), (99, ":")):
                    axs[i].plot(KS, [np.percentile(fn(n, arm, k), q) for k in KS], ls, color=COLOR[n], marker="o", label=f"{n} p{q}")
            axs[i].set_xticks(KS); axs[i].set_xlabel("k (x 50.05 ms)"); axs[i].set_ylabel(unit); axs[i].set_title(f"{arm} action {feat} by k"); axs[i].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(P / f"action_{feat}_norm_by_k.png", dpi=90); plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for i, arm in enumerate(ARMS):
        hist2(axs[i, 0], {n: F[n][f"{arm}_g"] for n in D}, f"{arm} state gripper (0 closed, 1 open)", 50, (0, 1))
        hist2(axs[i, 1], {n: F[n][f"{arm}_ag_k8"] for n in D}, f"{arm} action gripper k8", 50, (0, 1))
        hist2(axs[i, 2], {n: F[n][f"{arm}_ag_k16"] - F[n][f"{arm}_g"] for n in D}, f"{arm} dg = g_action(k16) - g_state", 80, (-1, 1), log=True)
    fig.tight_layout(); fig.savefig(P / "gripper_hist.png", dpi=90); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i, arm in enumerate(ARMS):
        for n in D: axs[i].plot(KS, [np.mean(np.abs(F[n][f"{arm}_ag_k{k}"] - F[n][f"{arm}_g"]) > 0.1) for k in KS], marker="o", color=COLOR[n], label=n)
        axs[i].set_xticks(KS); axs[i].set_title(f"{arm}: fraction |g(t+k) - g(t)| > 0.1"); axs[i].legend()
    fig.tight_layout(); fig.savefig(P / "gripper_transition.png", dpi=90); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for n in D:
        axs[0].plot(KS, [R["left_right_asymmetry"][f"{n}_k{k}"]["trans_ratio_R_over_L_p50"] for k in KS], marker="o", color=COLOR[n], label=n)
        for arm, ls in (("L", "-"), ("R", "--")):
            axs[1].plot(KS, [R["left_right_asymmetry"][f"{n}_k{k}"][f"mean_dxyz_mm_{arm}"][0] for k in KS], ls, marker="o", color=COLOR[n], label=f"{n} {arm} mean dx")
            axs[2].plot(KS, [R["left_right_asymmetry"][f"{n}_k{k}"][f"mean_dxyz_mm_{arm}"][1] for k in KS], ls, marker="o", color=COLOR[n], label=f"{n} {arm} mean dy")
    for ax, t in zip(axs, ("trans p50 ratio R/L", "mean dx (mm)", "mean dy (mm)")): ax.set_title(t); ax.set_xticks(KS); ax.legend(fontsize=7); ax.axhline(1 if "ratio" in t else 0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(P / "left_right_asymmetry.png", dpi=90); plt.close(fig)
    # PCA (raw + standardized), union fit, equal subsamples
    fig, axs = plt.subplots(2, 6, figsize=(28, 9))
    for c, k in enumerate((4, 8, 16)):
        Xe, Xr, _, _ = pcs[k]
        for r, mode in enumerate(("raw", "standardized")):
            sc = Scaler(Xe, Xr) if mode == "standardized" else (lambda X: X)
            Ze, Zr = sc(Xe), sc(Xr); V, ev = pca(np.concatenate([Ze, Zr])); mu = np.concatenate([Ze, Zr]).mean(0)
            pe, pr = (Ze - mu) @ V.T, (Zr - mu) @ V.T
            for j, (a_, b_) in enumerate(((0, 1), (0, 2))):
                ax = axs[r, 2 * c + j]
                ax.scatter(pr[:, a_], pr[:, b_], s=1, alpha=0.15, color=COLOR["robot"], label="robot"); ax.scatter(pe[:, a_], pe[:, b_], s=1, alpha=0.15, color=COLOR["ego"], label="ego")
                ax.set_title(f"CART20 k{k} {mode} PCA (union) PC{a_ + 1}/PC{b_ + 1} ev {ev[a_]:.2f}/{ev[b_]:.2f}", fontsize=8); ax.legend(markerscale=8, fontsize=7)
    fig.tight_layout(); fig.savefig(P / "cart20_pca.png", dpi=80); plt.close(fig)
    ie = sub(len(D["ego"]["state"]), 20000, rng); ir = sub(len(D["robot"]["state"]), 20000, rng)
    Se, Sr = D["ego"]["state"][ie][:, :18], D["robot"]["state"][ir][:, :18]
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.5))
    for j, mode in enumerate(("raw", "standardized")):
        sc = Scaler(Se, Sr) if mode == "standardized" else (lambda X: X); Ze, Zr = sc(Se), sc(Sr); V, ev = pca(np.concatenate([Ze, Zr])); mu = np.concatenate([Ze, Zr]).mean(0)
        for jj, (a_, b_) in enumerate(((0, 1), (0, 2))):
            ax = axs[2 * j + jj]; pe, pr = (Ze - mu) @ V.T, (Zr - mu) @ V.T
            ax.scatter(pr[:, a_], pr[:, b_], s=1, alpha=0.15, color=COLOR["robot"], label="robot"); ax.scatter(pe[:, a_], pe[:, b_], s=1, alpha=0.15, color=COLOR["ego"], label="ego")
            ax.set_title(f"state pose18 {mode} PCA PC{a_ + 1}/PC{b_ + 1}", fontsize=9); ax.legend(markerscale=8, fontsize=7)
    fig.tight_layout(); fig.savefig(P / "state_pca.png", dpi=80); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    for c, k in enumerate((4, 8, 16)):
        cv = R["coverage_nn"][f"k{k}_full_CART20"]
        names = ["robot_to_ego", "robot_to_robot_baseline", "ego_to_robot", "ego_to_ego_baseline"]
        for q, ls in (("p50", "o"), ("p95", "s"), ("p99", "^")): axs[c].plot(range(4), [cv[nm][q] for nm in names], ls, label=q)
        axs[c].set_xticks(range(4)); axs[c].set_xticklabels(["robot->ego", "robot->robot", "ego->robot", "ego->ego"], fontsize=8)
        axs[c].set_title(f"CART20 k{k} NN distance (common scaler)"); axs[c].legend()
    fig.tight_layout(); fig.savefig(P / "nearest_neighbor_distance.png", dpi=90); plt.close(fig)
    # workspace occupancy, common axes
    fig, axs = plt.subplots(2, 6, figsize=(26, 8))
    for i, arm in enumerate(ARMS):
        allx = np.concatenate([F[n][f"{arm}_sxyz"] for n in D]); lo, hi = np.percentile(allx, 0.5, 0), np.percentile(allx, 99.5, 0)
        for j, (a_, b_) in enumerate(((0, 1), (0, 2), (1, 2))):
            for c, n in enumerate(D):
                ax = axs[i, 2 * j + c]; x = F[n][f"{arm}_sxyz"]
                ax.hist2d(x[:, a_], x[:, b_], bins=60, range=[[lo[a_], hi[a_]], [lo[b_], hi[b_]]], cmin=1, cmap="viridis", norm=matplotlib.colors.LogNorm())
                ax.set_title(f"{n} {arm} state {'xyz'[a_]}{'xyz'[b_]} (common range)", fontsize=8)
    fig.tight_layout(); fig.savefig(P / "coverage_overlap.png", dpi=80); plt.close(fig)


def fmt(x, d=1):
    return "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def write_md(R, path, names):
    L = []; ap = L.append
    hz = {}
    for r in csv.DictReader(open(path.parent / "horizon_distribution.csv")): hz[(r["dataset"], r["arm"], int(r["k"]))] = r
    ap("# Ego CART20 vs R312c RELCART20: dataset comparison (read-only)\n")
    ap(f"- Ego: `{R['inputs']['ego']}`\n- Robot: `{R['inputs']['robot']}`\n- seed {R['inputs']['seed']}, joint subsample {R['inputs']['n_joint']} per dataset, runtime {R['runtime_s']} s\n")
    ap("## 1. Contract parity\n\n| Item | Ego | R312c |\n|---|---|---|")
    for r in R["contract_parity"]["rows"]: ap(f"| {r['item']} | {r['ego']} | {r['robot']} |")
    ap("\n| Documented contract | Ego | R312c | Match |\n|---|---|---|---|")
    for k, v in R["contract_parity"]["documented"].items(): ap(f"| {k} | {v['ego']} | {v['robot']} | {'yes' if v['match'] else 'NO'} |")
    ap(f"\nSemantic parity: **{'PASS' if R['contract_parity']['semantic_parity_pass'] else 'FAIL'}**.\n")
    ap("## 2. Dataset size\n\n| | Episodes | Training rows | Action targets | Row-time (min) | Rows/episode p5/p50/p95 |\n|---|---:|---:|---:|---:|---|")
    for n, s in R["size"].items(): ap(f"| {n} | {s['episodes']} | {s['training_rows']:,} | {s['action_targets']:,} | {s['row_time_min']} | {s['rows_per_episode']['p05']:.0f} / {s['rows_per_episode']['p50']:.0f} / {s['rows_per_episode']['p95']:.0f} |")
    ap("\n## 3. State / workspace (task-relative)\n\nSee `state_distribution.csv`. Workspace occupancy, 2 cm voxels, common grid:\n\n| Arm | Min count | Ego bins | Robot bins | Shared | Robot bins covered by ego | Robot SAMPLES in ego-visited bins |\n|---|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for mc in (1, 5):
            o = R["occupancy"][f"state_{arm}_xyz_2cm_min{mc}"]; ap(f"| {arm} | {mc} | {o['ego_bins']} | {o['robot_bins']} | {o['shared']} | {o['robot_bins_covered_by_ego']:.3f} | {o['robot_SAMPLES_in_bins_ego_visits']:.3f} |")
    ap("\n## 4–6. CART20 action by horizon (translation mm, rotation deg)\n\n| k (ms) | Arm | Ego trans p50/p95/p99 | R312c trans p50/p95/p99 | Ego rot p50/p95 | R312c rot p50/p95 |\n|---|---|---|---|---|---|")
    for k in KS:
        for arm in ARMS:
            e, r = hz[("ego", arm, k)], hz[("robot", arm, k)]; f_ = lambda x: f"{float(x):.1f}"
            ap(f"| {k} ({k * 50.05:.0f}) | {arm} | {f_(e['trans_p50_mm'])} / {f_(e['trans_p95_mm'])} / {f_(e['trans_p99_mm'])} | {f_(r['trans_p50_mm'])} / {f_(r['trans_p95_mm'])} / {f_(r['trans_p99_mm'])} | {f_(e['rot_p50_deg'])} / {f_(e['rot_p95_deg'])} | {f_(r['rot_p50_deg'])} / {f_(r['rot_p95_deg'])} |")
    ap("\nAction occupancy (1 cm voxels of dxyz, min 1 / 5 samples):\n\n| k | Arm | Robot bins covered by ego (min1 / min5) | Ego bins covered by robot (min1 / min5) | Robot samples in ego bins |\n|---|---|---|---|---|")
    for k in (4, 8, 16):
        for arm in ARMS:
            o1, o5 = R["occupancy"][f"action_k{k}_{arm}_dxyz_1cm_min1"], R["occupancy"][f"action_k{k}_{arm}_dxyz_1cm_min5"]
            ap(f"| {k} | {arm} | {o1['robot_bins_covered_by_ego']:.3f} / {o5['robot_bins_covered_by_ego']:.3f} | {o1['ego_bins_covered_by_robot']:.3f} / {o5['ego_bins_covered_by_robot']:.3f} | {o1['robot_SAMPLES_in_bins_ego_visits']:.3f} |")
    ap("\nMotion direction (moving samples > 5 mm, 12 x 6 azimuth/elevation bins):\n\n| Arm, k | JS divergence (bits) | Ego mean unit | R312c mean unit | Robot mass in bins ego lacks |\n|---|---:|---|---|---:|")
    for k_, v in R["direction"].items(): ap(f"| {k_} | {v['js_divergence_bits_12x6_bins']:.3f} | {v['mean_unit_ego']} | {v['mean_unit_robot']} | {v['robot_mass_in_bins_ego_lacks']:.3f} |")
    ap("\n## 7. Continuous gripper (0 closed / 1 open, NOT remapped)\n\nSee `gripper_distribution.csv` for the full table.\n")
    gd = list(csv.DictReader(open(path.parent / "gripper_distribution.csv")))
    ap("| Dataset | Arm | Signal | Mean | p10 / p50 / p90 | g<0.1 | 0.25–0.5 | 0.5–0.75 | ≥0.75 |\n|---|---|---|---:|---|---:|---:|---:|---:|")
    for r in gd:
        if r["signal"] in ("state", "action_k8"):
            ap(f"| {r['dataset']} | {r['arm']} | {r['signal']} | {float(r['mean']):.3f} | {float(r['p10']):.2f} / {float(r['p50']):.2f} / {float(r['p90']):.2f} | {float(r['frac_g<0.1']):.3f} | {float(r['frac_0.25<=g<0.5']):.3f} | {float(r['frac_0.5<=g<0.75']):.3f} | {float(r['frac_g>=0.75']):.3f} |")
    ap("\ndg = g_action(t+k) - g_state(t). For R312c that mixes two signals (follower state vs leader action), so the same-signal column action-internal = g_action(k) - g_action(k1) is the like-for-like one.\n")
    ap("| Dataset | Arm | k | fraction abs(dg) > 0.1 | abs(dg) p95 | Action-internal abs(dg) > 0.1 | Action-internal p95 | corr(abs(dg), translation) |\n|---|---|---:|---:|---:|---:|---:|---:|")
    for r in gd:
        if r["signal"].startswith("action_k") and r["signal"] in ("action_k4", "action_k16"):
            ap(f"| {r['dataset']} | {r['arm']} | {r['signal'][8:]} | {float(r['frac_abs_dg_gt_0p1']):.3f} | {float(r['dg_abs_p95']):.3f} | {float(r['frac_abs_dg_action_internal_gt_0p1']):.3f} | {float(r['dg_action_internal_abs_p95']):.3f} | {float(r['corr_dg_vs_trans']):.3f} |")
    ap("\nState gripper transitions per episode (hysteresis 0.4/0.6), p25/p50/p75: " + "; ".join(f"{k.split('_state')[0]} {v['p25']:.0f}/{v['p50']:.0f}/{v['p75']:.0f}" for k, v in R["gripper_transitions"].items()))
    ap(f"\nWithin-dataset state vs action(k1) gripper: robot {json.dumps(R['robot_gripper_state_vs_action_k1'])}; ego {json.dumps(R['ego_gripper_state_vs_action_k1'])}\n")
    ap("## 8. Left/right asymmetry\n\n| Dataset, k | Trans p50 L/R (mm) | Ratio R/L | Rot p95 L/R | Mean dxyz L (mm) | Mean dxyz R (mm) |\n|---|---|---:|---|---|---|")
    for k_, v in R["left_right_asymmetry"].items(): ap(f"| {k_} | {v['trans_p50_mm_L']:.1f} / {v['trans_p50_mm_R']:.1f} | {v['trans_ratio_R_over_L_p50']:.2f} | {v['rot_p95_deg_L']:.1f} / {v['rot_p95_deg_R']:.1f} | {v['mean_dxyz_mm_L']} | {v['mean_dxyz_mm_R']} |")
    ap("\n## 9. Near-zero actions (fraction of rows)\n\n| Dataset, arm, k | <1 mm | <2 mm | <5 mm | <10 mm | rot<1° | rot<2° | Both arms <2 mm |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for k_, v in R["near_zero"].items():
        if k_.split("_")[-1] in ("k1", "k4", "k8", "k16"): ap(f"| {k_} | {v['trans_lt_1mm']:.3f} | {v['trans_lt_2mm']:.3f} | {v['trans_lt_5mm']:.3f} | {v['trans_lt_10mm']:.3f} | {v['rot_lt_1deg']:.3f} | {v['rot_lt_2deg']:.3f} | {v['both_arms_trans_lt_2mm']:.3f} |")
    ap("\n## 10. Coverage / nearest neighbour (common scaler on the union, both directions, with in-domain baselines)\n\n| Block | robot→ego p50/p95 | robot→robot p50/p95 | ego→robot p50/p95 | ego→ego p50/p95 | Robot covered within robot-p95 | Ego covered within ego-p95 |\n|---|---|---|---|---|---:|---:|")
    for k_, v in R["coverage_nn"].items():
        if "robot_to_ego" not in v: continue
        g_ = lambda nm: f"{v[nm]['p50']:.2f} / {v[nm]['p95']:.2f}"
        ap(f"| {k_} | {g_('robot_to_ego')} | {g_('robot_to_robot_baseline')} | {g_('ego_to_robot')} | {g_('ego_to_ego_baseline')} | {v['frac_robot_covered_within_robot_p95']:.3f} | {fmt(v.get('frac_ego_covered_within_ego_p95'), 3)} |")
    ap("\nMMD (RBF, median heuristic, secondary): " + ", ".join(f"k{k} {R['coverage_nn'][f'k{k}_full_CART20']['mmd_rbf_secondary']:.4f}" for k in (4, 8, 16)))
    ap("\n## Right-arm low-x boundary (R312c base frame)\n")
    rb = R["right_arm_boundary"]
    if isinstance(rb, dict):
        ap(f"R312c right absolute x (mm) p1/p5/p50: {rb['robot_abs_right_x_mm']['p01']:.0f} / {rb['robot_abs_right_x_mm']['p05']:.0f} / {rb['robot_abs_right_x_mm']['p50']:.0f}\n")
        ap("| Abs x bin (mm) | Robot rows | Robot rows with ego within 2 cm (task-rel) | Robot k8 mean dxyz (mm) | Matched-ego k8 mean dxyz (mm) |\n|---|---:|---:|---|---|")
        for b in rb["bins"]: ap(f"| {b['x_mm']} | {b['robot_rows']} | {fmt(b.get('frac_robot_rows_with_ego_within_2cm_task_rel'), 3)} | {b.get('robot_k8_mean_dxyz_mm')} | {b.get('ego_matched_k8_mean_dxyz_mm')} |")
        ap(f"\n> {rb['caveat']}\n")
    ap("## Orders\n\n" + "\n".join(f"- {n}: episodes {json.dumps(v['episodes'])}" for n, v in R["orders"].items()))
    ap("\n## 11. Visual domain (low-level)\n")
    if isinstance(R["camera"], dict):
        ap("| View | Brightness p50 | Contrast p50 | Sharpness p50 | RGB mean |\n|---|---:|---:|---:|---|")
        for k_, v in R["camera"].items():
            if isinstance(v, dict): ap(f"| {k_} | {v['brightness']['p50']:.1f} | {v['contrast']['p50']:.1f} | {v['sharpness_laplacian_var']['p50']:.0f} | {v['rgb_mean']} |")
        ap(f"\n{R['camera']['note']} Contact sheets: `plots/camera_contact_sheets.png`.\n")
    ap("## 12–14. Interpretation\n\n(Filled in by hand from the numbers above. See `INTERPRETATION.md`.)\n")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ego", required=True); ap.add_argument("--robot", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--n-joint", type=int, default=20000); ap.add_argument("--n-images", type=int, default=5000)
    main(ap.parse_args())
