#!/usr/bin/env python3
"""READ-ONLY kinematic compatibility of ego CART20 with reBot (DIAGNOSTIC ONLY -- no q / dq is written to any dataset or
used as supervision; ego training stays human Cartesian -> CART20 -> AUX12 = 0).

IK = the deployment stack exactly: holobrain-mac-model/pink_ik.PinkIK (same URDF a117e024 as rebot_fk_torch / the dataset TCP,
TCP = gripper_link @ +0.07 m x, targets in the dataset frame), FrameTask pos 1.0 / ori 0.5, posture 1e-3 to the seed,
quadprog, dt 0.05, <= 200 iterations, stop at 0.5 mm / 0.5 deg, ConfigurationLimit = URDF limits, and the UI setting
V4_PINK_LOCK=joint5 (joint5 hard-held at the seed / measured q).  No relaxed configuration for the main numbers.
Extra modes (diagnostic, labelled): position-only (orientation cost 0, joint5 locked), full pose without the joint5 lock,
ori='yaw' (position + base-z rotation, joint5 locked).

Absolute targets: ego poses are TASK-RELATIVE (per arm, separate SLAM maps) and have no robot base frame.  Each ego episode is
placed at the task-start TCP of a real R312c episode (deterministic random assignment, seed): T_abs(t) = FK(q0_r) @ S_rel(t),
per arm independently, IK seeded at q0_r and then CONTINUOUSLY (q_t seeds t+1).  A second independent assignment is run for a
sensitivity check.  R312c reference = the same IK on R312c's own trajectories (T_abs = FK(q0) @ S_rel = FK(q_t)).
Success: |dp| < 2 mm and rotation error < 2 deg (full); |dp| < 2 mm (position-only); 'converged' = the solver's own 0.5 / 0.5 stop.
usage: kinematic_compat.py --ego <ego lerobot train> --robot <r312c lerobot> --output <dir> [--episodes 40] [--seed 1000]"""
import argparse
import collections
import json
import os
import pathlib
import sys
import time

import numpy as np

H = pathlib.Path.home()
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2]), str(H / "holobrain-mac-model"), str(H / "c8/c8old")]
os.environ.setdefault("REBOT_URDF", str(H / "holobrain-mac-model/reBot_B601_DM_dualarm.urdf"))
from ego_cart20.geometry.rotation6d import rotation_angle  # noqa: E402
from ego_cart20.geometry.transforms import pose9, pose9_to_T  # noqa: E402
from ego_cart20.scripts.compare_ego_r312c import load, pct  # noqa: E402

PM = "full"   # primary mode for the downstream metrics (q distribution, NN, dq, continuity, workspace); --primary-mode
ARMS = ("L", "R"); ST = {"L": slice(0, 9), "R": slice(9, 18)}; AC = {"L": slice(0, 9), "R": slice(10, 19)}; QI = {"L": slice(0, 6), "R": slice(6, 12)}
KS = (1, 4, 8, 16); POS_OK_MM, ROT_OK_DEG = 2.0, 2.0
COLOR = {"ego": "#d95f02", "robot": "#1b9e77", "ego_feasible": "#7570b3"}


class IK:
    def __init__(self):
        import pink_ik
        names = [f"left_joint{i}" for i in range(1, 7)] + [f"right_joint{i}" for i in range(1, 7)]
        self.full = pink_ik.PinkIK(names)
        self.pos = pink_ik.PinkIK(names)
        for t in self.pos.tasks.values(): t.set_orientation_cost(0.0)
        m = self.full.model; self.lo = np.array([m.lowerPositionLimit[i] for i in self.full.idx]); self.hi = np.array([m.upperPositionLimit[i] for i in self.full.idx])

    def fk(self, q): return self.full.fk(q)

    def solve(self, mode, seed, targets):
        """-> q12, pos_err_mm[2], rot_err_deg[2], iters, margin[2] (min normalized joint-limit margin per arm, locked joint excluded)"""
        solver = self.pos if mode == "pos" else self.full
        lock = () if mode == "full_nolock" else ("joint5",)
        ori = "yaw" if mode == "yaw" else "full"
        q, _, it = solver.solve(seed, targets, ori=ori, lock=lock)
        if not np.isfinite(q).all(): return q, np.full(2, np.inf), np.full(2, np.inf), it, np.full(2, np.nan)
        T = self.fk(q); pe = np.array([np.linalg.norm(T[i][:3, 3] - targets[i][:3, 3]) * 1e3 for i in range(2)])
        re = np.array([np.degrees(rotation_angle(targets[i][:3, :3].T @ T[i][:3, :3])) for i in range(2)])
        mg = np.minimum(q - self.lo, self.hi - q) / (self.hi - self.lo); keep = np.array([("joint5" not in n) or not lock for n in self.full.names])
        margin = np.array([mg[QI[a]][keep[QI[a]]].min() for a in ARMS])
        return q, pe, re, it, margin


def ok_of(mode, pe, re):
    return (pe < POS_OK_MM) if mode == "pos" else ((pe < POS_OK_MM) & (re < ROT_OK_DEG))


def episodes_rows(d, e):
    return np.flatnonzero(d["ep"] == e)


def run_traj(ik, mode, d, rows, T0, q0, cap=None):
    """continuous IK along consecutive dataset rows; targets T0[arm] @ S_rel(row). Returns per-row records."""
    rows = rows if cap is None else rows[:cap]; q = q0.copy(); out = []
    for r in rows:
        tg = [T0[i] @ pose9_to_T(d["state"][r, ST[a]]) for i, a in enumerate(ARMS)]
        qn, pe, re, it, mg = ik.solve(mode, q, tg)
        if np.isfinite(qn).all(): q = qn
        out.append(dict(row=int(r), q=qn, pe=pe, re=re, it=it, mg=mg, tg=tg))
    return out


def main(a):
    t0 = time.time(); OUT = pathlib.Path(a.output).expanduser(); (OUT / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed); ik = IK(); import torch, rebot_fk_torch
    D = {"ego": load(a.ego), "robot": load(a.robot)}; Rr = D["robot"]; R = collections.OrderedDict()
    fkt = rebot_fk_torch.ReBotFKTorch(device="cpu", dtype=torch.float64)

    # ------------------------------------------------ 15 / 9: tool-frame + FK consistency on R312c, before anything else
    idx = np.sort(rng.choice(len(Rr["state"]), 2000, replace=False)); first = {int(e): int(np.flatnonzero(Rr["ep"] == e)[0]) for e in np.unique(Rr["ep"])}
    sp, sr, ap_, ar_, pk = [], [], [], [], []
    for r in idx:
        q = Rr["q"][r]; q0 = Rr["q"][first[int(Rr["ep"][r])]]; T, T00 = ik.fk(q), ik.fk(q0)
        Tt = fkt.tcp(torch.tensor(q[None])).numpy()[0]; pk.append(max(np.linalg.norm(T[i][:3, 3] - Tt[i][:3, 3]) for i in range(2)) * 1e3)
        for i, a_ in enumerate(ARMS):
            S = np.linalg.inv(T00[i]) @ T[i]; Ss = pose9_to_T(Rr["state"][r, ST[a_]])
            sp.append(np.linalg.norm(S[:3, 3] - Ss[:3, 3]) * 1e3); sr.append(np.degrees(rotation_angle(S[:3, :3].T @ Ss[:3, :3])))
            for k in (1, 8, 16):
                Tf = ik.fk(q + Rr["action"][r, k - 1, 20:32])[i]; A = np.linalg.inv(T[i]) @ Tf; As = pose9_to_T(Rr["action"][r, k - 1, AC[a_]])
                ap_.append(np.linalg.norm(A[:3, 3] - As[:3, 3]) * 1e3); ar_.append(np.degrees(rotation_angle(A[:3, :3].T @ As[:3, :3])))
    R["robot_fk_consistency"] = dict(pink_fk_vs_rebot_fk_torch_mm=pct(pk, (50, 99)), state_from_fk_vs_stored_pos_mm=pct(sp, (50, 95, 99)), state_rot_deg=pct(sr, (50, 95, 99)),
                                     cart20_from_fk_q_plus_dq_vs_stored_pos_mm=pct(ap_, (50, 95, 99)), cart20_rot_deg=pct(ar_, (50, 95, 99)),
                                     note="q_future = q_t + stored dq12 (k = 1, 8, 16); the stored REL is the follower-measured TCP, dq12 the follower q")
    # tool-frame axis check: rotation-vector component spread of A_k in the TCP frame (an axis permutation would permute these)
    tf = {}
    for n, d in D.items():
        for a_ in ARMS:
            Rk = pose9_to_T(d["action"][::7, 7, AC[a_]].astype(np.float64))[:, :3, :3]
            v = np.stack([Rk[:, 2, 1] - Rk[:, 1, 2], Rk[:, 0, 2] - Rk[:, 2, 0], Rk[:, 1, 0] - Rk[:, 0, 1]], 1) / 2
            tr = d["action"][::7, 7, AC[a_]][:, :3]
            tf[f"{n}_{a_}"] = dict(rotvec_std_xyz_deg=np.degrees(v.std(0)).round(2).tolist(), trans_std_xyz_mm=(1e3 * tr.std(0)).round(1).tolist())
    R["tool_frame_axis_check_k8"] = tf

    # ------------------------------------------------ robot start poses + episode assignment
    eps_r = np.unique(Rr["ep"]); starts = {int(e): (Rr["q"][first[int(e)]], ik.fk(Rr["q"][first[int(e)]])) for e in eps_r}
    eps_e = np.unique(D["ego"]["ep"]); sel_e = np.sort(rng.choice(eps_e, min(a.episodes, len(eps_e)), replace=False))
    sel_r = np.sort(rng.choice(eps_r, min(a.episodes, len(eps_r)), replace=False))
    assignA = {int(e): int(rng.choice(eps_r)) for e in sel_e}; assignB = {int(e): int(rng.choice(eps_r)) for e in sel_e[: max(10, len(sel_e) // 2)]}
    R["setup"] = dict(ego_episodes=sel_e.tolist(), robot_episodes=sel_r.tolist(), assignment_A=assignA, assignment_B=assignB, success=f"pos < {POS_OK_MM} mm & rot < {ROT_OK_DEG} deg (pos-only: pos only)",
                      ik="pink_ik.PinkIK deployment settings, V4_PINK_LOCK=joint5", robot_cap_rows_per_episode=a.robot_cap)

    # ------------------------------------------------ 2/3/4/12/13 trajectory IK (state poses)
    traj = collections.defaultdict(list)
    for mode in ("full", "pos", "full_nolock", "yaw"):
        for e in sel_e:
            q0, T0 = starts[assignA[int(e)]]; traj[("ego", mode)] += run_traj(ik, mode, D["ego"], episodes_rows(D["ego"], e), T0, q0)
        for e in sel_r:
            q0, T0 = starts[int(e)]; traj[("robot", mode)] += run_traj(ik, mode, Rr, episodes_rows(Rr, e), T0, q0, cap=a.robot_cap)
        print(f"traj {mode} done {time.time() - t0:.0f}s", flush=True)
    for e in assignB:
        q0, T0 = starts[assignB[e]]; traj[("egoB", "full")] += run_traj(ik, "full", D["ego"], episodes_rows(D["ego"], e), T0, q0)
    feas = {}
    for (n, mode), recs in traj.items():
        pe = np.array([r["pe"] for r in recs]); re = np.array([r["re"] for r in recs]); it = np.array([r["it"] for r in recs]); mg = np.array([r["mg"] for r in recs])
        ok = ok_of(mode, pe, re); conv = (pe < 0.5 + 1e-9) & ((re < 0.5 + 1e-9) | (mode == "pos"))
        for i, a_ in enumerate(ARMS):
            feas[f"{n}_{mode}_{a_}"] = dict(n=int(len(pe)), success=float(ok[:, i].mean()), converged_0p5=float(conv[:, i].mean()),
                                            pos_err_mm=pct(pe[np.isfinite(pe[:, i]), i], (50, 90, 95, 99)), rot_err_deg=pct(re[np.isfinite(re[:, i]), i], (50, 90, 95, 99)),
                                            iters=pct(it, (50, 95)), margin_lt_10pct=float((mg[:, i] < .10).mean()), margin_lt_5pct=float((mg[:, i] < .05).mean()),
                                            margin_lt_2pct=float((mg[:, i] < .02).mean()), min_margin=pct(mg[:, i], (5, 50)))
        print(n, mode, {a_: round(feas[f"{n}_{mode}_{a_}"]["success"], 3) for a_ in ARMS}, flush=True)
    R["ik_state_feasibility"] = feas
    # 13 relaxed diagnostics on full-pose failures
    rel = {}
    for n in ("ego", "robot"):
        f_full = traj[(n, "full")]; idx_fail = {i: [j for j, r in enumerate(f_full) if not ok_of("full", r["pe"], r["re"])[i]] for i in range(2)}
        for i, a_ in enumerate(ARMS):
            nf = len(idx_fail[i]); rel[f"{n}_{a_}"] = dict(full_pose_failures=nf)
            for mode in ("pos", "yaw", "full_nolock"):
                rec = traj[(n, mode)]
                rel[f"{n}_{a_}"][f"recovered_by_{mode}"] = float(np.mean([ok_of(mode, rec[j]["pe"], rec[j]["re"])[i] for j in idx_fail[i]])) if nf else None
    R["relaxed_recovery_of_full_failures"] = rel

    # ------------------------------------------------ 5/6/10 joint-space distribution, margin, NN
    qa = Rr["q"]; span = ik.hi - ik.lo
    qego = {a_: np.array([r["q"][QI[a_]] for r in traj[("ego", PM)] if ok_of("full", r["pe"], r["re"])[ARMS.index(a_)]]) for a_ in ARMS}
    jd = {}
    for a_ in ARMS:
        for j in range(6):
            jd[f"{a_}_j{j + 1}"] = dict(ego_ik=pct(qego[a_][:, j], (1, 5, 50, 95, 99)) if len(qego[a_]) else None, robot_actual=pct(qa[:, QI[a_]][:, j], (1, 5, 50, 95, 99)))
    R["joint_distribution"] = jd
    mgr = np.minimum(qa - ik.lo, ik.hi - qa) / span
    R["robot_actual_joint_limit_margin"] = {a_: dict(lt_10pct=float((mgr[:, QI[a_]].min(1) < .1).mean()), lt_5pct=float((mgr[:, QI[a_]].min(1) < .05).mean()),
                                                     lt_2pct=float((mgr[:, QI[a_]].min(1) < .02).mean()),
                                                     per_joint_lt_2pct=[round(float((mgr[:, QI[a_]][:, j] < .02).mean()), 3) for j in range(6)]) for a_ in ARMS}
    R["ego_ik_joint_limit_margin_per_joint_lt_2pct"] = {a_: [round(float((np.minimum(qego[a_][:, j] - ik.lo[QI[a_]][j], ik.hi[QI[a_]][j] - qego[a_][:, j]) / span[QI[a_]][j] < .02).mean()), 3) for j in range(6)]
                                                        if len(qego[a_]) else None for a_ in ARMS}
    from scipy.spatial import cKDTree
    nn = {}
    for a_ in ARMS:
        Qr = qa[:, QI[a_]] / span[QI[a_]]; Qr = Qr[np.sort(rng.choice(len(Qr), min(40000, len(Qr)), replace=False))]
        Qe = qego[a_] / span[QI[a_]]
        if not len(Qe): continue
        h = len(Qr) // 2; tr_r = cKDTree(Qr[h:])
        e2r = tr_r.query(Qe, workers=-1)[0]; r2r = tr_r.query(Qr[:h], workers=-1)[0]; r2e = cKDTree(Qe).query(Qr[:h], workers=-1)[0]
        hh = len(Qe) // 2; e2e = cKDTree(Qe[hh:]).query(Qe[:hh], workers=-1)[0]
        nn[a_] = dict(ego_ik_to_robot=pct(e2r, (50, 90, 95, 99)), robot_to_robot_baseline=pct(r2r, (50, 90, 95, 99)), robot_to_ego_ik=pct(r2e, (50, 90, 95, 99)),
                      ego_to_ego_baseline=pct(e2e, (50, 90, 95, 99)), frac_ego_ik_within_robot_p95=float((e2r <= np.percentile(r2r, 95)).mean()),
                      frac_robot_within_ego_baseline_p95=float((r2e <= np.percentile(e2e, 95)).mean()),
                      note="q / (URDF range) per joint; robot side subsampled to 40k actual frames; ego = successful full-pose IK of the selected episodes")
    R["joint_space_nn"] = nn

    # ------------------------------------------------ 7 continuity (per 15 Hz row, consecutive rows only)
    cont = {}
    def dq_rows(recs, d):
        out = {a_: [] for a_ in ARMS}
        for x, y in zip(recs[:-1], recs[1:]):
            if d["ep"][x["row"]] != d["ep"][y["row"]]: continue
            for i, a_ in enumerate(ARMS):
                if ok_of("full", x["pe"], x["re"])[i] and ok_of("full", y["pe"], y["re"])[i]: out[a_].append(np.abs(y["q"][QI[a_]] - x["q"][QI[a_]]))
        return {a_: np.array(v) for a_, v in out.items()}
    dE = dq_rows(traj[("ego", PM)], D["ego"]); dRik = dq_rows(traj[("robot", PM)], Rr)
    same = (np.diff(Rr["ep"]) == 0)
    for a_ in ARMS:
        dRa = np.abs(np.diff(qa[:, QI[a_]], axis=0))[same]
        cont[a_] = {nm: dict(max_abs_dq_per_row_rad=pct(v.max(1), (50, 95, 99)), per_joint_p95=np.percentile(v, 95, axis=0).round(4).tolist(), n=int(len(v)))
                    for nm, v in (("ego_ik", dE[a_]), ("robot_ik", dRik[a_]), ("robot_actual", dRa)) if len(v)}
        # acceleration proxy: second difference on ego-IK and robot actual
    R["continuity_per_row_66ms"] = cont

    # ------------------------------------------------ 8 horizon dq induced by CART20 (seeded at the IK / actual q_t)
    hz = {}
    for n in ("ego", "robot"):
        d = D[n]; recs = traj[(n, PM)][::a.h_stride]; acc = {(a_, k): [] for a_ in ARMS for k in KS}; okc = {(a_, k): [] for a_ in ARMS for k in KS}
        for x in recs:
            if not ok_of("full", x["pe"], x["re"]).all(): continue
            qt = x["q"]; Tt = x["tg"]
            for k in KS:
                tg = [Tt[i] @ pose9_to_T(d["action"][x["row"], k - 1, AC[a_]]) for i, a_ in enumerate(ARMS)]
                qn, pe, re, it, mg = ik.solve(PM, qt, tg); ok = ok_of("full", pe, re)
                for i, a_ in enumerate(ARMS):
                    okc[(a_, k)].append(bool(ok[i]))
                    if ok[i]: acc[(a_, k)].append(qn[QI[a_]] - qt[QI[a_]])
        for a_ in ARMS:
            for k in KS:
                v = np.array(acc[(a_, k)]); rec = dict(target_ik_success=float(np.mean(okc[(a_, k)])) if okc[(a_, k)] else None, n=int(len(v)))
                if len(v): rec.update(max_abs_dq=pct(np.abs(v).max(1), (50, 95, 99)), per_joint_abs_p95=np.percentile(np.abs(v), 95, axis=0).round(4).tolist())
                if n == "robot":
                    act = Rr["action"][:, k - 1, 20:32][:, QI[a_]]
                    rec["robot_actual_dq"] = dict(max_abs_dq=pct(np.abs(act).max(1), (50, 95, 99)), per_joint_abs_p95=np.percentile(np.abs(act), 95, axis=0).round(4).tolist())
                hz[f"{n}_{a_}_k{k}"] = rec
        print("horizon", n, "done", f"{time.time() - t0:.0f}s", flush=True)
    R["horizon_dq"] = hz

    # ------------------------------------------------ 11 workspace overlap after IK (task-relative state xyz, 2 cm voxels)
    ws = {}
    for i, a_ in enumerate(ARMS):
        A_ = D["ego"]["state"][:, ST[a_]][:, :3]; recs = traj[("ego", PM)]
        B_ = np.array([D["ego"]["state"][r["row"], ST[a_]][:3] for r in recs if ok_of("full", r["pe"], r["re"])[i]]); Aall = np.array([D["ego"]["state"][r["row"], ST[a_]][:3] for r in recs])
        C_ = Rr["state"][:, ST[a_]][:, :3]; lo = np.minimum.reduce([A_.min(0), C_.min(0)])
        vox = lambda X: set(map(tuple, np.floor((X - lo) / 0.02).astype(int))) if len(X) else set()
        VA, VAs, VB, VC = vox(A_), vox(Aall), vox(B_), vox(C_)
        ws[a_] = dict(ego_sampled_rows_ik_feasible=float(len(B_) / max(len(Aall), 1)), ego_sampled_bins_with_a_feasible_row=float(len(VB) / max(len(VAs), 1)),
                      robot_bins_covered_by_ego_raw_ALL=float(len(VA & VC) / len(VC)), robot_bins_covered_by_ego_feasible_SAMPLED=float(len(VB & VC) / len(VC)),
                      robot_bins_covered_by_ego_raw_SAMPLED=float(len(VAs & VC) / len(VC)), note="feasible set is from the IK-sampled episodes only; compare it with raw_SAMPLED")
    R["workspace_after_ik"] = ws

    R["primary_mode"] = PM; R["runtime_s"] = round(time.time() - t0, 1)
    json.dump(R, open(OUT / "kinematic_summary.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    plots(D, traj, R, qego, qa, ik, OUT / "plots")
    write_md(R, OUT / "kinematic_summary.md")
    print("KINEMATIC DONE", OUT, R["runtime_s"], "s")


def plots(D, traj, R, qego, qa, ik, P):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for i, a_ in enumerate(ARMS):
        C_ = D["robot"]["state"][:, ST[a_]][:, :3]; A_ = D["ego"]["state"][:, ST[a_]][:, :3]
        rec = traj[("ego", PM)]; X = np.array([D["ego"]["state"][r["row"], ST[a_]][:3] for r in rec]); ok = np.array([ok_of("full", r["pe"], r["re"])[i] for r in rec])
        allx = np.concatenate([C_, A_]); lo, hi = np.percentile(allx, 0.5, 0), np.percentile(allx, 99.5, 0)
        fig, axs = plt.subplots(1, 3, figsize=(16, 5)); fig2, axs2 = plt.subplots(1, 3, figsize=(16, 5))
        for j, (p, q) in enumerate(((0, 1), (0, 2), (1, 2))):
            sr = np.random.default_rng(0).choice(len(C_), min(20000, len(C_)), replace=False); se = np.random.default_rng(0).choice(len(A_), min(20000, len(A_)), replace=False)
            axs[j].scatter(C_[sr, p], C_[sr, q], s=1, alpha=.2, color=COLOR["robot"], label="R312c"); axs[j].scatter(A_[se, p], A_[se, q], s=1, alpha=.2, color=COLOR["ego"], label="ego")
            axs2[j].scatter(X[~ok, p], X[~ok, q], s=2, alpha=.4, color="red", label="ego IK fail"); axs2[j].scatter(X[ok, p], X[ok, q], s=2, alpha=.3, color=COLOR["ego_feasible"], label="ego IK ok")
            for ax in (axs[j], axs2[j]): ax.set_xlim(lo[p], hi[p]); ax.set_ylim(lo[q], hi[q]); ax.set_xlabel("xyz"[p]); ax.set_ylabel("xyz"[q]); ax.legend(markerscale=6, fontsize=7)
        fig.suptitle(f"{a_} task-relative EEF state (common axes)"); fig.tight_layout(); fig.savefig(P / f"eef_workspace_overlay_{a_}.png", dpi=80); plt.close(fig)
        fig2.suptitle(f"{a_} ego full-pose IK (deploy Pink, joint5 locked) on the task-relative state"); fig2.tight_layout(); fig2.savefig(P / f"ik_success_workspace_{a_}.png", dpi=80); plt.close(fig2)
    for key, nm in (("pe", "position"), ("re", "rotation")):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for i, a_ in enumerate(ARMS):
            for n, c in (("ego", COLOR["ego"]), ("robot", COLOR["robot"])):
                v = np.array([r[key][i] for r in traj[(n, PM)]]); v = v[np.isfinite(v)]
                axs[i].hist(np.log10(np.maximum(v, 1e-4)), bins=80, histtype="step", density=True, color=c, label=n, lw=1.5)
            axs[i].axvline(np.log10(POS_OK_MM if key == "pe" else ROT_OK_DEG), color="k", ls=":"); axs[i].set_title(f"{a_} IK {nm} error, log10({'mm' if key == 'pe' else 'deg'})"); axs[i].legend()
        fig.tight_layout(); fig.savefig(P / f"ik_{nm}_error_hist.png", dpi=90); plt.close(fig)
    for a_ in ARMS:
        fig, axs = plt.subplots(1, 6, figsize=(22, 3.5))
        for j in range(6):
            lo, hi = ik.lo[QI[a_]][j], ik.hi[QI[a_]][j]
            axs[j].hist(qa[:, QI[a_]][:, j], bins=60, range=(lo, hi), density=True, histtype="step", color=COLOR["robot"], label="R312c actual", lw=1.5)
            if len(qego[a_]): axs[j].hist(qego[a_][:, j], bins=60, range=(lo, hi), density=True, histtype="step", color=COLOR["ego"], label="ego IK", lw=1.5)
            axs[j].set_title(f"{a_} j{j + 1} [{lo:.2f}, {hi:.2f}]"); axs[j].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(P / f"joint_distribution_{a_}.png", dpi=80); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    span = ik.hi - ik.lo; mgr = np.minimum(qa - ik.lo, ik.hi - qa) / span
    for i, a_ in enumerate(ARMS):
        me = np.array([r["mg"][i] for r in traj[("ego", PM)] if ok_of("full", r["pe"], r["re"])[i]])
        axs[i].hist(mgr[:, QI[a_]][:, [0, 1, 2, 3, 5]].min(1), bins=60, range=(0, .5), density=True, histtype="step", color=COLOR["robot"], label="R312c actual (excl. j5)")
        if len(me): axs[i].hist(me, bins=60, range=(0, .5), density=True, histtype="step", color=COLOR["ego"], label="ego IK (excl. locked j5)")
        axs[i].set_title(f"{a_} min normalized joint-limit margin"); axs[i].legend()
    fig.tight_layout(); fig.savefig(P / "joint_limit_margin.png", dpi=90); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for i, a_ in enumerate(ARMS):
        if not len(qego[a_]): continue
        Qr = qa[np.random.default_rng(0).choice(len(qa), 20000, replace=False)][:, QI[a_]] / span[QI[a_]]; Qe = qego[a_] / span[QI[a_]]
        Z = np.concatenate([Qr, Qe]); mu = Z.mean(0); V = np.linalg.svd(Z - mu, full_matrices=False)[2][:2]
        axs[i].scatter(*((Qr - mu) @ V.T).T, s=1, alpha=.2, color=COLOR["robot"], label="R312c actual q"); axs[i].scatter(*((Qe - mu) @ V.T).T, s=1, alpha=.3, color=COLOR["ego"], label="ego IK q")
        axs[i].set_title(f"{a_} joint-space PCA (union, q / range)"); axs[i].legend(markerscale=6)
    fig.tight_layout(); fig.savefig(P / "egoik_vs_r312c_q_pca.png", dpi=90); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i, a_ in enumerate(ARMS):
        for n, c, key in (("ego", COLOR["ego"], None), ("robot", COLOR["robot"], None), ("robot", "k", "robot_actual_dq")):
            ys = []
            for k in KS:
                rec = R["horizon_dq"][f"{n}_{a_}_k{k}"]; src = rec.get(key) if key else rec
                ys.append(src["max_abs_dq"]["p95"] if src and "max_abs_dq" in src else np.nan)
            axs[i].plot(KS, ys, marker="o", color=c, label=f"{n} {'actual dq' if key else 'IK dq'} p95")
        axs[i].set_xticks(KS); axs[i].set_title(f"{a_} max |dq| over joints by horizon (rad)"); axs[i].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(P / "delta_q_by_horizon.png", dpi=90); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4)); labels, ys, cs = [], [], []
    for n in ("ego", "robot"):
        for a_ in ARMS:
            for mode in ("full", "yaw", "full_nolock", "pos"):
                labels.append(f"{n}\n{a_} {mode}"); ys.append(R["ik_state_feasibility"][f"{n}_{mode}_{a_}"]["success"]); cs.append(COLOR[n])
    ax.bar(range(len(ys)), ys, color=cs); ax.set_xticks(range(len(ys))); ax.set_xticklabels(labels, fontsize=6); ax.set_ylim(0, 1.05); ax.set_title("IK success on state poses by mode")
    fig.tight_layout(); fig.savefig(P / "fullpose_vs_positiononly_ik.png", dpi=90); plt.close(fig)


def write_md(R, path):
    L = []; ap = L.append; F_ = R["ik_state_feasibility"]
    ap(f"# Ego → reBot kinematic compatibility (diagnostic only), downstream metrics from mode **{PM}**\n")
    ap("IK is the deployment Pink stack with V4_PINK_LOCK=joint5. Ego task-relative poses are placed at a real R312c episode's start TCP and solved continuously. "
       f"Success means pos < {POS_OK_MM} mm and rot < {ROT_OK_DEG}°. Nothing here feeds training.\n")
    c = R["robot_fk_consistency"]
    ap("## Consistency checks (R312c)\n")
    ap(f"- Pink FK vs rebot_fk_torch: p99 {c['pink_fk_vs_rebot_fk_torch_mm']['p99']:.4f} mm")
    ap(f"- State from FK(q) vs stored: pos p99 {c['state_from_fk_vs_stored_pos_mm']['p99']:.3f} mm, rot p99 {c['state_rot_deg']['p99']:.3f}°")
    ap(f"- CART20 from FK(q+dq) vs stored: pos p50/p95/p99 {c['cart20_from_fk_q_plus_dq_vs_stored_pos_mm']['p50']:.2f} / {c['cart20_from_fk_q_plus_dq_vs_stored_pos_mm']['p95']:.2f} / {c['cart20_from_fk_q_plus_dq_vs_stored_pos_mm']['p99']:.2f} mm, rot p95 {c['cart20_rot_deg']['p95']:.2f}°")
    ap("- Tool-frame axis check (k8, std of the rotation-vector / translation components in the TCP frame):")
    for k_, v in R["tool_frame_axis_check_k8"].items(): ap(f"  - {k_}: rotvec std xyz {v['rotvec_std_xyz_deg']}°, trans std xyz {v['trans_std_xyz_mm']} mm")
    ap("\n## Main table\n\n| Metric | Ego → reBot IK | R312c reference |\n|---|---:|---:|")
    for a_ in ARMS: ap(f"| {a_} full-pose IK success | {F_[f'ego_full_{a_}']['success']:.3f} | {F_[f'robot_full_{a_}']['success']:.3f} |")
    for a_ in ARMS: ap(f"| {a_} position-only success | {F_[f'ego_pos_{a_}']['success']:.3f} | {F_[f'robot_pos_{a_}']['success']:.3f} |")
    for a_ in ARMS: ap(f"| {a_} full pose, joint5 UNLOCKED (diag) | {F_[f'ego_full_nolock_{a_}']['success']:.3f} | {F_[f'robot_full_nolock_{a_}']['success']:.3f} |")
    for a_ in ARMS: ap(f"| {a_} ori=yaw (diag) | {F_[f'ego_yaw_{a_}']['success']:.3f} | {F_[f'robot_yaw_{a_}']['success']:.3f} |")
    for a_ in ARMS: ap(f"| {a_} full-pose success, 2nd start-pose assignment | {F_[f'egoB_full_{a_}']['success']:.3f} | – |")
    for a_ in ARMS:
        ap(f"| {a_} FK pos error p95 (mm, all solves) | {F_[f'ego_full_{a_}']['pos_err_mm']['p95']:.2f} | {F_[f'robot_full_{a_}']['pos_err_mm']['p95']:.2f} |")
        ap(f"| {a_} FK rot error p95 (deg, all solves) | {F_[f'ego_full_{a_}']['rot_err_deg']['p95']:.2f} | {F_[f'robot_full_{a_}']['rot_err_deg']['p95']:.2f} |")
    rm = R["robot_actual_joint_limit_margin"]
    for a_ in ARMS: ap(f"| {a_} joint-limit margin < 5 % (excl. locked j5 for IK) | {F_[f'ego_full_{a_}']['margin_lt_5pct']:.3f} | IK {F_[f'robot_full_{a_}']['margin_lt_5pct']:.3f} / actual {rm[a_]['lt_5pct']:.3f} |")
    for a_ in ARMS:
        if a_ in R["joint_space_nn"]: v = R["joint_space_nn"][a_]; ap(f"| {a_} q NN distance p95 (ego-IK→robot / robot→robot baseline) | {v['ego_ik_to_robot']['p95']:.3f} | {v['robot_to_robot_baseline']['p95']:.3f} |")
    for k in (4, 8, 16):
        for a_ in ARMS:
            e, r = R["horizon_dq"][f"ego_{a_}_k{k}"], R["horizon_dq"][f"robot_{a_}_k{k}"]
            ap(f"| {a_} k{k} max-joint abs(dq) p95 (rad) | {e['max_abs_dq']['p95'] if 'max_abs_dq' in e else float('nan'):.3f} | IK {r['max_abs_dq']['p95'] if 'max_abs_dq' in r else float('nan'):.3f} / actual {r['robot_actual_dq']['max_abs_dq']['p95']:.3f} |")
    ap("\n## Relaxed recovery of full-pose failures (diagnostic)\n\n| Dataset, arm | Full failures | Recovered by position-only | by ori=yaw | by joint5 unlocked |\n|---|---:|---:|---:|---:|")
    for k_, v in R["relaxed_recovery_of_full_failures"].items(): ap(f"| {k_} | {v['full_pose_failures']} | {fmt3(v['recovered_by_pos'])} | {fmt3(v['recovered_by_yaw'])} | {fmt3(v['recovered_by_full_nolock'])} |")
    ap("\n## Horizon targets: IK success of T(t) @ A_k seeded at q_t\n\n| Arm | k | Ego | R312c |\n|---|---:|---:|---:|")
    for a_ in ARMS:
        for k in KS: ap(f"| {a_} | {k} | {fmt3(R['horizon_dq'][f'ego_{a_}_k{k}']['target_ik_success'])} | {fmt3(R['horizon_dq'][f'robot_{a_}_k{k}']['target_ik_success'])} |")
    ap("\n## Continuity (per 66.7 ms row, max abs(dq) over joints, rad)\n\n| Arm | Ego IK p50/p95/p99 | R312c IK | R312c actual |\n|---|---|---|---|")
    for a_ in ARMS:
        cc = R["continuity_per_row_66ms"][a_]; g_ = lambda nm: (lambda v: f"{v['p50']:.4f} / {v['p95']:.4f} / {v['p99']:.4f}")(cc[nm]["max_abs_dq_per_row_rad"]) if nm in cc else "–"
        ap(f"| {a_} | {g_('ego_ik')} | {g_('robot_ik')} | {g_('robot_actual')} |")
    ap("\n## Joint-space overlap\n\n| Arm | ego-IK→robot p50/p95 | robot→robot p50/p95 | Ego-IK within robot p95 | robot→ego-IK p50/p95 | Robot within ego baseline p95 |\n|---|---|---|---:|---|---:|")
    for a_, v in R["joint_space_nn"].items():
        ap(f"| {a_} | {v['ego_ik_to_robot']['p50']:.3f} / {v['ego_ik_to_robot']['p95']:.3f} | {v['robot_to_robot_baseline']['p50']:.3f} / {v['robot_to_robot_baseline']['p95']:.3f} | {v['frac_ego_ik_within_robot_p95']:.3f} | {v['robot_to_ego_ik']['p50']:.3f} / {v['robot_to_ego_ik']['p95']:.3f} | {v['frac_robot_within_ego_baseline_p95']:.3f} |")
    ap(f"\nRobot actual per-joint fraction within 2 % of a limit: {json.dumps({a_: v['per_joint_lt_2pct'] for a_, v in rm.items()})}; ego IK: {json.dumps(R['ego_ik_joint_limit_margin_per_joint_lt_2pct'])}")
    ap("\n## Workspace after IK (task-relative, 2 cm voxels)\n")
    for a_, v in R["workspace_after_ik"].items(): ap(f"- {a_}: {json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})}")
    open(path, "w").write("\n".join(L) + "\n")


def fmt3(x): return "–" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ego", required=True); ap.add_argument("--robot", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--episodes", type=int, default=40); ap.add_argument("--robot-cap", type=int, default=250); ap.add_argument("--h-stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--primary-mode", default="full", choices=["full", "full_nolock"], help="mode whose successful solves feed the downstream metrics")
    args = ap.parse_args(); PM = args.primary_mode; main(args)
